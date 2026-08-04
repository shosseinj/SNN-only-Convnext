"""Controlled torchvision ConvNeXt-Tiny -> discrete TTFS Tiny initialization."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from models.discrete_ttfs_convnext import DiscreteTTFSConvNeXt, PositiveConv2d


TINY_DEPTHS = (3, 3, 9, 3)
TINY_DIMS = (96, 192, 384, 768)


def _architecture_signature(model: DiscreteTTFSConvNeXt) -> tuple[tuple[int, ...], tuple[int, ...]]:
    depths = tuple(len(stage) for stage in model.stages)
    dims = tuple(downsample.synapse.out_channels for downsample in model.downsamples)
    return depths, dims


def _mapping() -> list[tuple[str, str, str]]:
    """Return (source, target, transform) entries for justified affine mappings."""
    entries: list[tuple[str, str, str]] = []
    downsample_features = (0, 2, 4, 6)
    for stage, feature in enumerate(downsample_features):
        source_prefix = "features.0.0" if stage == 0 else f"features.{feature}.1"
        target_prefix = f"downsamples.{stage}.synapse"
        entries.extend([
            (f"{source_prefix}.weight", f"{target_prefix}.weight", "identity"),
            (f"{source_prefix}.bias", f"{target_prefix}.bias", "identity"),
        ])
    for stage, (feature, depth) in enumerate(zip((1, 3, 5, 7), TINY_DEPTHS)):
        for block in range(depth):
            source_prefix = f"features.{feature}.{block}.block"
            target_prefix = f"stages.{stage}.{block}"
            entries.extend([
                (f"{source_prefix}.0.weight", f"{target_prefix}.dw.weight", "identity"),
                (f"{source_prefix}.0.bias", f"{target_prefix}.dw.bias", "identity"),
                (f"{source_prefix}.3.weight", f"{target_prefix}.pw1.weight", "linear_to_conv1x1"),
                (f"{source_prefix}.3.bias", f"{target_prefix}.pw1.bias", "identity"),
                (f"{source_prefix}.5.weight", f"{target_prefix}.pw2.weight", "linear_to_conv1x1"),
                (f"{source_prefix}.5.bias", f"{target_prefix}.pw2.bias", "identity"),
            ])
    return entries


def load_torchvision_convnext_tiny(
    model: DiscreteTTFSConvNeXt,
    source_state_dict: Mapping[str, torch.Tensor] | None = None,
    positive_transform: str = "reject",
) -> dict[str, Any]:
    """Partially initialize a compatible TTFS Tiny and return a detailed report.

    If ``source_state_dict`` is omitted, torchvision's official ImageNet-1K V1
    weights are requested (and may be downloaded by torchvision). No mutation
    occurs until every proposed tensor has been validated.
    """
    depths, dims = _architecture_signature(model)
    if (depths, dims) != (TINY_DEPTHS, TINY_DIMS):
        raise ValueError(
            "ImageNet ConvNeXt-Tiny initialization requires model_size='tiny'; "
            f"received depths={depths}, dims={dims}"
        )
    if positive_transform not in {"reject", "abs"}:
        raise ValueError("positive_transform must be 'reject' or 'abs'")
    positive_targets = any(
        module.force_positive_weights for module in model.modules() if isinstance(module, PositiveConv2d)
    )
    if positive_targets and positive_transform == "reject":
        raise ValueError(
            "ImageNet weights are signed but this model enforces non-negative convolution weights. "
            "Set --imagenet_positive_transform abs explicitly, or train with "
            "--force_positive_weights false."
        )

    source_name = "provided_state_dict"
    if source_state_dict is None:
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        source_state_dict = convnext_tiny(weights=weights).state_dict()
        source_name = "torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1"

    target_state = model.state_dict()
    proposed: dict[str, torch.Tensor] = {}
    loaded: list[dict[str, Any]] = []
    skipped_incompatible: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    mapped_targets: set[str] = set()

    for source_key, target_key, reshape in _mapping():
        mapped_targets.add(target_key)
        if source_key not in source_state_dict:
            skipped_incompatible.append({
                "source": source_key, "target": target_key, "reason": "source_key_missing"
            })
            continue
        if target_key not in target_state:
            skipped_incompatible.append({
                "source": source_key, "target": target_key, "reason": "target_key_missing"
            })
            continue
        tensor = source_state_dict[source_key].detach()
        transforms: list[str] = []
        if reshape == "linear_to_conv1x1":
            tensor = tensor[:, :, None, None]
            transforms.append("linear_to_conv1x1")
        target_module_name = target_key.rsplit(".", 1)[0]
        target_module = model.get_submodule(target_module_name)
        if (
            target_key.endswith(".weight")
            and isinstance(target_module, PositiveConv2d)
            and target_module.force_positive_weights
        ):
            tensor = tensor.abs()
            transforms.append("absolute_value_for_non_negative_parameterization")
        if tensor.shape != target_state[target_key].shape:
            skipped_incompatible.append({
                "source": source_key,
                "target": target_key,
                "reason": "shape_mismatch",
                "source_shape": list(tensor.shape),
                "target_shape": list(target_state[target_key].shape),
            })
            continue
        proposed[target_key] = tensor.to(dtype=target_state[target_key].dtype)
        used_sources.add(source_key)
        loaded.append({
            "source": source_key,
            "target": target_key,
            "shape": list(tensor.shape),
            "transforms": transforms,
        })

    # Apply only after all entries have been inspected, preventing partial
    # mutation if validation above raises.
    with torch.no_grad():
        for target_key, tensor in proposed.items():
            target_state[target_key].copy_(tensor)

    skipped_source = []
    for key, tensor in source_state_dict.items():
        if key in used_sources:
            continue
        if key.startswith("classifier."):
            reason = "imagenet_classifier_or_final_norm_not_used"
        elif "block.2." in key or (key.startswith("features.") and key.split(".")[-2] == "0"):
            reason = "normalization_not_present_in_snn"
        elif key.endswith("layer_scale"):
            reason = "layer_scale_not_present_in_snn"
        else:
            reason = "no_justified_snn_mapping"
        skipped_source.append({"source": key, "shape": list(tensor.shape), "reason": reason})

    preserved_targets = []
    for key, tensor in target_state.items():
        if key not in proposed:
            if ".threshold" in key or "raw_threshold" in key:
                reason = "snn_threshold_newly_initialized"
            elif ".delay." in key:
                reason = "snn_delay_newly_initialized"
            elif key.startswith("classifier."):
                reason = "cifar10_classifier_newly_initialized"
            elif key.startswith("output_neuron."):
                reason = "snn_output_neuron_newly_initialized"
            elif key in mapped_targets:
                reason = "mapped_source_incompatible_or_missing"
            else:
                reason = "no_justified_imagenet_mapping"
            preserved_targets.append({"target": key, "shape": list(tensor.shape), "reason": reason})

    return {
        "status": "loaded_with_skips" if skipped_incompatible else "loaded",
        "source": source_name,
        "architecture": {"depths": list(depths), "dims": list(dims)},
        "positive_weight_transform": "abs" if positive_targets else "none",
        "loaded_tensor_count": len(loaded),
        "loaded_parameter_count": int(sum(proposed[k].numel() for k in proposed)),
        "loaded": loaded,
        "skipped_incompatible": skipped_incompatible,
        "skipped_source": skipped_source,
        "preserved_target": preserved_targets,
    }
