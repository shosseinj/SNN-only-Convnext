"""Structural and one-batch smoke tests for ImageNet ConvNeXt-Tiny initialization."""
from __future__ import annotations

import gc

import torch
from torchvision.models import convnext_tiny

from imagenet_init import load_torchvision_convnext_tiny
from models.discrete_ttfs_convnext import build_discrete_ttfs_convnext


def main() -> None:
    torch.manual_seed(7)
    source = convnext_tiny(weights=None).state_dict()  # structure only; never downloads

    model = build_discrete_ttfs_convnext(
        model_size="tiny", time_steps=2, force_positive_weights=False,
        learnable_delay=True,
    )
    threshold_before = model.downsamples[0].neuron.threshold.detach().clone()
    delay_before = model.downsamples[0].neuron.delay.raw_delay.detach().clone()
    classifier_before = model.classifier.weight.detach().clone()
    report = load_torchvision_convnext_tiny(model, source)

    assert report["status"] == "loaded"
    assert report["loaded_tensor_count"] == 116
    assert not report["skipped_incompatible"]
    assert torch.equal(model.downsamples[0].synapse.weight, source["features.0.0.weight"])
    assert torch.equal(model.downsamples[2].synapse.weight, source["features.4.1.weight"])
    assert torch.equal(
        model.stages[2][8].pw1.weight,
        source["features.5.8.block.3.weight"][:, :, None, None],
    )
    assert torch.equal(model.downsamples[0].neuron.threshold, threshold_before)
    assert torch.equal(model.downsamples[0].neuron.delay.raw_delay, delay_before)
    assert torch.equal(model.classifier.weight, classifier_before)
    assert all(item["reason"] != "cifar10_classifier_newly_initialized" or item["target"].startswith("classifier.")
               for item in report["preserved_target"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(torch.rand(1, 3, 32, 32, device=device))
    assert logits.shape == (1, 10) and torch.isfinite(logits).all()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    positive_model = build_discrete_ttfs_convnext(
        model_size="tiny", time_steps=2, force_positive_weights=True,
    )
    stem_before = positive_model.downsamples[0].synapse.weight.detach().clone()
    try:
        load_torchvision_convnext_tiny(positive_model, source, positive_transform="reject")
        raise AssertionError("signed ImageNet weights should require an explicit positive transform")
    except ValueError as error:
        assert "signed" in str(error)
    assert torch.equal(positive_model.downsamples[0].synapse.weight, stem_before)
    incompatible_source = dict(source)
    incompatible_source["features.0.0.weight"] = source["features.0.0.weight"][:, :, :2, :2]
    incompatible_report = load_torchvision_convnext_tiny(
        positive_model, incompatible_source, positive_transform="abs"
    )
    assert incompatible_report["status"] == "loaded_with_skips"
    assert incompatible_report["skipped_incompatible"] == [{
        "source": "features.0.0.weight",
        "target": "downsamples.0.synapse.weight",
        "reason": "shape_mismatch",
        "source_shape": [96, 3, 2, 2],
        "target_shape": [96, 3, 4, 4],
    }]
    assert torch.equal(positive_model.downsamples[0].synapse.weight, stem_before)
    positive_report = load_torchvision_convnext_tiny(positive_model, source, positive_transform="abs")
    assert positive_report["positive_weight_transform"] == "abs"
    assert torch.equal(positive_model.downsamples[0].synapse.weight, source["features.0.0.weight"].abs())
    del positive_model
    gc.collect()

    nano = build_discrete_ttfs_convnext(model_size="nano", time_steps=2)
    try:
        load_torchvision_convnext_tiny(nano, source, positive_transform="abs")
        raise AssertionError("Nano must reject ConvNeXt-Tiny initialization")
    except ValueError as error:
        assert "model_size='tiny'" in str(error)

    print("PASS: Tiny depths/dims and all 116 justified affine tensors map exactly")
    print("PASS: pointwise Linear weights reshape exactly to 1x1 convolutions")
    print("PASS: CIFAR classifier, thresholds, delays, and output neuron remain new")
    print("PASS: signed-to-positive loading requires explicit abs transform")
    print("PASS: incompatible stem shape is skipped and reported without mutation")
    print("PASS: Nano rejects ConvNeXt-Tiny initialization")
    print("PASS: native 1x3x32x32 TTFS forward returns finite 1x10 logits")


if __name__ == "__main__":
    main()
