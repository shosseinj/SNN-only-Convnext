"""Structural memory-safety checks for discrete TTFS ConvNeXt."""
from __future__ import annotations

import json

import torch
import torch.nn as nn

from models.discrete_ttfs_convnext import build_discrete_ttfs_convnext
from trainer import cuda_memory_snapshot, regional_gradient_norms


def verify_shape_only_initialization(model, images):
    calls = {"conv": 0, "norm": 0}
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(
                lambda *_args: calls.__setitem__("conv", calls["conv"] + 1)
            ))
        elif isinstance(module, nn.GroupNorm):
            handles.append(module.register_forward_hook(
                lambda *_args: calls.__setitem__("norm", calls["norm"] + 1)
            ))

    shape = tuple(images.shape)
    states = []
    for stage_index in range(4):
        downsample = model.downsamples[stage_index]
        state = downsample.init_state(shape, images.device, images.dtype, False)
        states.append(state)
        shape = tuple(state.membrane.shape)
        for block in model.stages[stage_index]:
            block_state = block.init_states(shape, images.device, images.dtype, False)
            states.extend(block_state[:3])
            assert block_state[4] is None
    for handle in handles:
        handle.remove()
    assert calls == {"conv": 0, "norm": 0}, calls
    assert all(state.first_spike is None for state in states)
    assert all(state.has_spiked.dtype == torch.bool for state in states)
    return calls


def run_size(model_size: str, device: torch.device) -> dict:
    torch.manual_seed(42)
    model = build_discrete_ttfs_convnext(
        model_size=model_size, time_steps=2, num_classes=10,
        readout_mode="spike_integrator", cifar_stem=True, current_norm=True,
        track_first_spike=False, use_checkpointing=False,
        threshold=0.2, threshold_mode="learnable_channel",
    ).to(device)
    images = torch.rand(2, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1], device=device)
    init_calls = verify_shape_only_initialization(model, images)

    captured = {}
    handle = model.classifier.register_forward_pre_hook(
        lambda _module, inputs: captured.update(feature=inputs[0].detach())
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if amp else torch.bfloat16,
        enabled=amp,
    ):
        logits, stats = model(images, return_stats=True)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    assert logits.shape == (2, 10)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    if amp:
        # No optimizer is needed: invert the scale for direct gradient checks.
        scale = scaler.get_scale()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(scale)
    norms = regional_gradient_norms(model)
    assert torch.isfinite(model.classifier.weight.grad).all()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert norms["stem"] > 0 and all(norms[f"stage_{i}"] > 0 for i in range(4)), norms
    assert all(torch.isfinite(torch.tensor(value)) for value in norms.values())
    assert all(
        stage["repeated_spike_ratio"] == 0.0
        for stage in stats["stage_spike_distribution"].values()
    )
    assert stats["classifier_input_source"] == "weighted_final_stage_hard_spikes_only"
    feature = captured["feature"]
    assert torch.all((feature >= 0) & (feature <= 1))
    final_spatial_size = model.downsamples[-1].output_shape((2, 384, 8, 8))[-2:]
    quantum = model.time_steps * final_spatial_size[0] * final_spatial_size[1]
    assert torch.allclose(feature * quantum, (feature * quantum).round(), atol=1e-3)
    handle.remove()
    return {
        "model_size": model_size,
        "loss": float(loss.detach().item()),
        "regional_gradient_norms": norms,
        "state_init_calls": init_calls,
        "global_sparsity": stats["global_sparsity"],
        "repeated_spike_ratio": max(
            stage["repeated_spike_ratio"]
            for stage in stats["stage_spike_distribution"].values()
        ),
        "amp": amp,
        "cuda_memory": cuda_memory_snapshot(device),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [run_size("nano", device), run_size("tiny", device)]
    print(json.dumps(results, indent=2))
    print("PASS: memory-safe nano and tiny structural checks")


if __name__ == "__main__":
    main()
