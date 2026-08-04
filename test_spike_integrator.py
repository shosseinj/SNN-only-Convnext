"""Strict structural smoke test for the event-only non-spiking TTFS readout."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from models.discrete_ttfs_convnext import build_discrete_ttfs_convnext
from trainer import regional_gradient_norms


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CIFAR10("../cifar_data", train=True, download=False, transform=ToTensor())
    images, labels = next(iter(DataLoader(Subset(dataset, range(16)), batch_size=16, shuffle=False)))
    images, labels = images.to(device), labels.to(device)
    model = build_discrete_ttfs_convnext(
        model_size="tiny", time_steps=2, threshold=0.2,
        force_positive_weights=False, residual=True, learnable_delay=True,
        threshold_mode="learnable_channel", readout_mode="spike_integrator",
    ).to(device)
    assert model.classifier.in_features == 768 and model.classifier.out_features == 10
    assert sum(p.numel() for p in model.classifier.parameters()) == 7690

    captured = {}
    handle = model.classifier.register_forward_pre_hook(
        lambda _module, inputs: captured.update({"feature": inputs[0].detach()})
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.05)
    passing_norms = None
    for _step in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(images, return_stats=True, return_layer_stats=False)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert logits.shape == (16, 10)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        assert stats["classifier_input_source"] == "weighted_final_stage_hard_spikes_only"
        # At native 32x32, the final grid is 1x1. With T=2 and one hard
        # first-spike event, every classifier input must be exactly 0, .5, or 1.
        feature = captured["feature"]
        assert feature.shape == (16, 768)
        assert torch.all((feature >= 0) & (feature <= 1))
        assert torch.allclose(feature * 2.0, (feature * 2.0).round())
        assert all(
            stage["repeated_spike_ratio"] <= 1e-7
            for stage in stats["stage_spike_distribution"].values()
        )
        loss.backward()
        norms = regional_gradient_norms(model)
        if norms["classifier"] > 0 and all(norms[f"stage_{stage}"] > 0 for stage in range(4)):
            passing_norms = norms
            break
        optimizer.step()
    handle.remove()
    assert passing_norms is not None, f"gradient-flow check failed: {norms}"
    print("PASS: spike_integrator uses only weighted hard final-stage events")
    print("PASS: finite 16x10 logits/loss and backward through every stage")
    print("PASS: repeated-spike ratio is zero")
    print("classifier_parameters=7690")
    print(f"regional_gradient_norms={passing_norms}")


if __name__ == "__main__":
    main()
