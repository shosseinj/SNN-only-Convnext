"""One-batch, one-epoch smoke tests for fixed and learnable TTFS thresholds."""
from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from models.discrete_ttfs_convnext import build_discrete_ttfs_convnext
from trainer import set_threshold_trainable, threshold_parameters, threshold_values


def run_case(mode: str, images: torch.Tensor, labels: torch.Tensor) -> dict:
    torch.manual_seed(42)
    model = build_discrete_ttfs_convnext(
        model_size="tiny", time_steps=2, threshold=0.2,
        threshold_mode=mode, threshold_min=0.05, threshold_max=0.8,
        residual=True, learnable_delay=True, force_positive_weights=False,
    ).to(images.device)
    params = threshold_parameters(model)
    assert torch.allclose(threshold_values(model), torch.full_like(threshold_values(model), 0.2), atol=1e-7)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.05)
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert all(optimizer_ids.count(id(p)) == 1 for p in params)

    # This smoke case represents an epoch after the configured freeze period.
    set_threshold_trainable(model, True)
    before = threshold_values(model).clone()
    optimizer.zero_grad(set_to_none=True)
    logits, stats = model(images, return_stats=True)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert logits.shape == (images.shape[0], 10)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad.detach() for p in params if p.grad is not None]
    gradient_mean = float(torch.cat([g.flatten() for g in gradients]).abs().mean().item()) if gradients else 0.0
    optimizer.step()
    after = threshold_values(model)
    update = float((after - before).abs().mean().item())

    for stage in stats["stage_spike_distribution"].values():
        assert stage["repeated_spike_ratio"] <= 1e-7
        assert all(torch.isfinite(torch.tensor(value)) for value in stage.values())
    if mode == "fixed":
        assert not params and gradient_mean == 0.0 and update == 0.0
    else:
        assert params and gradient_mean > 0.0 and update > 0.0

    result = {
        "mode": mode,
        "loss": float(loss.item()),
        "threshold_gradient_mean": gradient_mean,
        "threshold_update_magnitude": update,
        "threshold_mean": float(after.mean().item()),
        "output_shape": list(logits.shape),
        "stage_spike_distribution": stats["stage_spike_distribution"],
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CIFAR10("../cifar_data", train=True, download=False, transform=ToTensor())
    images, labels = next(iter(DataLoader(Subset(dataset, range(2)), batch_size=2, shuffle=False, num_workers=0)))
    images, labels = images.to(device), labels.to(device)
    results = [run_case("fixed", images, labels), run_case("learnable_channel", images, labels)]
    print(json.dumps(results, indent=2))
    print("PASS: fixed and learnable-channel Tiny T=2 one-epoch smoke tests")


if __name__ == "__main__":
    main()
