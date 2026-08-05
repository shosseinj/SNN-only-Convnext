#!/usr/bin/env python3
"""Deterministic 32-sample TTFS learnability and component ablation test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from models.discrete_ttfs_convnext import (
    FirstSpikeNeuron,
    SurrogateStep,
    build_discrete_ttfs_convnext,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="../cifar_data")
    p.add_argument("--output", default="./results/tiny_overfit/results.json")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def fixed_batch(data_path: str, device: torch.device):
    dataset = CIFAR10(data_path, train=True, download=False, transform=ToTensor())
    loader = DataLoader(Subset(dataset, range(32)), batch_size=32, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)


def structural_checks(device):
    # Exact surrogate derivative at and away from threshold.
    x = torch.tensor([-0.2, 0.0, 0.2], device=device, requires_grad=True)
    SurrogateStep.apply(x, 5.0).sum().backward()
    expected = 1.0 / (1.0 + 5.0 * x.detach().abs()).square()
    assert torch.allclose(x.grad, expected)

    # State persists and the hard first-spike mask permits at most one event.
    neuron = FirstSpikeNeuron(1, 2, threshold=0.5, learnable_delay=False).to(device)
    current = torch.tensor([[0.3]], device=device)
    state = neuron.init_state(current)
    emitted = []
    for step in range(2):
        spike, state = neuron.forward_step(current, state, step)
        emitted.append(spike)
    assert torch.allclose(state.membrane, torch.tensor([[0.6]], device=device))
    assert torch.stack(emitted).sum().item() == 1.0
    assert state.first_spike.item() == 1.0
    return {"surrogate_gradients": x.grad.detach().cpu().tolist(), "state_membrane": state.membrane.item()}


def run_case(name, images, labels, time_steps, residual, delay, args):
    torch.manual_seed(args.seed)
    model = build_discrete_ttfs_convnext(
        model_size="nano", time_steps=time_steps, threshold=0.05,
        learnable_delay=delay, residual=residual,
        force_positive_weights=False,
    ).to(images.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    initial_logits, initial_stats = model(images, return_stats=True)
    initial_loss = criterion(initial_logits, labels)
    initial_loss.backward()
    grad_values = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
    delay_grads = [p.grad.detach().flatten() for n, p in model.named_parameters() if "raw_delay" in n and p.grad is not None]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    parameter_deltas = {
        name: float((parameter.detach() - before[name]).abs().max().item())
        for name, parameter in model.named_parameters()
    }
    changed_name = max(parameter_deltas, key=parameter_deltas.get)

    best = 0.0
    success_epoch = None
    final_loss = float(initial_loss.item())
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(images, return_stats=True)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        accuracy = float((logits.argmax(dim=1) == labels).float().mean().mul(100).item())
        best = max(best, accuracy)
        final_loss = float(loss.item())
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"case": name, "epoch": epoch, "loss": final_loss, "accuracy": accuracy}), flush=True)
        if accuracy >= 95.0:
            success_epoch = epoch
            break

    collapse = next(
        (layer for layer, values in initial_stats["layer_event_stats"].items()
         if values["unique_sample_ttfs"] == 1), None
    )
    return {
        "configuration": name,
        "time_steps": time_steps,
        "residual": residual,
        "learnable_delay": delay,
        "success": best >= 95.0,
        "success_epoch": success_epoch,
        "best_training_accuracy": best,
        "final_loss": final_loss,
        "initial_unique_logits": int(torch.unique(initial_logits.detach(), dim=0).shape[0]),
        "first_collapsed_layer_at_initialization": collapse,
        "gradient_abs_mean": float(grad_values.abs().mean().item()),
        "gradient_abs_max": float(grad_values.abs().max().item()),
        "gradient_nonzero_fraction": float((grad_values != 0).float().mean().item()),
        "delay_gradient_abs_mean": float(torch.cat(delay_grads).abs().mean().item()) if delay_grads else None,
        "optimizer_max_parameter_delta": parameter_deltas[changed_name],
        "optimizer_max_delta_parameter": changed_name,
        "initial_spike_stats": initial_stats["layer_event_stats"],
        "final_spike_stats": stats["layer_event_stats"],
    }


def main():
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    images, labels = fixed_batch(args.data_path, device)
    results = {"structural_checks": structural_checks(device), "cases": []}
    for case in [
        ("A", 2, False, False),
        ("B", 2, True, False),
        ("C", 2, True, True),
        ("D", 5, True, True),
    ]:
        results["cases"].append(run_case(case[0], images, labels, *case[1:], args))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {target}")
    if not results["cases"][0]["success"]:
        raise SystemExit("Configuration A failed to reach 95% training accuracy")


if __name__ == "__main__":
    main()
