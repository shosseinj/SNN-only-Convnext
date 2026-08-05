"""Focused learnable-channel threshold gradient regression test."""
import torch

from models.discrete_ttfs_convnext_design_b import FirstSpikeNeuron


def main():
    neuron = FirstSpikeNeuron(
        channels=3,
        time_steps=4,
        threshold=0.4,
        learnable_delay=False,
        threshold_mode="learnable_channel",
        threshold_min=0.1,
        threshold_max=1.2,
    )
    state = neuron.init_state((2, 3), torch.device("cpu"), torch.float32, False)
    score = torch.zeros((), dtype=torch.float32)
    currents = (
        torch.full((2, 3), 0.22),
        torch.full((2, 3), 0.16),
        torch.full((2, 3), 0.10),
        torch.full((2, 3), 0.05),
    )
    for step, current in enumerate(currents):
        spikes, state = neuron.forward_step(current, state, step)
        score = score + spikes.sum() * float(4 - step)
    score.backward()

    gradient = neuron.raw_threshold.grad
    assert neuron.raw_threshold.requires_grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0
    print({
        "raw_threshold_count": neuron.raw_threshold.numel(),
        "gradient_mean_abs": float(gradient.abs().mean().item()),
        "gradient_max_abs": float(gradient.abs().max().item()),
        "threshold_mean": float(neuron.threshold_values().mean().item()),
    })
    print("PASS: learnable-channel raw_threshold gradient is finite and non-zero")


if __name__ == "__main__":
    main()
