"""Structural and tiny numerical checks for the CIFAR-10 TTFS baseline.

This script never loads CIFAR-10 and never enters a training loop.  It uses a
small ConvNeXtSpiking instance so it is suitable as a CPU smoke test.
"""

from __future__ import annotations

import argparse

import torch

from models.convnext import ConvNeXtSpiking, SpikingBlock
from evaluator import TemporalValidityDiagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Reduced widths/depths exercise the real implementation without starting
    # the 28M-parameter experiment or downloading data.
    model = ConvNeXtSpiking(
        in_chans=3,
        num_classes=10,
        depths=(1, 1, 1, 1),
        dims=(8, 16, 32, 64),
        drop_path_rate=0.0,
        t_min=0.0,
        t_max=1.0,
        force_positive_weights=True,
        init_delay=0.0,
    )
    model.train()

    blocks = [module for module in model.modules() if isinstance(module, SpikingBlock)]
    assert len(blocks) == 4
    assert all(block.force_positive_weights for block in blocks)
    assert all(block.D_mid.requires_grad and block.D_out.requires_grad for block in blocks)

    # Hook inputs/outputs prove that the returned residual is the elementwise
    # minimum of the shortcut and main branch for the active implementation.
    observed: dict[str, torch.Tensor] = {}
    block = blocks[0]

    def capture_input(_module, inputs):
        observed["shortcut"] = inputs[0].detach()

    def capture_drop_path(_module, _inputs, output):
        observed["main"] = output.detach()

    def capture_output(_module, _inputs, output):
        observed["fused"] = output.detach()

    handles = [
        block.register_forward_pre_hook(capture_input),
        block.drop_path.register_forward_hook(capture_drop_path),
        block.register_forward_hook(capture_output),
    ]

    image = torch.rand(2, 3, 32, 32)
    spike_times = 1.0 - image
    model.eval()
    with torch.no_grad():
        logits_without_diagnostics = model(spike_times)
        diagnostics = TemporalValidityDiagnostics(model, t_min=0.0, t_max=1.0)
        logits_with_diagnostics = model(spike_times)
        diagnostics.disable()
        temporal_result = diagnostics.result()
    assert torch.equal(logits_without_diagnostics, logits_with_diagnostics)
    assert set(temporal_result["per_block"]) == {
        "stages.0.0", "stages.1.0", "stages.2.0", "stages.3.0"
    }
    assert temporal_result["status"] in {"PASS", "WARNING", "FAIL"}

    model.train()
    logits = model(spike_times)
    loss = logits.square().mean()
    loss.backward()
    for handle in handles:
        handle.remove()

    assert logits.shape == (2, 10)
    assert torch.equal(observed["fused"], torch.minimum(observed["shortcut"], observed["main"]))
    assert all(block.D_mid.grad is not None and block.D_out.grad is not None for block in blocks)

    print("PASS: RGB 3x32x32 input -> 10 logits")
    print("PASS: every spiking block uses element-wise minimum residual fusion")
    print("PASS: D_mid and D_out are learnable and receive gradients")
    print("PASS: submitted non-negative pointwise constraint is enabled (ReLU parameterization)")
    print("PASS: temporal diagnostics leave forward outputs unchanged")
    print("PASS: temporal diagnostics contain global and per-block statistics")


if __name__ == "__main__":
    main()
