# Pure Discrete-Time TTFS Spiking ConvNeXt

This repository studies a fully discrete-time, first-spike-only version of a ConvNeXt-style network. It was created to separate a pure spiking implementation from earlier hybrid experiments and to make temporal behavior, spike activity, and computational proxies directly measurable.

## Model Definition

The pure TTFS model uses:

- explicit simulation over `T` discrete timesteps;
- latency-coded input spikes with at most one input spike per pixel;
- first-spike-only neurons after stem, downsampling, depthwise, pointwise, and classifier synaptic operations;
- event-space residual fusion based on the earliest valid spike;
- optional learnable per-channel temporal delays;
- differentiable output scores derived from class-output spike times.

The no-spike sentinel is `T`, while valid spike times are `0 ... T-1`.

## Purpose

The repository is intended for controlled experiments on the trade-off between classification performance and event-driven computation. It keeps the historical hybrid model separately while providing a dedicated pure-SNN path.

## Evaluation

`evaluator.py` reports prediction metrics together with temporal and efficiency-oriented measurements, including spike counts by timestep, approximate event-fanout SynOps, parameter counts, dense MAC/FLOP estimates, learned delays, inference latency, and peak GPU memory where available.

Approximate SynOps are a software-side proxy and are not presented as direct hardware energy measurements.

## Quick Smoke Test

```bash
python trainer.py --model_size nano --synthetic_data --dry_run \
  --batch_size 2 --max_train_batches 2 --max_val_batches 2 \
  --num_workers 0 --output_dir ./results/smoke_test
```

## Main Files

- `models/discrete_ttfs_convnext.py` — pure TTFS model
- `trainer.py` — training entry point
- `evaluator.py` — quantitative evaluation
- `evaluate_sparsity_synops.py` — spike/SynOps analysis
- `BASELINE_AUDIT.md` — baseline audit
- `PURE_SNN_TTFS_README.md` — detailed implementation notes

## Reproducibility

Experiment configuration, model size, number of timesteps, thresholds, delay settings, optimizer parameters, seed, and output path are explicitly controlled from the command line so that individual runs can be reproduced and compared.

## Upstream Attribution

The training utilities and ConvNeXt baseline files retain code and copyright notices from Meta's [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) implementation. The discrete TTFS model and related experiments extend that baseline; this repository is not an original implementation of the ConvNeXt backbone.
