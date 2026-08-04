# Implementation summary

## Added

- `models/discrete_ttfs_convnext.py`: separate pure discrete-time TTFS SNN.
- `trainer.py`: CIFAR-10 baseline trainer with deterministic seed support, validation checkpointing, and JSON logs.
- `evaluator.py`: accuracy, loss, spike timing, SynOps estimate, parameter count, model size, MAC/FLOP estimate, delay and latency metrics.
- `reporter.py`: CSV/JSON/Markdown experiment aggregation and multi-seed mean/sample-standard-deviation summaries.
- `reproducibility.py`, `experiment_utils.py`.
- `PURE_SNN_TTFS_README.md` with commands.

## Model semantics

- Explicit `T`-step simulation.
- First-spike-only neurons; at most one spike per neuron.
- Valid first-spike indices are `0..T-1`; `T` is the no-spike sentinel.
- Event-space earliest-spike residual fusion.
- All feature operations are synaptic convolution followed by a spiking neuron.
- The classifier is a signed synaptic linear layer followed by first-spike output neurons.
- Earlier output spikes create larger differentiable class scores; prediction uses `argmax`.
- Per-channel learnable delay gates are measured in timestep units.

## Validation performed

A CPU-only synthetic smoke test completed for the nano configuration with `T=5`, including forward, backward, optimizer update, checkpoint save/load, evaluator, and reporter. Full CIFAR-10 training was not run.

## Important scientific note

This is a new pure discrete-time SNN, not a reproduction of the submitted hybrid continuous TTFS model. Its accuracy and stability must be established experimentally before manuscript claims are changed.
