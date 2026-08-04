# Pure discrete-time TTFS Spiking ConvNeXt

This package preserves the historical hybrid model in `models/convnext.py` and adds a separate **pure SNN** in `models/discrete_ttfs_convnext.py`.

## Definition

- Explicit simulation over `T` timesteps.
- Input pixels use latency coding and emit at most one spike.
- Every stem/downsampling/depthwise/pointwise/classifier synaptic operation is followed by a first-spike-only neuron.
- Residual fusion is event-space OR with a first-spike mask, equivalent to selecting the earliest valid spike.
- The no-spike sentinel is `T`; valid times are `0 ... T-1`.
- Learnable delays are differentiable per-channel delay gates measured in timestep units.
- With `force_positive_weights=true`, feature-extraction Conv effective weights use `ReLU(raw_weight)`. The classifier synapse remains signed to preserve class discrimination.
- Output logits are differentiable scores accumulated from class-output first spikes; earlier spikes receive larger scores, so `argmax` and cross-entropy are valid.

This is a new model and does **not** reproduce the submitted hybrid model by construction.

## Baseline

The baseline uses CIFAR-10 at native RGB 32x32, ConvNeXt-Tiny dimensions, `T=5`, earliest-spike residual fusion, learnable delay, positive effective synaptic weights, AdamW, batch size 150, 300 epochs, learning rate 4e-4, cosine decay to 1e-6, weight decay 0.05, label smoothing 0.1, and seed 42.

### Offline smoke test

```bash
python trainer.py --model_size nano --synthetic_data --dry_run \
  --batch_size 2 --max_train_batches 2 --max_val_batches 2 --num_workers 0 \
  --output_dir ./results/smoke_test
```

### Full baseline

```bash
python trainer.py \
  --experiment_name cifar10_baseline_pure_snn_ttfs_t5_seed42 \
  --data_path ./cifar_data --model_size tiny --time_steps 5 --threshold 0.2 \
  --learnable_delay true --force_positive_weights true \
  --batch_size 150 --epochs 300 --lr 4e-4 --min_lr 1e-6 \
  --weight_decay 0.05 --label_smoothing 0.1 --seed 42 \
  --output_dir ./results/cifar10_baseline_pure_snn_ttfs_t5_seed42
```

### Evaluation

```bash
python evaluator.py \
  --checkpoint ./results/cifar10_baseline_pure_snn_ttfs_t5_seed42/best_checkpoint.pth \
  --data_path ./cifar_data \
  --output_dir ./results/cifar10_baseline_pure_snn_ttfs_t5_seed42
```

### Report

```bash
python reporter.py --results_root ./results --output ./reports/experiment_report
```

## Metrics

`evaluator.py` writes accuracy, loss, spike count by timestep, approximate event-fanout SynOps, trainable/total parameters, model size, dense MACs and FLOPs accumulated across all `T` calls, delays, effective constrained-weight statistics, latency, and peak GPU memory when CUDA is used.

SynOps are labeled as an approximation and must not be presented as direct hardware energy measurements.
