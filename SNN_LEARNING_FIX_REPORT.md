# Pure Discrete-Time TTFS SNN Learning Fix Report

## Outcome

The pure TTFS Nano model is learnable. Configuration A memorized 31/32 fixed
CIFAR-10 images (96.875%) at epoch 78. The inputs were unaugmented, the loader
did not shuffle, and the model used two explicit timesteps. No rate code or
analog feature activation was introduced.

Raw machine-readable results are in `results/tiny_overfit/results.json` and can
be reproduced with:

```powershell
..\.venv\Scripts\python.exe tiny_overfit_test.py --data_path ..\cifar_data --output .\results\tiny_overfit\results.json
```

## Root cause

This was a backward-path and initialization failure, not merely an optimizer
hyperparameter issue.

1. The claimed fast-sigmoid surrogate returned
   `slope / (1 + slope*abs(x))^2`. Its peak derivative was 10 and that gain was
   compounded through approximately 19 spike operations in Nano. Optimization
   rapidly drove hard events into identical saturated/silent patterns. The
   corrected SuperSpike/fast-sigmoid surrogate has unit peak gain:
   `1 / (1 + slope*abs(x))^2`; slope 5 controls width only.
2. Positive synapses used `ReLU(raw_weight)`. About half of every zero-centered
   kernel began negative and therefore had exactly zero gradient forever.
   Effective positive weights now use `abs(raw_weight)`, which preserves the
   non-negative constraint without killing negative raw parameters. Their
   initialization is also fan-in-aware; fixed `std=0.02` caused membrane charge
   to grow with width for positive-only kernels.
3. A requested zero learnable delay was represented by a saturated sigmoid
   logit near -9.21. This reduced delay gradients by roughly four orders of
   magnitude. Delays now use a straight-through bounded timestep value, and the
   discrete gate is centered on a timestep bin (`step + 0.5`).
4. Residual behavior could not previously be ablated. A real `--residual`
   switch was added. Enabled residuals still fuse spike events with logical OR
   and a first-spike mask, which is the event-space equivalent of taking the
   minimum valid first-spike time. Membrane values are never fused.

Before the fixes, a diagnostic batch had identical logits for all four sampled
images. In the no-residual positive-weight trace, TTFS diversity fell from 32
unique inputs to 25 after `stages.0.0`, 17 after `downsamples.1`, and 1 at
`stages.1.0`; later layers were identical or silent. This was the first collapse
point. The fixed harness records both event occurrence and time-weighted TTFS
diversity at every downsample, block, and output.

## Verification of the ten requested points

1. Gradients are non-zero and `optimizer.step()` changes parameters. In A the
   global mean absolute gradient was 7.364e-6, maximum was 0.1240, and 36.25% of
   parameter-gradient entries were non-zero. The largest one-step parameter
   change was 0.0010000002 (`classifier.weight`).
2. The surrogate derivative is checked numerically at x = [-0.2, 0, 0.2]; the
   observed gradients are [0.25, 1.0, 0.25].
3. Every recorded layer in every case had `max_spikes_per_neuron = 1.0`.
4. A two-step unit test integrates currents 0.3 + 0.3 into membrane 0.6 and
   emits exactly one event at step 1, proving state persistence.
5. All 32 images have distinct input TTFS representations. Encoding is
   `round((1-x)*(T-1))`, and each pixel emits once at its encoded timestep.
6. Residual fusion operates on `main` and `x_spike` events, followed by a hard
   first-spike mask. No membrane tensor enters the residual operation.
7. C has mean absolute delay gradient 5.766e-6 and 32 distinct initial output
   TTFS patterns; D has 7.056e-6 and 31. Delays therefore receive gradients and
   do not initially collapse all samples.
8. Fixed A begins with 3 distinct logit rows and finishes with 9 distinct output
   TTFS patterns. C begins with 32 distinct logit rows.
9. `CrossEntropyLoss` receives the raw event-derived `logits`; predictions use
   `logits.argmax(dim=1)`. There is no softmax before the loss.
10. The pre-fix first identical representation was `stages.1.0`. After the fix,
    configuration A has no fully collapsed layer at initialization.

## Tiny-overfit component results

All cases use the same first 32 CIFAR-10 training samples, `ToTensor()` only,
one fixed batch, no shuffle, seed 42, Nano, raw cross-entropy, Adam at 1e-3,
signed feature synapses, and threshold 0.05.

| Case | T | Residual | Delay | Best train accuracy | First >=95% epoch | Result |
|---|---:|---|---|---:|---:|---|
| A | 2 | off | off | **96.875%** | **78** | pass |
| B | 2 | minimum/event OR | off | 93.75% | - | did not pass by 200 |
| C | 2 | minimum/event OR | learnable | **96.875%** | **26** | pass |
| D | 5 | minimum/event OR | learnable | 90.625% | - | did not pass by 200 |

The first configuration that succeeds is A. C also succeeds and demonstrates
that the repaired learnable delay is trainable. B and D are reported as bounded
negative results; they must not be described as passing the 95% criterion.

Selected output spike statistics:

| Case | Initial output spike fraction | Initial unique TTFS | Final output spike fraction | Final unique TTFS |
|---|---:|---:|---:|---:|
| A | 0.13125 | 3 | 0.103125 | 9 |
| B | 0.47500 | 22 | 0.084375 | 7 |
| C | 0.553125 | 32 | 0.13125 | 17 |
| D | 0.54375 | 31 | 0.109375 | 10 |

## Files and lines changed

- `models/discrete_ttfs_convnext.py`: surrogate derivative (line 34),
  differentiable positive constraint (lines 54 and 66), delay parameterization
  and gate (lines 82-96), residual switch and event fusion (lines 155-184),
  fan-in-aware positive initialization (lines 227-238), and layer TTFS/spike
  diagnostics (lines 267-348).
- `trainer.py`: `--residual` option (line 35), ten-iteration logs (line 117),
  model wiring (around line 142), and accurate residual metadata.
- `tiny_overfit_test.py`: new fixed-data structural checks and A-D overfit
  harness (lines 1-146).
- `results/tiny_overfit/results.json`: generated detailed evidence.

Line numbers refer to the fixed files in this workspace.

## Exact fixed-Nano command

This command uses configuration C, which retains the minimum residual and the
repaired learnable delays, has 596,910 parameters, and exceeded 95% in the
isolated learnability test:

```powershell
..\.venv\Scripts\python.exe trainer.py `
  --data_path ..\cifar_data --model_size nano `
  --time_steps 2 --threshold 0.05 `
  --residual true --learnable_delay true `
  --force_positive_weights false `
  --batch_size 128 --epochs 300 --lr 1e-3 --min_lr 1e-6 `
  --weight_decay 0 --label_smoothing 0 --seed 42 `
  --output_dir .\results\cifar10_nano_fixed_ttfs
```

The model remains explicitly unrolled over T, input pixels remain latency
encoded, hidden and output neurons remain first-spike-only, and logits remain
derived from output first-spike events.
