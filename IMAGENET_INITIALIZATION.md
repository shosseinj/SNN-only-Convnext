# Optional ImageNet initialization for the discrete TTFS Tiny

## Compatibility audit

The project Tiny uses depths `[3, 3, 9, 3]` and dimensions
`[96, 192, 384, 768]`, matching torchvision ConvNeXt-Tiny.

The following tensors have justified mappings:

- The current CIFAR stem is a 3→96, 4×4, stride-4 convolution. Its parameter
  shape exactly matches torchvision's stem. Keeping 32×32 inputs changes the
  output grid from 56×56 to 8×8 but does not change the kernel mapping.
- All three 2×2 stride-2 downsampling convolution weights and biases match.
- Every 7×7 depthwise convolution weight and bias matches.
- Torchvision pointwise `Linear` matrices `[out, in]` map exactly to the SNN's
  1×1 convolution tensors `[out, in, 1, 1]` by adding two singleton axes.

The following state is deliberately not loaded:

- Stem, block, downsampling, and final LayerNorm parameters: the pure SNN has no
  corresponding normalization modules.
- ConvNeXt layer-scale parameters: the pure SNN has no layer-scale modules.
- ImageNet's 1000-class classifier and final normalization: CIFAR uses a newly
  initialized signed 10-class synapse.
- All neuron thresholds, learnable delays, output-neuron state, surrogate
  behavior, and residual event logic: there is no ANN counterpart.

This is structurally compatible partial initialization, not functional ANN/SNN
equivalence. In particular, normalization and GELU are replaced by explicit
first-spike neurons and temporal state.

## Training from scratch

ImageNet initialization is off by default:

```powershell
..\.venv\Scripts\python.exe trainer.py `
  --model_size tiny --time_steps 5 `
  --imagenet_pretrained false
```

## Partial ImageNet initialization

For signed feature synapses, torchvision weights load directly:

```powershell
..\.venv\Scripts\python.exe trainer.py `
  --model_size tiny --time_steps 5 `
  --force_positive_weights false `
  --imagenet_pretrained true
```

For non-negative feature synapses, signed ImageNet kernels cannot be represented
exactly. Loading is rejected by default. To explicitly accept magnitude-only
initialization, request the absolute-value transform:

```powershell
..\.venv\Scripts\python.exe trainer.py `
  --model_size tiny --time_steps 5 `
  --force_positive_weights true `
  --imagenet_pretrained true `
  --imagenet_positive_transform abs
```

The `abs` transform is recorded for every affected tensor. It preserves kernel
magnitudes but discards pretrained signs, so signed loading is the more faithful
transfer when the experiment permits it.

The first enabled run may download torchvision's official
`ConvNeXt_Tiny_Weights.IMAGENET1K_V1` checkpoint. No download occurs when the
option is false.

## Safety and report

- Nano is rejected before loading because its depths and dimensions differ.
- `--resume` and `--imagenet_pretrained true` are mutually exclusive.
- Every mapped tensor is checked for source presence, target presence, and exact
  post-transform shape before it is copied.
- An incompatible tensor is left newly initialized and listed under
  `skipped_incompatible`; it is never silently loaded.
- The complete report is written to
  `imagenet_initialization_report.json` inside the automatically generated run
  directory. It lists loaded mappings, transformations, skipped source state,
  incompatible mappings, and every preserved SNN target parameter.

Run the download-free structural smoke test with:

```powershell
..\.venv\Scripts\python.exe test_imagenet_init.py
```
