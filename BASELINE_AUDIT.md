# Baseline Audit

## Temporal Semantics and TTFS Classification

### Classification

**HYBRID_CONTINUOUS_TTFS**

The model contains analytic, real-valued TTFS mappings and has no discrete-time
simulation loop. It is classified as hybrid because ordinary `Conv2d` layers
operate directly on tensors interpreted as spike times and the network ends in
a conventional linear logit head rather than a class-spike-time output layer.

### Search for discrete-time simulation

A global search of the repository's Python, Markdown, and text files found no
`for t in range(...)`, `time_steps`, `timesteps`, or `num_steps` construct used
for neural simulation. It also found no membrane-potential recurrence, neuron
state reset, persistent temporal state, or tensor dimension representing a
sequence of discrete timesteps. Occurrences of `range(...)` are architecture,
epoch, batch, or reporting loops. Variables named `t` in `datasets.py` are
transform-list variables, not simulation time.

Active spike-time tensors have shapes such as `(N, C, H, W)` and
`(N*H*W, C)`. There is no separate `T` dimension. Consequently, no active
layer performs explicit discrete-time simulation.

### Analytic TTFS mapping

`models/convnext.py:909-935` defines `call_spiking_torch(tj, W, D_i,
t_min_prev, t_min, t_max)`. Here `tj` is one scalar spike-time value per input
neuron, with the input-feature dimension last; `W` maps input neurons to output
neurons; and `D_i` is a per-output-neuron delay. `t_min_prev` is unused.

The exact computation is

```text
threshold = t_max - t_min - D_i
delta     = tj - t_min
raw_ti    = delta @ W + threshold + t_min
ti        = raw_ti if raw_ti < t_max else t_max
```

Equivalently, `raw_ti = (tj - t_min) @ W + t_max - D_i`. `ti` represents one
floating-point output spike time per output neuron. The operation is a single
analytic matrix expression followed by elementwise upper clipping; it is not
iterative.

The code guarantees `ti <= t_max`, but it does **not** clamp at `t_min`.
Therefore the documented/intended window is `[t_min, t_max]`, while the actual
numeric range is `(-infinity, t_max]` for finite inputs and parameters. Negative
weights, ordinary convolutions, or sufficiently large affine terms can produce
values below `t_min`.

### CIFAR intensity-to-spike encoding

`datasets.py:58-65` first converts the image to `float32`, clamps intensity
elementwise to `[0,1]`, and applies

```text
t = t_min + (1 - I) * (t_max - t_min).
```

Higher intensity produces an earlier (smaller) spike time. `I=1` maps to
`t_min`; `I=0` maps to `t_max`. The encoder output is therefore in the closed
range `[t_min, t_max]`. For CIFAR-10, the three RGB channels remain three
separate channels: the same scalar mapping is applied independently to every
R, G, and B pixel value, preserving shape `(3, H, W)`.

### Model output and prediction rule

The active path in `models/convnext.py:1148-1163` is:

```python
for i in range(4):
    x_t = self.downsample_layers[i](x_t)
    x_t = self.stages[i](x_t)
x_pool = x_t.mean([-2, -1])
logits = self.head(-x_pool)
return logits
```

`x_pool` is a spatial average of final feature values interpreted as spike
times. Negation makes earlier/smaller feature times larger before a learned
`nn.Linear(dims[-1], num_classes)` transformation. The returned values are raw
class logits, **not class spike times**. Prediction must therefore use
`argmax(logits)`, as done in `main.py:125`, `main_32_32.py:533`,
`main_95.26.py:534`, and `evaluate_sparsity_synops.py:206`. `argmin` would only
be appropriate if the outputs themselves were class spike times, which they
are not.

### Training loss compatibility

`main.py:741-749` selects `SoftTargetCrossEntropy` when Mixup/CutMix is active,
otherwise label-smoothed cross-entropy when smoothing is positive, otherwise
ordinary `CrossEntropyLoss`. These losses are compatible with the model's raw
logits. The spike-time-to-logit transformation is exactly

```text
logits = Linear(-mean_spatial(final_feature_times)).
```

Cross-entropy raises the correct-class **logit** relative to other logits. It
does not directly require a correct-class output neuron to spike earlier,
because the model has no class spike-time output neurons; the learned linear
head can mix all negated pooled features. Thus the loss is compatible with the
implemented classifier semantics, but an “earliest class spike wins” claim
would not describe this implementation.

The current checkout imports `train_one_epoch` from a missing `engine.py`, so
the complete active training step is unavailable. Historical repository code
applied the selected criterion directly to `model(...)` output, consistent with
the logit semantics above.

### Meaning of `t_max`

`t_max` has multiple meanings:

1. It is the intended latest time in the configured TTFS window.
2. It is the upper clipping boundary in `call_spiking_torch`.
3. It is the operational no-spike/silent sentinel: sparsity code counts values
   `>= t_max - 1e-6` as neurons that did not spike within the window.
4. In the input encoder, zero intensity maps exactly to `t_max`.

The code therefore does not distinguish “a spike exactly at the latest valid
time” from “no spike”; both are treated as silent by reporting code.

### At-most-one-spike semantics

Each represented neuron has one scalar time, not a time-indexed spike train.
There is no mechanism capable of producing multiple events for one neuron.
Thus the representation encodes at most one spike per neuron, with `t_max`
used as the no-spike sentinel. This is representational TTFS rather than an
event simulator that explicitly emits and resets after a spike.

### Minimum residual merge

`models/convnext.py:1117` computes

```python
out = torch.minimum(tj, self.drop_path(t_out))
```

Within a `SpikingBlock`, `tj` and `t_out` have identical `(N,C,H,W)` shapes and
are nominally expressed in the same configured time units. Under the baseline
`drop_path_rate=0`, `drop_path` is identity, so it does not rescale `t_out`.

However, strict spike-time validity is not guaranteed for both inputs. Before
each stage, unconstrained stem/downsampling `Conv2d` layers operate directly on
time values, and the depthwise convolution does the same inside each block.
Their outputs are not clamped to `[t_min,t_max]`. The analytic mapping also
lacks a lower clamp. With nonzero stochastic depth, `DropPath` can additionally
rescale retained values. Hence minimum fusion is shape-compatible and
nominally time-domain, but the implementation does not prove that both operands
are valid spike times in the same declared range.

### Explicit checks

| Check | Result | Evidence |
|---|---|---|
| No discrete timestep loop | **PASS** | No neural `for t in range`, timestep count, temporal recurrence, reset, or `T` tensor axis exists. |
| Floating-point spike times | **PASS** | Encoder casts to `float32`; analytic layers propagate real-valued scalar times. |
| At-most-one spike per neuron | **PASS** | One scalar time per neuron; no event sequence or repeated firing state. |
| Prediction rule correctness | **PASS** | Output is `head(-x_pool)` logits and all active evaluation paths use `argmax`. |
| Loss compatibility | **PASS** | Cross-entropy variants consume the returned logits; no `argmin`/class-time loss is assumed. |
| No-spike convention | **PASS, with ambiguity** | `t_max` is consistently counted as silent, but is also described as the latest time and used as the clipping boundary. |
| Minimum residual semantic validity | **FAIL** | Shapes and nominal units match, but ordinary convolutions and missing lower clamps do not guarantee both operands lie in `[t_min,t_max]`. |

