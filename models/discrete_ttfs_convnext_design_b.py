"""Corrected pure discrete-time TTFS Spiking ConvNeXt.

Key properties
--------------
- Explicit simulation over T discrete time bins.
- Design B keeps the depthwise internal path analog; TTFS replaces GELU and converts each block output back to spikes.
- Hidden synapses use bias=False, preventing bias accumulation at every timestep.
- Signed synapses use fan-in-aware initialization.
- Optional current normalization is applied before membrane integration.
- Residual fusion implements earliest-spike semantics with a gradient-preserving
  hard OR in event space and explicit min bookkeeping in spike-time space.
- CIFAR stem is used by default: 3x3, stride 1, padding 1.
- Optional no-spike input state for low-intensity pixels.
- spike_integrator readout uses only final-stage hard spikes; the final linear
  classifier is non-spiking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate spike
# ---------------------------------------------------------------------------

class SurrogateStep(torch.autograd.Function):
    """Binary threshold in forward; unit-gain fast-sigmoid in backward."""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, slope: float, grad_clip: float
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        ctx.grad_clip = float(grad_clip)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad_clip = ctx.grad_clip
        # Deep explicitly-unrolled SNNs can create very large local adjoints
        # even when the final global parameter norm is moderate. Bound the
        # surrogate adjoint before convolution backward and do the derivative
        # math in FP32; the hard-spike forward path is unchanged.
        grad_output_fp32 = grad_output.float().clamp_(-grad_clip, grad_clip)
        x_fp32 = x.float()
        surrogate = 1.0 / (1.0 + slope * x_fp32.abs()).pow(2)
        return (grad_output_fp32 * surrogate).to(grad_output.dtype), None, None


def spike_fn(
    x: torch.Tensor, slope: float = 5.0, grad_clip: float = 16.0
) -> torch.Tensor:
    return SurrogateStep.apply(x, slope, grad_clip)


# ---------------------------------------------------------------------------
# Synapses
# ---------------------------------------------------------------------------

class EffectiveConv2d(nn.Conv2d):
    """Conv2d with optional non-negative effective weights."""

    def __init__(
        self, *args, force_positive_weights: bool = False,
        force_fp32: bool = False, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.force_positive_weights = bool(force_positive_weights)
        self.force_fp32 = bool(force_fp32)

    def effective_weight(self) -> torch.Tensor:
        return self.weight.abs() if self.force_positive_weights else self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_fp32 and x.device.type == "cuda" and torch.is_autocast_enabled():
            with torch.autocast(device_type="cuda", enabled=False):
                bias = self.bias.float() if self.bias is not None else None
                return F.conv2d(
                    x.float(), self.effective_weight().float(), bias,
                    self.stride, self.padding, self.dilation, self.groups,
                )
        return F.conv2d(
            x,
            self.effective_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )



class EffectiveLinear(nn.Linear):
    """Linear layer with optional non-negative effective weights."""

    def __init__(self, *args, force_positive_weights: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_positive_weights = bool(force_positive_weights)

    def effective_weight(self) -> torch.Tensor:
        return self.weight.abs() if self.force_positive_weights else self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)
PositiveConv2d = EffectiveConv2d
PositiveLinear = EffectiveLinear

class ChannelCurrentNorm(nn.Module):
    """Normalize synaptic current, never hard spike events.

    GroupNorm(1, C) is independent of batch statistics and works at 1x1 spatial
    resolution. It is optional because it changes the architecture and should
    be reported as an ablation.
    """

    def __init__(self, channels: int, enabled: bool, force_fp32: bool = False):
        super().__init__()
        # Sparse event currents can have nearly zero variance. A larger epsilon
        # bounds GroupNorm's inverse-standard-deviation backward gain and avoids
        # deep-stack NaNs without changing the event path or tensor shapes.
        self.norm = (
            nn.GroupNorm(1, channels, eps=1e-3, affine=True)
            if enabled else nn.Identity()
        )
        self.force_fp32 = bool(force_fp32 and enabled)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.force_fp32 and current.device.type == "cuda" and torch.is_autocast_enabled():
            with torch.autocast(device_type="cuda", enabled=False):
                return self.norm(current.float())
        return self.norm(current)


# ---------------------------------------------------------------------------
# Learnable temporal parameters
# ---------------------------------------------------------------------------

class LearnableDelay(nn.Module):
    """Bounded per-channel delay measured in discrete timestep units."""

    def __init__(
        self,
        channels: int,
        time_steps: int,
        init_delay: float = 0.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("delay temperature must be positive")
        self.time_steps = int(time_steps)
        self.temperature = float(temperature)

        max_delay = float(max(self.time_steps - 1, 0))
        if max_delay == 0:
            init_probability = 0.5
        else:
            init_probability = min(max(float(init_delay) / max_delay, 1e-4), 1.0 - 1e-4)

        raw_init = torch.logit(torch.tensor(init_probability))
        self.raw_delay = nn.Parameter(torch.full((channels,), float(raw_init)))

    def values(self) -> torch.Tensor:
        max_delay = float(max(self.time_steps - 1, 0))
        if max_delay == 0:
            return torch.zeros_like(self.raw_delay)
        return max_delay * torch.sigmoid(self.raw_delay)

    def hard_values(self) -> torch.Tensor:
        return self.values().round()

    def gate(self, step: int, ndim: int) -> torch.Tensor:
        delay = self.values()
        shape = [1, delay.numel()] + [1] * max(ndim - 2, 0)
        # At delay=0 and step=0 this is close to one for a sufficiently small
        # temperature, unlike the previous 0.731 attenuation.
        return torch.sigmoid(
            (float(step) + 1.0 - delay.view(*shape)) / self.temperature
        )


@dataclass
class NeuronState:
    membrane: torch.Tensor
    has_spiked: torch.Tensor
    first_spike: Optional[torch.Tensor] = None


class FirstSpikeNeuron(nn.Module):
    """Integrate-and-fire neuron that emits at most one spike."""

    def __init__(
        self,
        channels: int,
        time_steps: int,
        threshold: float = 0.2,
        learnable_delay: bool = True,
        init_delay: float = 0.0,
        delay_temperature: float = 0.25,
        surrogate_slope: float = 5.0,
        surrogate_grad_clip: float = 16.0,
        threshold_mode: str = "fixed",
        threshold_min: float = 0.05,
        threshold_max: float = 0.8,
    ):
        super().__init__()

        if threshold_mode not in {"fixed", "learnable_layer", "learnable_channel"}:
            raise ValueError(f"unknown threshold_mode={threshold_mode!r}")
        if threshold_mode != "fixed" and not threshold_min < threshold < threshold_max:
            raise ValueError(
                "learnable threshold initialization must satisfy "
                "threshold_min < threshold < threshold_max"
            )

        self.time_steps = int(time_steps)
        self.threshold_mode = threshold_mode
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.surrogate_slope = float(surrogate_slope)
        if surrogate_grad_clip <= 0:
            raise ValueError("surrogate_grad_clip must be positive")
        self.surrogate_grad_clip = float(surrogate_grad_clip)

        if threshold_mode == "fixed":
            self.register_buffer(
                "fixed_threshold",
                torch.full((channels,), float(threshold)),
            )
            self.register_parameter("raw_threshold", None)
        else:
            size = 1 if threshold_mode == "learnable_layer" else channels
            probability = (
                (float(threshold) - self.threshold_min)
                / (self.threshold_max - self.threshold_min)
            )
            raw = torch.logit(torch.full((size,), probability))
            self.raw_threshold = nn.Parameter(raw)
            self.register_buffer("fixed_threshold", None)

        self.delay = (
            LearnableDelay(
                channels=channels,
                time_steps=time_steps,
                init_delay=init_delay,
                temperature=delay_temperature,
            )
            if learnable_delay
            else None
        )

    def threshold_values(self) -> torch.Tensor:
        if self.threshold_mode == "fixed":
            return self.fixed_threshold
        return self.threshold_min + (
            self.threshold_max - self.threshold_min
        ) * torch.sigmoid(self.raw_threshold)

    def init_state(
        self,
        shape_or_sample,
        device=None,
        dtype=None,
        track_first_spike: Optional[bool] = None,
    ) -> NeuronState:
        """Allocate state directly; never execute a synapse or normalization."""
        legacy_tensor_call = isinstance(shape_or_sample, torch.Tensor)
        if legacy_tensor_call:
            shape = tuple(shape_or_sample.shape)
            device = shape_or_sample.device
            dtype = shape_or_sample.dtype
        else:
            shape = tuple(shape_or_sample)
        if device is None or dtype is None:
            raise ValueError("device and dtype are required for shape-based state initialization")
        if track_first_spike is None:
            track_first_spike = legacy_tensor_call
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        sentinel = (
            torch.full(shape, float(self.time_steps), device=device, dtype=dtype)
            if track_first_spike
            else None
        )
        return NeuronState(
            membrane=zeros,
            has_spiked=torch.zeros(shape, device=device, dtype=torch.bool),
            first_spike=sentinel,
        )

    def forward_step(
        self,
        current: torch.Tensor,
        state: NeuronState,
        step: int,
    ) -> Tuple[torch.Tensor, NeuronState]:
        if self.delay is not None:
            current = current * self.delay.gate(step, current.ndim)

        # Stop integrating after the first spike. This keeps membrane-based
        # diagnostics from accumulating unrelated post-spike current.
        active = (~state.has_spiked).to(current.dtype)
        membrane = state.membrane + current * active

        threshold = self.threshold_values()
        shape = [1, threshold.numel()] + [1] * max(current.ndim - 2, 0)
        candidate = spike_fn(
            membrane - threshold.view(*shape),
            self.surrogate_slope,
            self.surrogate_grad_clip,
        )
        spike = candidate * active

        newly = spike.detach().bool() & (~state.has_spiked)
        first_spike = state.first_spike
        if first_spike is not None:
            first_spike = torch.where(
                newly,
                torch.full_like(first_spike, float(step)),
                first_spike,
            )

        return spike, NeuronState(
            membrane=membrane,
            has_spiked=state.has_spiked | newly,
            first_spike=first_spike,
        )


# ---------------------------------------------------------------------------
# Spiking ConvNeXt building blocks
# ---------------------------------------------------------------------------

class SpikingDownsample(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        time_steps: int,
        force_positive_weights: bool,
        learnable_delay: bool,
        init_delay: float,
        threshold: float,
        threshold_mode: str,
        threshold_min: float,
        threshold_max: float,
        current_norm: bool,
        surrogate_grad_clip: float = 16.0,
    ):
        super().__init__()
        self.synapse = EffectiveConv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            force_positive_weights=force_positive_weights,
        )
        self.current_norm = ChannelCurrentNorm(out_ch, current_norm)
        self.neuron = FirstSpikeNeuron(
            out_ch,
            time_steps,
            threshold,
            learnable_delay,
            init_delay,
            threshold_mode=threshold_mode,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            surrogate_grad_clip=surrogate_grad_clip,
        )

    def current(self, spikes: torch.Tensor) -> torch.Tensor:
        return self.current_norm(self.synapse(spikes))

    def output_shape(self, input_shape: Sequence[int]) -> Tuple[int, int, int, int]:
        batch, _, height, width = input_shape
        kernel_h, kernel_w = self.synapse.kernel_size
        stride_h, stride_w = self.synapse.stride
        padding_h, padding_w = self.synapse.padding
        dilation_h, dilation_w = self.synapse.dilation
        out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        return batch, self.synapse.out_channels, out_h, out_w

    def init_state(
        self,
        input_shape: Sequence[int],
        device,
        dtype,
        track_first_spike: bool = False,
    ) -> NeuronState:
        return self.neuron.init_state(
            self.output_shape(input_shape), device, dtype, track_first_spike
        )

    def forward_step(
        self,
        spikes: torch.Tensor,
        state: NeuronState,
        step: int,
    ) -> Tuple[torch.Tensor, NeuronState, torch.Tensor]:
        current = self.current(spikes)
        output, state = self.neuron.forward_step(current, state, step)
        return output, state, current


def hard_or_ste(
    main: torch.Tensor,
    residual: torch.Tensor,
    main_gradient_scale: float = 0.5,
) -> torch.Tensor:
    """Hard logical OR forward with a stable residual-style surrogate.

    Forward:
        0 OR 0 = 0
        1 OR 0 = 1
        0 OR 1 = 1
        1 OR 1 = 1

    ``a`` is the transformed main branch and ``b`` is the residual branch.
    Backward preserves an identity gradient along the residual path and sends
    quarter-strength gradient into the main path. Sending the full upstream
    gradient into both paths caused overflow across the 18-block Tiny model;
    averaging both paths instead made early-stage gradients vanish.
    """
    hard = torch.clamp(main + residual, 0.0, 1.0)
    soft = residual + float(main_gradient_scale) * main
    return soft + (hard - soft).detach()


class SpikingConvNeXtBlock(nn.Module):
    """ConvNeXt-faithful Design B block.

    Structure:
        spike input
        -> depthwise convolution (analog current)
        -> current normalization
        -> pointwise expansion
        -> first-spike neuron (replaces GELU)
        -> pointwise projection
        -> first-spike block-output neuron
        -> earliest-event residual fusion

    Unlike Design A, no neuron is inserted directly after the depthwise
    convolution. The block still has spike input and spike output.
    """

    def __init__(
        self,
        dim: int,
        time_steps: int,
        force_positive_weights: bool = False,
        learnable_delay: bool = True,
        init_delay: float = 0.0,
        threshold: float = 0.2,
        residual: bool = True,
        threshold_mode: str = "fixed",
        threshold_min: float = 0.05,
        threshold_max: float = 0.8,
        current_norm: bool = True,
        force_fp32_norm: bool = False,
        residual_main_gradient_scale: float = 0.25,
        surrogate_grad_clip: float = 16.0,
    ):
        super().__init__()

        neuron_kwargs = dict(
            threshold_mode=threshold_mode,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            surrogate_grad_clip=surrogate_grad_clip,
        )

        # The depthwise convolution remains an ordinary affine/synaptic
        # operation. There is intentionally no spiking neuron after it.
        self.dw = EffectiveConv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            bias=False,
            force_positive_weights=force_positive_weights,
        )
        self.dw_norm = ChannelCurrentNorm(dim, current_norm, force_fp32_norm)

        # ConvNeXt expansion layer. Its first-spike neuron replaces GELU.
        self.pw1 = EffectiveConv2d(
            dim,
            4 * dim,
            kernel_size=1,
            bias=False,
            force_positive_weights=force_positive_weights,
            force_fp32=force_fp32_norm,
        )
        self.pw1_norm = ChannelCurrentNorm(4 * dim, current_norm, force_fp32_norm)
        self.activation_neuron = FirstSpikeNeuron(
            4 * dim,
            time_steps,
            threshold,
            learnable_delay,
            init_delay,
            **neuron_kwargs,
        )

        # Projection back to dim, followed by a first-spike output neuron so
        # every block still communicates spikes to the next block.
        self.pw2 = EffectiveConv2d(
            4 * dim,
            dim,
            kernel_size=1,
            bias=False,
            force_positive_weights=force_positive_weights,
        )
        self.pw2_norm = ChannelCurrentNorm(dim, current_norm, force_fp32_norm)
        self.output_neuron = FirstSpikeNeuron(
            dim,
            time_steps,
            threshold,
            learnable_delay,
            init_delay,
            **neuron_kwargs,
        )

        # Backward-compatible attribute names for utilities that inspect them.
        self.pw1_neuron = self.activation_neuron
        self.pw2_neuron = self.output_neuron

        self.time_steps = int(time_steps)
        self.residual = bool(residual)
        self.residual_main_gradient_scale = float(residual_main_gradient_scale)

    def init_states(
        self,
        input_shape: Sequence[int],
        device,
        dtype,
        track_first_spike: bool = False,
    ):
        batch, channels, height, width = input_shape
        main_shape = (batch, channels, height, width)
        expanded_shape = (batch, 4 * channels, height, width)

        state_activation = self.activation_neuron.init_state(
            expanded_shape, device, dtype, track_first_spike
        )
        state_output = self.output_neuron.init_state(
            main_shape, device, dtype, track_first_spike
        )

        # Separate block-output mask is needed because the residual branch can
        # fire earlier than the transformed branch.
        output_has_spiked = torch.zeros(
            main_shape, device=device, dtype=torch.bool
        )
        output_first_spike = (
            torch.full(
                main_shape,
                float(self.time_steps),
                device=device,
                dtype=dtype,
            )
            if track_first_spike
            else None
        )

        return [
            state_activation,
            state_output,
            output_has_spiked,
            output_first_spike,
        ]

    def forward_step(
        self,
        x_spike: torch.Tensor,
        states,
        step: int,
    ):
        (
            state_activation,
            state_output,
            out_has_spiked,
            out_first,
        ) = states

        # No spike threshold here: this is the analog internal ConvNeXt path.
        dw_current = self.dw_norm(self.dw(x_spike))

        # The first-spike neuron replaces ConvNeXt's GELU activation.
        pw1_current = self.pw1_norm(self.pw1(dw_current))
        activation_spike, state_activation = (
            self.activation_neuron.forward_step(
                pw1_current,
                state_activation,
                step,
            )
        )

        # Projection followed by a spike conversion at block output.
        pw2_current = self.pw2_norm(self.pw2(activation_spike))
        main_spike, state_output = self.output_neuron.forward_step(
            pw2_current,
            state_output,
            step,
        )

        if self.residual:
            # Hard forward is binary OR. With first-spike-only masking this is
            # equivalent to selecting the earlier main/residual event.
            fused_event = hard_or_ste(
                main_spike, x_spike, self.residual_main_gradient_scale
            )

            if out_first is not None and state_output.first_spike is not None:
                no_spike = torch.full_like(
                    out_first, float(self.time_steps)
                )
                residual_first_now = torch.where(
                    x_spike.detach().bool(),
                    torch.full_like(out_first, float(step)),
                    no_spike,
                )
                candidate_first = torch.minimum(
                    state_output.first_spike,
                    residual_first_now,
                )
            else:
                candidate_first = None
        else:
            fused_event = main_spike
            candidate_first = state_output.first_spike

        active = (~out_has_spiked).to(fused_event.dtype)
        out = fused_event * active
        newly = out.detach().bool() & (~out_has_spiked)

        if out_first is not None and candidate_first is not None:
            out_first = torch.where(
                newly,
                torch.minimum(out_first, candidate_first),
                out_first,
            )

        return (
            out,
            [
                state_activation,
                state_output,
                out_has_spiked | newly,
                out_first,
            ],
            {
                # dw_current is analog and must not be counted as spikes.
                "dw_current": dw_current,
                "pw1_spike": activation_spike,
                "main": main_spike,
            },
        )


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class DiscreteTTFSConvNeXt(nn.Module):
    """Design-B discrete-time TTFS ConvNeXt with non-spiking output integrator."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 10,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        time_steps: int = 4,
        threshold: float = 0.4,
        force_positive_weights: bool = False,
        learnable_delay: bool = True,
        init_delay: float = 0.5,
        residual: bool = True,
        threshold_mode: str = "learnable_channel",
        threshold_min: float = 0.1,
        threshold_max: float = 1.2,
        readout_mode: str = "spike_integrator",
        soft_time_beta: float = 10.0,
        current_norm: bool = True,
        cifar_stem: bool = True,
        input_no_spike_threshold: float = 0.05,
        residual_main_gradient_scale: float = 0.25,
        surrogate_grad_clip: float = 16.0,
        track_first_spike: bool = False,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        if time_steps < 2:
            raise ValueError("time_steps must be >= 2")
        if readout_mode not in {"spike_integrator", "ttfs"}:
            raise ValueError(
                "This corrected model supports readout_mode='spike_integrator' "
                "or 'ttfs'."
            )
        if not 0.0 <= input_no_spike_threshold < 1.0:
            raise ValueError("input_no_spike_threshold must be in [0, 1)")

        self.time_steps = int(time_steps)
        self.num_classes = int(num_classes)
        self.temporal_model_type = "DISCRETE_TIME_TTFS_SNN_DESIGN_B"
        self.block_design = "B_convnext_faithful"
        self.maximum_spikes_per_neuron = 1
        self.readout_mode = readout_mode
        self.soft_time_beta = float(soft_time_beta)
        self.input_no_spike_threshold = float(input_no_spike_threshold)
        self.residual_main_gradient_scale = float(residual_main_gradient_scale)
        self.surrogate_grad_clip = float(surrogate_grad_clip)
        self.track_first_spike = bool(track_first_spike)
        self.use_checkpointing = bool(use_checkpointing)
        self.cifar_stem = bool(cifar_stem)
        self.current_norm = bool(current_norm)
        if self.use_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing is disabled because this model carries "
                "mutable temporal neuron state; a safe explicit-state checkpoint "
                "implementation is not yet available."
            )

        self.downsamples = nn.ModuleList()

        if cifar_stem:
            stem_kernel, stem_stride, stem_padding = 3, 1, 1
        else:
            stem_kernel, stem_stride, stem_padding = 4, 4, 0

        self.downsamples.append(
            SpikingDownsample(
                in_ch=in_chans,
                out_ch=dims[0],
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=stem_padding,
                time_steps=time_steps,
                force_positive_weights=force_positive_weights,
                learnable_delay=learnable_delay,
                init_delay=init_delay,
                threshold=threshold,
                threshold_mode=threshold_mode,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                current_norm=current_norm,
                surrogate_grad_clip=surrogate_grad_clip,
            )
        )

        for index in range(3):
            self.downsamples.append(
                SpikingDownsample(
                    in_ch=dims[index],
                    out_ch=dims[index + 1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    time_steps=time_steps,
                    force_positive_weights=force_positive_weights,
                    learnable_delay=learnable_delay,
                    init_delay=init_delay,
                    threshold=threshold,
                    threshold_mode=threshold_mode,
                    threshold_min=threshold_min,
                    threshold_max=threshold_max,
                    current_norm=current_norm,
                    surrogate_grad_clip=surrogate_grad_clip,
                )
            )

        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SpikingConvNeXtBlock(
                            dim=dims[stage],
                            time_steps=time_steps,
                            force_positive_weights=force_positive_weights,
                            learnable_delay=learnable_delay,
                            init_delay=init_delay,
                            threshold=threshold,
                            residual=residual,
                            threshold_mode=threshold_mode,
                            threshold_min=threshold_min,
                            threshold_max=threshold_max,
                            current_norm=current_norm,
                            force_fp32_norm=(stage == 3),
                            residual_main_gradient_scale=residual_main_gradient_scale,
                            surrogate_grad_clip=surrogate_grad_clip,
                        )
                        for _ in range(depths[stage])
                    ]
                )
                for stage in range(4)
            ]
        )

        # Signed, non-spiking classifier. Bias is safe here because it is
        # evaluated once after temporal integration.
        self.classifier = EffectiveLinear(
            dims[-1],
            num_classes,
            bias=True,
            force_positive_weights=False,
        )

        # Only needed by the historical class-spike readout.
        self.output_neuron: Optional[FirstSpikeNeuron]
        if readout_mode == "ttfs":
            self.output_neuron = FirstSpikeNeuron(
                num_classes,
                time_steps,
                threshold,
                learnable_delay,
                init_delay,
                threshold_mode=threshold_mode,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                surrogate_grad_clip=surrogate_grad_clip,
            )
        else:
            self.output_neuron = None

        self._initialize_weights()

    # ------------------------- initialization -------------------------

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, EffectiveConv2d):
                if module.force_positive_weights:
                    fan_in = (
                        module.in_channels // module.groups
                    ) * module.kernel_size[0] * module.kernel_size[1]
                    nn.init.trunc_normal_(
                        module.weight,
                        std=1.0 / max(fan_in, 1),
                    )
                else:
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_in",
                        nonlinearity="linear",
                    )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, EffectiveLinear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.GroupNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --------------------------- input coding -------------------------

    def latency_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode values in [0,1] into {0,...,T-1,T}, where T means no spike."""
        if torch.is_floating_point(x):
            # Fail loudly instead of silently destroying normalized inputs.
            min_value = float(x.detach().amin().item())
            max_value = float(x.detach().amax().item())
            if min_value < -1e-4 or max_value > 1.0001:
                raise ValueError(
                    f"latency_encode expected input in [0,1], got "
                    f"min={min_value:.4f}, max={max_value:.4f}. "
                    "Remove dataset Normalize(mean,std) before TTFS encoding."
                )

        x = x.clamp(0.0, 1.0)
        spike_time = torch.round(
            (1.0 - x) * (self.time_steps - 1)
        ).long()

        active = x > self.input_no_spike_threshold
        no_spike = torch.full_like(spike_time, self.time_steps)
        return torch.where(active, spike_time, no_spike)

    # ----------------------------- metrics ----------------------------

    @staticmethod
    def _synops(module: nn.Module, input_spikes: torch.Tensor) -> float:
        number_of_spikes = float(input_spikes.detach().sum().item())

        if isinstance(module, nn.Conv2d):
            kernel_h, kernel_w = module.kernel_size
            fanout = (
                module.out_channels / module.groups
            ) * kernel_h * kernel_w
        elif isinstance(module, nn.Linear):
            fanout = module.out_features
        else:
            fanout = 0.0

        return number_of_spikes * fanout

    # ------------------------------ forward ---------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ):
        batch_size = x.shape[0]
        first_idx = self.latency_encode(x)
        device, dtype = x.device, x.dtype
        track_timestamps = self.track_first_spike

        # Allocate states from known convolution shapes. No Conv2d, GroupNorm,
        # or autograd-tracked operation is executed during initialization.
        current_shape = tuple(x.shape)
        downsample_states: List[NeuronState] = []
        block_states: List[List] = []
        for stage_index in range(4):
            downsample = self.downsamples[stage_index]
            downsample_state = downsample.init_state(
                current_shape, device, dtype, track_timestamps
            )
            downsample_states.append(downsample_state)
            current_shape = tuple(downsample_state.membrane.shape)
            states_for_stage = [
                block.init_states(current_shape, device, dtype, track_timestamps)
                for block in self.stages[stage_index]
            ]
            block_states.append(states_for_stage)

        final_spike_score = torch.zeros(current_shape, device=device, dtype=dtype)
        class_score = torch.zeros(
            batch_size,
            self.num_classes,
            device=device,
            dtype=dtype,
        )

        output_state = None
        if self.output_neuron is not None:
            output_state = self.output_neuron.init_state(
                class_score.shape, device, dtype, False
            )

        total_synops = 0.0
        layer_spike_counts: Dict[str, float] = {} if return_stats else None
        layer_unit_counts: Dict[str, int] = {} if return_stats else None
        block_output_spikes = 0.0
        block_output_units = 0
        if return_stats:
            stage_step_spike_counts = [
                [0.0 for _ in range(self.time_steps)] for _ in range(4)
            ]
            stage_seen = [None for _ in range(4)]
            stage_repeated_counts = [0.0 for _ in range(4)]
            stage_unit_counts = [0 for _ in range(4)]

        for step in range(self.time_steps):
            spikes = (first_idx == step).to(dtype)

            if return_stats:
                layer_spike_counts["input"] = (
                    layer_spike_counts.get("input", 0.0)
                    + float(spikes.detach().sum().item())
                )
                layer_unit_counts["input"] = spikes.numel()

            for stage_index in range(4):
                downsample = self.downsamples[stage_index]

                if return_stats:
                    total_synops += self._synops(
                        downsample.synapse,
                        spikes,
                    )

                spikes, downsample_states[stage_index], _ = downsample.forward_step(
                    spikes,
                    downsample_states[stage_index],
                    step,
                )

                if return_stats:
                    name = f"downsamples.{stage_index}"
                    layer_spike_counts[name] = (
                        layer_spike_counts.get(name, 0.0)
                        + float(spikes.detach().sum().item())
                    )
                    layer_unit_counts[name] = spikes.numel()

                for block_index, block in enumerate(self.stages[stage_index]):
                    block_input = spikes
                    (
                        spikes,
                        block_states[stage_index][block_index],
                        internal,
                    ) = block.forward_step(
                        block_input,
                        block_states[stage_index][block_index],
                        step,
                    )

                    if return_stats:
                        # Design B:
                        # - dw consumes spikes and can be estimated as SynOps.
                        # - pw1 consumes analog depthwise current, so it is not
                        #   included in spike-driven SynOps.
                        # - pw2 consumes TTFS activation spikes.
                        total_synops += self._synops(
                            block.dw,
                            block_input,
                        )
                        total_synops += self._synops(
                            block.pw2,
                            internal["pw1_spike"],
                        )

                        populations = {
                            "activation": internal["pw1_spike"],
                            "main": internal["main"],
                        }
                        for population_name, population in populations.items():
                            name = (
                                f"stages.{stage_index}.{block_index}."
                                f"{population_name}"
                            )
                            layer_spike_counts[name] = (
                                layer_spike_counts.get(name, 0.0)
                                + float(population.detach().sum().item())
                            )
                            layer_unit_counts[name] = population.numel()
                        block_output_spikes += float(spikes.detach().sum().item())
                        block_output_units += spikes.numel()

                if return_stats:
                    events = spikes.detach().bool()
                    event_count = float(events.sum().item())
                    stage_step_spike_counts[stage_index][step] += event_count
                    stage_unit_counts[stage_index] = events.numel()
                    if stage_seen[stage_index] is None:
                        stage_seen[stage_index] = events.clone()
                    else:
                        stage_repeated_counts[stage_index] += float(
                            (stage_seen[stage_index] & events).sum().item()
                        )
                        stage_seen[stage_index].logical_or_(events)

            time_weight = (
                self.time_steps - step
            ) / float(self.time_steps)
            final_spike_score = (
                final_spike_score + spikes * time_weight
            )

            if self.readout_mode == "ttfs":
                pooled = spikes.mean(dim=(-2, -1))
                current = self.classifier(pooled)
                class_spike, output_state = self.output_neuron.forward_step(
                    current,
                    output_state,
                    step,
                )
                class_score = class_score + class_spike * time_weight

        final_features = final_spike_score.mean(dim=(-2, -1))

        if self.readout_mode == "spike_integrator":
            logits = self.classifier(final_features)
        else:
            logits = class_score

        if not return_stats:
            return logits

        # Synaptic-neuron sparsity counts each distinct neuron population once:
        # downsample, TTFS activation, and pw2/main. The depthwise path is analog
        # in Design B and is therefore excluded. Residual block outputs are
        # reported separately because they represent the same channel/spatial
        # population as pw2 and would otherwise double-count the denominator.
        hidden_spikes = sum(
            count
            for name, count in layer_spike_counts.items()
            if name != "input"
        )
        hidden_units = sum(
            units
            for name, units in layer_unit_counts.items()
            if name != "input"
        )

        # layer_unit_counts stores one timestep's population size. Since TTFS
        # allows at most one spike per neuron, denominator is the population
        # size, not population_size*T.
        global_spikes_per_neuron = (
            hidden_spikes / hidden_units
            if hidden_units > 0
            else 0.0
        )
        global_sparsity = 1.0 - global_spikes_per_neuron

        stats: Dict[str, object] = {
            "time_steps": self.time_steps,
            "readout_mode": self.readout_mode,
            "classifier_input_source": (
                "weighted_final_stage_hard_spikes_only"
                if self.readout_mode == "spike_integrator"
                else "class_first_spikes"
            ),
            "total_synops_estimate": total_synops,
            "global_hidden_spikes": hidden_spikes,
            "global_hidden_units": hidden_units,
            "global_spikes_per_neuron": global_spikes_per_neuron,
            "global_sparsity": global_sparsity,
            "synaptic_neuron_spikes_per_neuron": global_spikes_per_neuron,
            "synaptic_neuron_sparsity": global_sparsity,
            "block_output_events_per_neuron": (
                block_output_spikes / block_output_units
                if block_output_units > 0 else 0.0
            ),
            "layer_spike_counts": layer_spike_counts,
            "layer_unit_counts": layer_unit_counts,
            "stage_spike_distribution": {
                f"stage_{stage}": {
                    **{
                        f"spike_fraction_t{step}": (
                            stage_step_spike_counts[stage][step]
                            / max(stage_unit_counts[stage], 1)
                        )
                        for step in range(self.time_steps)
                    },
                    "silent_neuron_fraction": (
                        1.0 - float(stage_seen[stage].sum().item())
                        / max(stage_unit_counts[stage], 1)
                    ),
                    "repeated_spike_ratio": (
                        stage_repeated_counts[stage]
                        / max(stage_unit_counts[stage], 1)
                    ),
                    "mean_first_spike_time": (
                        sum(
                            step * stage_step_spike_counts[stage][step]
                            for step in range(self.time_steps)
                        )
                        / max(float(stage_seen[stage].sum().item()), 1.0)
                    ),
                }
                for stage in range(4)
            },
            "final_feature_diagnostics": {
                "feature_std_across_batch": float(
                    final_features.detach().float().std(
                        dim=0,
                        unbiased=False,
                    ).mean().item()
                ),
                "classwise_logit_std": logits.detach().float().std(
                    dim=0,
                    unbiased=False,
                ).cpu().tolist(),
            },
        }

        return logits, stats


def build_discrete_ttfs_convnext(
    model_size: str = "tiny",
    **kwargs,
) -> DiscreteTTFSConvNeXt:
    configs = {
        "nano": ((1, 1, 2, 1), (24, 48, 96, 192)),
        "tiny": ((3, 3, 9, 3), (96, 192, 384, 768)),
    }

    if model_size not in configs:
        raise ValueError(
            f"Unknown model_size={model_size!r}; "
            f"choose from {sorted(configs)}"
        )

    depths, dims = configs[model_size]
    return DiscreteTTFSConvNeXt(
        depths=depths,
        dims=dims,
        **kwargs,
    )
