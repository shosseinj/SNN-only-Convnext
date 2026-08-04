"""Pure discrete-time TTFS Spiking ConvNeXt.

Unlike the historical hybrid model, every feature-producing affine operation is
followed by a first-spike-only neuron and the network is simulated explicitly
for ``T`` timesteps. Ordinary convolutions/linears are synaptic operators on
binary spike tensors; analog activations are not propagated between blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateStep(torch.autograd.Function):
    """Binary spike in forward, fast-sigmoid surrogate in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        # Unit-gain SuperSpike / fast-sigmoid surrogate.  Multiplying by
        # ``slope`` makes the peak derivative ``slope`` and compounds
        # catastrophically through a deep explicitly-unrolled SNN.
        grad = 1.0 / (1.0 + slope * x.abs()).pow(2)
        return grad_output * grad, None


def spike_fn(x: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    return SurrogateStep.apply(x, slope)


class PositiveConv2d(nn.Conv2d):
    """Conv2d with optional non-negative effective weights.

    ``abs`` preserves the intended initialization scale while, unlike ReLU,
    allowing raw weights initialized below zero to receive gradients.
    """

    def __init__(self, *args, force_positive_weights: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_positive_weights = force_positive_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.abs() if self.force_positive_weights else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PositiveLinear(nn.Linear):
    """Linear with optional functional ReLU non-negative weights."""

    def __init__(self, *args, force_positive_weights: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_positive_weights = force_positive_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.abs() if self.force_positive_weights else self.weight
        return F.linear(x, w, self.bias)


class LearnableDelay(nn.Module):
    """Differentiable per-channel delay measured in discrete timesteps.

    The simulator remains discrete. During training a sigmoid gate controls when
    synaptic current becomes effective, allowing gradients to reach the delay.
    ``hard_values`` gives rounded delays for reporting.
    """

    def __init__(self, channels: int, time_steps: int, init_delay: float = 0.0, temperature: float = 0.5):
        super().__init__()
        self.time_steps = int(time_steps)
        self.temperature = float(temperature)
        self.raw_delay = nn.Parameter(torch.full((channels,), float(init_delay)))

    def values(self) -> torch.Tensor:
        # Straight-through bounds retain a useful gradient at exact delay 0;
        # sigmoid parameterization put zero at a saturated logit (~-9.21).
        bounded = self.raw_delay.clamp(0.0, float(max(self.time_steps - 1, 0)))
        return self.raw_delay + (bounded - self.raw_delay).detach()

    def hard_values(self) -> torch.Tensor:
        return self.values().round()

    def gate(self, step: int, ndim: int) -> torch.Tensor:
        d = self.values()
        shape = [1, d.numel()] + [1] * max(ndim - 2, 0)
        return torch.sigmoid((float(step) + 0.5 - d.view(*shape)) / self.temperature)


@dataclass
class NeuronState:
    membrane: torch.Tensor
    has_spiked: torch.Tensor
    first_spike: torch.Tensor


class FirstSpikeNeuron(nn.Module):
    """Integrate-and-fire neuron that can emit at most one spike."""

    def __init__(self, channels: int, time_steps: int, threshold: float = 1.0,
                 learnable_delay: bool = True, init_delay: float = 0.0,
                 delay_temperature: float = 0.5, surrogate_slope: float = 5.0,
                 threshold_mode: str = "fixed", threshold_min: float = 0.05,
                 threshold_max: float = 0.8):
        super().__init__()
        self.time_steps = int(time_steps)
        if threshold_mode not in {"fixed", "learnable_layer", "learnable_channel"}:
            raise ValueError(f"unknown threshold_mode={threshold_mode!r}")
        if threshold_mode != "fixed" and not threshold_min < threshold < threshold_max:
            raise ValueError("learnable threshold initialization must satisfy threshold_min < threshold < threshold_max")
        self.threshold_mode = threshold_mode
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        if threshold_mode == "fixed":
            self.threshold = nn.Parameter(torch.full((channels,), float(threshold)), requires_grad=False)
            self.register_parameter("raw_threshold", None)
        else:
            self.register_parameter("threshold", None)
            size = 1 if threshold_mode == "learnable_layer" else channels
            probability = (float(threshold) - threshold_min) / (threshold_max - threshold_min)
            raw = torch.logit(torch.full((size,), probability))
            self.raw_threshold = nn.Parameter(raw)
        self.delay = LearnableDelay(channels, time_steps, init_delay, delay_temperature) if learnable_delay else None
        self.surrogate_slope = float(surrogate_slope)

    def threshold_values(self) -> torch.Tensor:
        if self.threshold_mode == "fixed":
            return self.threshold
        return self.threshold_min + (self.threshold_max - self.threshold_min) * torch.sigmoid(self.raw_threshold)

    def init_state(self, current: torch.Tensor) -> NeuronState:
        zeros = torch.zeros_like(current)
        sentinel = torch.full_like(current, float(self.time_steps))
        return NeuronState(zeros, torch.zeros_like(current, dtype=torch.bool), sentinel)

    def forward_step(self, current: torch.Tensor, state: NeuronState, step: int) -> Tuple[torch.Tensor, NeuronState]:
        if self.delay is not None:
            current = current * self.delay.gate(step, current.ndim)
        membrane = state.membrane + current
        threshold = self.threshold_values()
        shape = [1, threshold.numel()] + [1] * max(current.ndim - 2, 0)
        candidate = spike_fn(membrane - threshold.view(*shape), self.surrogate_slope)
        spike = candidate * (~state.has_spiked).to(candidate.dtype)
        newly = spike.detach().bool() & (~state.has_spiked)
        first_spike = torch.where(newly, torch.full_like(state.first_spike, float(step)), state.first_spike)
        return spike, NeuronState(membrane, state.has_spiked | newly, first_spike)


class SpikingDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, time_steps: int,
                 force_positive_weights: bool, learnable_delay: bool, init_delay: float, threshold: float,
                 threshold_mode: str, threshold_min: float, threshold_max: float):
        super().__init__()
        self.synapse = PositiveConv2d(in_ch, out_ch, kernel_size, stride=stride,
                                      force_positive_weights=force_positive_weights)
        self.neuron = FirstSpikeNeuron(out_ch, time_steps, threshold, learnable_delay, init_delay,
                                       threshold_mode=threshold_mode, threshold_min=threshold_min,
                                       threshold_max=threshold_max)

    def init_state(self, sample_spikes: torch.Tensor) -> NeuronState:
        return self.neuron.init_state(self.synapse(sample_spikes))

    def forward_step(self, spikes: torch.Tensor, state: NeuronState, step: int):
        return self.neuron.forward_step(self.synapse(spikes), state, step)


class SpikingConvNeXtBlock(nn.Module):
    """Depthwise -> pointwise expansion -> pointwise projection, all spiking."""

    def __init__(self, dim: int, time_steps: int, force_positive_weights: bool = True,
                 learnable_delay: bool = True, init_delay: float = 0.0, threshold: float = 1.0,
                 residual: bool = True, threshold_mode: str = "fixed",
                 threshold_min: float = 0.05, threshold_max: float = 0.8):
        super().__init__()
        self.dw = PositiveConv2d(dim, dim, 7, padding=3, groups=dim,
                                 force_positive_weights=force_positive_weights)
        neuron_kwargs = dict(threshold_mode=threshold_mode, threshold_min=threshold_min,
                             threshold_max=threshold_max)
        self.dw_neuron = FirstSpikeNeuron(dim, time_steps, threshold, learnable_delay, init_delay, **neuron_kwargs)
        self.pw1 = PositiveConv2d(dim, 4 * dim, 1, force_positive_weights=force_positive_weights)
        self.pw1_neuron = FirstSpikeNeuron(4 * dim, time_steps, threshold, learnable_delay, init_delay, **neuron_kwargs)
        self.pw2 = PositiveConv2d(4 * dim, dim, 1, force_positive_weights=force_positive_weights)
        self.pw2_neuron = FirstSpikeNeuron(dim, time_steps, threshold, learnable_delay, init_delay, **neuron_kwargs)
        self.time_steps = time_steps
        self.residual = bool(residual)

    def init_states(self, sample: torch.Tensor):
        dw_cur = self.dw(sample)
        s1 = self.dw_neuron.init_state(dw_cur)
        pw1_cur = self.pw1(torch.zeros_like(dw_cur))
        s2 = self.pw1_neuron.init_state(pw1_cur)
        pw2_cur = self.pw2(torch.zeros_like(pw1_cur))
        s3 = self.pw2_neuron.init_state(pw2_cur)
        residual_has_spiked = torch.zeros_like(sample, dtype=torch.bool)
        residual_first = torch.full_like(sample, float(self.time_steps))
        return [s1, s2, s3, residual_has_spiked, residual_first]

    def forward_step(self, x_spike: torch.Tensor, states, step: int):
        s1, s2, s3, out_has_spiked, out_first = states
        z, s1 = self.dw_neuron.forward_step(self.dw(x_spike), s1, step)
        z, s2 = self.pw1_neuron.forward_step(self.pw1(z), s2, step)
        main, s3 = self.pw2_neuron.forward_step(self.pw2(z), s3, step)
        # Earliest-spike residual fusion in event space (logical OR), then first-spike mask.
        union = torch.clamp(main + x_spike, 0.0, 1.0) if self.residual else main
        out = union * (~out_has_spiked).to(union.dtype)
        newly = out.detach().bool() & (~out_has_spiked)
        out_first = torch.where(newly, torch.full_like(out_first, float(step)), out_first)
        return out, [s1, s2, s3, out_has_spiked | newly, out_first]


class DiscreteTTFSConvNeXt(nn.Module):
    """A pure, explicitly simulated, first-spike-only SNN."""

    def __init__(self, in_chans: int = 3, num_classes: int = 10,
                 depths: Sequence[int] = (3, 3, 9, 3), dims: Sequence[int] = (96, 192, 384, 768),
                 time_steps: int = 5, threshold: float = 1.0,
                 force_positive_weights: bool = True, learnable_delay: bool = True,
                 init_delay: float = 0.0, residual: bool = True,
                 threshold_mode: str = "fixed", threshold_min: float = 0.05,
                 threshold_max: float = 0.8, readout_mode: str = "ttfs",
                 soft_time_beta: float = 10.0):
        super().__init__()
        if time_steps < 2:
            raise ValueError("time_steps must be >= 2 for TTFS ordering")
        self.time_steps = int(time_steps)
        self.num_classes = int(num_classes)
        self.temporal_model_type = "DISCRETE_TIME_TTFS_SNN"
        self.maximum_spikes_per_neuron = 1
        self.residual = bool(residual)
        self.threshold_mode = threshold_mode
        if readout_mode not in {"ttfs", "membrane", "hybrid", "soft_time"}:
            raise ValueError(f"unknown readout_mode={readout_mode!r}")
        if soft_time_beta <= 0:
            raise ValueError("soft_time_beta must be positive")
        self.readout_mode = readout_mode
        self.soft_time_beta = float(soft_time_beta)

        self.downsamples = nn.ModuleList()
        self.downsamples.append(SpikingDownsample(in_chans, dims[0], 4, 4, time_steps,
                                                  force_positive_weights, learnable_delay, init_delay, threshold,
                                                  threshold_mode, threshold_min, threshold_max))
        for i in range(3):
            self.downsamples.append(SpikingDownsample(dims[i], dims[i + 1], 2, 2, time_steps,
                                                      force_positive_weights, learnable_delay, init_delay, threshold,
                                                      threshold_mode, threshold_min, threshold_max))
        self.stages = nn.ModuleList([
            nn.ModuleList([SpikingConvNeXtBlock(dims[i], time_steps, force_positive_weights,
                                                learnable_delay, init_delay, threshold, residual,
                                                threshold_mode, threshold_min, threshold_max)
                           for _ in range(depths[i])]) for i in range(4)
        ])
        # Readout synapse is signed so classes can receive distinct excitatory/inhibitory evidence.
        # The positive-weight constraint is applied to feature-extraction synapses.
        self.classifier = PositiveLinear(dims[-1], num_classes,
                                         force_positive_weights=False)
        self.hybrid_classifier = (
            PositiveLinear(2 * dims[-1], num_classes, force_positive_weights=False)
            if readout_mode == "hybrid" else None
        )
        self.output_neuron = FirstSpikeNeuron(num_classes, time_steps, threshold,
                                              learnable_delay, init_delay, threshold_mode=threshold_mode,
                                              threshold_min=threshold_min, threshold_max=threshold_max)
        self._initialize_weights()

    def active_classifier_parameters(self):
        classifier = self.hybrid_classifier if self.readout_mode == "hybrid" else self.classifier
        return classifier.parameters()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, PositiveConv2d) and m.force_positive_weights:
                # Positive weights have a non-zero mean, so the usual fixed
                # normal std makes fan-in grow the membrane with network
                # width and immediately saturates deep neurons.  Initialize
                # the total available first-spike charge near 2*threshold.
                fan_in = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
                # Every encoded input pixel emits exactly once, so using the
                # same 2x charge budget as sparse hidden layers saturates the
                # wide Tiny network from its stem onward.  Center the stem at
                # one threshold; retain 2x for progressively sparse hidden
                # event tensors.
                if m is self.downsamples[0].synapse:
                    charge_budget = 1.0
                else:
                    charge_budget = 2.0
                initial_threshold = float(self.downsamples[0].neuron.threshold_values()[0].detach())
                std = (charge_budget * initial_threshold) / (0.8 * fan_in)
                nn.init.trunc_normal_(m.weight, std=std)
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def latency_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized intensity [0,1] to first-spike indices [0,T-1]."""
        x = x.clamp(0.0, 1.0)
        return torch.round((1.0 - x) * (self.time_steps - 1)).long()

    def _count_synops(self, module: nn.Module, input_spikes: torch.Tensor) -> float:
        spikes = float(input_spikes.detach().sum().item())
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            fanout = (module.out_channels / module.groups) * kh * kw
        elif isinstance(module, nn.Linear):
            fanout = module.out_features
        else:
            fanout = 0.0
        return spikes * fanout

    def forward(self, x: torch.Tensor, return_stats: bool = False, return_layer_stats: bool | None = None):
        if return_layer_stats is None:
            return_layer_stats = return_stats
        first_idx = self.latency_encode(x)
        layer_events: Dict[str, torch.Tensor] = {}
        layer_times: Dict[str, torch.Tensor] = {}
        stage_events: Dict[int, List[torch.Tensor]] = {stage: [] for stage in range(4)}
        stage_membrane_traces: Dict[int, List[torch.Tensor]] = {stage: [] for stage in range(4)}

        def record(name: str, event: torch.Tensor) -> None:
            if return_layer_stats:
                detached = event.detach()
                layer_events[name] = layer_events.get(name, torch.zeros_like(detached)) + detached
                layer_times[name] = layer_times.get(name, torch.zeros_like(detached)) + detached * (
                    (self.time_steps - t) / float(self.time_steps)
                )
        # Initial zero spikes define all state shapes without consuming an event.
        input_zero = torch.zeros_like(x)
        ds_states = []
        block_states: List[List] = []
        sample = input_zero
        for stage_idx in range(4):
            ds_state = self.downsamples[stage_idx].init_state(sample)
            ds_states.append(ds_state)
            sample = torch.zeros_like(ds_state.membrane)
            stage_states = []
            for block in self.stages[stage_idx]:
                st = block.init_states(sample)
                stage_states.append(st)
            block_states.append(stage_states)
        pooled_sample = sample.mean((-2, -1))
        output_state = self.output_neuron.init_state(self.classifier(pooled_sample))

        spike_count_per_t = []
        synops_per_t = []
        class_score = torch.zeros((x.shape[0], self.num_classes), device=x.device, dtype=x.dtype)
        ttfs_feature_score = torch.zeros_like(sample)
        final_membrane_trace: List[torch.Tensor] = []
        for t in range(self.time_steps):
            spikes = (first_idx == t).to(x.dtype)
            record("input", spikes)
            step_spikes = float(spikes.detach().sum().item()) if return_stats else 0.0
            step_synops = 0.0
            for stage_idx in range(4):
                if return_stats:
                    step_synops += self._count_synops(self.downsamples[stage_idx].synapse, spikes)
                spikes, ds_states[stage_idx] = self.downsamples[stage_idx].forward_step(spikes, ds_states[stage_idx], t)
                record(f"downsamples.{stage_idx}", spikes)
                if return_stats:
                    step_spikes += float(spikes.detach().sum().item())
                for block_idx, block in enumerate(self.stages[stage_idx]):
                    if return_stats:
                        step_synops += self._count_synops(block.dw, spikes)
                        # Approximate internal event SynOps from actual intermediate spikes is not exposed;
                        # count block input for all three synapses and label this an estimate.
                        step_synops += self._count_synops(block.pw1, spikes)
                        step_synops += self._count_synops(block.pw2, spikes)
                    spikes, block_states[stage_idx][block_idx] = block.forward_step(
                        spikes, block_states[stage_idx][block_idx], t
                    )
                    record(f"stages.{stage_idx}.{block_idx}", spikes)
                    if return_stats:
                        step_spikes += float(spikes.detach().sum().item())
                if return_stats:
                    stage_events[stage_idx].append(spikes.detach())
                    stage_membrane_traces[stage_idx].append(
                        block_states[stage_idx][-1][2].membrane.detach()
                    )
            ttfs_feature_score = ttfs_feature_score + spikes * (
                (self.time_steps - t) / float(self.time_steps)
            )
            final_membrane_trace.append(block_states[3][-1][2].membrane)
            pooled = spikes.mean((-2, -1))
            if return_stats:
                step_synops += self._count_synops(self.classifier, pooled)
            class_current = self.classifier(pooled)
            class_spike, output_state = self.output_neuron.forward_step(class_current, output_state, t)
            record("output", class_spike)
            # Differentiable TTFS score: an earlier first spike receives a larger score.
            class_score = class_score + class_spike * ((self.time_steps - t) / float(self.time_steps))
            if return_stats:
                step_spikes += float(class_spike.detach().sum().item())
                spike_count_per_t.append(step_spikes)
                synops_per_t.append(step_synops)

        first_time = output_state.first_spike
        pooled_ttfs = ttfs_feature_score.mean((-2, -1))
        final_membrane = final_membrane_trace[-1]
        pooled_membrane = final_membrane.mean((-2, -1))
        if self.readout_mode == "ttfs":
            # Preserve the historical differentiable class first-spike score exactly.
            final_features = pooled_ttfs
            logits = class_score
        elif self.readout_mode == "membrane":
            final_features = pooled_membrane
            logits = self.classifier(final_features)
        elif self.readout_mode == "hybrid":
            final_features = torch.cat((pooled_ttfs, pooled_membrane), dim=1)
            logits = self.hybrid_classifier(final_features)
        else:
            threshold = self.stages[3][-1].pw2_neuron.threshold_values()
            shape = [1, 1, threshold.numel()] + [1] * (final_membrane.dim() - 2)
            membrane_sequence = torch.stack(final_membrane_trace, dim=0)
            crossing = torch.sigmoid(
                self.soft_time_beta * (membrane_sequence - threshold.view(*shape))
            )
            survival = torch.ones_like(crossing[0])
            expected_time = torch.zeros_like(crossing[0])
            for step in range(self.time_steps):
                first_crossing = survival * crossing[step]
                expected_time = expected_time + float(step) * first_crossing
                survival = survival * (1.0 - crossing[step])
            expected_time = expected_time + float(self.time_steps) * survival
            final_features = expected_time.mean((-2, -1))
            logits = self.classifier(final_features)
        if not return_stats:
            return logits
        total_units = sum(s.numel() for s in [output_state.has_spiked])
        stats: Dict[str, object] = {
            "time_steps": self.time_steps,
            "spike_count_per_timestep": spike_count_per_t,
            "total_spikes": float(sum(spike_count_per_t)),
            "synops_per_timestep_estimate": synops_per_t,
            "total_synops_estimate": float(sum(synops_per_t)),
            "output_silent_fraction": float((~output_state.has_spiked).float().mean().item()),
            "output_first_spike_times": first_time.detach(),
            "synops_status": "approximate_event_fanout",
            "readout_mode": self.readout_mode,
            "stage_membrane_diagnostics": {
                f"stage_{stage}": {
                    "membrane_mean": float(membranes[-1].float().mean().item()),
                    "membrane_std": float(membranes[-1].float().std().item()),
                    "effective_threshold_mean": float(
                        self.stages[stage][-1].pw2_neuron.threshold_values().detach().float().mean().item()
                    ),
                    "membrane_minus_threshold_mean": float((
                        membranes[-1] - self.stages[stage][-1].pw2_neuron.threshold_values().detach().view(
                            1, -1, *([1] * (membranes[-1].dim() - 2))
                        )
                    ).float().mean().item()),
                    "near_threshold_fraction": float((
                        (membranes[-1] - self.stages[stage][-1].pw2_neuron.threshold_values().detach().view(
                            1, -1, *([1] * (membranes[-1].dim() - 2))
                        )).abs() < 0.05
                    ).float().mean().item()),
                }
                for stage, membranes in stage_membrane_traces.items() if membranes
            },
            "final_feature_diagnostics": {
                "feature_std_across_batch": float(
                    final_features.detach().float().std(dim=0).mean().item()
                ),
                "mean_pairwise_feature_distance": float(
                    torch.pdist(final_features.detach().float()).mean().item()
                ) if final_features.shape[0] > 1 else 0.0,
                "distinct_logit_vectors": int(torch.unique(logits.detach(), dim=0).shape[0]),
                "classwise_logit_std": logits.detach().float().std(dim=0).cpu().tolist(),
            },
            "stage_spike_distribution": {
                f"stage_{stage}": {
                    **{
                        f"spike_fraction_t{step}": float(events[step].float().mean().item())
                        for step in range(len(events))
                    },
                    "silent_neuron_fraction": float(
                        (torch.stack(events).sum(0) == 0).float().mean().item()
                    ),
                    "repeated_spike_ratio": float(
                        (torch.stack(events).sum(0) > 1).float().mean().item()
                    ),
                }
                for stage, events in stage_events.items() if events
            },
            "layer_event_stats": {
                name: {
                    "spikes": float(events.sum().item()),
                    "spike_fraction": float(events.mean().item()),
                    "max_spikes_per_neuron": float(events.max().item()),
                    "unique_sample_representations": int(torch.unique(events.flatten(1), dim=0).shape[0]),
                    "unique_sample_ttfs": int(torch.unique(layer_times[name].flatten(1), dim=0).shape[0]),
                }
                for name, events in layer_events.items()
            },
        }
        return logits, stats


def build_discrete_ttfs_convnext(model_size: str = "tiny", **kwargs) -> DiscreteTTFSConvNeXt:
    configs = {
        "nano": ((1, 1, 2, 1), (24, 48, 96, 192)),
        "tiny": ((3, 3, 9, 3), (96, 192, 384, 768)),
    }
    if model_size not in configs:
        raise ValueError(f"Unknown model_size={model_size!r}; choose from {sorted(configs)}")
    depths, dims = configs[model_size]
    return DiscreteTTFSConvNeXt(depths=depths, dims=dims, **kwargs)
