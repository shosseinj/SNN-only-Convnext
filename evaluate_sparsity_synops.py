#!/usr/bin/env python3
"""
Evaluate accuracy + sparsity + SynOps (synaptic operations) for CIFAR-10 Spiking ConvNeXt.
SynOps = sum_over_layers( spike_count * fan_out ), following Sorbaro et al. 2020.
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict

from datasets import build_dataset
from models.convnext import ConvNeXtSpiking
from evaluator import (
    TemporalValidityDiagnostics,
    classifier_path_verification,
    temporal_diagnostics_markdown,
)
import utils

# ---------- Try to import thop for MAC calculation ----------
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' not installed. Please install it (pip install thop) to auto-compute analog MACs, or provide --analog_macs manually.")

# ---------- sparsity hooks (same as before) ----------
class SparsityHook:
    """Measure activation sparsity: fraction of outputs == t_max (no spike). Also stores output shape."""
    def __init__(self, layer_name, t_max=1.0):
        self.layer_name = layer_name
        self.t_max = t_max
        self.sparsity = 0.0
        self.num_silent = 0
        self.num_total = 0
        self.output_shape = None   # will be set during first forward

    def __call__(self, module, input, output):
        if not isinstance(output, torch.Tensor):
            return
        # output are spike times (t_min..t_max)
        if self.output_shape is None:
            self.output_shape = output.shape[1:]   # drop batch dimension
        silent = (output >= self.t_max - 1e-6)   # neurons that never spiked
        self.num_silent = silent.sum().item()
        self.num_total = output.numel()
        self.sparsity = self.num_silent / self.num_total if self.num_total > 0 else 0.0


class WeightSparsityHook:
    """Measure weight sparsity for downsampling Conv2d layers."""
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.sparsity = 0.0
        self.num_zero = 0
        self.num_total = 0

    def __call__(self, module, input, output):
        if isinstance(module, nn.Conv2d) and module.weight is not None:
            w = module.weight.data
            self.num_zero = (w == 0).sum().item()
            self.num_total = w.numel()
            self.sparsity = self.num_zero / self.num_total if self.num_total > 0 else 0.0


def register_sparsity_hooks(model, t_max):
    """Register hooks for all spiking blocks and downsampling convs."""
    hooks = {}
    for name, module in model.named_modules():
        if hasattr(module, "t_max"):          # spiking block (e.g., TTFS layer)
            hook = SparsityHook(name, t_max)
            module.register_forward_hook(hook)
            hooks[name] = hook
        elif isinstance(module, nn.Conv2d) and 'downsample_layers' in name:
            hook = WeightSparsityHook(name)
            module.register_forward_hook(hook)
            hooks[name] = hook
    print(f"Registered {len(hooks)} sparsity hooks")
    return hooks


# ---------- Build mapping: spiking module -> next trainable layer ----------
def build_spike_to_weight_map_via_forward(model, device, sample_input):
    """
    Run a forward pass with a dummy input, record the order of module calls,
    then for each spiking module (has t_max) find the next trainable module
    (Conv2d/Linear) in that order. Returns dict: spike_module_name -> (next_module, fan_out)
    """
    from collections import OrderedDict

    # Hook to record module names during forward
    order = []
    def record_order(module, inp, out):
        # Get full name by searching in model.named_modules()
        for name, m in model.named_modules():
            if m is module:
                order.append(name)
                break

    # Register forward hook on all modules
    handles = []
    for module in model.modules():
        handles.append(module.register_forward_hook(record_order))

    # Run dummy forward
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)

    # Remove hooks
    for h in handles:
        h.remove()

    # Identify which modules are spiking and which are trainable
    spike_names = set()
    trainable_names = set()
    for name, module in model.named_modules():
        if hasattr(module, "t_max"):
            spike_names.add(name)
        if isinstance(module, (nn.Conv2d, nn.Linear)) and any(p.requires_grad for p in module.parameters()):
            trainable_names.add(name)

    # Build mapping: for each spike module, find next trainable module in order
    spike_to_next = {}
    for i, mod_name in enumerate(order):
        if mod_name in spike_names:
            # Look forward from i+1 to find first trainable module
            next_trainable = None
            for j in range(i+1, len(order)):
                if order[j] in trainable_names:
                    next_trainable = order[j]
                    break
            if next_trainable is not None:
                next_mod = dict(model.named_modules())[next_trainable]
                # Compute fan_out per spike
                if isinstance(next_mod, nn.Conv2d):
                    # For Conv2d, each input neuron (spike) connects to:
                    # out_channels * kernel_h * kernel_w target neurons
                    k_h, k_w = next_mod.kernel_size
                    fan_out = next_mod.out_channels * k_h * k_w
                elif isinstance(next_mod, nn.Linear):
                    fan_out = next_mod.out_features
                else:
                    fan_out = 1  # fallback
                spike_to_next[mod_name] = (next_mod, fan_out)
            else:
                print(f"Warning: No trainable module after spiking layer {mod_name}")
        elif mod_name in trainable_names:
            # Not needed
            pass

    print(f"Found mapping for {len(spike_to_next)} spiking layers")
    return spike_to_next

# ---------- Compute SynOps from collected sparsity ----------
def compute_synops(spike_hooks, spike_to_weight_map, model, device):
    """
    spike_hooks: dict name -> SparsityHook (contains sparsity, output_shape)
    spike_to_weight_map: dict spike_name -> (next_layer, fan_out)
    Returns total SynOps (int) and a dict per-spike layer details.
    """
    total_synops = 0
    layer_details = {}
    for spike_name, (next_layer, fan_out) in spike_to_weight_map.items():
        hook = spike_hooks.get(spike_name)
        if hook is None:
            print(f"Warning: no sparsity hook for {spike_name}")
            continue
        num_neurons = np.prod(hook.output_shape)   # total neurons in this layer
        spike_count = num_neurons * (1 - hook.sparsity)   # number of neurons that spiked
        synops_this = spike_count * fan_out
        total_synops += synops_this
        layer_details[spike_name] = {
            'num_neurons': num_neurons,
            'sparsity (%)': hook.sparsity * 100,
            'spike_count': spike_count,
            'fan_out': fan_out,
            'SynOps': synops_this
        }
    return total_synops, layer_details


# ---------- Evaluate accuracy + sparsity + SynOps ----------
def evaluate_sparsity_and_synops(model, dataloader, device, args):
    """Run evaluation and return accuracy, global sparsity, total SynOps, and per-layer details."""
    model.eval()
    
    # Get a single batch to build mapping (using first batch images)
    sample_images, _ = next(iter(dataloader))
    sample_images = sample_images[:1].to(device)
    spike_to_weight_map = build_spike_to_weight_map_via_forward(model, device, sample_images)
    
    # Register sparsity hooks (same as before)
    hooks = register_sparsity_hooks(model, args.ttfs_tmax)

    temporal_diagnostics = None
    if args.temporal_diagnostics:
        temporal_diagnostics = TemporalValidityDiagnostics(
            model, args.ttfs_tmin, args.ttfs_tmax
        )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                acc_sofar = 100.0 * correct / total
                print(f"Batch {batch_idx+1}/{len(dataloader)} | Acc={acc_sofar:.2f}%")

    if temporal_diagnostics is not None:
        temporal_diagnostics.disable()
        temporal_validity = temporal_diagnostics.result()
    else:
        temporal_validity = None

    accuracy = 100.0 * correct / total

    # Collect sparsity from hooks (only SparsityHook)
    spike_hooks = {name: h for name, h in hooks.items() if isinstance(h, SparsityHook)}
    
    # Compute SynOps using the mapping
    total_synops = 0
    synops_details = {}
    for spike_name, (next_mod, fan_out) in spike_to_weight_map.items():
        hook = spike_hooks.get(spike_name)
        if hook is None:
            # Try to find hook by partial match (e.g., if names differ due to container nesting)
            # Simple fallback: find hook where hook.layer_name ends with spike_name
            found = None
            for hn, h in spike_hooks.items():
                if hn.endswith(spike_name):
                    found = h
                    break
            if found is None:
                print(f"Warning: No sparsity hook for {spike_name}")
                continue
            hook = found

        num_neurons = np.prod(hook.output_shape) if hook.output_shape is not None else 0
        if num_neurons == 0:
            continue
        spike_count = num_neurons * (1 - hook.sparsity)   # number of neurons that spiked
        synops_this = spike_count * fan_out
        total_synops += synops_this
        synops_details[spike_name] = {
            'num_neurons': num_neurons,
            'sparsity (%)': hook.sparsity * 100,
            'spike_count': spike_count,
            'fan_out': fan_out,
            'SynOps': synops_this
        }

    # Also compute global activation sparsity
    total_silent = sum(h.num_silent for h in spike_hooks.values())
    total_neurons = sum(h.num_total for h in spike_hooks.values())
    global_sparsity = (total_silent / total_neurons * 100) if total_neurons > 0 else 0.0

    return accuracy, global_sparsity, total_synops, synops_details, temporal_validity

# ---------- Compute analog MACs (optional) ----------
def get_analog_macs(model_func, input_size=(3, 32, 32), device='cuda'):
    """Create an analog version of ConvNeXt (non-spiking) and compute MACs using thop."""
    if not THOP_AVAILABLE:
        return None
    # We need to define the analog counterpart. For simplicity, use the same architecture but with standard ReLU.
    # Since your ConvNeXtSpiking might differ, we'll assume the user provides MACs manually.
    # Here we create a dummy analog model (you may need to import your analog ConvNeXt).
    try:
        from models.convnext import ConvNeXt  # hypothetical analog version
        analog_model = ConvNeXt(num_classes=args.nb_classes)
    except ImportError:
        print("Analog ConvNeXt not found. Please install `thop` and provide --analog_macs.")
        return None
    analog_model.to(device).eval()
    dummy = torch.randn(1, *input_size).to(device)
    macs, params = profile(analog_model, inputs=(dummy,), verbose=False)
    return macs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# ---------- Main ----------
def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt evaluation (accuracy + sparsity + SynOps)', add_help=False)

    parser.add_argument('--data_path', default='./cifar_data/', type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--nb_classes', default=10, type=int)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='CIFAR', choices=['CIFAR', 'IMNET', 'image_folder'])
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=False)
    parser.add_argument('--crop_pct', default=None, type=float)

    # Augmentation (still needed for dataset builder)
    parser.add_argument('--color_jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', type=str2bool, default=False)

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float)
    parser.add_argument('--head_init_scale', default=1.0, type=float)
    parser.add_argument('--ttfs_tmin', type=float, default=0.0)
    parser.add_argument('--spiking', type=str2bool, default=True)
    parser.add_argument('--run_eval', type=str2bool, default=True)
    parser.add_argument('--ttfs_tmax', type=float, default=1.0)
    parser.add_argument('--ttfs_force_pos_weights', type=str2bool, default=False)
    parser.add_argument('--ttfs_init_delay', type=float, default=0.0)
    parser.add_argument('--ttfs_stage_delays', type=str, default="0.4,0.0,0.00,0.0")

    # Checkpoint
    # parser.add_argument('--load_weights', default='./ckpt_residual_Di_96.09/checkpoint-298.pth', type=str)
    parser.add_argument('--load_weights', default='./weights/ckpt_residual_Di_96.09/checkpoint_residual_Di_96.09.pth', type=str)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Evaluation
    parser.add_argument('--eval', type=str2bool, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_amp', type=str2bool, default=False)
    parser.add_argument('--temporal_diagnostics', type=str2bool, default=False,
                        help='Collect read-only temporal validity statistics for every SpikingBlock')
    parser.add_argument('--max_eval_batches', type=int, default=None,
                        help='Optional evaluation batch limit for smoke tests only')

    # Dummy arguments
    parser.add_argument('--dist_eval', type=str2bool, default=True)
    parser.add_argument('--disable_eval', type=str2bool, default=False)
    parser.add_argument('--distributed', type=str2bool, default=False)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--finetune', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--analog_macs', type=float, default=306000000,
                        help='Pre-computed MACs of the analog ConvNeXt. If not provided and thop is available, will try to compute.')
    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build dataset and loader (as in your code)
    args.disable_eval = False
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    print(f"Validation samples: {len(dataset_val)}")
    loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # Create spiking model
    stage_delays = None
    if args.ttfs_stage_delays:
        try:
            stage_delays = [float(d.strip()) for d in args.ttfs_stage_delays.split(',')]
            if len(stage_delays) != 4:
                stage_delays = None
        except Exception:
            stage_delays = None

    model = ConvNeXtSpiking(
        in_chans=3,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
        t_min=args.ttfs_tmin,
        t_max=args.ttfs_tmax,
        force_positive_weights=args.ttfs_force_pos_weights,
        init_delay=args.ttfs_init_delay,
        stage_delays=stage_delays
    )

    # Load checkpoint (as in your code)
    if args.load_weights:
        load_path = os.path.normpath(args.load_weights)
        if os.path.isdir(load_path):
            candidates = [f for f in os.listdir(load_path) if f.endswith(('.pth', '.pt'))]
            if not candidates:
                raise FileNotFoundError(f"No checkpoint in {load_path}")
            load_path = os.path.join(load_path, sorted(candidates, key=os.path.getmtime)[-1])
        print(f"Loading weights: {load_path}")
        ckpt = torch.load(load_path, map_location='cpu', weights_only=False)
        state_dict = None
        for key in args.model_key.split('|'):
            if key in ckpt:
                state_dict = ckpt[key]
                break
        if state_dict is None:
            state_dict = ckpt
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
        print("Checkpoint loaded.")

    model = model.to(device)
    model.eval()

    acc = 0.0
    global_sparsity = 0.0
    total_synops = 0.0
    synops_details = {}
    temporal_validity = None
    
    print("\n" + "="*60 )
    if args.run_eval:
    # Build spike -> weight mapping for SynOps
        acc, global_sparsity, total_synops, synops_details, temporal_validity = evaluate_sparsity_and_synops(
                    model, loader, device, args
                )
        print(f"Accuracy: {acc:.2f}%")

    # Compute or retrieve analog MACs
    if args.analog_macs is None and THOP_AVAILABLE:
        print("\nAttempting to compute analog MACs automatically...")
        analog_macs = get_analog_macs(model, input_size=(3, 32, 32), device=device)
    else:
        analog_macs = args.analog_macs

    # Report results
    print(f"Global sparsity (silent neurons): {global_sparsity:.2f}%")
    print(f"  -> Efficiency: {100 - global_sparsity:.2f}% of neurons spike")
    print(f"\nTotal Synaptic Operations (SynOps): {total_synops:.0f}  ({total_synops/1e6:.2f}M)")
    if analog_macs is not None:
        print(f"Analog ConvNeXt MACs: {analog_macs:.0f}  ({analog_macs/1e6:.2f}M)")
        ratio = total_synops / analog_macs
        print(f"SynOps / MACs ratio: {ratio:.3f}")
        if ratio < 1.0:
            print("✓ SNN is more efficient than analog CNN (SynOps < MACs).")
        else:
            print("✗ SNN currently uses more operations than analog. Try stronger SynOp regularization.")
    else:
        print("Analog MACs not provided. Use --analog_macs to enable comparison.")
    print("="*60)

    # Optional: per-layer SynOps breakdown
    print("\nPer-layer SynOps (top 10 contributors):")
    sorted_layers = sorted(synops_details.items(), key=lambda x: -x[1]['SynOps'])
    for name, detail in sorted_layers[:10]:
        print(f"  {name:<50} SynOps={detail['SynOps']:.0f}  (sparsity={detail['sparsity (%)']:.1f}%, fan_out={detail['fan_out']})")


    # Save machine-readable results and a compact generated report.
    results_dir = args.output_dir or "."
    os.makedirs(results_dir, exist_ok=True)
    evaluation_results = {
        "accuracy": acc,
        "global_sparsity_percent": global_sparsity,
        "total_synops": total_synops,
        "temporal_model_type": "HYBRID_CONTINUOUS_TTFS",
        "discrete_simulation_steps": None,
        "temporal_window": [args.ttfs_tmin, args.ttfs_tmax],
        "prediction_rule": "argmax_on_negative_pooled_time",
        "classifier_path_verification": classifier_path_verification(),
        "temporal_validity_status": (
            temporal_validity["status"] if temporal_validity is not None else "NOT_RUN"
        ),
        "temporal_validity": temporal_validity,
    }
    with open(os.path.join(results_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, allow_nan=False)
        f.write("\n")
    with open(os.path.join(results_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(temporal_diagnostics_markdown(evaluation_results) if temporal_validity is not None else
                "## Temporal Validity Diagnostics\n\nDiagnostics disabled.\n")

    if args.output_dir:
        out_file = os.path.join(results_dir, "sparsity_synops_results.txt")
        with open(out_file, 'w') as f:
            f.write(f"Checkpoint: {args.load_weights}\n")
            f.write(f"Accuracy: {acc:.2f}%\n")
            f.write(f"Global sparsity: {global_sparsity:.2f}%\n")
            f.write(f"Total SynOps: {total_synops:.0f} ({total_synops/1e6:.2f}M)\n")
            if analog_macs:
                f.write(f"Analog MACs: {analog_macs:.0f} ({analog_macs/1e6:.2f}M)\n")
                f.write(f"SynOps/MACs ratio: {ratio:.3f}\n")
            f.write("Per-layer SynOps:\n")
            for name, detail in sorted_layers:
                f.write(f"  {name}: SynOps={detail['SynOps']:.0f}, sparsity={detail['sparsity (%)']:.1f}%, fan_out={detail['fan_out']}\n")
        print(f"\nResults saved to {out_file}")

    return acc, global_sparsity, total_synops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt evaluation with sparsity and SynOps', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
