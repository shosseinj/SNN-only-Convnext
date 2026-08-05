#!/usr/bin/env python3
"""Train the pure discrete-time TTFS Spiking ConvNeXt baseline."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# from models.discrete_ttfs_convnext import FirstSpikeNeuron, build_discrete_ttfs_convnext
# from models.discrete_ttfs_convnext_design_b import FirstSpikeNeuron, build_discrete_ttfs_convnext
from models.discrete_ttfs_convnext_design_b_gelu import FirstSpikeNeuron, build_discrete_ttfs_convnext
from test.imagenet_init import load_torchvision_convnext_tiny
from test.reproducibility import seed_everything, seed_worker
from experiment_utils import atomic_json_dump, append_jsonl, count_parameters, model_size_mb, save_checkpoint


def str2bool(v):
    if isinstance(v, bool): return v
    if str(v).lower() in {"1","true","yes","y"}: return True
    if str(v).lower() in {"0","false","no","n"}: return False
    raise argparse.ArgumentTypeError("expected boolean")


def parser():
    p = argparse.ArgumentParser("Pure discrete TTFS ConvNeXt trainer")
    p.add_argument("--experiment_name", default="cifar10_baseline_pure_snn_ttfs_t5_seed42")
    p.add_argument("--data_path", default="../cifar_data")
    p.add_argument("--output_dir", default="", help="generated from training settings when omitted")
    p.add_argument("--model_size", choices=["nano","tiny"], default="tiny")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--time_steps", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--threshold_mode", choices=["fixed", "learnable_layer", "learnable_channel"], default="learnable_channel")
    p.add_argument("--threshold_min", type=float, default=0.1)
    p.add_argument("--threshold_max", type=float, default=1.2)
    p.add_argument("--threshold_freeze_epochs", type=int, default=3)
    p.add_argument(
        "--readout_mode",
        choices=["ttfs", "spike_integrator"],
        default="spike_integrator",
    )
    p.add_argument("--learnable_delay", type=str2bool, default=False)
    p.add_argument("--residual", type=str2bool, default=True)
    p.add_argument("--init_delay", type=float, default=0.5)
    p.add_argument("--input_no_spike_threshold", type=float, default=0.05)
    p.add_argument("--residual_main_gradient_scale", type=float, default=0.25)
    p.add_argument("--surrogate_grad_clip", type=float, default=16.0)
    p.add_argument("--force_positive_weights", type=str2bool, default=False)
    p.add_argument("--imagenet_pretrained", type=str2bool, default=False,
                   help="partially initialize Tiny from torchvision ImageNet-1K ConvNeXt-Tiny")
    p.add_argument("--imagenet_positive_transform", choices=["reject", "abs"], default="reject",
                   help="explicit handling of signed ImageNet conv weights when positivity is enabled")
    p.add_argument("--batch_size", type=int, default=90)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument(
        "--amp_init_scale", type=float, default=1.0,
        help="initial CUDA GradScaler scale; conservative default for deep SNN backward graphs",
    )
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="clip the global gradient norm before each optimizer step; 0 disables clipping",
    )
    p.add_argument("--detect_anomaly_batch", type=int, default=-1,
                   help="zero-based training batch to trace with autograd anomaly detection")
    p.add_argument("--resume", default="")
    p.add_argument("--download", type=str2bool, default=False)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--max_test_batches", type=int, default=0)
    p.add_argument("--train_subset_size", type=int, default=0)
    p.add_argument("--val_subset_size", type=int, default=0)
    p.add_argument("--synthetic_data", action="store_true", help="offline smoke test only")
    p.add_argument("--cifar_stem", type=str2bool, default=True)
    p.add_argument("--current_norm", type=str2bool, default=True)
    p.add_argument("--track_first_spike", type=str2bool, default=False)
    p.add_argument("--use_checkpointing", type=str2bool, default=False)
    return p


def build_loaders(args, generator):
    # Native 32x32 RGB CIFAR-10. Normalize first, then map back to [0,1] inside model is avoided;
    # therefore latency encoding receives ToTensor values directly.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    if args.synthetic_data:
        train_all = datasets.FakeData(size=64, image_size=(3,32,32), num_classes=10, transform=train_tf)
        test_set = datasets.FakeData(size=32, image_size=(3,32,32), num_classes=10, transform=test_tf)
    else:
        train_all = datasets.CIFAR10(args.data_path, train=True, download=args.download, transform=train_tf)
        test_set = datasets.CIFAR10(args.data_path, train=False, download=args.download, transform=test_tf)
    val_n = max(1, int(len(train_all) * args.val_fraction))
    train_n = len(train_all) - val_n
    train_set, val_indices = random_split(train_all, [train_n, val_n], generator=generator)
    # Validation must not use random training augmentation: construct a second dataset and reuse indices.
    if args.synthetic_data:
        val_base = datasets.FakeData(size=len(train_all), image_size=(3,32,32), num_classes=10, transform=test_tf, random_offset=0)
    else:
        val_base = datasets.CIFAR10(args.data_path, train=True, download=False, transform=test_tf)
    val_set = Subset(val_base, val_indices.indices)
    if args.train_subset_size:
        train_set = Subset(train_set, range(min(args.train_subset_size, len(train_set))))
    if args.val_subset_size:
        val_set = Subset(val_set, range(min(args.val_subset_size, len(val_set))))
    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator, drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


def automatic_output_dir(args) -> Path:
    dataset = "synthetic" if args.synthetic_data else "cifar10"
    learning_rate = format(args.lr, ".10g").replace(".", "p").replace("-", "m")
    positive_weights = str(args.force_positive_weights).lower()
    initialization = "imagenet" if args.imagenet_pretrained else "scratch"
    run_name = (
        f"{dataset}_seed{args.seed}_epochs{args.epochs}_t{args.time_steps}_"
        f"positiveweights{positive_weights}_bs{args.batch_size}_lr{learning_rate}_"
        f"threshold{args.threshold_mode}_readout{args.readout_mode}_init{initialization}"
    )
    return Path("results") / run_name


def global_gradient_norm(parameters) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0


def nonfinite_gradient_names(model, limit: int = 8) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            names.append(name)
            if len(names) >= limit:
                break
    return names


def cuda_memory_snapshot(device) -> dict:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    gib = float(1024 ** 3)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def regional_gradient_norms(model):
    regions = {name: [] for name in (
        "stem", "stage_0", "stage_1", "stage_2", "stage_3",
        "classifier", "thresholds", "delays",
    )}
    classifier = model.classifier
    classifier_ids = {id(parameter) for parameter in classifier.parameters()}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if id(parameter) in classifier_ids:
            region = "classifier"
        elif "raw_threshold" in name:
            region = "thresholds"
        elif "raw_delay" in name:
            region = "delays"
        elif name.startswith("downsamples.0"):
            region = "stem"
        elif name.startswith("stages.0"):
            region = "stage_0"
        elif name.startswith("downsamples.1") or name.startswith("stages.1"):
            region = "stage_1"
        elif name.startswith("downsamples.2") or name.startswith("stages.2"):
            region = "stage_2"
        elif name.startswith("downsamples.3") or name.startswith("stages.3"):
            region = "stage_3"
        else:
            continue
        regions[region].append(parameter)
    return {name: global_gradient_norm(parameters) for name, parameters in regions.items()}


def threshold_parameters(model):
    return [
        neuron.raw_threshold
        for neuron in model.modules()
        if isinstance(neuron, FirstSpikeNeuron) and neuron.raw_threshold is not None
    ]


def threshold_values(model) -> torch.Tensor:
    values = [
        neuron.threshold_values().detach().flatten()
        for neuron in model.modules()
        if isinstance(neuron, FirstSpikeNeuron)
    ]
    return torch.cat(values)


def set_threshold_trainable(model, enabled: bool) -> None:
    for parameter in threshold_parameters(model):
        parameter.requires_grad_(enabled)


def threshold_backward_diagnostics(model) -> dict:
    parameters = threshold_parameters(model)
    gradients = [
        parameter.grad.detach().float().flatten()
        for parameter in parameters
        if parameter.grad is not None
    ]
    absolute = torch.cat([gradient.abs() for gradient in gradients]) if gradients else None
    return {
        "raw_threshold_count": len(parameters),
        "requires_grad_count": sum(parameter.requires_grad for parameter in parameters),
        "requires_grad_all": all(parameter.requires_grad for parameter in parameters),
        "grad_none_count": sum(parameter.grad is None for parameter in parameters),
        "grad_none_any": any(parameter.grad is None for parameter in parameters),
        "gradient_mean_abs": float(absolute.mean().item()) if absolute is not None else 0.0,
        "gradient_max_abs": float(absolute.max().item()) if absolute is not None else 0.0,
        "threshold_mean": float(threshold_values(model).mean().item()),
    }


def _accumulate_nested(total, values):
    if not total:
        return json.loads(json.dumps(values))
    for key, value in values.items():
        if isinstance(value, dict):
            total[key] = _accumulate_nested(total.get(key, {}), value)
        elif isinstance(value, list):
            total[key] = [a + float(b) for a, b in zip(total.get(key, [0.0] * len(value)), value)]
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + float(value)
    return total


def _average_nested(values, count):
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result[key] = _average_nested(value, count)
        elif isinstance(value, list):
            result[key] = [item / max(count, 1) for item in value]
        else:
            result[key] = value / max(count, 1)
    return result


def run_epoch(
    model, loader, criterion, device, optimizer=None, scaler=None, amp=False,
    max_batches=0, grad_clip=0.0, gradient_accumulation_steps=1,
    detect_anomaly_batch=-1,
):
    training = optimizer is not None
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    model.train(training)
    total_loss = total_correct = total = 0
    total_spikes = total_hidden_units = total_synops = 0.0
    total_grad_norm = grad_norm_steps = 0.0
    total_post_clip_grad_norm = 0.0
    total_threshold_grad = threshold_grad_steps = 0.0
    total_classifier_grad_norm = 0.0
    diagnostic_batches = 0
    stage_spike_diagnostics = {}
    final_feature_diagnostics = {}
    regional_grad_totals = {name: 0.0 for name in (
        "stem", "stage_0", "stage_1", "stage_2", "stage_3",
        "classifier", "thresholds", "delays",
    )}
    memory_checkpoints = {}
    effective_batches = min(len(loader), max_batches) if max_batches else len(loader)
    optimizer_step_index = 0
    total_optimizer_steps = (
        math.ceil(effective_batches / gradient_accumulation_steps) if training else 0
    )
    latest_grad_norm = None
    latest_post_clip_grad_norm = None
    latest_threshold_diagnostics = None
    start = time.time()
    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches: break
        images, labels = images.to(device), labels.to(device)
        tracing_this_batch = training and i == detect_anomaly_batch
        diagnostic_handles = []
        if tracing_this_batch:
            torch.autograd.set_detect_anomaly(True, check_nan=True)
            first_large_gradient_module = [None]
            def diagnostic_hook(name):
                def hook(_module, grad_input, grad_output):
                    input_norms = [
                        float(torch.linalg.vector_norm(tensor.detach().float()).item())
                        for tensor in grad_input if tensor is not None
                    ]
                    output_norms = [
                        float(torch.linalg.vector_norm(tensor.detach().float()).item())
                        for tensor in grad_output if tensor is not None
                    ]
                    input_bad = any(
                        tensor is not None and not torch.isfinite(tensor).all()
                        for tensor in grad_input
                    )
                    output_bad = any(
                        tensor is not None and not torch.isfinite(tensor).all()
                        for tensor in grad_output
                    )
                    if input_bad or output_bad:
                        print(json.dumps({
                            "nonfinite_backward_module": name,
                            "input_gradient_nonfinite": input_bad,
                            "output_gradient_nonfinite": output_bad,
                        }), flush=True)
                    input_max = max(input_norms, default=0.0)
                    output_max = max(output_norms, default=0.0)
                    if (
                        first_large_gradient_module[0] is None
                        and input_max > 1e5
                        and output_max <= 1e5
                    ):
                        first_large_gradient_module[0] = name
                        print(json.dumps({
                            "first_large_gradient_module": name,
                            "input_gradient_norm": input_max,
                            "output_gradient_norm": output_max,
                            "threshold": 1e5,
                        }), flush=True)
                return hook
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.GroupNorm, FirstSpikeNeuron)):
                    diagnostic_handles.append(
                        module.register_full_backward_hook(diagnostic_hook(name))
                    )
        accumulation_index = i % gradient_accumulation_steps
        if training and accumulation_index == 0:
            optimizer.zero_grad(set_to_none=True)
        group_start = i - accumulation_index
        group_size = min(gradient_accumulation_steps, effective_batches - group_start)
        should_step = training and (
            accumulation_index + 1 == group_size or i + 1 == effective_batches
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            logits, stats = model(images, return_stats=True)
            loss = criterion(logits, labels)
        if i == 0:
            memory_checkpoints["first_forward"] = cuda_memory_snapshot(device)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at batch {i}: {loss.item()}")
        if training:
            backward_loss = loss / group_size
            if scaler is not None and scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            if i == 0:
                memory_checkpoints["first_backward"] = cuda_memory_snapshot(device)

            if should_step:
                optimizer_step_index += 1
                should_print_diagnostics = (
                    optimizer_step_index == 1
                    or optimizer_step_index % 100 == 0
                    or optimizer_step_index == total_optimizer_steps
                )
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = global_gradient_norm(model.parameters())
                latest_grad_norm = grad_norm
                if not math.isfinite(grad_norm):
                    bad_names = nonfinite_gradient_names(model)
                    raise FloatingPointError(
                        f"non-finite gradient norm at batch {i}: {grad_norm}; "
                        f"first affected parameters: {bad_names or ['norm_overflow']}"
                    )
                total_grad_norm += grad_norm
                grad_norm_steps += 1
                if grad_norm > 1e5:
                    print(json.dumps({
                        "warning": "excessive_pre_clip_gradient_norm",
                        "batch": i,
                        "pre_clip_grad_norm": grad_norm,
                        "limit": 1e5,
                    }), flush=True)
                threshold_grads = [p.grad.detach().abs().mean() for p in threshold_parameters(model) if p.grad is not None]
                threshold_before_step = threshold_values(model)
                latest_threshold_diagnostics = threshold_backward_diagnostics(model)
                if threshold_grads:
                    total_threshold_grad += float(torch.stack(threshold_grads).mean().item())
                    threshold_grad_steps += 1
                classifier = model.classifier
                if classifier.weight.grad is not None:
                    total_classifier_grad_norm += float(classifier.weight.grad.detach().float().norm().item())
                step_regional_gradients = regional_gradient_norms(model)
                for region, norm in step_regional_gradients.items():
                    regional_grad_totals[region] += norm
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                post_clip_grad_norm = global_gradient_norm(model.parameters())
                latest_post_clip_grad_norm = post_clip_grad_norm
                total_post_clip_grad_norm += post_clip_grad_norm
                if should_print_diagnostics:
                    print(json.dumps({
                        "phase": "gradient_diagnostics",
                        "optimizer_step": optimizer_step_index,
                        "batch": i,
                        "pre_clip_grad_norm": grad_norm,
                        "post_clip_grad_norm": post_clip_grad_norm,
                        "pre_clip_regional_gradient_norms": {
                            region: step_regional_gradients[region]
                            for region in ("stem", "stage_0", "stage_1", "stage_2", "stage_3")
                        },
                    }), flush=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                threshold_after_step = threshold_values(model)
                latest_threshold_diagnostics["update_mean_abs"] = float(
                    (threshold_after_step - threshold_before_step).abs().mean().item()
                )
                if should_print_diagnostics:
                    if (
                        latest_threshold_diagnostics["raw_threshold_count"] > 0
                        and latest_threshold_diagnostics["requires_grad_count"] == 0
                    ):
                        print(json.dumps({
                            "phase": "threshold_diagnostics",
                            "status": "frozen",
                            "raw_threshold_count": latest_threshold_diagnostics["raw_threshold_count"],
                            "threshold_mean": latest_threshold_diagnostics["threshold_mean"],
                        }), flush=True)
                    else:
                        print(json.dumps({
                            "phase": "threshold_diagnostics",
                            "optimizer_step": optimizer_step_index,
                            "batch": i,
                            **latest_threshold_diagnostics,
                        }), flush=True)
                if "first_optimizer_step" not in memory_checkpoints:
                    memory_checkpoints["first_optimizer_step"] = cuda_memory_snapshot(device)
            if tracing_this_batch:
                for handle in diagnostic_handles:
                    handle.remove()
                torch.autograd.set_detect_anomaly(False)
        n = labels.numel()
        total += n; total_loss += loss.item() * n
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_spikes += float(stats["global_hidden_spikes"])
        total_hidden_units += float(stats["global_hidden_units"])
        total_synops += float(stats["total_synops_estimate"])
        stage_spike_diagnostics = _accumulate_nested(stage_spike_diagnostics, stats["stage_spike_distribution"])
        final_feature_diagnostics = _accumulate_nested(final_feature_diagnostics, stats["final_feature_diagnostics"])
        diagnostic_batches += 1
        if (i + 1) % 10 == 0:
            iteration_log = {
                "phase": "train" if training else "validation",
                "iteration": i + 1,
                "loss": total_loss / max(total, 1),
                "accuracy": 100.0 * total_correct / max(total, 1),
            }
            if training:
                iteration_log["pre_clip_grad_norm"] = latest_grad_norm
                iteration_log["post_clip_grad_norm"] = latest_post_clip_grad_norm
            print(json.dumps(iteration_log), flush=True)
    averaged_stage_diagnostics = _average_nested(
        stage_spike_diagnostics, diagnostic_batches
    )
    for stage_name, stage_stats in averaged_stage_diagnostics.items():
        if stage_stats.get("spike_fraction_t0", 0.0) > 0.70:
            print(json.dumps({
                "warning": "excessive_t0_spiking",
                "phase": "train" if training else "validation",
                "stage": stage_name,
                "spike_fraction_t0": stage_stats["spike_fraction_t0"],
                "limit": 0.70,
            }), flush=True)
    return {
        "loss": total_loss / max(total,1), "accuracy": 100.0 * total_correct / max(total,1),
        "samples": total, "spikes_per_sample": total_spikes / max(total,1),
        "global_spikes_per_neuron": total_spikes / max(total_hidden_units, 1.0),
        "global_sparsity": 1.0 - total_spikes / max(total_hidden_units, 1.0),
        "synops_per_sample_estimate": total_synops / max(total,1), "seconds": time.time()-start,
        "grad_norm": total_grad_norm / max(grad_norm_steps, 1) if training else None,
        "post_clip_grad_norm": (
            total_post_clip_grad_norm / max(grad_norm_steps, 1)
            if training else None
        ),
        "threshold_gradient_mean": total_threshold_grad / max(threshold_grad_steps, 1) if threshold_grad_steps else 0.0,
        "classifier_weight_gradient_norm": total_classifier_grad_norm / max(grad_norm_steps, 1) if training else None,
        "stage_spike_distribution": averaged_stage_diagnostics,
        "final_feature_diagnostics": _average_nested(final_feature_diagnostics, diagnostic_batches),
        "regional_gradient_norms": {
            region: value / max(grad_norm_steps, 1) for region, value in regional_grad_totals.items()
        } if training else None,
        "memory_checkpoints": memory_checkpoints,
        "threshold_diagnostics": latest_threshold_diagnostics,
    }


def main(args):
    if args.imagenet_pretrained and args.resume:
        raise ValueError("--imagenet_pretrained and --resume are mutually exclusive initialization modes")
    out = Path(args.output_dir) if args.output_dir else automatic_output_dir(args)
    args.output_dir = str(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")
    config = vars(args).copy(); config.update({
        "temporal_model_type": "DISCRETE_TIME_TTFS_SNN",
        "residual_fusion": "earliest_spike_or" if args.residual else "disabled",
        "input_size": [3, 32, 32],
    })
    atomic_json_dump(config, out/"config.json")
    generator = seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(args, generator)
    model = build_discrete_ttfs_convnext(model_size=args.model_size, in_chans=3, num_classes=args.num_classes,
        time_steps=args.time_steps, threshold=args.threshold, force_positive_weights=args.force_positive_weights,
        learnable_delay=args.learnable_delay, init_delay=args.init_delay, residual=args.residual,
        threshold_mode=args.threshold_mode, threshold_min=args.threshold_min, threshold_max=args.threshold_max,
        readout_mode=args.readout_mode, cifar_stem=args.cifar_stem,
        current_norm=args.current_norm, track_first_spike=args.track_first_spike,
        use_checkpointing=args.use_checkpointing,
        input_no_spike_threshold=args.input_no_spike_threshold,
        residual_main_gradient_scale=args.residual_main_gradient_scale,
        surrogate_grad_clip=args.surrogate_grad_clip)
    if args.imagenet_pretrained:
        imagenet_report = load_torchvision_convnext_tiny(
            model, positive_transform=args.imagenet_positive_transform
        )
        atomic_json_dump(imagenet_report, out/"imagenet_initialization_report.json")
        print(json.dumps({
            "imagenet_initialization": imagenet_report["status"],
            "loaded_tensor_count": imagenet_report["loaded_tensor_count"],
            "loaded_parameter_count": imagenet_report["loaded_parameter_count"],
            "report": str(out/"imagenet_initialization_report.json"),
        }, indent=2))
    model = model.to(device)
    model_creation_memory = cuda_memory_snapshot(device)
    if model_creation_memory:
        print(json.dumps({"cuda_memory_after_model_creation": model_creation_memory}))
    trainable, total_params = count_parameters(model)
    print(f"Model type: {model.temporal_model_type}; T={args.time_steps}; params={total_params:,}; device={device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids)):
        raise RuntimeError("optimizer contains duplicate parameters")
    if any(optimizer_parameter_ids.count(id(parameter)) != 1 for parameter in threshold_parameters(model)):
        raise RuntimeError("each learnable threshold parameter must appear in the optimizer exactly once")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs,1), eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
        init_scale=args.amp_init_scale,
    )
    start_epoch=0; best=-math.inf
    if args.resume:
        ckpt=torch.load(args.resume,map_location="cpu",weights_only=False); model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"]); start_epoch=ckpt["epoch"]+1; best=ckpt.get("best_accuracy",best)
        if ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    epochs = 1 if args.dry_run else args.epochs
    started=time.time()
    last_threshold_stats = None
    for epoch in range(start_epoch, epochs):
        set_threshold_trainable(model, epoch >= args.threshold_freeze_epochs)
        thresholds_before = threshold_values(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tr=run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, args.amp,
            args.max_train_batches, args.grad_clip, args.gradient_accumulation_steps,
            args.detect_anomaly_batch,
        )
        train_memory = cuda_memory_snapshot(device)
        tr["peak_allocated_gib"] = train_memory.get("peak_allocated_gib", 0.0)
        tr["peak_reserved_gib"] = train_memory.get("peak_reserved_gib", 0.0)
        thresholds_after = threshold_values(model)
        last_threshold_stats = {
            "mean": float(thresholds_after.mean().item()),
            "min": float(thresholds_after.min().item()),
            "max": float(thresholds_after.max().item()),
            "gradient_mean": float(tr["threshold_gradient_mean"]),
            "update_magnitude": float((thresholds_after - thresholds_before).abs().mean().item()),
            "frozen": epoch < args.threshold_freeze_epochs,
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            va=run_epoch(model,val_loader,criterion,device,max_batches=args.max_val_batches)
        val_memory = cuda_memory_snapshot(device)
        va["peak_allocated_gib"] = val_memory.get("peak_allocated_gib", 0.0)
        va["peak_reserved_gib"] = val_memory.get("peak_reserved_gib", 0.0)
        row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"threshold_stats":last_threshold_stats,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}}
        append_jsonl(row,out/"train_log.jsonl"); print(json.dumps(row))
        save_checkpoint(out/"last_checkpoint.pth",model,optimizer,scheduler,epoch,best,config)
        if va["accuracy"] > best:
            best=va["accuracy"]; save_checkpoint(out/"best_checkpoint.pth",model,optimizer,scheduler,epoch,best,config)
        scheduler.step()
    best_ckpt=torch.load(out/"best_checkpoint.pth",map_location=device,weights_only=False); model.load_state_dict(best_ckpt["model"])
    with torch.inference_mode():
        test=run_epoch(model,test_loader,criterion,device,max_batches=args.max_test_batches)
    summary={"experiment_name":args.experiment_name,"status":"completed","dataset":"CIFAR-10" if not args.synthetic_data else "FakeData",
      "temporal_model_type":model.temporal_model_type,"time_steps":args.time_steps,"seed":args.seed,
      "residual_fusion":"earliest_spike_or" if args.residual else "disabled","learnable_delay":args.learnable_delay,"force_positive_weights":args.force_positive_weights,
      "best_epoch":best_ckpt["epoch"],"best_validation_accuracy":best,"test_accuracy":test["accuracy"],"test_loss":test["loss"],
      "spikes_per_sample":test["spikes_per_sample"],"synops_per_sample_estimate":test["synops_per_sample_estimate"],
      "threshold_mode":args.threshold_mode,"threshold_stats":last_threshold_stats,
      "readout_mode":args.readout_mode,
      "batch_size":args.batch_size,"amp_enabled":bool(args.amp and device.type == "cuda"),
      "gradient_accumulation_steps":args.gradient_accumulation_steps,
      "cifar_stem":args.cifar_stem,"current_norm":args.current_norm,
      "track_first_spike":args.track_first_spike,"use_checkpointing":args.use_checkpointing,
      "input_no_spike_threshold":args.input_no_spike_threshold,
      "residual_main_gradient_scale":args.residual_main_gradient_scale,
      "surrogate_grad_clip":args.surrogate_grad_clip,
      "model_creation_memory":model_creation_memory,
      "train_peak_allocated_gib":tr["peak_allocated_gib"],
      "train_peak_reserved_gib":tr["peak_reserved_gib"],
      "val_peak_allocated_gib":va["peak_allocated_gib"],
      "val_peak_reserved_gib":va["peak_reserved_gib"],
      "global_spikes_per_neuron":test["global_spikes_per_neuron"],
      "global_sparsity":test["global_sparsity"],
      "stage_spike_distribution":test["stage_spike_distribution"],
      "final_feature_diagnostics":test["final_feature_diagnostics"],
      "trainable_parameters":trainable,"total_parameters":total_params,"model_size_mb":model_size_mb(model),
      "training_time_seconds":time.time()-started,"checkpoint":str(out/"best_checkpoint.pth")}
    atomic_json_dump(summary,out/"training_summary.json"); print(json.dumps(summary,indent=2)); return summary

if __name__=="__main__": main(parser().parse_args())
