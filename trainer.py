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

from models.discrete_ttfs_convnext import FirstSpikeNeuron, build_discrete_ttfs_convnext
from imagenet_init import load_torchvision_convnext_tiny
from reproducibility import seed_everything, seed_worker
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
    p.add_argument("--time_steps", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--threshold_mode", choices=["fixed", "learnable_layer", "learnable_channel"], default="fixed")
    p.add_argument("--threshold_min", type=float, default=0.05)
    p.add_argument("--threshold_max", type=float, default=0.8)
    p.add_argument("--threshold_freeze_epochs", type=int, default=3)
    p.add_argument("--readout_mode", choices=["ttfs", "membrane", "hybrid", "soft_time"], default="hybrid")
    p.add_argument("--soft_time_beta", type=float, default=10.0)
    p.add_argument("--learnable_delay", type=str2bool, default=True)
    p.add_argument("--residual", type=str2bool, default=True)
    p.add_argument("--init_delay", type=float, default=0.0)
    p.add_argument("--force_positive_weights", type=str2bool, default=False)
    p.add_argument("--imagenet_pretrained", type=str2bool, default=False,
                   help="partially initialize Tiny from torchvision ImageNet-1K ConvNeXt-Tiny")
    p.add_argument("--imagenet_positive_transform", choices=["reject", "abs"], default="reject",
                   help="explicit handling of signed ImageNet conv weights when positivity is enabled")
    p.add_argument("--batch_size", type=int, default=150)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", type=str2bool, default=False)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--resume", default="")
    p.add_argument("--download", type=str2bool, default=False)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--max_test_batches", type=int, default=0)
    p.add_argument("--train_subset_size", type=int, default=0)
    p.add_argument("--val_subset_size", type=int, default=0)
    p.add_argument("--synthetic_data", action="store_true", help="offline smoke test only")
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
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            contribution = grad.square().sum()
            squared_norm = contribution if squared_norm is None else squared_norm + contribution
    return float(squared_norm.sqrt().item()) if squared_norm is not None else 0.0


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


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, max_batches=0, grad_clip=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    total_spikes = total_synops = 0.0
    total_grad_norm = grad_norm_steps = 0.0
    total_threshold_grad = threshold_grad_steps = 0.0
    total_classifier_grad_norm = 0.0
    diagnostic_batches = 0
    stage_spike_diagnostics = {}
    stage_membrane_diagnostics = {}
    final_feature_diagnostics = {}
    start = time.time()
    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches: break
        images, labels = images.to(device), labels.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, stats = model(images, return_stats=True, return_layer_stats=False)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at batch {i}: {loss.item()}")
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = global_gradient_norm(model.parameters())
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                grad_norm = global_gradient_norm(model.parameters())
                if grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_grad_norm += grad_norm
            grad_norm_steps += 1
            threshold_grads = [p.grad.detach().abs().mean() for p in threshold_parameters(model) if p.grad is not None]
            if threshold_grads:
                total_threshold_grad += float(torch.stack(threshold_grads).mean().item())
                threshold_grad_steps += 1
            classifier = model.hybrid_classifier if model.readout_mode == "hybrid" else model.classifier
            if classifier.weight.grad is not None:
                total_classifier_grad_norm += float(classifier.weight.grad.detach().float().norm().item())
        n = labels.numel()
        total += n; total_loss += loss.item() * n
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_spikes += float(stats["total_spikes"])
        total_synops += float(stats["total_synops_estimate"])
        stage_spike_diagnostics = _accumulate_nested(stage_spike_diagnostics, stats["stage_spike_distribution"])
        stage_membrane_diagnostics = _accumulate_nested(stage_membrane_diagnostics, stats["stage_membrane_diagnostics"])
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
                iteration_log["grad_norm"] = grad_norm
            print(json.dumps(iteration_log), flush=True)
    return {
        "loss": total_loss / max(total,1), "accuracy": 100.0 * total_correct / max(total,1),
        "samples": total, "spikes_per_sample": total_spikes / max(total,1),
        "synops_per_sample_estimate": total_synops / max(total,1), "seconds": time.time()-start,
        "grad_norm": total_grad_norm / max(grad_norm_steps, 1) if training else None,
        "threshold_gradient_mean": total_threshold_grad / max(threshold_grad_steps, 1) if threshold_grad_steps else 0.0,
        "classifier_weight_gradient_norm": total_classifier_grad_norm / max(grad_norm_steps, 1) if training else None,
        "stage_spike_distribution": _average_nested(stage_spike_diagnostics, diagnostic_batches),
        "stage_membrane_diagnostics": _average_nested(stage_membrane_diagnostics, diagnostic_batches),
        "final_feature_diagnostics": _average_nested(final_feature_diagnostics, diagnostic_batches),
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
        readout_mode=args.readout_mode, soft_time_beta=args.soft_time_beta)
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
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
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
        tr=run_epoch(model,train_loader,criterion,device,optimizer,scaler,args.amp,args.max_train_batches, args.grad_clip)
        thresholds_after = threshold_values(model)
        last_threshold_stats = {
            "mean": float(thresholds_after.mean().item()),
            "min": float(thresholds_after.min().item()),
            "max": float(thresholds_after.max().item()),
            "gradient_mean": float(tr["threshold_gradient_mean"]),
            "update_magnitude": float((thresholds_after - thresholds_before).abs().mean().item()),
            "frozen": epoch < args.threshold_freeze_epochs,
        }
        with torch.no_grad(): va=run_epoch(model,val_loader,criterion,device,max_batches=args.max_val_batches)
        row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"threshold_stats":last_threshold_stats,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}}
        append_jsonl(row,out/"train_log.jsonl"); print(json.dumps(row))
        save_checkpoint(out/"last_checkpoint.pth",model,optimizer,scheduler,epoch,best,config)
        if va["accuracy"] > best:
            best=va["accuracy"]; save_checkpoint(out/"best_checkpoint.pth",model,optimizer,scheduler,epoch,best,config)
        scheduler.step()
    best_ckpt=torch.load(out/"best_checkpoint.pth",map_location=device,weights_only=False); model.load_state_dict(best_ckpt["model"])
    with torch.no_grad(): test=run_epoch(model,test_loader,criterion,device,max_batches=args.max_test_batches)
    summary={"experiment_name":args.experiment_name,"status":"completed","dataset":"CIFAR-10" if not args.synthetic_data else "FakeData",
      "temporal_model_type":model.temporal_model_type,"time_steps":args.time_steps,"seed":args.seed,
      "residual_fusion":"earliest_spike_or" if args.residual else "disabled","learnable_delay":args.learnable_delay,"force_positive_weights":args.force_positive_weights,
      "best_epoch":best_ckpt["epoch"],"best_validation_accuracy":best,"test_accuracy":test["accuracy"],"test_loss":test["loss"],
      "spikes_per_sample":test["spikes_per_sample"],"synops_per_sample_estimate":test["synops_per_sample_estimate"],
      "threshold_mode":args.threshold_mode,"threshold_stats":last_threshold_stats,
      "readout_mode":args.readout_mode,"soft_time_beta":args.soft_time_beta,
      "stage_spike_distribution":test["stage_spike_distribution"],
      "stage_membrane_diagnostics":test["stage_membrane_diagnostics"],
      "final_feature_diagnostics":test["final_feature_diagnostics"],
      "trainable_parameters":trainable,"total_parameters":total_params,"model_size_mb":model_size_mb(model),
      "training_time_seconds":time.time()-started,"checkpoint":str(out/"best_checkpoint.pth")}
    atomic_json_dump(summary,out/"training_summary.json"); print(json.dumps(summary,indent=2)); return summary

if __name__=="__main__": main(parser().parse_args())
