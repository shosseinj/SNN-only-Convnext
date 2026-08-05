#!/usr/bin/env python3
"""Train the ReLU ConvNeXt ANN teacher for ANN-to-TTFS conversion."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from models.ann_convnext_relu_for_ttfs_conversion import build_ann_convnext_relu
from test.reproducibility import seed_everything, seed_worker
from experiment_utils import (
    atomic_json_dump,
    append_jsonl,
    count_parameters,
    model_size_mb,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected boolean")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ReLU ConvNeXt ANN teacher trainer")

    p.add_argument(
        "--experiment_name",
        default="cifar10_ann_convnext_relu_tiny_seed42",
    )
    p.add_argument("--data_path", default="../cifar_data")
    p.add_argument(
        "--output_dir",
        default="",
        help="generated from training settings when omitted",
    )

    p.add_argument("--model_size", choices=["nano", "tiny"], default="tiny")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--cifar_stem", type=str2bool, default=True)
    p.add_argument("--current_norm", type=str2bool, default=True)
    p.add_argument("--residual", type=str2bool, default=True)
    p.add_argument("--force_positive_weights", type=str2bool, default=False)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=5)

    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--amp_init_scale", type=float, default=65536.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--resume", default="")
    p.add_argument("--download", type=str2bool, default=False)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--max_test_batches", type=int, default=0)
    p.add_argument("--train_subset_size", type=int, default=0)
    p.add_argument("--val_subset_size", type=int, default=0)
    p.add_argument(
        "--synthetic_data",
        action="store_true",
        help="offline smoke test only",
    )

    return p


def build_loaders(args, generator):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    if args.synthetic_data:
        train_all = datasets.FakeData(
            size=64,
            image_size=(3, 32, 32),
            num_classes=args.num_classes,
            transform=train_tf,
        )
        val_base = datasets.FakeData(
            size=64,
            image_size=(3, 32, 32),
            num_classes=args.num_classes,
            transform=eval_tf,
            random_offset=0,
        )
        test_set = datasets.FakeData(
            size=32,
            image_size=(3, 32, 32),
            num_classes=args.num_classes,
            transform=eval_tf,
            random_offset=1000,
        )
    else:
        train_all = datasets.CIFAR10(
            args.data_path,
            train=True,
            download=args.download,
            transform=train_tf,
        )
        val_base = datasets.CIFAR10(
            args.data_path,
            train=True,
            download=False,
            transform=eval_tf,
        )
        test_set = datasets.CIFAR10(
            args.data_path,
            train=False,
            download=args.download,
            transform=eval_tf,
        )

    val_n = max(1, int(len(train_all) * args.val_fraction))
    train_n = len(train_all) - val_n
    train_set, val_indices = random_split(
        train_all,
        [train_n, val_n],
        generator=generator,
    )
    val_set = Subset(val_base, val_indices.indices)

    if args.train_subset_size:
        train_set = Subset(
            train_set,
            range(min(args.train_subset_size, len(train_set))),
        )
    if args.val_subset_size:
        val_set = Subset(
            val_set,
            range(min(args.val_subset_size, len(val_set))),
        )

    common = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def automatic_output_dir(args) -> Path:
    dataset = "synthetic" if args.synthetic_data else "cifar10"
    lr = format(args.lr, ".10g").replace(".", "p").replace("-", "m")
    run_name = (
        f"{dataset}_ann_relu_{args.model_size}_seed{args.seed}_"
        f"epochs{args.epochs}_bs{args.batch_size}_lr{lr}"
    )
    return Path("results") / run_name


def global_gradient_norm(parameters) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def regional_gradient_norms(model):
    regions = {
        name: []
        for name in (
            "stem",
            "stage_0",
            "stage_1",
            "stage_2",
            "stage_3",
            "classifier",
        )
    }

    classifier_ids = {id(p) for p in model.classifier.parameters()}

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        if id(parameter) in classifier_ids:
            region = "classifier"
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

    return {
        name: global_gradient_norm(parameters)
        for name, parameters in regions.items()
    }


def cuda_memory_snapshot(device) -> dict:
    if device.type != "cuda":
        return {}

    torch.cuda.synchronize(device)
    gib = float(1024**3)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def save_ann_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_accuracy: float,
    config: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "transferable_model": model.transferable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_accuracy": float(best_accuracy),
        "config": config,
        "model_type": model.model_type,
        "block_design": model.block_design,
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def set_learning_rate(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def cosine_lr_for_epoch(
    epoch: int,
    epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)

    cosine_epochs = max(epochs - warmup_epochs, 1)
    progress = float(epoch - warmup_epochs) / float(max(cosine_epochs - 1, 1))
    progress = min(max(progress, 0.0), 1.0)

    return min_lr + 0.5 * (base_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp=False,
    max_batches=0,
    grad_clip=0.0,
    gradient_accumulation_steps=1,
):
    training = optimizer is not None
    model.train(training)

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    total_grad_norm = 0.0
    total_post_clip_grad_norm = 0.0
    grad_steps = 0

    regional_totals = {
        name: 0.0
        for name in (
            "stem",
            "stage_0",
            "stage_1",
            "stage_2",
            "stage_3",
            "classifier",
        )
    }

    effective_batches = (
        min(len(loader), max_batches)
        if max_batches
        else len(loader)
    )

    latest_grad_norm = None
    latest_post_clip_grad_norm = None
    optimizer_step_index = 0
    start = time.time()

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        accumulation_index = batch_index % gradient_accumulation_steps
        if training and accumulation_index == 0:
            optimizer.zero_grad(set_to_none=True)

        group_start = batch_index - accumulation_index
        group_size = min(
            gradient_accumulation_steps,
            effective_batches - group_start,
        )
        should_step = training and (
            accumulation_index + 1 == group_size
            or batch_index + 1 == effective_batches
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at batch {batch_index}: {loss.item()}"
            )

        if training:
            backward_loss = loss / group_size
            if scaler is not None and scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            if should_step:
                optimizer_step_index += 1

                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)

                grad_norm = global_gradient_norm(model.parameters())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite gradient norm at batch {batch_index}: "
                        f"{grad_norm}"
                    )

                region_norms = regional_gradient_norms(model)
                for name, value in region_norms.items():
                    regional_totals[name] += value

                latest_grad_norm = grad_norm
                total_grad_norm += grad_norm

                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                post_clip_grad_norm = global_gradient_norm(model.parameters())
                latest_post_clip_grad_norm = post_clip_grad_norm
                total_post_clip_grad_norm += post_clip_grad_norm
                grad_steps += 1

                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if (
                    optimizer_step_index == 1
                    or optimizer_step_index % 100 == 0
                ):
                    print(
                        json.dumps(
                            {
                                "phase": "gradient_diagnostics",
                                "optimizer_step": optimizer_step_index,
                                "batch": batch_index,
                                "pre_clip_grad_norm": grad_norm,
                                "post_clip_grad_norm": post_clip_grad_norm,
                                "regional_gradient_norms": region_norms,
                            }
                        ),
                        flush=True,
                    )

        batch_size = labels.numel()
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

        if (batch_index + 1) % 10 == 0:
            row = {
                "phase": "train" if training else "validation",
                "iteration": batch_index + 1,
                "loss": total_loss / max(total_samples, 1),
                "accuracy": 100.0 * total_correct / max(total_samples, 1),
            }
            if training:
                row["pre_clip_grad_norm"] = latest_grad_norm
                row["post_clip_grad_norm"] = latest_post_clip_grad_norm

            print(json.dumps(row), flush=True)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": 100.0 * total_correct / max(total_samples, 1),
        "samples": total_samples,
        "seconds": time.time() - start,
        "grad_norm": (
            total_grad_norm / max(grad_steps, 1)
            if training
            else None
        ),
        "post_clip_grad_norm": (
            total_post_clip_grad_norm / max(grad_steps, 1)
            if training
            else None
        ),
        "regional_gradient_norms": (
            {
                name: value / max(grad_steps, 1)
                for name, value in regional_totals.items()
            }
            if training
            else None
        ),
    }


def main(args):
    out = (
        Path(args.output_dir)
        if args.output_dir
        else automatic_output_dir(args)
    )
    args.output_dir = str(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out}")

    config = vars(args).copy()
    config.update(
        {
            "model_type": "ANN_CONVNEXT_RELU_TTFS_TEACHER",
            "dataset": "FakeData" if args.synthetic_data else "CIFAR-10",
            "input_size": [3, 32, 32],
            "normalization_mean": list(CIFAR10_MEAN),
            "normalization_std": list(CIFAR10_STD),
            "conversion_target": "two_TTFS_design_B",
        }
    )
    atomic_json_dump(config, out / "config.json")

    generator = seed_everything(args.seed, deterministic=True)
    device = torch.device(
        args.device
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    train_loader, val_loader, test_loader = build_loaders(args, generator)

    model = build_ann_convnext_relu(
        model_size=args.model_size,
        in_chans=3,
        num_classes=args.num_classes,
        cifar_stem=args.cifar_stem,
        current_norm=args.current_norm,
        residual=args.residual,
        force_positive_weights=args.force_positive_weights,
    ).to(device)

    model_creation_memory = cuda_memory_snapshot(device)
    if model_creation_memory:
        print(
            json.dumps(
                {"cuda_memory_after_model_creation": model_creation_memory}
            )
        )

    trainable_params, total_params = count_parameters(model)
    print(
        f"Model type: {model.model_type}; "
        f"params={total_params:,}; device={device}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
        init_scale=args.amp_init_scale,
    )

    start_epoch = 0
    best_accuracy = -math.inf

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(
            checkpoint.get("best_accuracy", best_accuracy)
        )

    epochs = 1 if args.dry_run else args.epochs
    started = time.time()

    last_train = None
    last_val = None

    for epoch in range(start_epoch, epochs):
        current_lr = cosine_lr_for_epoch(
            epoch=epoch,
            epochs=epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
        )
        set_learning_rate(optimizer, current_lr)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
            max_batches=args.max_train_batches,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        train_memory = cuda_memory_snapshot(device)
        train_metrics["peak_allocated_gib"] = train_memory.get(
            "peak_allocated_gib",
            0.0,
        )
        train_metrics["peak_reserved_gib"] = train_memory.get(
            "peak_reserved_gib",
            0.0,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=args.max_val_batches,
            )

        val_memory = cuda_memory_snapshot(device)
        val_metrics["peak_allocated_gib"] = val_memory.get(
            "peak_allocated_gib",
            0.0,
        )
        val_metrics["peak_reserved_gib"] = val_memory.get(
            "peak_reserved_gib",
            0.0,
        )

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
        }

        append_jsonl(row, out / "train_log.jsonl")
        print(json.dumps(row), flush=True)

        save_ann_checkpoint(
            out / "last_checkpoint.pth",
            model,
            optimizer,
            None,
            epoch,
            best_accuracy,
            config,
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_ann_checkpoint(
                out / "best_checkpoint.pth",
                model,
                optimizer,
                None,
                epoch,
                best_accuracy,
                config,
            )

        last_train = train_metrics
        last_val = val_metrics

    best_checkpoint = torch.load(
        out / "best_checkpoint.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])

    with torch.inference_mode():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_test_batches,
        )

    # A separate lightweight file for the conversion script.
    torch.save(
        {
            "transferable_model": model.transferable_state_dict(),
            "source_model": model.model_type,
            "source_checkpoint": str(out / "best_checkpoint.pth"),
            "best_validation_accuracy": best_accuracy,
            "test_accuracy": test_metrics["accuracy"],
            "normalization_mean": CIFAR10_MEAN,
            "normalization_std": CIFAR10_STD,
        },
        out / "ttfs_transferable_weights.pth",
    )

    summary = {
        "experiment_name": args.experiment_name,
        "status": "completed",
        "dataset": "FakeData" if args.synthetic_data else "CIFAR-10",
        "model_type": model.model_type,
        "block_design": model.block_design,
        "seed": args.seed,
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "batch_size": args.batch_size,
        "amp_enabled": bool(args.amp and device.type == "cuda"),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "cifar_stem": args.cifar_stem,
        "current_norm": args.current_norm,
        "residual": args.residual,
        "force_positive_weights": args.force_positive_weights,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "model_size_mb": model_size_mb(model),
        "training_time_seconds": time.time() - started,
        "best_checkpoint": str(out / "best_checkpoint.pth"),
        "transferable_weights": str(
            out / "ttfs_transferable_weights.pth"
        ),
        "model_creation_memory": model_creation_memory,
        "last_train_metrics": last_train,
        "last_validation_metrics": last_val,
    }

    atomic_json_dump(summary, out / "training_summary.json")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(parser().parse_args())
