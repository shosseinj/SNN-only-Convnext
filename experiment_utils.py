from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Any, Dict
import torch


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)


def atomic_json_dump(data: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_jsonl(data: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, config):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "best_accuracy": best_acc, "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)
