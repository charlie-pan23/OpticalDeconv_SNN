"""Shared preprocessing/training utilities for HIPSA SNN experiments.

The functions in this file are intentionally config-driven. Dataset-specific
entry scripts should be thin wrappers so that train/eval stages can reuse the
same split files, preprocessing mode, and checkpoint conventions.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from spikingjelly.activation_based import functional as sf

try:
    from utils.logger import logger
except Exception:  # pragma: no cover - fallback only for standalone syntax checks
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


# =============================================================================
# General config / reproducibility helpers
# =============================================================================


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def save_json(obj: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)


def choose_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def create_run_dir(config: Dict[str, Any], config_path: str | Path, dataset_tag: str) -> Path:
    run_root = Path(config["paths"]["run_root"])
    run_name = f"run_{timestamp()}_{dataset_tag}"
    run_dir = run_root / run_name
    for sub in ["checkpoints", "logs", "metrics", "configs", "activity", "baseline", "power", "robustness"]:
        ensure_dir(run_dir / sub)

    shutil.copy2(config_path, run_dir / "configs" / Path(config_path).name)
    for extra_key in ["hardware_config", "device_config"]:
        extra = config.get("paths", {}).get(extra_key)
        if extra and Path(extra).exists():
            shutil.copy2(extra, run_dir / "configs" / Path(extra).name)

    save_json({"run_dir": str(run_dir), "created_at": timestamp()}, run_dir / "run_info.json")
    return run_dir


# =============================================================================
# Dataset label extraction and split helpers
# =============================================================================


def get_dataset_labels(dataset: Dataset, class_names: Optional[List[str]] = None) -> List[int]:
    """Extract labels from common dataset representations.

    SpikingJelly neuromorphic datasets may expose labels through samples, targets,
    labels, or file paths. If none are available, this function falls back to a
    one-pass scan. The fallback is slower but only used in preprocessing.
    """
    for attr in ["targets", "labels"]:
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            try:
                return [int(v) for v in values]
            except Exception:
                pass

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        labels = []
        ok = True
        for item in samples:
            try:
                labels.append(int(item[1]))
            except Exception:
                ok = False
                break
        if ok and labels:
            return labels

    if class_names and hasattr(dataset, "samples"):
        class_to_idx = {str(c): i for i, c in enumerate(class_names)}
        labels = []
        for item in getattr(dataset, "samples"):
            path = str(item[0])
            parts = Path(path).parts
            found = None
            for p in parts:
                if p in class_to_idx:
                    found = class_to_idx[p]
                    break
            if found is None:
                break
            labels.append(found)
        if len(labels) == len(dataset):
            return labels

    logger.warning("Could not read dataset labels from metadata; scanning samples once. This may take time.")
    labels = []
    for i in tqdm(range(len(dataset)), desc="Scanning labels", leave=False):
        _, target = dataset[i]
        if torch.is_tensor(target):
            target = int(target.item())
        labels.append(int(target))
    return labels


def class_counts(labels: Sequence[int], num_classes: int) -> Dict[str, int]:
    counts = {str(i): 0 for i in range(num_classes)}
    for y in labels:
        counts[str(int(y))] = counts.get(str(int(y)), 0) + 1
    return counts


def split_class_balanced(
    labels: Sequence[int],
    num_classes: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    labels_np = np.asarray(labels, dtype=np.int64)
    for cls in range(num_classes):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure all samples are assigned exactly once.
        n_train = min(n_train, n)
        n_val = min(n_val, max(n - n_train, 0))
        train_indices.extend(idx[:n_train].tolist())
        val_indices.extend(idx[n_train:n_train + n_val].tolist())
        test_indices.extend(idx[n_train + n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return {
        "train_indices": [int(i) for i in train_indices],
        "val_indices": [int(i) for i in val_indices],
        "test_indices": [int(i) for i in test_indices],
    }


def split_train_val_class_balanced(
    labels: Sequence[int],
    num_classes: int,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    labels_np = np.asarray(labels, dtype=np.int64)
    for cls in range(num_classes):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return {"train_indices": [int(i) for i in train_indices], "val_indices": [int(i) for i in val_indices]}


def save_split(split: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(split, path)


def load_split(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# =============================================================================
# Frame preprocessing and augmentation
# =============================================================================


def encode_frames(x: torch.Tensor, preprocess_cfg: Dict[str, Any]) -> torch.Tensor:
    """Encode event frames according to config.

    Accepts either [B,T,C,H,W] or [T,B,C,H,W] and preserves the input shape.
    """
    x = x.float()
    mode = str(preprocess_cfg.get("input_encoding", "binary")).lower()
    if mode == "binary":
        threshold = float(preprocess_cfg.get("binary_threshold", 0.0))
        return (x > threshold).float()
    if mode == "clipped_count":
        max_v = float(preprocess_cfg.get("clipped_count_max", 3.0))
        return torch.clamp(x, 0.0, max_v) / max(max_v, 1e-12)
    if mode == "normalized_count":
        eps = float(preprocess_cfg.get("normalize_eps", 1.0e-6))
        dims = tuple(range(2, x.dim()))
        denom = x.amax(dim=dims, keepdim=True).clamp_min(eps)
        return x / denom
    raise ValueError(f"Unsupported input_encoding: {mode}")


def to_time_first(data: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Convert DataLoader batch to [T,B,C,H,W]."""
    if data.dim() != 5:
        raise ValueError(f"Expected 5D event frames, got {tuple(data.shape)}")
    if data.shape[0] == time_steps:
        return data.contiguous()
    if data.shape[1] == time_steps:
        return data.transpose(0, 1).contiguous()
    raise ValueError(f"Cannot infer time dimension from shape={tuple(data.shape)} and T={time_steps}")


def prepare_batch(
    data: torch.Tensor,
    target: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
    train: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train:
        data = apply_augmentation(data, config)
    data = encode_frames(data, config.get("preprocess", {}))
    data = to_time_first(data, int(config["dataset"]["time_steps"]))
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True).long().view(-1)
    return data, target


def apply_augmentation(data: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    aug = config.get("augmentation", {})
    if not aug.get("enabled", False):
        return data
    if data.dim() != 5:
        return data

    # DataLoader normally gives [B,T,C,H,W]. If it is already [T,B,C,H,W], avoid
    # augmentation because batch dimension is ambiguous in this helper.
    time_steps = int(config["dataset"]["time_steps"])
    if data.shape[1] != time_steps:
        return data

    x = data
    spatial = aug.get("spatial", {})
    temporal = aug.get("temporal", {})
    reg = aug.get("regularization", {})

    if temporal.get("random_temporal_shift", False):
        max_shift = int(temporal.get("max_shift_steps", 1))
        if max_shift > 0:
            shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],))
            x = x.clone()
            for b, s in enumerate(shifts.tolist()):
                if s:
                    x[b] = torch.roll(x[b], shifts=s, dims=0)

    if spatial.get("random_horizontal_flip", False):
        p = float(spatial.get("horizontal_flip_p", 0.5))
        mask = torch.rand(x.shape[0]) < p
        if mask.any():
            x = x.clone()
            x[mask] = torch.flip(x[mask], dims=(-1,))

    if spatial.get("random_shift", False):
        max_shift = int(spatial.get("max_shift_pixels", 4))
        if max_shift > 0:
            x = x.clone()
            for b in range(x.shape[0]):
                dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                x[b] = torch.roll(x[b], shifts=(dy, dx), dims=(-2, -1))

    if spatial.get("random_crop", False):
        padding = int(spatial.get("crop_padding", 4))
        if padding > 0:
            b, t, c, h, w = x.shape
            flat = x.reshape(b * t, c, h, w)
            flat = F.pad(flat, (padding, padding, padding, padding))
            hp, wp = h + 2 * padding, w + 2 * padding
            cropped = torch.empty_like(x)
            flat = flat.reshape(b, t, c, hp, wp)
            for bi in range(b):
                top = int(torch.randint(0, 2 * padding + 1, (1,)).item())
                left = int(torch.randint(0, 2 * padding + 1, (1,)).item())
                cropped[bi] = flat[bi, :, :, top:top + h, left:left + w]
            x = cropped

    if temporal.get("event_drop", False):
        p = float(temporal.get("event_drop_p", 0.0))
        if p > 0:
            keep = torch.rand_like(x) > p
            x = x * keep.to(dtype=x.dtype)

    if reg.get("cutout", False):
        p = float(reg.get("cutout_p", 0.0))
        size = int(reg.get("cutout_size", 16))
        if p > 0 and size > 0:
            x = x.clone()
            b, t, c, h, w = x.shape
            for bi in range(b):
                if random.random() < p:
                    cy = random.randint(0, h - 1)
                    cx = random.randint(0, w - 1)
                    y0 = max(0, cy - size // 2)
                    y1 = min(h, y0 + size)
                    x0 = max(0, cx - size // 2)
                    x1 = min(w, x0 + size)
                    x[bi, :, :, y0:y1, x0:x1] = 0
    return x


# =============================================================================
# Training helpers
# =============================================================================


def reset_net(model: nn.Module) -> None:
    try:
        sf.reset_net(model)
    except Exception:
        pass


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = int((preds == targets).sum().item())
    total = int(targets.numel())
    return correct, total


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config["training"].get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(opt_cfg.get("momentum", 0.9)), weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    tr = config["training"]
    sched_cfg = tr.get("scheduler", {})
    name = str(sched_cfg.get("name", "cosine_warmup")).lower()
    epochs = int(tr.get("epochs", 100))
    if name in {"none", "null", "off"}:
        return None
    if name == "cosine":
        min_lr = float(sched_cfg.get("min_lr", 1e-5))
        base_lr = float(tr.get("optimizer", {}).get("lr", 1e-3))
        eta_min_ratio = min_lr / max(base_lr, 1e-12)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * eta_min_ratio)
    if name == "cosine_warmup":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-5))
        base_lr = float(tr.get("optimizer", {}).get("lr", 1e-3))
        min_ratio = min_lr / max(base_lr, 1e-12)

        def lr_lambda(epoch: int):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max((epoch + 1) / warmup_epochs, min_ratio)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported scheduler: {name}")


def build_criterion(config: Dict[str, Any]) -> nn.Module:
    loss_cfg = config["training"].get("loss", {})
    smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    try:
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    except TypeError:
        if smoothing > 0:
            logger.warning("This PyTorch version does not support label_smoothing; using standard CrossEntropyLoss.")
        return nn.CrossEntropyLoss()


def compute_loss(logits_t: torch.Tensor, targets: torch.Tensor, criterion: nn.Module, config: Dict[str, Any]) -> torch.Tensor:
    from models.snn_common import aggregate_time_logits

    tr = config["training"]
    loss_cfg = tr.get("loss", {})
    logits = aggregate_time_logits(logits_t, mode=tr.get("logits_aggregation", "mean_time"))
    loss = criterion(logits, targets)

    if loss_cfg.get("use_tet_loss", False):
        tet_weight = float(loss_cfg.get("tet_loss_weight", 0.5))
        loss_t = torch.stack([criterion(logits_t[t], targets) for t in range(logits_t.shape[0])]).mean()
        loss = loss + tet_weight * loss_t
    return loss


def get_grad_scaler(enabled: bool):
    enabled = bool(enabled and torch.cuda.is_available())
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool, dtype: str = "bf16"):
    enabled = bool(enabled and torch.cuda.is_available())

    dtype = str(dtype).lower()
    if dtype in ["bf16", "bfloat16"]:
        amp_dtype = torch.bfloat16
    elif dtype in ["fp16", "float16"]:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=amp_dtype, enabled=enabled)

    return torch.cuda.amp.autocast(dtype=amp_dtype, enabled=enabled)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, config: Dict[str, Any], drop_last: bool) -> DataLoader:
    tr = config["training"]
    num_workers = int(tr.get("num_workers", 4))
    persistent = bool(tr.get("persistent_workers", True) and num_workers > 0)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(tr.get("pin_memory", True) and torch.cuda.is_available()),
        drop_last=drop_last,
        persistent_workers=persistent,
    )

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(tr.get("prefetch_factor", 4))

    return DataLoader(**kwargs)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    from models.snn_common import aggregate_time_logits

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, targets in tqdm(loader, desc=desc, leave=False):
        reset_net(model)
        data, targets = prepare_batch(data, targets, config, device, train=False)
        logits_t = model(data)
        logits = aggregate_time_logits(logits_t, mode=config["training"].get("logits_aggregation", "mean_time"))
        loss = criterion(logits, targets)
        c, n = accuracy_from_logits(logits, targets)
        correct += c
        total += n
        total_loss += float(loss.item()) * n
    reset_net(model)
    return {"loss": total_loss / max(total, 1), "acc": 100.0 * correct / max(total, 1), "total": float(total)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    scaler: Any,
    epoch: int,
) -> Dict[str, float]:
    from models.snn_common import aggregate_time_logits

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = bool(config["training"].get("amp", False))
    grad_clip = config["training"].get("grad_clip_norm", None)

    for data, targets in tqdm(loader, desc=f"Train {epoch}", leave=False):
        reset_net(model)
        data, targets = prepare_batch(data, targets, config, device, train=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(
                amp_enabled,
                config["training"].get("amp_dtype", "bf16")
        ):
            logits_t = model(data)
            loss = compute_loss(logits_t, targets, criterion, config)
            logits = aggregate_time_logits(logits_t, mode=config["training"].get("logits_aggregation", "mean_time"))
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None and float(grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        c, n = accuracy_from_logits(logits.detach(), targets)
        correct += c
        total += n
        total_loss += float(loss.item()) * n
    reset_net(model)
    return {"loss": total_loss / max(total, 1), "acc": 100.0 * correct / max(total, 1), "total": float(total)}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": make_json_safe(metrics),
        "config": make_json_safe(config),
    }
    torch.save(state, path)


def load_model_weights(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> Dict[str, Any]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items() if torch.is_tensor(v)}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"Missing checkpoint keys: {len(missing)}. First few: {missing[:8]}")
    if unexpected:
        logger.warning(f"Unexpected checkpoint keys: {len(unexpected)}. First few: {unexpected[:8]}")
    return ckpt if isinstance(ckpt, dict) else {"state_dict": state}


def append_csv_row(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(make_json_safe(row))


def summarize_split(labels: Sequence[int], split: Dict[str, Sequence[int]], num_classes: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels_np = np.asarray(labels, dtype=np.int64)
    for key, indices in split.items():
        if not key.endswith("indices"):
            continue
        values = labels_np[np.asarray(indices, dtype=np.int64)] if len(indices) else np.asarray([], dtype=np.int64)
        out[key.replace("_indices", "")] = {
            "num_samples": int(len(indices)),
            "class_counts": class_counts(values.tolist(), num_classes),
        }
    return out
