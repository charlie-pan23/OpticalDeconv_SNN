"""Checkpoint utilities for HIPSA evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "net", "network", "module")


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract a model state_dict from common checkpoint formats."""
    obj = checkpoint
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, Mapping):
        for key in STATE_DICT_KEYS:
            value = obj.get(key)
            if isinstance(value, Mapping):
                obj = value
                break
    if not isinstance(obj, Mapping):
        raise TypeError(f"Cannot extract state_dict from object of type {type(checkpoint).__name__}")

    # Heuristic: a real state_dict is a dict of tensor-like values.
    tensor_items = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
    if not tensor_items:
        raise TypeError("Checkpoint dict does not contain tensor values for a model state_dict")
    return dict(tensor_items)


def strip_prefix_from_state_dict(state_dict: Mapping[str, torch.Tensor], prefixes: Sequence[str] = ("module.", "model.")) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def add_prefix_to_state_dict(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    ckpt = safe_torch_load(path, map_location=map_location)
    if isinstance(ckpt, Mapping):
        return dict(ckpt)
    return {"state_dict": extract_state_dict(ckpt)}


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
    strip_prefix: bool = True,
) -> Dict[str, Any]:
    """Load model weights and return metadata including missing/unexpected keys."""
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    if strip_prefix:
        state_dict = strip_prefix_from_state_dict(state_dict)

    result = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))

    if missing:
        logger.warning("Missing keys when loading %s: %s", checkpoint_path, missing)
    if unexpected:
        logger.warning("Unexpected keys when loading %s: %s", checkpoint_path, unexpected)

    metadata: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict": bool(strict),
    }
    if isinstance(checkpoint, Mapping):
        for key in ["epoch", "metrics", "config", "best_acc", "best_val", "val_acc", "test_acc"]:
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    return metadata


def clone_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Deep-copy current model state to CPU memory."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_model_state(model: nn.Module, state: Mapping[str, torch.Tensor], device: str | torch.device = "cpu") -> None:
    state_on_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}
    model.load_state_dict(state_on_device, strict=True)


def find_best_checkpoint(run_dir: str | Path, filename: str = "best_val.pth") -> Path:
    """Find best checkpoint in a run directory."""
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "checkpoints" / filename,
        run_dir / filename,
        run_dir / "checkpoints" / "best.pth",
        run_dir / "checkpoints" / "model_best.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(run_dir.glob("**/*best*.pth")) + sorted(run_dir.glob("**/best_val.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find best checkpoint under {run_dir}")


def checkpoint_summary(path: str | Path) -> Dict[str, Any]:
    ckpt = safe_torch_load(path, map_location="cpu")
    summary: Dict[str, Any] = {"path": str(path), "type": type(ckpt).__name__}
    if isinstance(ckpt, Mapping):
        summary["keys"] = sorted([str(k) for k in ckpt.keys()])
        for key in ["epoch", "metrics", "best_acc", "best_val", "val_acc", "test_acc"]:
            if key in ckpt:
                summary[key] = ckpt[key]
    state_dict = extract_state_dict(ckpt)
    summary["num_tensors"] = len(state_dict)
    summary["num_parameters"] = int(sum(v.numel() for v in state_dict.values() if torch.is_tensor(v)))
    summary["first_keys"] = list(state_dict.keys())[:10]
    return summary


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    if config is not None:
        payload["config"] = dict(config)
    torch.save(payload, path)
