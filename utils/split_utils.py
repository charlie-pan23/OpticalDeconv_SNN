"""Split loading and validation helpers for HIPSA evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


SPLIT_ALIASES = {
    "train": ("train", "train_indices", "train_idx", "training"),
    "val": ("val", "val_indices", "valid", "valid_indices", "validation", "validation_indices"),
    "test": ("test", "test_indices", "test_idx", "official_test"),
}


def save_split(split: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(split), path)


def load_split(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    try:
        split = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        split = torch.load(path, map_location="cpu")
    if not isinstance(split, dict):
        raise TypeError(f"Expected split dict in {path}, got {type(split).__name__}")
    return normalize_split_dict(split)


def normalize_split_dict(split: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize split keys to train_indices/val_indices/test_indices when possible."""
    out: Dict[str, Any] = dict(split)
    for canonical, aliases in SPLIT_ALIASES.items():
        target_key = f"{canonical}_indices" if canonical != "val" else "val_indices"
        if target_key in out:
            out[target_key] = _to_int_list(out[target_key])
            continue
        for alias in aliases:
            if alias in out:
                out[target_key] = _to_int_list(out[alias])
                break
    return out


def _to_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return [int(v) for v in list(values)]


def get_split_indices(split: Mapping[str, Any], split_name: str) -> List[int]:
    split = normalize_split_dict(split)
    name = split_name.lower()
    if name in {"valid", "validation"}:
        name = "val"
    key = f"{name}_indices" if name != "val" else "val_indices"
    if key not in split:
        raise KeyError(f"Split '{split_name}' not found. Available keys: {sorted(split.keys())}")
    return _to_int_list(split[key])


def subset_from_split(dataset: Dataset, split: Mapping[str, Any], split_name: str) -> Subset:
    indices = get_split_indices(split, split_name)
    return Subset(dataset, indices)


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

    rng = np.random.default_rng(int(seed))
    labels_np = np.asarray(labels, dtype=np.int64)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for cls in range(int(num_classes)):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
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
        "seed": int(seed),
        "method": "class_balanced_random",
    }


def split_train_val_class_balanced(
    labels: Sequence[int],
    num_classes: int,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(int(seed))
    labels_np = np.asarray(labels, dtype=np.int64)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for cls in range(int(num_classes)):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return {
        "train_indices": [int(i) for i in train_indices],
        "val_indices": [int(i) for i in val_indices],
        "seed": int(seed),
        "method": "train_val_class_balanced",
    }


def summarize_split(split: Mapping[str, Any], labels: Optional[Sequence[int]] = None, num_classes: Optional[int] = None) -> Dict[str, Any]:
    split = normalize_split_dict(split)
    summary: Dict[str, Any] = {}
    for name in ["train", "val", "test"]:
        key = f"{name}_indices" if name != "val" else "val_indices"
        indices = _to_int_list(split.get(key, []))
        info: Dict[str, Any] = {"num_samples": len(indices)}
        if labels is not None and len(indices) > 0:
            if num_classes is None:
                num_classes = max([int(y) for y in labels], default=-1) + 1
            counts = {str(i): 0 for i in range(int(num_classes))}
            for idx in indices:
                y = int(labels[idx])
                counts[str(y)] = counts.get(str(y), 0) + 1
            info["class_counts"] = counts
        summary[name] = info
    return summary


def validate_split(split: Mapping[str, Any], dataset_len: int, require_test: bool = True) -> None:
    split = normalize_split_dict(split)
    all_indices: List[int] = []
    for name in ["train", "val", "test"]:
        key = f"{name}_indices" if name != "val" else "val_indices"
        if key not in split:
            if name == "test" and require_test:
                raise KeyError("Split file does not contain test_indices")
            continue
        indices = _to_int_list(split[key])
        bad = [i for i in indices if i < 0 or i >= dataset_len]
        if bad:
            raise ValueError(f"Split {name} has out-of-range indices, e.g. {bad[:5]} for dataset_len={dataset_len}")
        all_indices.extend(indices)
    if len(all_indices) != len(set(all_indices)):
        logger.warning("Split contains overlapping indices across train/val/test.")
