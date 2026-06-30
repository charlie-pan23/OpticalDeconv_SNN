"""Data and batch helpers shared by HIPSA evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from spikingjelly.activation_based import functional as sf
except Exception:  # pragma: no cover
    sf = None

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


def get_dataset_labels(dataset: Dataset, class_names: Optional[Sequence[str]] = None) -> List[int]:
    """Extract integer labels from common Dataset representations.

    SpikingJelly datasets do not always expose labels the same way across
    versions. This function tries metadata first, and only scans samples as a
    slow fallback. Use it during preprocessing/split creation, not per batch.
    """
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            try:
                return [int(v) for v in values]
            except Exception:
                pass

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        labels: List[int] = []
        ok = True
        for item in samples:
            try:
                labels.append(int(item[1]))
            except Exception:
                ok = False
                break
        if ok and len(labels) == len(dataset):
            return labels

        if class_names is not None:
            class_to_idx = {str(c): i for i, c in enumerate(class_names)}
            labels = []
            for item in samples:
                path = str(item[0])
                found = None
                for part in Path(path).parts:
                    if part in class_to_idx:
                        found = class_to_idx[part]
                        break
                if found is None:
                    labels = []
                    break
                labels.append(found)
            if len(labels) == len(dataset):
                return labels

    logger.warning("Could not read labels from dataset metadata; scanning samples once.")
    labels = []
    for i in tqdm(range(len(dataset)), desc="Scanning labels", leave=False):
        _, target = dataset[i]
        if torch.is_tensor(target):
            target = int(target.item())
        labels.append(int(target))
    return labels


def encode_frames(x: torch.Tensor, preprocess_cfg: Mapping[str, Any]) -> torch.Tensor:
    """Encode event frames according to config.

    Supports:
    - binary:        (x > threshold).float()
    - clipped_count: clamp(x, 0, max) / max
    - normalized_count: sample-wise max normalization
    - none/raw:      float frames without value conversion
    """
    x = x.float()
    mode = str(preprocess_cfg.get("input_encoding", preprocess_cfg.get("encoding", "binary"))).lower()

    if mode in {"none", "raw", "float", "count"}:
        return x
    if mode == "binary":
        threshold = float(preprocess_cfg.get("binary_threshold", 0.0))
        return (x > threshold).float()
    if mode == "clipped_count":
        max_v = float(preprocess_cfg.get("clipped_count_max", 3.0))
        if max_v <= 0:
            raise ValueError("preprocess.clipped_count_max must be positive")
        return torch.clamp(x, 0.0, max_v) / max_v
    if mode == "normalized_count":
        eps = float(preprocess_cfg.get("normalize_eps", 1.0e-6))
        # For [B,T,C,H,W], normalize per sample over T,C,H,W.
        # For [T,B,C,H,W], normalize per batch item over T,C,H,W after transpose.
        if x.dim() == 5:
            dims = tuple(range(1, x.dim())) if x.shape[0] != int(preprocess_cfg.get("time_steps", -1)) else (0, 2, 3, 4)
        else:
            dims = tuple(range(1, x.dim()))
        denom = x.amax(dim=dims, keepdim=True).clamp_min(eps)
        return x / denom
    raise ValueError(f"Unsupported input_encoding: {mode}")


def infer_time_dim(data: torch.Tensor, time_steps: int) -> int:
    if data.dim() != 5:
        raise ValueError(f"Expected 5D event frame tensor, got shape={tuple(data.shape)}")
    if data.shape[0] == int(time_steps):
        return 0
    if data.shape[1] == int(time_steps):
        return 1
    raise ValueError(f"Cannot infer time dimension from shape={tuple(data.shape)} with T={time_steps}")


def to_time_first(data: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Convert event frames to [T,B,C,H,W]."""
    time_dim = infer_time_dim(data, time_steps)
    if time_dim == 0:
        return data.contiguous()
    return data.transpose(0, 1).contiguous()


def prepare_snn_batch(
    data: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    non_blocking: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a SpikingJelly frame batch for SNN inference.

    Dataset output is usually [B,T,C,H,W]; model input is [T,B,C,H,W].
    """
    dataset_cfg = config.get("dataset", {}) if isinstance(config, Mapping) else {}
    preprocess_cfg = dict(config.get("preprocess", {})) if isinstance(config, Mapping) else {}
    T = int(dataset_cfg.get("time_steps", preprocess_cfg.get("time_steps", 10)))
    preprocess_cfg.setdefault("time_steps", T)

    data = encode_frames(data, preprocess_cfg)
    data = to_time_first(data, T)
    data = data.to(device, non_blocking=non_blocking)
    target = target.to(device, non_blocking=non_blocking).long().view(-1)
    return data, target


def reset_snn_state(model: torch.nn.Module) -> None:
    """Reset SpikingJelly neuron states if available."""
    if sf is None:
        return
    try:
        sf.reset_net(model)
    except Exception as exc:
        logger.debug("reset_net failed: %s", exc)


def logits_aggregation_from_config(config: Mapping[str, Any], default: str = "mean_time") -> str:
    """Read the timestep-logit aggregation mode from a config mapping.

    Training and evaluation must use the same aggregation rule. The HIPSA
    training configs normally store this value under ``training.logits_aggregation``.
    This helper also checks a few fallback locations for compatibility with older
    configs and returns ``mean_time`` by default.
    """
    if not isinstance(config, Mapping):
        return str(default)

    candidate = config.get("logits_aggregation")
    if candidate is not None:
        return str(candidate)

    for section_name in ("training", "evaluation", "eval", "model", "network"):
        section = config.get(section_name, {})
        if isinstance(section, Mapping):
            candidate = section.get("logits_aggregation")
            if candidate is not None:
                return str(candidate)
            candidate = section.get("aggregation")
            if candidate is not None:
                return str(candidate)

    return str(default)


def aggregate_time_logits(logits_t: torch.Tensor, mode: str = "mean_time") -> torch.Tensor:
    """Aggregate [T,B,C] logits into [B,C]."""
    mode = str(mode).lower()
    if logits_t.dim() != 3:
        raise ValueError(f"Expected logits_t shape [T,B,C], got {tuple(logits_t.shape)}")
    if mode in {"mean", "mean_time", "avg"}:
        return logits_t.mean(dim=0)
    if mode in {"sum", "sum_time"}:
        return logits_t.sum(dim=0)
    if mode in {"last", "last_time"}:
        return logits_t[-1]
    raise ValueError(f"Unsupported logits aggregation mode: {mode}")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = int((preds == targets).sum().item())
    total = int(targets.numel())
    return correct, total


def class_counts(labels: Sequence[int], num_classes: Optional[int] = None) -> Dict[str, int]:
    if num_classes is None:
        num_classes = max([int(x) for x in labels], default=-1) + 1
    counts = {str(i): 0 for i in range(int(num_classes))}
    for y in labels:
        key = str(int(y))
        counts[key] = counts.get(key, 0) + 1
    return counts
