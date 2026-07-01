"""Shared utilities for HIPSA eval_00-06 scripts.

Design rules:
1. eval scripts collect and save data only.
2. plot scripts read saved JSON/CSV/NPY data and generate figures.
3. eval outputs go to:
   results/eval_v2/<dataset>/<eval_xx>/
4. plot outputs go to:
   plot/results/<plot_xx>/<dataset>/
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.model_registry import build_model, describe_model
from utils.checkpoint_utils import load_model_weights
from utils.config_utils import (
    batch_size_eval,
    dataset_root_from_config,
    dataset_tag,
    load_eval_config,
    load_json,
    load_yaml,
    num_classes,
    num_workers_eval,
    resolve_eval_artifacts,
    save_json,
    split_file_from_config,
    time_steps,
)
from utils.data_utils import (
    accuracy_from_logits,
    aggregate_time_logits,
    logits_aggregation_from_config,
    prepare_snn_batch,
    reset_snn_state,
)
from utils.seed_utils import make_generator, seed_worker
from utils.split_utils import get_split_indices, load_split, validate_split

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("HIPSA")


@dataclass
class EvalContext:
    """Runtime context shared by eval scripts."""

    config_path: Path
    config: Dict[str, Any]
    artifacts: Dict[str, Any]
    dataset: str
    device: torch.device
    output_dir: Path
    checkpoint: Path
    run_dir: Optional[Path]
    hardware_cfg: Dict[str, Any]
    device_params: Dict[str, Any]


def choose_device(device: str = "auto") -> torch.device:
    """Choose evaluation device.

    Use CUDA when available unless the user explicitly passes cpu/cuda.
    """

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_output_dir(
    config_path: str | Path,
    eval_name: str,
    dataset: Optional[str] = None,
    root: Optional[str | Path] = None,
) -> Path:
    """Create standard eval output directory.

    Default:
      results/eval_v2/<dataset>/<eval_name>/

    Example:
      results/eval_v2/cifar10dvs/eval_00/
    """

    cfg = load_eval_config(config_path)
    tag = dataset or dataset_tag(cfg)

    base = Path(root) if root is not None else Path("results") / "eval_v2"
    out = base / tag / eval_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_eval_context(
    config_path: str | Path,
    checkpoint: Optional[str | Path] = None,
    run_dir: Optional[str | Path] = None,
    hardware_config: str | Path = "configs/hardware_hipsa.yaml",
    device_params: str | Path = "configs/device_params.yaml",
    output_dir: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    eval_name: str = "eval_00",
    device: str = "auto",
) -> EvalContext:
    """Load config, checkpoint, hardware config, device params, and output dir.

    `output_dir` is treated as the exact final directory.
    `output_root` is treated as the root and expands to:
      <output_root>/<dataset>/<eval_name>/

    If neither is provided, use:
      results/eval_v2/<dataset>/<eval_name>/
    """

    config_path = Path(config_path)

    hw_path = Path(hardware_config)
    dp_path = Path(device_params)

    cfg = load_eval_config(
        config_path,
        hw_path if hw_path.exists() else None,
        dp_path if dp_path.exists() else None,
    )

    artifacts = resolve_eval_artifacts(
        config_path,
        checkpoint=checkpoint,
        run_dir=run_dir,
    )

    tag = dataset_tag(cfg)
    dev = choose_device(device)

    ckpt = Path(checkpoint) if checkpoint is not None else artifacts.get("checkpoint")
    if ckpt is None:
        raise FileNotFoundError(
            "Could not resolve checkpoint. Pass --checkpoint explicitly."
        )

    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = make_output_dir(
            config_path=config_path,
            eval_name=eval_name,
            dataset=tag,
            root=output_root,
        )

    hw = cfg.get("hardware", {}) if isinstance(cfg.get("hardware", {}), dict) else {}
    dp = (
        cfg.get("device_params", {})
        if isinstance(cfg.get("device_params", {}), dict)
        else {}
    )

    if not hw and hw_path.exists():
        hw = load_yaml(hw_path)
    if not dp and dp_path.exists():
        dp = load_yaml(dp_path)

    resolved_run_dir = None
    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
    elif artifacts.get("run_dir"):
        resolved_run_dir = Path(artifacts["run_dir"])

    return EvalContext(
        config_path=config_path,
        config=cfg,
        artifacts=artifacts,
        dataset=tag,
        device=dev,
        output_dir=out,
        checkpoint=ckpt,
        run_dir=resolved_run_dir,
        hardware_cfg=hw,
        device_params=dp,
    )


def build_dataset(config: Mapping[str, Any], split_name: str = "test") -> Any:
    """Build the raw SpikingJelly dataset for the requested split."""

    tag = dataset_tag(config)
    T = time_steps(config)

    ds_cfg = (
        config.get("dataset", {})
        if isinstance(config.get("dataset", {}), Mapping)
        else {}
    )

    root = dataset_root_from_config(config)
    if root is None:
        root = Path("datasets") / (
            "CIFAR10DVS" if tag == "cifar10dvs" else "DVSGesture"
        )

    data_type = str(ds_cfg.get("data_type", "frame"))
    split_by = str(ds_cfg.get("split_by", "number"))

    if tag == "cifar10dvs":
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

        return CIFAR10DVS(
            root=str(root),
            data_type=data_type,
            frames_number=T,
            split_by=split_by,
        )

    if tag == "dvsgesture":
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

        # DVS128Gesture has separate train=True / train=False datasets.
        # For train/val we load train=True; for test we load train=False.
        train_flag = split_name.lower() in {"train", "val", "valid", "validation"}

        return DVS128Gesture(
            root=str(root),
            train=train_flag,
            data_type=data_type,
            frames_number=T,
            split_by=split_by,
        )

    raise ValueError(f"Unsupported dataset for eval: {tag}")


def _validate_selected_indices(
    indices: Sequence[int],
    dataset_len: int,
    split_name: str,
    dataset_name: str,
) -> None:
    """Validate selected split indices against the currently loaded dataset."""

    if len(indices) == 0:
        raise ValueError(f"Empty split '{split_name}' for dataset '{dataset_name}'.")

    min_idx = min(indices)
    max_idx = max(indices)

    if min_idx < 0 or max_idx >= dataset_len:
        raise IndexError(
            f"Split '{split_name}' for dataset '{dataset_name}' has invalid indices: "
            f"min={min_idx}, max={max_idx}, dataset_len={dataset_len}. "
            "For DVS Gesture, remember train/val indices are relative to train=True "
            "and test indices are relative to train=False."
        )


def build_dataset_subset(
    config: Mapping[str, Any],
    split_name: str = "test",
    allow_no_split: bool = False,
) -> Any:
    """Build dataset subset using the frozen split file.

    Important:
    - CIFAR10-DVS uses one dataset and split indices are validated globally.
    - DVS Gesture uses train=True for train/val and train=False for test, so we
      validate only selected indices against the loaded subset dataset.
    """

    dataset = build_dataset(config, split_name=split_name)
    tag = dataset_tag(config)

    split_path = split_file_from_config(config)
    if split_path is None or not Path(split_path).exists():
        if allow_no_split:
            logger.warning("No split file found; using full %s dataset.", split_name)
            return dataset

        raise FileNotFoundError(
            "Split file not found. Evaluation must use the frozen training split. "
            "Pass --allow-no-split only for debugging."
        )

    split = load_split(split_path)
    indices = get_split_indices(split, split_name)

    if tag == "dvsgesture":
        _validate_selected_indices(
            indices=indices,
            dataset_len=len(dataset),
            split_name=split_name,
            dataset_name=tag,
        )
    else:
        validate_split(
            split,
            len(dataset),
            require_test=(split_name == "test"),
        )

    return Subset(dataset, indices)


def _default_num_workers(
    config: Mapping[str, Any],
    num_workers: Optional[int],
) -> int:
    """Return stable DataLoader num_workers.

    On Windows local machines, default to 0 to avoid multiprocessing issues with
    SpikingJelly datasets. On Linux servers, use config value or default 4.
    """

    if num_workers is not None:
        return int(num_workers)

    if os.name == "nt":
        return 0

    return int(num_workers_eval(config, default=4))


def build_loader(
    config: Mapping[str, Any],
    split_name: str = "test",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    allow_no_split: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """Build deterministic evaluation DataLoader."""

    ds = build_dataset_subset(
        config,
        split_name=split_name,
        allow_no_split=allow_no_split,
    )

    bs = int(batch_size if batch_size is not None else batch_size_eval(config, default=1))
    nw = _default_num_workers(config, num_workers)

    seed = int(
        config.get("experiment", {}).get(
            "seed",
            config.get("split", {}).get("seed", 42),
        )
    )
    gen = make_generator(seed)

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker if nw > 0 else None,
        generator=gen,
    )


def build_eval_model(context: EvalContext, strict: bool = True) -> nn.Module:
    """Build model from config and load checkpoint weights."""

    model = build_model(context.config).to(context.device)
    meta = load_model_weights(
        model,
        context.checkpoint,
        device=context.device,
        strict=strict,
    )

    model.eval()

    logger.info("Loaded model: %s", describe_model(model))
    logger.info("Checkpoint metadata: %s", meta)

    return model


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Quick clean accuracy evaluation.

    This helper is useful for sanity checks.
    Formal eval_00 should still save predictions, confusion matrix, and summary.
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    agg_mode = logits_aggregation_from_config(config)

    for i, (data, target) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        reset_snn_state(model)
        data, target = prepare_snn_batch(data, target, config, device)

        logits_t = model(data)
        logits = aggregate_time_logits(logits_t, agg_mode)

        loss = criterion(logits, target)
        c, n = accuracy_from_logits(logits, target)

        correct += c
        total += n
        loss_sum += float(loss.item()) * n

    return {
        "loss": loss_sum / max(total, 1),
        "acc": 100.0 * correct / max(total, 1),
        "correct": int(correct),
        "total": int(total),
    }


def save_csv_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Save list-like dictionaries to CSV with stable field order."""

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            key = str(key)
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})