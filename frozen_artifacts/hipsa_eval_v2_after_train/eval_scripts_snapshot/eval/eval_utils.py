"""Shared utilities for HIPSA eval00-03 scripts."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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
    frame_dir_from_config,
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
from utils.data_utils import prepare_snn_batch, reset_snn_state, aggregate_time_logits, logits_aggregation_from_config, accuracy_from_logits
from utils.seed_utils import make_generator, seed_worker
from utils.split_utils import get_split_indices, load_split, validate_split

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


@dataclass
class EvalContext:
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
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_output_dir(config_path: str | Path, eval_name: str, dataset: Optional[str] = None, root: Optional[str | Path] = None) -> Path:
    cfg = load_eval_config(config_path)
    tag = dataset or dataset_tag(cfg)
    base = Path(root) if root is not None else Path("results") / tag / "eval"
    out = base / eval_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_eval_context(
    config_path: str | Path,
    checkpoint: Optional[str | Path] = None,
    run_dir: Optional[str | Path] = None,
    hardware_config: str | Path = "configs/hardware_hipsa.yaml",
    device_params: str | Path = "configs/device_params.yaml",
    output_dir: Optional[str | Path] = None,
    eval_name: str = "eval",
    device: str = "auto",
) -> EvalContext:
    config_path = Path(config_path)
    cfg = load_eval_config(config_path, hardware_config if Path(hardware_config).exists() else None, device_params if Path(device_params).exists() else None)
    artifacts = resolve_eval_artifacts(config_path, checkpoint=checkpoint, run_dir=run_dir)
    tag = dataset_tag(cfg)
    dev = choose_device(device)

    ckpt = Path(checkpoint) if checkpoint is not None else artifacts.get("checkpoint")
    if ckpt is None:
        raise FileNotFoundError("Could not resolve checkpoint. Pass --checkpoint explicitly.")
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    out = Path(output_dir) if output_dir is not None else make_output_dir(config_path, eval_name, tag)
    out.mkdir(parents=True, exist_ok=True)

    hw = cfg.get("hardware", {}) if isinstance(cfg.get("hardware", {}), dict) else {}
    dp = cfg.get("device_params", {}) if isinstance(cfg.get("device_params", {}), dict) else {}
    if not hw and Path(hardware_config).exists():
        hw = load_yaml(hardware_config)
    if not dp and Path(device_params).exists():
        dp = load_yaml(device_params)

    return EvalContext(
        config_path=config_path,
        config=cfg,
        artifacts=artifacts,
        dataset=tag,
        device=dev,
        output_dir=out,
        checkpoint=ckpt,
        run_dir=Path(run_dir) if run_dir is not None else (Path(artifacts["run_dir"]) if artifacts.get("run_dir") else None),
        hardware_cfg=hw,
        device_params=dp,
    )


def build_dataset(config: Mapping[str, Any], split_name: str = "test") -> Any:
    tag = dataset_tag(config)
    T = time_steps(config)
    ds_cfg = config.get("dataset", {}) if isinstance(config.get("dataset", {}), Mapping) else {}
    root = dataset_root_from_config(config) or Path("datasets") / ("CIFAR10DVS" if tag == "cifar10dvs" else "DVSGesture")
    data_type = str(ds_cfg.get("data_type", "frame"))
    split_by = str(ds_cfg.get("split_by", "number"))
    if tag == "cifar10dvs":
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        return CIFAR10DVS(root=str(root), data_type=data_type, frames_number=T, split_by=split_by)
    if tag == "dvsgesture":
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        train = split_name.lower() in {"train", "val", "valid", "validation"}
        return DVS128Gesture(root=str(root), train=train, data_type=data_type, frames_number=T, split_by=split_by)
    raise ValueError(f"Unsupported dataset for eval: {tag}")


def build_dataset_subset(config: Mapping[str, Any], split_name: str = "test", allow_no_split: bool = False) -> Any:
    dataset = build_dataset(config, split_name=split_name)
    split_path = split_file_from_config(config)
    if split_path is None or not Path(split_path).exists():
        if allow_no_split:
            logger.warning("No split file found; using full %s dataset.", split_name)
            return dataset
        raise FileNotFoundError("Split file not found. Evaluation must use training split. Pass --allow-no-split only for debugging.")
    split = load_split(split_path)
    try:
        validate_split(split, len(dataset), require_test=(split_name == "test" and dataset_tag(config) != "dvsgesture"))
    except Exception as exc:
        # DVS split uses train indices relative to train=True and test indices relative to train=False.
        if dataset_tag(config) != "dvsgesture":
            raise exc
    indices = get_split_indices(split, split_name)
    return Subset(dataset, indices)


def build_loader(config: Mapping[str, Any], split_name: str = "test", batch_size: Optional[int] = None, num_workers: Optional[int] = None, allow_no_split: bool = False, shuffle: bool = False) -> DataLoader:
    ds = build_dataset_subset(config, split_name=split_name, allow_no_split=allow_no_split)
    bs = int(batch_size if batch_size is not None else batch_size_eval(config, default=1))
    nw = int(num_workers if num_workers is not None else num_workers_eval(config, default=4))
    gen = make_generator(int(config.get("experiment", {}).get("seed", config.get("split", {}).get("seed", 42))))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=False, worker_init_fn=seed_worker if nw > 0 else None, generator=gen)


def build_eval_model(context: EvalContext, strict: bool = True) -> nn.Module:
    model = build_model(context.config).to(context.device)
    meta = load_model_weights(model, context.checkpoint, device=context.device, strict=strict)
    model.eval()
    logger.info("Loaded model: %s", describe_model(model))
    logger.info("Checkpoint metadata: %s", meta)
    return model


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, config: Mapping[str, Any], device: torch.device, max_batches: Optional[int] = None) -> Dict[str, Any]:
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
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(str(k))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
