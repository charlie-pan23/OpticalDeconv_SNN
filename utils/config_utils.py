"""Configuration helpers for HIPSA evaluation.

These utilities keep eval scripts config-driven. They intentionally avoid
hard-coding dataset-specific checkpoints, split files, or preprocessing modes.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


PathLike = str | os.PathLike[str]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file as a dict. Missing or empty files raise clear errors."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"Expected YAML dict in {path}, got {type(obj).__name__}")
    return obj


def save_yaml(obj: Mapping[str, Any], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(make_json_safe(obj), f, allow_unicode=True, sort_keys=False)


def load_json(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON dict in {path}, got {type(obj).__name__}")
    return obj


def save_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, indent=indent, ensure_ascii=False)


def make_json_safe(obj: Any) -> Any:
    """Convert common non-JSON objects into JSON-serializable objects."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None

    if isinstance(obj, Mapping):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if torch is not None and torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of base updated recursively by updates."""
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_by_path(cfg: Mapping[str, Any], key_path: str, default: Any = None, sep: str = ".") -> Any:
    """Read nested config value, e.g. get_by_path(cfg, 'training.optimizer.lr')."""
    cur: Any = cfg
    for key in key_path.split(sep):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_by_path(cfg: Dict[str, Any], key_path: str, value: Any, sep: str = ".") -> None:
    """Set nested config value in-place."""
    cur = cfg
    keys = key_path.split(sep)
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def project_root_from_config(config_path: PathLike) -> Path:
    """Infer project root from a config path.

    For typical `configs/foo.yaml`, project root is the parent of `configs/`.
    Otherwise, use current working directory.
    """
    p = Path(config_path).resolve()
    if p.parent.name == "configs":
        return p.parent.parent
    return Path.cwd().resolve()


def resolve_path(path: Optional[str], root: Optional[PathLike] = None) -> Optional[Path]:
    """Resolve a possibly relative path against root or cwd."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    base = Path(root) if root is not None else Path.cwd()
    return (base / p).resolve()


def dataset_tag(cfg: Mapping[str, Any]) -> str:
    name = str(get_by_path(cfg, "dataset.name", "dataset")).lower()
    aliases = {
        "cifar10-dvs": "cifar10dvs",
        "cifar10_dvs": "cifar10dvs",
        "cifar10dvs": "cifar10dvs",
        "dvs128gesture": "dvsgesture",
        "dvs_gesture": "dvsgesture",
        "dvs-gesture": "dvsgesture",
        "dvsgesture": "dvsgesture",
    }
    return aliases.get(name, name.replace(" ", "").replace("-", "_"))


def num_classes(cfg: Mapping[str, Any]) -> int:
    value = get_by_path(cfg, "dataset.num_classes", get_by_path(cfg, "model.num_classes", None))
    if value is None:
        tag = dataset_tag(cfg)
        if tag == "cifar10dvs":
            return 10
        if tag == "dvsgesture":
            return 11
        raise KeyError("num_classes not found in config and cannot infer from dataset.name")
    return int(value)


def time_steps(cfg: Mapping[str, Any]) -> int:
    value = get_by_path(cfg, "dataset.time_steps", get_by_path(cfg, "model.time_steps", None))
    if value is None:
        raise KeyError("time_steps not found in dataset/model config")
    return int(value)


def batch_size_eval(cfg: Mapping[str, Any], default: int = 1) -> int:
    candidates = [
        "evaluation.batch_size",
        "eval.batch_size",
        "training.batch_size_eval",
        "dataset.batch_size_eval",
        "dataset.batch_size",
    ]
    for key in candidates:
        value = get_by_path(cfg, key, None)
        if value is not None:
            return int(value)
    return int(default)


def num_workers_eval(cfg: Mapping[str, Any], default: int = 4) -> int:
    for key in ["evaluation.num_workers", "eval.num_workers", "training.num_workers"]:
        value = get_by_path(cfg, key, None)
        if value is not None:
            return int(value)
    return int(default)


def checkpoint_path_from_config(cfg: Mapping[str, Any], root: Optional[PathLike] = None) -> Optional[Path]:
    """Find checkpoint path from common config keys."""
    candidates = [
        "evaluation.checkpoint",
        "eval.checkpoint",
        "checkpoint.path",
        "paths.checkpoint",
        "paths.best_checkpoint",
        "output.checkpoint",
    ]
    for key in candidates:
        value = get_by_path(cfg, key, None)
        if value:
            return resolve_path(str(value), root)
    return None


def split_file_from_config(cfg: Mapping[str, Any], root: Optional[PathLike] = None) -> Optional[Path]:
    for key in ["split.split_file", "dataset.split_file", "paths.split_file"]:
        value = get_by_path(cfg, key, None)
        if value:
            return resolve_path(str(value), root)
    return None


def frame_dir_from_config(cfg: Mapping[str, Any], root: Optional[PathLike] = None) -> Optional[Path]:
    """Infer frame cache directory from config.

    Supports explicit dataset.frame_dir or constructs
    <dataset.root_dir>/<DatasetName>/frames_number_T_split_by_number for older configs.
    """
    explicit = get_by_path(cfg, "dataset.frame_dir", None) or get_by_path(cfg, "paths.frame_dir", None)
    if explicit:
        return resolve_path(str(explicit), root)

    root_dir = get_by_path(cfg, "dataset.root_dir", None)
    if root_dir is None:
        return None
    base = resolve_path(str(root_dir), root)
    tag = dataset_tag(cfg)
    folder_name = "CIFAR10DVS" if tag == "cifar10dvs" else "DVSGesture" if tag == "dvsgesture" else ""
    T = time_steps(cfg)
    if folder_name:
        return base / folder_name / f"frames_number_{T}_split_by_number"
    return base / f"frames_number_{T}_split_by_number"


def load_config(config_path: PathLike, required: bool = True) -> Dict[str, Any]:
    if required:
        return load_yaml(config_path)
    path = Path(config_path)
    return load_yaml(path) if path.exists() else {}


def load_eval_config(
    config_path: PathLike,
    hardware_path: Optional[PathLike] = None,
    device_params_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Load dataset/model config and optionally attach hardware/device configs."""
    cfg = load_config(config_path)
    root = project_root_from_config(config_path)

    if hardware_path is None:
        candidate = root / "configs" / "hardware_hipsa.yaml"
        hardware_path = candidate if candidate.exists() else None
    if device_params_path is None:
        candidate = root / "configs" / "device_params.yaml"
        device_params_path = candidate if candidate.exists() else None

    if hardware_path is not None and Path(hardware_path).exists():
        cfg["hardware"] = deep_update(cfg.get("hardware", {}), load_yaml(hardware_path))
        cfg.setdefault("paths", {})["hardware_config"] = str(hardware_path)
    if device_params_path is not None and Path(device_params_path).exists():
        cfg["device_params"] = deep_update(cfg.get("device_params", {}), load_yaml(device_params_path))
        cfg.setdefault("paths", {})["device_params"] = str(device_params_path)
    cfg.setdefault("_meta", {})["config_path"] = str(config_path)
    cfg["_meta"]["project_root"] = str(root)
    return cfg


def copy_config_snapshot(
    config_paths: Sequence[PathLike],
    output_dir: PathLike,
    snapshot_name: str = "config_snapshot.yaml",
    merged_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save a merged snapshot and copy source config files for reproducibility."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if merged_config is not None:
        save_yaml(dict(merged_config), out / snapshot_name)
    for p in config_paths:
        src = Path(p)
        if src.exists():
            shutil.copy2(src, out / src.name)


def validate_eval_config(cfg: Mapping[str, Any]) -> Tuple[bool, list[str]]:
    """Lightweight checks before running eval scripts."""
    errors: list[str] = []
    for key in ["dataset.name", "dataset.time_steps", "model.class_name"]:
        if get_by_path(cfg, key, None) is None:
            errors.append(f"Missing config key: {key}")
    if split_file_from_config(cfg, get_by_path(cfg, "_meta.project_root", None)) is None:
        errors.append("Missing split file: split.split_file")
    return (len(errors) == 0), errors

# =============================================================================
# Eval artifact resolution helpers (added for eval_00-03 compatibility)
# =============================================================================

CANONICAL_EVAL_ARTIFACTS: Dict[str, Dict[str, Any]] = {
    "cifar10dvs": {
        "config": "configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml",
        "checkpoint": "results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth",
        "train_run": "results/cifar10dvs/cifar10dvs/run_20260629_194703_cifar10dvs",
        "test_run": "results/cifar10dvs/run_20260629_223733_cifar10dvs",
        "input_encoding": "clipped_count",
        "clipped_count_max": 3.0,
        "expected_test_acc": 76.40,
        "expected_test_loss": 0.9317,
    },
    "dvsgesture": {
        "config": "results/dvsgesture/config_dvsgesture_acc88p54.yaml",
        "checkpoint": "results/dvsgesture/best_dvsgesture_acc88p54.pth",
        "final_test": "results/dvsgesture/final_test_acc88p54.json",
        "train_run": "results/dvsgesture/run_20260629_191844_dvsgesture",
        "test_run": "results/dvsgesture/run_20260629_191844_dvsgesture",
        "input_encoding": "binary",
        "expected_test_acc": 88.54,
        "expected_test_loss": 0.4111,
    },
}


def _path_exists(path: Optional[Path]) -> bool:
    return path is not None and Path(path).exists()


def _first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def _project_root_from_cfg_or_cwd(cfg: Mapping[str, Any], root: Optional[PathLike] = None) -> Path:
    if root is not None:
        return Path(root).resolve()
    meta_root = get_by_path(cfg, "_meta.project_root", None)
    if meta_root:
        return Path(str(meta_root)).resolve()
    return Path.cwd().resolve()


def dataset_root_from_config(cfg: Mapping[str, Any], root: Optional[PathLike] = None) -> Path:
    """Return the dataset root path expected by SpikingJelly.

    For CIFAR10-DVS this returns `<project>/datasets/CIFAR10DVS`.
    For DVS Gesture this returns `<project>/datasets/DVSGesture`.

    The function accepts both styles in config:
    - dataset.root_dir: ./datasets
    - dataset.root_dir: ./datasets/CIFAR10DVS or ./datasets/DVSGesture
    """
    project_root = _project_root_from_cfg_or_cwd(cfg, root)
    tag = dataset_tag(cfg)
    dataset_folder = "CIFAR10DVS" if tag == "cifar10dvs" else "DVSGesture" if tag == "dvsgesture" else tag

    root_dir = get_by_path(cfg, "dataset.root_dir", None)
    if root_dir is None:
        return (project_root / "datasets" / dataset_folder).resolve()

    base = resolve_path(str(root_dir), project_root)
    if base is None:
        return (project_root / "datasets" / dataset_folder).resolve()

    # If root_dir already points to the dataset-specific folder, do not append again.
    if base.name.lower() == dataset_folder.lower():
        return base.resolve()
    return (base / dataset_folder).resolve()


# Redefine split_file_from_config with safe fallback discovery.
def split_file_from_config(cfg: Mapping[str, Any], root: Optional[PathLike] = None) -> Optional[Path]:  # type: ignore[no-redef]
    project_root = _project_root_from_cfg_or_cwd(cfg, root)
    for key in ["split.split_file", "dataset.split_file", "paths.split_file"]:
        value = get_by_path(cfg, key, None)
        if value:
            p = resolve_path(str(value), project_root)
            if p is not None:
                return p

    tag = dataset_tag(cfg)
    T = time_steps(cfg)
    ds_root = dataset_root_from_config(cfg, project_root)
    split_dir = ds_root / "splits"
    if not split_dir.exists():
        return None

    patterns = []
    if tag == "dvsgesture":
        patterns.extend([f"dvsgesture_T{T}_seed*_split.pth", f"*T{T}*split*.pth", "*.pth"])
    elif tag == "cifar10dvs":
        patterns.extend([f"*T{T}*split*.pth", f"*{T}*split*.pth", "*.pth"])
    else:
        patterns.extend([f"*T{T}*split*.pth", "*.pth"])

    for pattern in patterns:
        matches = sorted(split_dir.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


# Redefine frame_dir_from_config using dataset_root_from_config to avoid duplicated paths.
def frame_dir_from_config(cfg: Mapping[str, Any], root: Optional[PathLike] = None) -> Optional[Path]:  # type: ignore[no-redef]
    project_root = _project_root_from_cfg_or_cwd(cfg, root)
    explicit = get_by_path(cfg, "dataset.frame_dir", None) or get_by_path(cfg, "paths.frame_dir", None)
    if explicit:
        return resolve_path(str(explicit), project_root)
    ds_root = dataset_root_from_config(cfg, project_root)
    return (ds_root / f"frames_number_{time_steps(cfg)}_split_by_number").resolve()


def _canonical_path(tag: str, key: str, project_root: Path) -> Optional[Path]:
    value = CANONICAL_EVAL_ARTIFACTS.get(tag, {}).get(key)
    if not value:
        return None
    return resolve_path(str(value), project_root)


def resolve_eval_artifacts(
    config_path: PathLike,
    checkpoint: Optional[PathLike] = None,
    run_dir: Optional[PathLike] = None,
    root: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Resolve checkpoint, dataset paths, split file, and frozen eval metadata.

    Command-line arguments have the highest priority, followed by values inside
    config, then the frozen artifact names currently used by this HIPSA project.
    """
    cfg = load_config(config_path)
    project_root = Path(root).resolve() if root is not None else project_root_from_config(config_path)
    cfg.setdefault("_meta", {})["project_root"] = str(project_root)
    cfg["_meta"]["config_path"] = str(config_path)

    tag = dataset_tag(cfg)
    canonical = CANONICAL_EVAL_ARTIFACTS.get(tag, {})

    ckpt_candidates = [
        resolve_path(str(checkpoint), project_root) if checkpoint is not None else None,
        checkpoint_path_from_config(cfg, project_root),
        _canonical_path(tag, "checkpoint", project_root),
    ]
    ckpt = _first_existing(ckpt_candidates) or ckpt_candidates[0] or ckpt_candidates[1] or ckpt_candidates[2]

    run_candidates = [
        resolve_path(str(run_dir), project_root) if run_dir is not None else None,
        resolve_path(str(get_by_path(cfg, "paths.run_dir", "")), project_root) if get_by_path(cfg, "paths.run_dir", None) else None,
        _canonical_path(tag, "train_run", project_root),
    ]
    resolved_run = _first_existing(run_candidates) or run_candidates[0] or run_candidates[1] or run_candidates[2]

    final_test = None
    for key in ["paths.final_test", "evaluation.final_test", "eval.final_test"]:
        value = get_by_path(cfg, key, None)
        if value:
            final_test = resolve_path(str(value), project_root)
            break
    if final_test is None:
        final_test = _canonical_path(tag, "final_test", project_root)

    input_encoding = str(
        get_by_path(cfg, "preprocess.input_encoding", canonical.get("input_encoding", "binary"))
    ).lower()

    # Keep exported eval artifacts semantically clean.
    # `clipped_count_max` is meaningful only when the active input encoding is
    # clipped_count. For binary/raw/normalized_count encodings, store None so
    # JSON output becomes null and downstream scripts do not accidentally treat
    # it as an active clipped-count setting.
    if input_encoding == "clipped_count":
        clipped_count_max = get_by_path(
            cfg, "preprocess.clipped_count_max", canonical.get("clipped_count_max", 3.0)
        )
    else:
        clipped_count_max = None

    artifacts: Dict[str, Any] = {
        "dataset": tag,
        "config_path": Path(config_path).resolve(),
        "project_root": project_root,
        "checkpoint": ckpt,
        "run_dir": resolved_run,
        "train_run": _canonical_path(tag, "train_run", project_root),
        "test_run": _canonical_path(tag, "test_run", project_root),
        "final_test_json": final_test,
        "dataset_root": dataset_root_from_config(cfg, project_root),
        "frame_dir": frame_dir_from_config(cfg, project_root),
        "split_file": split_file_from_config(cfg, project_root),
        "time_steps": time_steps(cfg),
        "num_classes": num_classes(cfg),
        "model_class": get_by_path(cfg, "model.class_name", None),
        "input_encoding": input_encoding,
        "clipped_count_max": clipped_count_max,
        "expected_test_acc": canonical.get("expected_test_acc"),
        "expected_test_loss": canonical.get("expected_test_loss"),
    }
    return artifacts


def validate_eval_artifacts(artifacts: Mapping[str, Any], require_split: bool = True) -> Tuple[bool, list[str]]:
    """Check whether resolved eval artifacts exist on disk."""
    errors: list[str] = []
    for key in ["config_path", "checkpoint", "dataset_root"]:
        value = artifacts.get(key)
        if value is None or not Path(value).exists():
            errors.append(f"Missing {key}: {value}")
    frame_dir = artifacts.get("frame_dir")
    if frame_dir is not None and not Path(frame_dir).exists():
        errors.append(f"Missing frame_dir: {frame_dir}")
    split_file = artifacts.get("split_file")
    if require_split and (split_file is None or not Path(split_file).exists()):
        errors.append(f"Missing split_file: {split_file}")
    return len(errors) == 0, errors
