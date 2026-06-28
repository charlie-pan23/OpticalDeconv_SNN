"""Robust CIFAR10-DVS preprocessing for HIPSA.

This version avoids the common SpikingJelly failure where empty
``events_np`` or ``frames_number_*`` folders are created and then torchvision
thinks the dataset cache is complete but finds no .npz files.

Run from project root:
  python train/pre_cifar10dvs.py --config configs/config_cifar10dvs.yaml

Main behavior:
  1. Inspect raw zip / extract / events_np / frame caches.
  2. If event .npz files are missing, manually convert .aedat -> event .npz
     using CIFAR10DVS.read_aedat_save_to_np, similar to fix_dataset.py.
  3. Remove empty/incomplete frame cache before asking SpikingJelly to build frames.
  4. Instantiate CIFAR10DVS(frame mode), then save deterministic split + manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

from train.train_utils import (
    class_counts,
    ensure_dir,
    get_dataset_labels,
    load_config,
    load_split,
    save_json,
    save_split,
    set_seed,
    split_class_balanced,
    summarize_split,
)

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    from train.train_utils import logger


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_ZIPS = [f"{c}.zip" for c in CLASSES]


def apply_numpy_legacy_patch() -> None:
    """Compatibility patch for old SpikingJelly readers under new NumPy.

    This is intentionally local to preprocessing. It should not affect training
    math because training uses tensors loaded from generated frame .npz files.
    """
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]


def sha1_file(path: Path, max_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def count_npz_by_class(root: Path) -> Dict[str, int]:
    return {cls: len(list((root / cls).glob("*.npz"))) if (root / cls).exists() else 0 for cls in CLASSES}


def count_aedat_by_class(root: Path) -> Dict[str, int]:
    return {cls: len(list((root / cls).rglob("*.aedat"))) if (root / cls).exists() else 0 for cls in CLASSES}


def total_count(counts: Dict[str, int]) -> int:
    return int(sum(counts.values()))


def find_zip_for_class(raw_dir: Path, nested_dir: Path, cls: str) -> Optional[Path]:
    candidates = [raw_dir / f"{cls}.zip", nested_dir / f"{cls}.zip"]
    for p in candidates:
        if p.exists():
            return p
    return None


def get_class_workers(config: Dict[str, Any], default: int = 10) -> int:
    """Number of class-level workers for CIFAR10-DVS preprocessing.

    CIFAR10-DVS has exactly 10 classes. A class-level worker avoids creating
    10,000 tiny tasks while still using multicore CPUs well.
    """
    v = config.get("preprocess", {}).get("class_workers", default)
    try:
        workers = int(v)
    except Exception:
        workers = default
    return max(1, min(workers, len(CLASSES)))


def _extract_one_class(cls: str, raw_dir: Path, nested_dir: Path, extract_dir: Path) -> Tuple[str, int]:
    cls_extract = extract_dir / cls
    current = len(list(cls_extract.rglob("*.aedat"))) if cls_extract.exists() else 0
    if current >= 1000:
        return cls, current

    # Remove incomplete class extraction to avoid mixing partial files from an interrupted run.
    if cls_extract.exists():
        shutil.rmtree(cls_extract)
    cls_extract.mkdir(parents=True, exist_ok=True)

    zip_path = find_zip_for_class(raw_dir, nested_dir, cls)
    if zip_path is None:
        raise FileNotFoundError(
            f"Cannot find {cls}.zip under {raw_dir} or {nested_dir}. Check CIFAR10-DVS raw zip layout."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cls_extract)

    return cls, len(list(cls_extract.rglob("*.aedat")))


def _convert_one_class(cls: str, extract_dir: Path, events_np_dir: Path) -> Tuple[str, int, int, List[str]]:
    """Convert one CIFAR10-DVS class sequentially inside a class-level worker."""
    # Keep the legacy patch inside each worker too, because ThreadPool workers
    # may enter this function before the main conversion loop has touched NumPy.
    apply_numpy_legacy_patch()

    cls_extract = extract_dir / cls
    cls_events = events_np_dir / cls
    cls_events.mkdir(parents=True, exist_ok=True)

    aedat_files = sorted(cls_extract.rglob("*.aedat"))
    total_files = len(aedat_files)
    success_files = 0
    failed: List[str] = []

    for src in aedat_files:
        dst = cls_events / (src.stem + ".npz")
        if dst.exists() and dst.stat().st_size > 1024:
            success_files += 1
            continue
        try:
            CIFAR10DVS.read_aedat_save_to_np(str(src), str(dst))
            if dst.exists() and dst.stat().st_size > 1024:
                success_files += 1
            else:
                failed.append(str(src))
        except Exception as exc:
            failed.append(f"{src}: {exc}")

    return cls, total_files, success_files, failed


def extract_class_zips_if_needed(config: Dict[str, Any], workers: Optional[int] = None) -> None:
    raw_dir = Path(config["paths"]["raw_download_dir"])
    nested_dir = Path(config["paths"].get("raw_nested_download_dir", raw_dir / "CIFAR10DVS"))
    extract_dir = Path(config["paths"].get("extract_dir", Path(config["paths"]["dataset_root"]) / "extract"))
    ensure_dir(extract_dir)

    aedat_counts = count_aedat_by_class(extract_dir)
    if total_count(aedat_counts) >= 10000 and all(aedat_counts[c] >= 1000 for c in CLASSES):
        logger.info(f"Existing extracted .aedat files found: {aedat_counts}")
        return

    workers = get_class_workers(config) if workers is None else max(1, min(int(workers), len(CLASSES)))
    todo = [cls for cls in CLASSES if aedat_counts.get(cls, 0) < 1000]
    logger.info(
        f"Extracted .aedat cache is incomplete. Extracting {len(todo)} classes with {workers} class workers."
    )

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as executor:
        futures = {
            executor.submit(_extract_one_class, cls, raw_dir, nested_dir, extract_dir): cls
            for cls in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extract/classes"):
            cls = futures[fut]
            try:
                done_cls, n = fut.result()
                logger.info(f"Extracted {done_cls}: {n} .aedat files")
            except Exception as exc:
                raise RuntimeError(f"Failed to extract class {cls}: {exc}") from exc

    aedat_counts = count_aedat_by_class(extract_dir)
    logger.info(f"Extracted .aedat counts: {aedat_counts}")
    if total_count(aedat_counts) == 0:
        raise RuntimeError("No .aedat files found after extraction. Check zip contents or extension spelling.")



def manual_convert_events_if_needed(
    config: Dict[str, Any],
    force: bool = False,
    workers: Optional[int] = None,
) -> Dict[str, int]:
    """Convert extracted .aedat files to events_np .npz class folders.

    This version uses class-level parallelism: up to 10 workers process the 10
    CIFAR10-DVS classes concurrently. Each worker converts files within one
    class sequentially, which avoids creating 10,000 tiny tasks and is much more
    stable on shared servers.
    """
    apply_numpy_legacy_patch()
    extract_dir = Path(config["paths"].get("extract_dir", Path(config["paths"]["dataset_root"]) / "extract"))
    events_np_dir = Path(config["paths"].get("events_np_dir", Path(config["paths"]["dataset_root"]) / "events_np"))
    ensure_dir(events_np_dir)

    event_counts = count_npz_by_class(events_np_dir)
    if (not force) and total_count(event_counts) >= 10000 and all(event_counts[c] >= 1000 for c in CLASSES):
        logger.info(f"Existing events_np cache looks complete: {event_counts}")
        return event_counts

    if force and events_np_dir.exists():
        logger.warning(f"Force rebuilding event cache: removing {events_np_dir}")
        shutil.rmtree(events_np_dir)
        ensure_dir(events_np_dir)

    workers = get_class_workers(config) if workers is None else max(1, min(int(workers), len(CLASSES)))
    extract_class_zips_if_needed(config, workers=workers)

    event_counts = count_npz_by_class(events_np_dir)
    todo = [cls for cls in CLASSES if force or event_counts.get(cls, 0) < 1000]

    if not todo:
        logger.info(f"Existing events_np cache looks complete: {event_counts}")
        return event_counts

    logger.info(f"Converting events for {len(todo)} classes with {workers} class workers.")
    total_files = 0
    success_files = 0
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as executor:
        futures = {
            executor.submit(_convert_one_class, cls, extract_dir, events_np_dir): cls
            for cls in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="events_np/classes"):
            cls = futures[fut]
            try:
                done_cls, n_total, n_success, cls_failed = fut.result()
                total_files += n_total
                success_files += n_success
                failed.extend(cls_failed)
                logger.info(f"Converted {done_cls}: {n_success}/{n_total} event npz files")
            except Exception as exc:
                raise RuntimeError(f"Failed to convert class {cls}: {exc}") from exc

    event_counts = count_npz_by_class(events_np_dir)
    logger.info(f"Event conversion success: {success_files}/{total_files}")
    logger.info(f"events_np counts: {event_counts}")
    if failed:
        logger.warning(f"Failed conversions: {len(failed)}. First few: {failed[:5]}")
    if total_count(event_counts) == 0:
        raise RuntimeError(
            "Manual .aedat -> events_np conversion produced zero .npz files. "
            "This usually indicates SpikingJelly/NumPy/Python incompatibility or corrupted raw .aedat files."
        )
    return event_counts



def remove_incomplete_frame_cache(config: Dict[str, Any], force: bool = False) -> None:
    frame_dir = Path(config["paths"].get("frame_dir"))
    if not frame_dir.exists():
        return
    frame_counts = count_npz_by_class(frame_dir)
    n = total_count(frame_counts)
    if force or n == 0:
        logger.warning(f"Removing incomplete frame cache {frame_dir}; current npz count={n}, counts={frame_counts}")
        shutil.rmtree(frame_dir)
    else:
        logger.info(f"Existing frame cache count={n}, counts={frame_counts}")


def inspect_raw_layout(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config["paths"]
    raw_dir = Path(paths["raw_download_dir"])
    nested_dir = Path(paths.get("raw_nested_download_dir", raw_dir / "CIFAR10DVS"))
    found = {}
    missing: List[str] = []
    duplicated: List[str] = []
    for name in CLASS_ZIPS:
        locations = []
        for d in [raw_dir, nested_dir]:
            p = d / name
            if p.exists():
                locations.append(str(p))
        if not locations:
            missing.append(name)
        else:
            found[name] = locations
            if len(locations) > 1:
                duplicated.append(name)
    return {
        "raw_download_dir": str(raw_dir),
        "raw_nested_download_dir": str(nested_dir),
        "found_zip_locations": found,
        "missing_class_zips": missing,
        "duplicated_class_zips": duplicated,
    }


def build_dataset(config: Dict[str, Any]) -> CIFAR10DVS:
    ds_cfg = config["dataset"]
    return CIFAR10DVS(
        root=config["paths"]["dataset_root"],
        data_type=ds_cfg.get("data_type", "frame"),
        frames_number=int(ds_cfg["time_steps"]),
        split_by=ds_cfg.get("split_by", "number"),
    )


def maybe_reuse_split(config: Dict[str, Any]) -> Dict[str, Any] | None:
    split_cfg = config["split"]
    split_file = Path(split_cfg["split_file"])
    if split_cfg.get("reuse_if_exists", True) and split_file.exists():
        logger.info(f"Reusing existing split file: {split_file}")
        return load_split(split_file)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CIFAR10-DVS frames and deterministic splits.")
    parser.add_argument("--config", type=str, default="configs/config_cifar10dvs.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild event/frame caches and split.")
    parser.add_argument("--force-events", action="store_true", help="Rebuild events_np cache only.")
    parser.add_argument("--force-frames", action="store_true", help="Remove existing frame cache before frame integration.")
    parser.add_argument("--class-workers", type=int, default=None, help="Class-level workers for extract/event conversion. Default: preprocess.class_workers or 10.")
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config.get("experiment", {})
    set_seed(int(exp.get("seed", 42)), bool(exp.get("deterministic", True)), bool(exp.get("benchmark", False)))

    # Only create bookkeeping directories. Do NOT pre-create SpikingJelly frame cache.
    for key in ["split_dir", "manifest_dir"]:
        ensure_dir(config["paths"][key])

    logger.info("=== CIFAR10-DVS preprocessing ===")
    logger.info(f"Config: {args.config}")

    raw_report = inspect_raw_layout(config)
    if raw_report["missing_class_zips"]:
        logger.warning(f"Missing CIFAR10-DVS class zip files: {raw_report['missing_class_zips']}")
    if raw_report["duplicated_class_zips"]:
        logger.warning("Duplicated class zips detected. This is tolerated but should be cleaned for final reproducibility.")

    if args.class_workers is not None:
        config.setdefault("preprocess", {})["class_workers"] = int(args.class_workers)
    workers = get_class_workers(config)
    logger.info(f"Using {workers} class-level workers for extraction/event conversion.")

    # Step 1: make sure events_np contains actual .npz files.
    event_counts = manual_convert_events_if_needed(config, force=bool(args.force or args.force_events), workers=workers)

    # Step 2: remove empty frame dir before SpikingJelly builds frame cache.
    remove_incomplete_frame_cache(config, force=bool(args.force or args.force_frames))

    # Step 3: ask SpikingJelly to integrate frames and load ImageFolder-like dataset.
    dataset = build_dataset(config)
    frame_counts = count_npz_by_class(Path(config["paths"]["frame_dir"]))
    logger.info(f"Frame npz counts: {frame_counts}")

    num_samples = len(dataset)
    labels = get_dataset_labels(dataset, config["dataset"].get("class_names"))
    num_classes = int(config["dataset"]["num_classes"])
    counts = class_counts(labels, num_classes)

    logger.info(f"Dataset samples: {num_samples}")
    logger.info(f"Class counts: {counts}")

    expected = config.get("preprocess", {}).get("expected_num_samples")
    if expected is not None and int(expected) != int(num_samples):
        logger.warning(f"Expected {expected} samples, but found {num_samples}. Check raw/cache layout.")

    split = None if args.force else maybe_reuse_split(config)
    if split is None:
        split_cfg = config["split"]
        split = split_class_balanced(
            labels=labels,
            num_classes=num_classes,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            seed=int(split_cfg["seed"]),
        )
        split.update({
            "dataset": "cifar10dvs",
            "split_method": split_cfg.get("method", "class_balanced_random"),
            "seed": int(split_cfg["seed"]),
            "time_steps": int(config["dataset"]["time_steps"]),
            "num_classes": num_classes,
            "labels": [int(x) for x in labels],
            "class_names": config["dataset"].get("class_names", []),
        })
        save_split(split, split_cfg["split_file"])
        logger.info(f"Saved split file: {split_cfg['split_file']}")

    split_summary = summarize_split(labels, split, num_classes)
    logger.info(f"Split summary: {split_summary}")

    frame_dir = Path(config["paths"]["frame_dir"])
    manifest: Dict[str, Any] = {
        "dataset": "cifar10dvs",
        "config_path": args.config,
        "dataset_root": config["paths"]["dataset_root"],
        "time_steps": int(config["dataset"]["time_steps"]),
        "data_type": config["dataset"].get("data_type", "frame"),
        "split_by": config["dataset"].get("split_by", "number"),
        "num_samples": int(num_samples),
        "class_counts": counts,
        "split_file": config["split"]["split_file"],
        "split_summary": split_summary,
        "event_counts": event_counts,
        "frame_counts": frame_counts,
        "frame_dir": str(frame_dir),
        "frame_npz_count": int(total_count(frame_counts)),
        "raw_layout": raw_report,
    }

    zip_hashes = {}
    for name, locations in raw_report["found_zip_locations"].items():
        p = Path(locations[0])
        try:
            zip_hashes[name] = {"path": str(p), "size_bytes": p.stat().st_size, "sha1_first_1MB": sha1_file(p)}
        except Exception as exc:
            zip_hashes[name] = {"path": str(p), "error": str(exc)}
    manifest["raw_zip_hashes"] = zip_hashes

    manifest_file = Path(config["preprocess"]["manifest_file"])
    save_json(manifest, manifest_file)
    logger.info(f"Saved manifest: {manifest_file}")
    logger.info("=== CIFAR10-DVS preprocessing complete ===")


if __name__ == "__main__":
    main()
