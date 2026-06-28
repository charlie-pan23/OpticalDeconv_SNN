"""Robust DVS Gesture preprocessing for HIPSA.

This version fixes two common SpikingJelly layout problems:
1) It does NOT create frames_number_10_split_by_number before SpikingJelly.
2) If root/download is incomplete but root/DvsGesture already exists, it directly
   creates root/events_np from the extracted AEDAT/CSV files, similar to the
   CIFAR10-DVS fix_dataset.py workflow.

Run from project root:
  python train/pre_dvsgesture.py --config configs/config_dvsgesture.yaml --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import spikingjelly.datasets.dvs128_gesture.DVS128Gesture. "
        "Please check your SpikingJelly installation."
    ) from exc

from train.train_utils import (
    class_counts,
    ensure_dir,
    get_dataset_labels,
    load_config,
    load_split,
    save_json,
    save_split,
    set_seed,
    split_train_val_class_balanced,
)

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    try:
        from utils.Logger import logger
    except Exception:
        from train.train_utils import logger


REQUIRED_DOWNLOAD_FILES = ["DvsGesture.tar.gz", "gesture_mapping.csv", "LICENSE.txt", "README.txt"]


def count_npz(root: Path) -> int:
    return len(list(root.rglob("*.npz"))) if root.exists() else 0


def is_nonempty_events_np(events_np_dir: Path) -> bool:
    return (events_np_dir / "train").is_dir() and (events_np_dir / "test").is_dir() and count_npz(events_np_dir) > 0


def is_nonempty_frame_cache(frame_dir: Path) -> bool:
    return (frame_dir / "train").is_dir() and (frame_dir / "test").is_dir() and count_npz(frame_dir) > 0


def safe_remove(path: Path, reason: str) -> None:
    if path.exists():
        logger.warning(f"Removing {reason}: {path}")
        shutil.rmtree(path)


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
        logger.info(f"Symlinked {dst} -> {src}")
    except Exception:
        shutil.copy2(src, dst)
        logger.info(f"Copied {src} -> {dst}")


def normalize_download_layout(config: Dict[str, Any]) -> Dict[str, Any]:
    """Put known root-level resource files into root/download when available.

    Current SpikingJelly versions expect manual resources under root/download.
    This function does not fabricate missing files; it only links/copies files that
    already exist somewhere in the user's dataset root.
    """
    root = Path(config["paths"]["dataset_root"])
    download = root / "download"
    download.mkdir(parents=True, exist_ok=True)

    candidates = [root, Path(config["paths"].get("extracted_dir", root / "DvsGesture")), root / "DvsGesture"]
    report = {"download_dir": str(download), "linked_or_existing": {}, "missing": []}

    for name in REQUIRED_DOWNLOAD_FILES:
        dst = download / name
        if dst.exists():
            report["linked_or_existing"][name] = str(dst)
            continue

        found = None
        for base in candidates:
            p = base / name
            if p.exists():
                found = p
                break
        if found is not None:
            symlink_or_copy(found, dst)
            report["linked_or_existing"][name] = str(dst)
        else:
            report["missing"].append(name)

    return report


def ensure_events_np_from_extracted(config: Dict[str, Any], force: bool = False) -> None:
    """Create root/events_np/train,test from an already extracted DvsGesture folder.

    This bypasses SpikingJelly's manual-download resource check when the extracted
    AEDAT/CSV files already exist locally.
    """
    root = Path(config["paths"]["dataset_root"])
    events_np_dir = Path(config["paths"].get("events_np_dir", root / "events_np"))
    extracted_dir = Path(config["paths"].get("extracted_dir", root / "DvsGesture"))

    if force and events_np_dir.exists() and not is_nonempty_events_np(events_np_dir):
        safe_remove(events_np_dir, "empty/incomplete DVS Gesture events_np cache")

    if is_nonempty_events_np(events_np_dir):
        logger.info(f"Existing DVS Gesture events_np cache is valid: {events_np_dir} ({count_npz(events_np_dir)} npz files)")
        return

    trials_train = extracted_dir / "trials_to_train.txt"
    trials_test = extracted_dir / "trials_to_test.txt"
    if not extracted_dir.exists():
        logger.warning(f"Extracted DVS Gesture directory not found: {extracted_dir}")
        return
    if not trials_train.exists() or not trials_test.exists():
        logger.warning(
            f"Extracted DVS Gesture exists but trials_to_train/test.txt are missing under {extracted_dir}. "
            "SpikingJelly official split cannot be created from this folder."
        )
        return

    # Latest SpikingJelly expects extract_root/DvsGesture. Since extracted_dir is
    # normally root/DvsGesture, extract_root should be root.
    extract_root = extracted_dir.parent

    if events_np_dir.exists():
        safe_remove(events_np_dir, "incomplete DVS Gesture events_np cache before rebuilding")

    # IMPORTANT:
    # SpikingJelly's DVS128Gesture.create_events_np_files() uses os.mkdir()
    # to create events_np/train and events_np/test. os.mkdir() does not create
    # missing parent directories, so the events_np root itself must already exist.
    # Creating only this parent directory is safe; we still do NOT pre-create
    # frames_number_10_split_by_number, because that would make SpikingJelly
    # falsely assume the frame cache is complete.
    events_np_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating DVS Gesture events_np from extracted data: extract_root={extract_root}, events_np={events_np_dir}")
    if hasattr(DVS128Gesture, "create_raw_from_extracted"):
        DVS128Gesture.create_raw_from_extracted(extract_root, events_np_dir)
    elif hasattr(DVS128Gesture, "create_events_np_files"):
        DVS128Gesture.create_events_np_files(extract_root, events_np_dir)
    else:
        raise RuntimeError("This SpikingJelly DVS128Gesture class has neither create_raw_from_extracted nor create_events_np_files.")

    if not is_nonempty_events_np(events_np_dir):
        raise RuntimeError(f"Failed to create valid events_np cache at {events_np_dir}")
    logger.info(f"DVS Gesture events_np created: {events_np_dir} ({count_npz(events_np_dir)} npz files)")


def build_dataset(config: Dict[str, Any], train: bool) -> DVS128Gesture:
    ds_cfg = config["dataset"]
    return DVS128Gesture(
        root=config["paths"]["dataset_root"],
        train=train,
        data_type=ds_cfg.get("data_type", "frame"),
        frames_number=int(ds_cfg["time_steps"]),
        split_by=ds_cfg.get("split_by", "number"),
    )


def maybe_reuse_split(config: Dict[str, Any]) -> Dict[str, Any] | None:
    split_file = Path(config["split"]["split_file"])
    if config["split"].get("reuse_if_exists", True) and split_file.exists():
        logger.info(f"Reusing existing split file: {split_file}")
        return load_split(split_file)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DVS Gesture frames and deterministic validation split.")
    parser.add_argument("--config", type=str, default="configs/config_dvsgesture.yaml")
    parser.add_argument("--force", action="store_true", help="Remove incomplete caches and rebuild split/cache.")
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config.get("experiment", {})
    set_seed(int(exp.get("seed", 42)), bool(exp.get("deterministic", True)), bool(exp.get("benchmark", False)))

    root = Path(config["paths"]["dataset_root"])
    frame_dir = Path(config["paths"].get("frame_dir", root / "frames_number_10_split_by_number"))
    events_np_dir = Path(config["paths"].get("events_np_dir", root / "events_np"))

    logger.info("=== DVS Gesture preprocessing v3 ===")
    logger.info(f"Config: {args.config}")

    # Only create folders that are controlled by our pipeline. Do not create
    # frame_dir in advance, because SpikingJelly uses its existence as a cache signal.
    ensure_dir(config["paths"]["split_dir"])
    ensure_dir(config["paths"]["manifest_dir"])

    download_report = normalize_download_layout(config)
    if download_report["missing"]:
        logger.warning(
            "Missing manual-download resources under root/download: "
            f"{download_report['missing']}. If extracted data is complete, the script will try events_np-from-extracted fallback."
        )

    if args.force and frame_dir.exists():
        safe_remove(frame_dir, "existing DVS Gesture frame cache")

    # If the user already has extracted root/DvsGesture, create events_np directly.
    # This is the DVS equivalent of the previous CIFAR10-DVS fix_dataset idea.
    ensure_events_np_from_extracted(config, force=args.force)

    logger.info("Instantiating DVS128Gesture(train=True) to build/load frame cache.")
    train_full = build_dataset(config, train=True)
    logger.info("Instantiating DVS128Gesture(train=False) to build/load frame cache.")
    test_full = build_dataset(config, train=False)

    num_classes = int(config["dataset"]["num_classes"])
    train_labels = get_dataset_labels(train_full, config["dataset"].get("class_names"))
    test_labels = get_dataset_labels(test_full, config["dataset"].get("class_names"))

    logger.info(f"Official train samples: {len(train_full)}")
    logger.info(f"Official test samples:  {len(test_full)}")
    logger.info(f"Train class counts: {class_counts(train_labels, num_classes)}")
    logger.info(f"Test class counts:  {class_counts(test_labels, num_classes)}")

    split = None if args.force else maybe_reuse_split(config)
    if split is None:
        split_cfg = config["split"]
        train_val = split_train_val_class_balanced(
            labels=train_labels,
            num_classes=num_classes,
            val_ratio=float(split_cfg["val_ratio_from_train"]),
            seed=int(split_cfg["seed"]),
        )
        split = {
            "dataset": "dvsgesture",
            "split_method": "official_train_test_with_val_from_train",
            "seed": int(split_cfg["seed"]),
            "time_steps": int(config["dataset"]["time_steps"]),
            "num_classes": num_classes,
            "train_indices": train_val["train_indices"],
            "val_indices": train_val["val_indices"],
            "test_indices": list(range(len(test_full))),
            "train_labels": [int(x) for x in train_labels],
            "test_labels": [int(x) for x in test_labels],
            "class_names": config["dataset"].get("class_names", []),
        }
        save_split(split, split_cfg["split_file"])
        logger.info(f"Saved split file: {split_cfg['split_file']}")

    def count_subset(labels: List[int], indices: List[int]) -> Dict[str, int]:
        return class_counts([labels[i] for i in indices], num_classes)

    split_summary = {
        "train": {"num_samples": len(split["train_indices"]), "class_counts": count_subset(train_labels, split["train_indices"])},
        "val": {"num_samples": len(split["val_indices"]), "class_counts": count_subset(train_labels, split["val_indices"])},
        "test": {"num_samples": len(split["test_indices"]), "class_counts": count_subset(test_labels, split["test_indices"])},
    }

    manifest = {
        "dataset": "dvsgesture",
        "config_path": args.config,
        "dataset_root": str(root),
        "time_steps": int(config["dataset"]["time_steps"]),
        "data_type": config["dataset"].get("data_type", "frame"),
        "split_by": config["dataset"].get("split_by", "number"),
        "official_train_samples": int(len(train_full)),
        "official_test_samples": int(len(test_full)),
        "train_class_counts": class_counts(train_labels, num_classes),
        "test_class_counts": class_counts(test_labels, num_classes),
        "split_file": config["split"]["split_file"],
        "split_summary": split_summary,
        "events_np_dir": str(events_np_dir),
        "events_np_count": int(count_npz(events_np_dir)),
        "frame_dir": str(frame_dir),
        "frame_npz_count": int(count_npz(frame_dir)),
        "download_layout": download_report,
    }
    save_json(manifest, config["preprocess"]["manifest_file"])
    logger.info(f"Saved manifest: {config['preprocess']['manifest_file']}")
    logger.info(f"Split summary: {split_summary}")
    logger.info("=== DVS Gesture preprocessing complete ===")


if __name__ == "__main__":
    main()
