"""Robust DVS Gesture preprocessing for HIPSA.

Version v6: matches the current SpikingJelly converter semantics.
DVS128Gesture.create_events_np_files(extract_root, events_np_root) internally
looks for extract_root/DvsGesture, so the staging directory is now built as
_spikingjelly_flat_extracted/DvsGesture and the parent directory is passed to
the converter.

Fixes common SpikingJelly compatibility issues:
1) Some DVS Gesture archives are extracted as DvsGesture/userXX/*.aedat, while
   SpikingJelly's create_events_np_files expects .aedat/.csv files directly under
   the extracted root. This script creates a flat staging directory with symlinks.
2) SpikingJelly creates events_np/train using os.mkdir, so the events_np parent
   directory must exist before calling create_events_np_files.

Run:
  python train/pre_dvsgesture.py --config configs/config_dvsgesture.yaml --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    from train.train_utils import logger


RESOURCE_FILES = ["DvsGesture.tar.gz", "gesture_mapping.csv", "Gesture_mapping.csv", "LICENSE.txt", "README.txt", "errata.txt"]
SPLIT_FILES = ["trials_to_train.txt", "trials_to_test.txt"]


def find_existing_extracted_root(config: Dict[str, Any]) -> Path:
    """Find the real DVS Gesture extracted directory.

    Different archives/users may produce one of:
      datasets/DVSGesture/DvsGesture
      datasets/DVSGesture/DVSGesture
      datasets/DVSGesture/download/DvsGesture

    We choose a directory that contains trials_to_train/test and has .aedat files
    either directly or recursively under userXX subdirectories.
    """
    dataset_root = Path(config["paths"]["dataset_root"])
    configured = Path(config["paths"].get("extracted_dir", dataset_root / "DvsGesture"))
    candidates = [
        configured,
        dataset_root / "DvsGesture",
        dataset_root / "DVSGesture",
        dataset_root / "download" / "DvsGesture",
        dataset_root / "download" / "DVSGesture",
    ]

    # Also scan one level below dataset_root for any directory containing the split files.
    if dataset_root.exists():
        for p in dataset_root.iterdir():
            if p.is_dir() and p not in candidates:
                candidates.append(p)

    scored = []
    for cand in candidates:
        if not cand.exists() or not cand.is_dir():
            continue
        has_split = all((cand / name).exists() for name in SPLIT_FILES)
        aedat_count = len(list(cand.rglob("*.aedat")))
        csv_count = len(list(cand.rglob("*.csv")))
        scored.append((has_split, aedat_count, csv_count, cand))

    # Prefer dirs with split files and aedat files.
    for has_split, aedat_count, csv_count, cand in sorted(scored, key=lambda x: (not x[0], -x[1])):
        if has_split and aedat_count > 0:
            logger.info(f"Resolved DVS Gesture extracted root: {cand} (aedat={aedat_count}, csv={csv_count})")
            return cand

    # Fallback: any dir with aedat files, but this may later fail due missing split files.
    for has_split, aedat_count, csv_count, cand in sorted(scored, key=lambda x: -x[1]):
        if aedat_count > 0:
            logger.warning(f"Using extracted root without confirmed split files: {cand} (aedat={aedat_count}, csv={csv_count})")
            return cand

    details = [{"path": str(c), "exists": c.exists()} for c in candidates]
    raise FileNotFoundError(f"Could not locate extracted DVS Gesture root. Checked: {details}")


def safe_remove(path: Path, reason: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    logger.warning(f"Removing {reason}: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def count_npz(root: Path) -> int:
    return len(list(root.rglob("*.npz"))) if root.exists() else 0


def inspect_tree(root: Path) -> Dict[str, Any]:
    return {
        "root": str(root),
        "exists": root.exists(),
        "aedat_count_recursive": len(list(root.rglob("*.aedat"))) if root.exists() else 0,
        "csv_count_recursive": len(list(root.rglob("*.csv"))) if root.exists() else 0,
        "npz_count_recursive": len(list(root.rglob("*.npz"))) if root.exists() else 0,
        "top_level_files": sorted([p.name for p in root.iterdir() if p.is_file()])[:30] if root.exists() else [],
        "top_level_dirs": sorted([p.name for p in root.iterdir() if p.is_dir()])[:30] if root.exists() else [],
    }


def find_resource(dataset_root: Path, extracted_dir: Path, name: str) -> Path | None:
    """Find a resource under common DVS Gesture layouts, case-insensitively."""
    candidates = [dataset_root / name, extracted_dir / name, dataset_root / "download" / name]
    for p in candidates:
        if p.exists():
            return p

    # Case-insensitive direct search first.
    lname = name.lower()
    for base in [extracted_dir, dataset_root, dataset_root / "download"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.name.lower() == lname:
                return p
    return None

def prepare_download_dir(config: Dict[str, Any]) -> None:
    """Put or link required raw resource files under dataset_root/download.

    Some SpikingJelly versions check this directory before preprocessing. Missing
    README/LICENSE/mapping is not fatal if we can build events_np from extracted data,
    but linking existing files reduces needless warnings and avoids download checks.
    """
    dataset_root = Path(config["paths"]["dataset_root"])
    extracted_dir = find_existing_extracted_root(config)
    download_dir = dataset_root / "download"
    download_dir.mkdir(parents=True, exist_ok=True)

    for name in RESOURCE_FILES:
        src = find_resource(dataset_root, extracted_dir, name)
        if src is None:
            logger.warning(f"Required resource not found locally: {name}")
            continue
        dst = download_dir / name
        if not dst.exists() and not dst.is_symlink():
            link_or_copy(src, dst)
            logger.info(f"Linked/copied {name} -> {dst}")


def prepare_flat_extracted_dir(config: Dict[str, Any], force: bool = False) -> Path:
    """Create a SpikingJelly-compatible extracted root.

    IMPORTANT for the installed SpikingJelly version:
        DVS128Gesture.create_events_np_files(extract_root, events_np_root)
        internally does:
            aedat_dir = os.path.join(extract_root, 'DvsGesture')

    Therefore this function must return the PARENT directory that contains a
    child directory named exactly ``DvsGesture``. All .aedat/.csv/split files
    are placed flat inside that child directory.

    If the user's extracted directory is already:
        <parent>/DvsGesture/*.aedat
    then return <parent>. Otherwise, create:
        datasets/DVSGesture/_spikingjelly_flat_extracted/DvsGesture/*.aedat
    and return:
        datasets/DVSGesture/_spikingjelly_flat_extracted
    """
    dataset_root = Path(config["paths"]["dataset_root"])
    extracted_dir = find_existing_extracted_root(config)

    # Case 1: original extracted layout is already exactly what the converter expects:
    # parent/DvsGesture contains top-level .aedat and label .csv files.
    direct_aedat = list(extracted_dir.glob("*.aedat"))
    direct_label_csv = [p for p in extracted_dir.glob("*.csv") if "mapping" not in p.name.lower()]
    if extracted_dir.name == "DvsGesture" and direct_aedat and direct_label_csv:
        logger.info(f"Using original SpikingJelly-compatible extracted root directly: {extracted_dir.parent}")
        return extracted_dir.parent

    # Case 2: common user layout is DvsGesture/user01/*.aedat. Build a staging parent
    # whose child DvsGesture is flat.
    staging_parent = dataset_root / "_spikingjelly_flat_extracted"
    staging_aedat_dir = staging_parent / "DvsGesture"

    if force and staging_parent.exists():
        safe_remove(staging_parent, "old flat DVS Gesture staging directory")
    staging_aedat_dir.mkdir(parents=True, exist_ok=True)

    # Link split files into staging_parent/DvsGesture.
    for name in SPLIT_FILES:
        src = find_resource(dataset_root, extracted_dir, name)
        if src is None:
            raise FileNotFoundError(
                f"Required split file {name} was not found. extracted_root={extracted_dir}, dataset_root={dataset_root}"
            )
        link_or_copy(src, staging_aedat_dir / name)

    # Link optional resource files. The converter only needs split files, .aedat, and *_labels.csv,
    # but these files make the staging directory closer to the official archive layout.
    resource_name_map = {
        "gesture_mapping.csv": ["gesture_mapping.csv", "Gesture_mapping.csv"],
        "LICENSE.txt": ["LICENSE.txt"],
        "README.txt": ["README.txt"],
        "errata.txt": ["errata.txt"],
    }
    for dst_name, possible_names in resource_name_map.items():
        for src_name in possible_names:
            src = find_resource(dataset_root, extracted_dir, src_name)
            if src is not None:
                link_or_copy(src, staging_aedat_dir / dst_name)
                break

    # Link all aedat files and csv label files from nested user directories into the flat
    # DvsGesture staging child. File basenames must remain unchanged because trials_to_train.txt
    # contains names like user01_fluorescent.aedat and the converter then searches for
    # user01_fluorescent_labels.csv.
    linked = 0
    duplicates: List[str] = []
    for src in sorted(extracted_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".aedat", ".csv"}:
            continue
        if src.name.lower() in {"gesture_mapping.csv"}:
            continue
        if src.name in SPLIT_FILES:
            continue
        dst = staging_aedat_dir / src.name
        if dst.exists() or dst.is_symlink():
            # Duplicate basenames should not happen for official DVS Gesture trials.
            # If it does, keep a renamed copy for diagnostics but warn because the converter
            # will only use the exact basename listed in trials_to_train/test.
            rel = src.relative_to(extracted_dir)
            dst = staging_aedat_dir / ("__".join(rel.parts))
            duplicates.append(src.name)
        link_or_copy(src, dst)
        linked += 1

    parent_info = inspect_tree(staging_parent)
    child_info = inspect_tree(staging_aedat_dir)
    logger.info(f"Prepared SpikingJelly staging parent: {parent_info}")
    logger.info(f"Prepared SpikingJelly staging DvsGesture child: {child_info}")

    missing_split = [name for name in SPLIT_FILES if not (staging_aedat_dir / name).exists()]
    if missing_split:
        raise FileNotFoundError(
            f"Staging DvsGesture dir is missing required split files: {missing_split}. dir={staging_aedat_dir}"
        )
    direct_aedat_count = len(list(staging_aedat_dir.glob("*.aedat")))
    if direct_aedat_count == 0:
        raise RuntimeError(
            f"Staging DvsGesture dir contains no top-level .aedat files. dir={staging_aedat_dir}"
        )
    if duplicates:
        logger.warning(f"Duplicate basenames found while flattening. Count={len(duplicates)}. First few={duplicates[:5]}")

    # Return the parent because SpikingJelly will append /DvsGesture internally.
    return staging_parent

def validate_events_np(events_np_dir: Path) -> Tuple[int, Dict[str, Any]]:
    train_dir = events_np_dir / "train"
    test_dir = events_np_dir / "test"
    train_npz = count_npz(train_dir)
    test_npz = count_npz(test_dir)
    summary = {
        "events_np_dir": str(events_np_dir),
        "train_dir_exists": train_dir.exists(),
        "test_dir_exists": test_dir.exists(),
        "train_npz": train_npz,
        "test_npz": test_npz,
        "total_npz": train_npz + test_npz,
        "class_dirs_train": sorted([p.name for p in train_dir.iterdir() if p.is_dir()]) if train_dir.exists() else [],
        "class_dirs_test": sorted([p.name for p in test_dir.iterdir() if p.is_dir()]) if test_dir.exists() else [],
    }
    return train_npz + test_npz, summary


def ensure_events_np_from_extracted(config: Dict[str, Any], force: bool = False) -> None:
    dataset_root = Path(config["paths"]["dataset_root"])
    events_np_dir = Path(config["paths"]["events_np_dir"])

    existing_count, existing_summary = validate_events_np(events_np_dir)
    if existing_count > 0 and not force:
        logger.info(f"Using existing DVS Gesture events_np cache: {existing_summary}")
        return

    if events_np_dir.exists():
        safe_remove(events_np_dir, "empty/incomplete DVS Gesture events_np cache")
    events_np_dir.mkdir(parents=True, exist_ok=True)

    prepare_download_dir(config)
    flat_or_extract_root = prepare_flat_extracted_dir(config, force=force)

    logger.info(f"Creating DVS Gesture events_np from: {flat_or_extract_root}")
    logger.info(f"Saving DVS Gesture events_np to: {events_np_dir}")
    DVS128Gesture.create_events_np_files(str(flat_or_extract_root), str(events_np_dir))

    new_count, new_summary = validate_events_np(events_np_dir)
    logger.info(f"DVS Gesture events_np summary after conversion: {new_summary}")
    if new_count <= 0:
        staging_info = inspect_tree(flat_or_extract_root)
        raise RuntimeError(
            "SpikingJelly converter finished but produced 0 .npz samples.\n"
            f"events_np summary: {new_summary}\n"
            f"staging/extract summary: {staging_info}\n"
            "This usually means the converter did not see .aedat files at the expected root, "
            "or the label csv filenames do not match the aedat files."
        )


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
    split_cfg = config["split"]
    split_file = Path(split_cfg["split_file"])
    if split_cfg.get("reuse_if_exists", True) and split_file.exists():
        logger.info(f"Reusing existing split file: {split_file}")
        return load_split(split_file)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DVS Gesture frames and deterministic validation split.")
    parser.add_argument("--config", type=str, default="configs/config_dvsgesture.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild events/frame/split caches.")
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config.get("experiment", {})
    set_seed(int(exp.get("seed", 42)), bool(exp.get("deterministic", True)), bool(exp.get("benchmark", False)))

    dataset_root = Path(config["paths"]["dataset_root"])
    frame_dir = Path(config["paths"]["frame_dir"])
    split_dir = Path(config["paths"]["split_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])

    ensure_dir(split_dir)
    ensure_dir(manifest_dir)

    logger.info("=== DVS Gesture preprocessing ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset root: {dataset_root}")

    if args.force and frame_dir.exists():
        safe_remove(frame_dir, "existing DVS Gesture frame cache")

    if config.get("preprocess", {}).get("allow_numpy_legacy_patch", False):
        import numpy as np
        np.bool = bool  # type: ignore[attr-defined]
        np.object = object  # type: ignore[attr-defined]
        logger.warning("Applied NumPy legacy monkey patch because allow_numpy_legacy_patch=true.")

    ensure_events_np_from_extracted(config, force=args.force)

    # Build frame datasets. SpikingJelly will integrate existing events_np into frames if needed.
    train_full = build_dataset(config, train=True)
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

    def count_subset(labels, indices):
        return class_counts([labels[i] for i in indices], num_classes)

    split_summary = {
        "train": {"num_samples": len(split["train_indices"]), "class_counts": count_subset(train_labels, split["train_indices"])},
        "val": {"num_samples": len(split["val_indices"]), "class_counts": count_subset(train_labels, split["val_indices"])},
        "test": {"num_samples": len(split["test_indices"]), "class_counts": count_subset(test_labels, split["test_indices"])},
    }
    logger.info(f"Split summary: {split_summary}")

    _, events_summary = validate_events_np(Path(config["paths"]["events_np_dir"]))
    manifest = {
        "dataset": "dvsgesture",
        "config_path": args.config,
        "dataset_root": str(dataset_root),
        "time_steps": int(config["dataset"]["time_steps"]),
        "data_type": config["dataset"].get("data_type", "frame"),
        "split_by": config["dataset"].get("split_by", "number"),
        "official_train_samples": int(len(train_full)),
        "official_test_samples": int(len(test_full)),
        "train_class_counts": class_counts(train_labels, num_classes),
        "test_class_counts": class_counts(test_labels, num_classes),
        "split_file": config["split"]["split_file"],
        "split_summary": split_summary,
        "events_np_summary": events_summary,
        "frame_dir": str(frame_dir),
        "frame_npz_count": int(count_npz(frame_dir)),
        "extracted_tree": inspect_tree(find_existing_extracted_root(config)),
    }
    manifest_file = Path(config["preprocess"]["manifest_file"])
    save_json(manifest, manifest_file)
    logger.info(f"Saved manifest: {manifest_file}")
    logger.info("=== DVS Gesture preprocessing complete ===")


if __name__ == "__main__":
    main()
