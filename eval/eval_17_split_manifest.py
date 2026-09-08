"""Create immutable calibration/test manifests without reading test samples for tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_indices(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    split_path = Path(args.split_file)
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    if not isinstance(split, dict):
        raise TypeError("Frozen split must be a dictionary.")
    partitions = {}
    ownership: dict[tuple[str, int], str] = {}
    overlap = []
    for name in ("train", "val", "test"):
        aliases = [name, f"{name}_indices"]
        if name == "val":
            aliases.extend(["valid", "validation", "valid_indices", "validation_indices"])
        raw = next((split[key] for key in aliases if key in split), None)
        if raw is None:
            raise KeyError(f"Missing {name} indices in frozen split.")
        indices = normalize_indices(raw)
        index_scope = "official_test_dataset" if args.dataset.lower() == "dvsgesture" and name == "test" else "training_dataset"
        for index in indices:
            scoped_index = (index_scope, index)
            if scoped_index in ownership:
                overlap.append({"scope": index_scope, "index": index, "first": ownership[scoped_index], "second": name})
            ownership[scoped_index] = name
        payload = "\n".join(map(str, indices)).encode()
        partitions[name] = {
            "count": len(indices),
            "indices": indices,
            "indices_sha256": hashlib.sha256(payload).hexdigest(),
            "role": "calibration_or_training" if name != "test" else "reporting_only_no_tuning",
            "index_scope": index_scope,
        }
    manifest = {
        "dataset": args.dataset,
        "split_file": str(split_path.resolve()),
        "split_file_sha256": sha256(split_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "partitions": partitions,
        "partition_overlap": overlap,
        "fairness_pass": not overlap,
        "policy": "ADC ranges, fixed-point formats, thresholds, architecture choices and seeds are frozen using train/val only; test is reporting-only.",
    }
    if overlap:
        raise SystemExit("Split overlap detected; refusing publication manifest.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
