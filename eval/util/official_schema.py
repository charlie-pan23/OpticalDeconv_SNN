"""Shared helpers for the official eval_v10 result contract.

The official entries use this module to keep paths, configuration snapshots,
validation fields, and provenance consistent.  It deliberately does not move
or import legacy analysis code beyond the thin adapters in eval_101/eval_104.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(value: str | Path, default: str) -> Path:
    raw = Path(value if value is not None and str(value) else default)
    return raw if raw.is_absolute() else REPO_ROOT / raw


def option_value(argv: list[str], name: str, default: str) -> str:
    try:
        index = argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        return default
    return argv[index + 1]


def option_values(argv: list[str], name: str, default: Iterable[str]) -> list[str]:
    try:
        index = argv.index(name)
    except ValueError:
        return list(default)
    values: list[str] = []
    for item in argv[index + 1:]:
        if item.startswith("--"):
            break
        values.append(item)
    return values or list(default)


def relative_repo_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> Any:
    from utils.result_io import load_yaml
    return load_yaml(path)


def write_standard_artifacts(
    output_dir: str | Path,
    *,
    eval_name: str,
    metrics: Mapping[str, Any],
    validation: Mapping[str, Any],
    config_paths: Mapping[str, str | Path],
    evidence_class: str,
    extra_provenance: Mapping[str, Any] | None = None,
) -> None:
    """Write the common official artifact set without machine-specific defaults."""
    from utils.result_io import save_json, save_yaml

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, Any] = {}
    source_files: dict[str, str] = {}
    source_hashes: dict[str, str | None] = {}
    for key, raw_path in config_paths.items():
        path = Path(raw_path)
        source_files[key] = relative_repo_path(path)
        source_hashes[key] = _sha256(path)
        if path.exists():
            try:
                snapshots[key] = _load_yaml(path)
            except Exception as exc:  # pragma: no cover - defensive boundary
                snapshots[key] = {"unreadable": f"{type(exc).__name__}: {exc}"}

    save_yaml(
        {
            "eval_name": eval_name,
            "evidence_class": evidence_class,
            "source_files": source_files,
            "snapshots": snapshots,
        },
        out / "config_snapshot.yaml",
    )
    save_json(dict(metrics), out / "metrics.json")
    save_json(dict(validation), out / "validation.json")

    provenance: dict[str, Any] = {
        "eval_name": eval_name,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "evidence_class": evidence_class,
        "command": " ".join(sys.argv),
        "source_files": source_files,
        "source_sha256": source_hashes,
    }
    if extra_provenance:
        provenance.update(dict(extra_provenance))
    save_json(provenance, out / "runtime_provenance.json")


def matched_accounting_contract(
    *,
    lock_fraction: float,
    tile_load_cycles: int,
    precision: str = "W6/ADC6/Vm16",
) -> dict[str, Any]:
    """Return the scope that every architecture variant must share.

    This is a comparison contract, not a claim that a CPU/GPU or silicon
    baseline has been measured.  Numeric variant rows remain the source of
    architecture-level estimates; this object prevents scope drift.
    """
    return {
        "baseline_id": "matched_four_tile_w6_adc6_vm16_lock_and_configuration_accounting",
        "comparison_scope": "architecture_level_system_accounting",
        "same_four_photonic_tiles": True,
        "same_array_shape": "64x64_per_tile",
        "same_laser_and_link_budget": True,
        "same_precision": precision,
        "same_timestep_count": True,
        "same_input_activity_trace": True,
        "same_static_power_terms": True,
        "same_non_mvm_accounting": True,
        "same_mrr_lock_fraction": float(lock_fraction),
        "same_tile_load_cycles_per_tile": int(tile_load_cycles),
        "same_configuration_protocol": "layer_weight_tile_boundary; amortized across timesteps",
        "accuracy_policy": "only dedicated eval_105 replay may provide accuracy; architecture counters never infer variant accuracy",
        "external_software_speedup_comparison": False,
        "evidence_class": "architecture_level_matched_scope_contract",
    }
