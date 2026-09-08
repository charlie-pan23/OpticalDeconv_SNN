"""Fail-closed audit for externally adopted architecture-model parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED = {"parameter", "adopted_value", "adopted_unit", "source", "venue", "source_location", "compatibility", "mapping", "uncertainty", "evidence_class"}
FORBIDDEN_EXTERNAL_CLASSES = {"measured_HIPSA", "post_layout", "signoff", "silicon_validated"}


def load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping in {path}")
    return value


def audit(registry: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = registry.get("parameters", [])
    if not isinstance(rows, list) or not rows:
        return ["parameters must be a non-empty list"]
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"parameters[{index}] is not a mapping")
            continue
        missing = sorted(REQUIRED - set(row))
        if missing:
            errors.append(f"{row.get('parameter', index)} missing: {', '.join(missing)}")
        name = str(row.get("parameter", ""))
        if name in seen:
            errors.append(f"duplicate parameter: {name}")
        seen.add(name)
        if str(row.get("evidence_class")) == "external_reference":
            labels = set(row.get("labels", []))
            bad = sorted(labels & FORBIDDEN_EXTERNAL_CLASSES)
            if bad:
                errors.append(f"{name} external reference has forbidden labels: {bad}")
            uncertainty = row.get("uncertainty", {})
            if not isinstance(uncertainty, Mapping) or len(uncertainty) == 0:
                errors.append(f"{name} lacks an uncertainty/sensitivity declaration")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/reference_parameter_registry.yaml")
    args = parser.parse_args()
    errors = audit(load_yaml(Path(args.registry)))
    if errors:
        raise SystemExit("REFERENCE PARAMETER AUDIT FAILED\n- " + "\n- ".join(errors))
    print("REFERENCE PARAMETER AUDIT PASS")


if __name__ == "__main__":
    main()
