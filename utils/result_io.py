"""
result_io.py

Shared I/O helpers for HIPSA evaluation and plotting scripts.

Design rule:
- eval scripts save raw data to results/eval_v2/<dataset>/<eval_xx>/
- plot scripts read saved data and write figures to plot/results/<eval_xx>/
- model inference, hardware modeling, and plotting should not duplicate JSON/CSV I/O logic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def ensure_dir(path: str | Path, is_file: bool = False) -> Path:
    """Create a directory.

    Args:
        path: Directory path, or file path when is_file=True.
        is_file: If True, create path.parent.

    Returns:
        Path object.
    """

    p = Path(path)
    target = p.parent if is_file else p
    target.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for common scientific Python objects."""

    if isinstance(obj, Path):
        return str(obj)

    # numpy support
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass

    # torch support
    try:
        import torch

        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(obj, set):
        return sorted(list(obj))

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def atomic_write_text(text: str, path: str | Path, encoding: str = "utf-8") -> None:
    """Atomically write text to a file."""

    path = ensure_dir(path, is_file=True)
    directory = path.parent

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        delete=False,
        dir=str(directory),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_text(text, path, encoding=encoding)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON as dictionary."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(obj).__name__}")

    return obj


def save_json(
    obj: Mapping[str, Any],
    path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    atomic: bool = True,
) -> None:
    """Save dictionary-like object to JSON."""

    text = json.dumps(
        obj,
        indent=indent,
        ensure_ascii=False,
        sort_keys=sort_keys,
        default=_json_default,
    )

    if atomic:
        atomic_write_text(text + "\n", path)
    else:
        path = ensure_dir(path, is_file=True)
        path.write_text(text + "\n", encoding="utf-8")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML as dictionary."""

    if yaml is None:
        raise ImportError("PyYAML is required to load YAML files.")

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if obj is None:
        return {}

    if not isinstance(obj, dict):
        raise TypeError(f"Expected YAML object in {path}, got {type(obj).__name__}")

    return obj


def save_yaml(obj: Mapping[str, Any], path: str | Path, *, atomic: bool = True) -> None:
    """Save dictionary-like object to YAML."""

    if yaml is None:
        raise ImportError("PyYAML is required to save YAML files.")

    text = yaml.safe_dump(
        dict(obj),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

    if atomic:
        atomic_write_text(text, path)
    else:
        path = ensure_dir(path, is_file=True)
        path.write_text(text, encoding="utf-8")


def save_csv_rows(
    rows: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Save rows to CSV.

    Field order:
    - Use provided fieldnames when given.
    - Otherwise preserve first-seen key order across rows.
    """

    rows = list(rows)
    path = ensure_dir(path, is_file=True)

    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row.keys():
                key = str(key)
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    else:
        fieldnames = [str(x) for x in fieldnames]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_scalar(value: str) -> Any:
    """Best-effort scalar parser for CSV values."""

    if value == "":
        return ""

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none" or lower == "null":
        return None

    try:
        if "." not in value and "e" not in lower:
            return int(value)
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return value


def load_csv_rows(path: str | Path, *, parse_numbers: bool = False) -> List[Dict[str, Any]]:
    """Load CSV rows.

    Args:
        parse_numbers: If True, parse ints/floats/bools where possible.
    """

    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if parse_numbers:
                rows.append({k: _parse_scalar(v) for k, v in row.items()})
            else:
                rows.append(dict(row))

    return rows


def append_jsonl(obj: Mapping[str, Any], path: str | Path) -> None:
    """Append one JSON object as a JSONL row."""

    path = ensure_dir(path, is_file=True)
    line = json.dumps(obj, ensure_ascii=False, default=_json_default)

    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSONL as list of dictionaries."""

    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected JSONL object in {path}")
            rows.append(obj)

    return rows


def copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy a file and create target directory."""

    src = Path(src)
    dst = ensure_dir(dst, is_file=True)
    shutil.copy2(src, dst)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file."""

    h = hashlib.sha256()
    path = Path(path)

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def write_sha256sums(root: str | Path, output: str | Path = "sha256sums.txt") -> Path:
    """Write SHA256 checksums for files under root.

    The checksum file itself is excluded.
    """

    root = Path(root)
    output = root / output if not Path(output).is_absolute() else Path(output)
    output_name = output.name

    rows: List[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == output_name:
            continue

        digest = sha256_file(path)
        rel = path.relative_to(root).as_posix()
        rows.append(f"{digest}  ./{rel}")

    write_text(output, "\n".join(rows) + "\n")
    return output


def save_run_manifest(
    output_dir: str | Path,
    *,
    eval_name: str,
    command: str,
    inputs: Optional[Mapping[str, Any]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    filename: str = "run_manifest.json",
) -> Path:
    """Save a lightweight run manifest for one eval stage."""

    output_dir = ensure_dir(output_dir)
    path = output_dir / filename

    obj: Dict[str, Any] = {
        "eval_name": eval_name,
        "command": command,
        "inputs": dict(inputs or {}),
        "outputs": dict(outputs or {}),
        "extra": dict(extra or {}),
    }

    save_json(obj, path)
    return path


def require_files(paths: Iterable[str | Path]) -> None:
    """Raise FileNotFoundError if any required file is missing."""

    missing = [str(Path(p)) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def dataset_eval_dir(
    dataset: str,
    eval_name: str,
    root: str | Path = "results/eval_v2",
) -> Path:
    """Return standard eval output/input directory."""

    return Path(root) / dataset / eval_name


def plot_result_dir(
    plot_name: str,
    dataset: Optional[str] = None,
    root: str | Path = "plot/results",
) -> Path:
    """Return standard plot result directory."""

    base = Path(root) / plot_name
    if dataset is not None:
        base = base / dataset
    base.mkdir(parents=True, exist_ok=True)
    return base