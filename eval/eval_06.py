"""
eval_06.py

CPU/GPU software runtime baseline for HIPSA evaluation.

This script runs the same trained SNN workload on PyTorch CPU / CUDA backends
and measures per-image latency.

It does NOT compute HIPSA hardware latency.
It does NOT generate figures.

Outputs:
  results/eval_v2/<dataset>/eval_06/
    runtime_summary.json
    runtime_samples.csv
    platform_info.json
    run_manifest.json

Energy:
  If --cpu-active-power-w or --gpu-active-power-w is provided, this script also
  estimates software energy as:
    energy = active_power_w * latency_s
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from eval.eval_utils import build_eval_model, build_loader, load_eval_context
from utils.data_utils import (
    accuracy_from_logits,
    aggregate_time_logits,
    logits_aggregation_from_config,
    prepare_snn_batch,
    reset_snn_state,
)
from utils.result_io import save_csv_rows, save_json, save_run_manifest


def maybe_run_command(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def get_platform_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }

    if torch.cuda.is_available():
        devices = []
        for idx in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(prop.total_memory),
                    "major": int(prop.major),
                    "minor": int(prop.minor),
                    "multi_processor_count": int(prop.multi_processor_count),
                }
            )
        info["cuda_devices"] = devices
        info["nvidia_smi"] = maybe_run_command(["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"])
    else:
        info["cuda_devices"] = []

    info["lscpu"] = maybe_run_command(["bash", "-lc", "lscpu | head -40"])
    return info


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize_latencies(latencies_s: List[float]) -> Dict[str, Any]:
    if not latencies_s:
        return {
            "num_timed_batches": 0,
            "mean_latency_s": None,
            "median_latency_s": None,
            "std_latency_s": None,
            "p50_latency_s": None,
            "p90_latency_s": None,
            "p95_latency_s": None,
            "p99_latency_s": None,
            "min_latency_s": None,
            "max_latency_s": None,
            "throughput_images_per_s": None,
        }

    mean_s = statistics.mean(latencies_s)
    median_s = statistics.median(latencies_s)
    std_s = statistics.pstdev(latencies_s) if len(latencies_s) > 1 else 0.0

    return {
        "num_timed_batches": len(latencies_s),
        "mean_latency_s": mean_s,
        "mean_latency_ms": mean_s * 1000.0,
        "median_latency_s": median_s,
        "median_latency_ms": median_s * 1000.0,
        "std_latency_s": std_s,
        "std_latency_ms": std_s * 1000.0,
        "p50_latency_s": percentile(latencies_s, 50),
        "p50_latency_ms": percentile(latencies_s, 50) * 1000.0,
        "p90_latency_s": percentile(latencies_s, 90),
        "p90_latency_ms": percentile(latencies_s, 90) * 1000.0,
        "p95_latency_s": percentile(latencies_s, 95),
        "p95_latency_ms": percentile(latencies_s, 95) * 1000.0,
        "p99_latency_s": percentile(latencies_s, 99),
        "p99_latency_ms": percentile(latencies_s, 99) * 1000.0,
        "min_latency_s": min(latencies_s),
        "min_latency_ms": min(latencies_s) * 1000.0,
        "max_latency_s": max(latencies_s),
        "max_latency_ms": max(latencies_s) * 1000.0,
        "throughput_images_per_s": 1.0 / mean_s if mean_s > 0 else None,
    }


def device_available(device_name: str) -> bool:
    if device_name == "cpu":
        return True
    if device_name in {"cuda", "gpu"}:
        return torch.cuda.is_available()
    if device_name.startswith("cuda"):
        return torch.cuda.is_available()
    return False


def normalize_device_name(device_name: str) -> str:
    if device_name == "gpu":
        return "cuda"
    return device_name


@torch.inference_mode()
def benchmark_one_device(
    *,
    config_path: str,
    checkpoint: str,
    run_dir: Optional[str],
    split: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
    num_warmup: int,
    num_runs: int,
    max_batches: Optional[int],
    cpu_active_power_w: Optional[float],
    gpu_active_power_w: Optional[float],
    non_strict: bool,
    allow_no_split: bool,
) -> Dict[str, Any]:
    device_name = normalize_device_name(device_name)

    if not device_available(device_name):
        return {
            "device": device_name,
            "available": False,
            "skipped_reason": f"Device not available: {device_name}",
        }

    ctx = load_eval_context(
        config_path=config_path,
        checkpoint=checkpoint,
        run_dir=run_dir,
        output_root="results/eval_v2",
        eval_name="eval_06_tmp",
        device=device_name,
    )

    model = build_eval_model(ctx, strict=not non_strict)

    loader = build_loader(
        ctx.config,
        split_name=split,
        batch_size=batch_size,
        num_workers=num_workers,
        allow_no_split=allow_no_split,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(reduction="sum")
    agg_mode = logits_aggregation_from_config(ctx.config)

    timed_rows: List[Dict[str, Any]] = []
    latencies_s: List[float] = []

    total_seen = 0
    total_timed = 0
    correct = 0
    loss_sum = 0.0
    sample_index = 0

    target_total_batches = num_warmup + num_runs
    if max_batches is not None:
        target_total_batches = min(target_total_batches, max_batches)

    if device_name.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= target_total_batches:
            break

        reset_snn_state(model)
        data, target = prepare_snn_batch(data, target, ctx.config, ctx.device)

        if device_name.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        logits_t = model(data)

        if device_name.startswith("cuda"):
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        logits = aggregate_time_logits(logits_t, agg_mode)
        loss = criterion(logits, target)
        c, n = accuracy_from_logits(logits, target)

        total_seen += int(n)

        is_warmup = batch_idx < num_warmup
        elapsed_s = t1 - t0

        if not is_warmup:
            latencies_s.append(elapsed_s)
            total_timed += int(n)
            correct += int(c)
            loss_sum += float(loss.item())

            timed_rows.append(
                {
                    "device": device_name,
                    "batch_index": batch_idx,
                    "sample_index_start": sample_index,
                    "batch_size": int(n),
                    "latency_s": elapsed_s,
                    "latency_ms": elapsed_s * 1000.0,
                    "latency_ms_per_image": elapsed_s * 1000.0 / max(int(n), 1),
                    "correct": int(c),
                    "loss_sum": float(loss.item()),
                    "warmup": 0,
                }
            )
        else:
            timed_rows.append(
                {
                    "device": device_name,
                    "batch_index": batch_idx,
                    "sample_index_start": sample_index,
                    "batch_size": int(n),
                    "latency_s": elapsed_s,
                    "latency_ms": elapsed_s * 1000.0,
                    "latency_ms_per_image": elapsed_s * 1000.0 / max(int(n), 1),
                    "correct": int(c),
                    "loss_sum": float(loss.item()),
                    "warmup": 1,
                }
            )

        sample_index += int(n)

    summary = summarize_latencies(latencies_s)

    # If batch_size > 1, convert batch latency to per-image latency.
    # For Section 4, use batch_size=1.
    mean_latency_s_per_image = None
    if latencies_s and batch_size > 0:
        mean_latency_s_per_image = statistics.mean(
            [row["latency_s"] / max(int(row["batch_size"]), 1) for row in timed_rows if int(row["warmup"]) == 0]
        )

    if mean_latency_s_per_image is not None:
        summary["mean_latency_s_per_image"] = mean_latency_s_per_image
        summary["mean_latency_ms_per_image"] = mean_latency_s_per_image * 1000.0
        summary["throughput_images_per_s_per_image_latency"] = 1.0 / mean_latency_s_per_image if mean_latency_s_per_image > 0 else None

    power_w = None
    if device_name == "cpu":
        power_w = cpu_active_power_w
    elif device_name.startswith("cuda"):
        power_w = gpu_active_power_w

    if power_w is not None and mean_latency_s_per_image is not None:
        summary["active_power_w"] = float(power_w)
        summary["energy_J_per_image"] = float(power_w) * mean_latency_s_per_image
        summary["energy_mJ_per_image"] = float(power_w) * mean_latency_s_per_image * 1000.0
    else:
        summary["active_power_w"] = power_w
        summary["energy_J_per_image"] = None
        summary["energy_mJ_per_image"] = None

    acc = 100.0 * correct / max(total_timed, 1)
    avg_loss = loss_sum / max(total_timed, 1)

    return {
        "device": device_name,
        "available": True,
        "dataset": ctx.dataset,
        "num_warmup": num_warmup,
        "num_runs_requested": num_runs,
        "num_timed_batches": summary.get("num_timed_batches"),
        "batch_size": batch_size,
        "total_seen_samples": total_seen,
        "total_timed_samples": total_timed,
        "accuracy_percent_timed": acc,
        "loss_timed": avg_loss,
        "summary": summary,
        "rows": timed_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_06: CPU/GPU runtime baseline")

    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--run-dir", default=None, type=str)

    parser.add_argument("--output-root", default="results/eval_v2", type=str)
    parser.add_argument("--split", default="test", type=str)

    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--num-warmup", default=20, type=int)
    parser.add_argument("--num-runs", default=100, type=int)
    parser.add_argument("--max-batches", default=None, type=int)

    parser.add_argument("--cpu-active-power-w", default=None, type=float)
    parser.add_argument("--gpu-active-power-w", default=None, type=float)

    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--non-strict", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_root) / args.dataset / "eval_06"
    output_dir.mkdir(parents=True, exist_ok=True)

    platform_info = get_platform_info()
    save_json(platform_info, output_dir / "platform_info.json")

    all_device_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("[eval_06] CPU/GPU runtime baseline")
    print(f"dataset    : {args.dataset}")
    print(f"devices    : {args.devices}")
    print(f"batch_size : {args.batch_size}")
    print(f"warmup/runs: {args.num_warmup}/{args.num_runs}")
    print("=" * 80)

    for device_name in args.devices:
        print(f"[eval_06] benchmarking device={device_name}")

        result = benchmark_one_device(
            config_path=args.config,
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=device_name,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            max_batches=args.max_batches,
            cpu_active_power_w=args.cpu_active_power_w,
            gpu_active_power_w=args.gpu_active_power_w,
            non_strict=args.non_strict,
            allow_no_split=args.allow_no_split,
        )

        rows = result.pop("rows", [])
        all_rows.extend(rows)
        all_device_summaries.append(result)

        if result.get("available"):
            s = result["summary"]
            print(
                "  mean={:.3f} ms/image, p95={:.3f} ms, acc={:.2f}%".format(
                    float(s.get("mean_latency_ms_per_image") or s.get("mean_latency_ms") or 0.0),
                    float(s.get("p95_latency_ms") or 0.0),
                    float(result.get("accuracy_percent_timed") or 0.0),
                )
            )
        else:
            print(f"  skipped: {result.get('skipped_reason')}")

    summary = {
        "eval_name": "eval_06",
        "purpose": "cpu_gpu_software_runtime_baseline",
        "command": " ".join(sys.argv),
        "dataset": args.dataset,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "devices": args.devices,
        "cpu_active_power_w": args.cpu_active_power_w,
        "gpu_active_power_w": args.gpu_active_power_w,
        "device_results": all_device_summaries,
        "notes": {
            "latency": "Use batch_size=1 for Section 4 latency-oriented edge inference.",
            "energy": "Software energy is reported only when active power is provided.",
            "platform_warning": "Do not mix AutoDL server runtime with laptop CPU/GPU names in the paper.",
        },
    }

    save_json(summary, output_dir / "runtime_summary.json")
    save_csv_rows(all_rows, output_dir / "runtime_samples.csv")

    save_run_manifest(
        output_dir,
        eval_name="eval_06",
        command=" ".join(sys.argv),
        inputs={
            "config": args.config,
            "checkpoint": args.checkpoint,
        },
        outputs={
            "runtime_summary": "runtime_summary.json",
            "runtime_samples": "runtime_samples.csv",
            "platform_info": "platform_info.json",
        },
        extra={
            "dataset": args.dataset,
            "devices": args.devices,
        },
    )

    print("=" * 80)
    print("[eval_06] complete")
    print(f"output_dir: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()