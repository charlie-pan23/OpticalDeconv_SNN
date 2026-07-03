"""
eval_06.py

Final CPU/CUDA software baseline for HIPSA evaluation.

This version does not force batch size = 1 as the only software baseline.
Instead, it performs a compact, paper-ready batch-size sweep for each software
device and reports:

1. batch-1 latency:
   Single-image PyTorch forward latency. Useful as a latency sanity check.

2. selected throughput-equivalent per-image latency:
   The best effective per-image latency over a compact batch sweep:

       effective_latency = total_timed_forward_time / total_timed_samples

   This is the recommended main software baseline for comparison against
   high-throughput accelerators, because GPUs require batching to expose their
   natural parallelism.

Default sweep:
    batch sizes = 1, 8, 32

Default run budget:
    Each batch point runs enough timed iterations to cover roughly
    --target-timed-samples images, capped by --num-runs. This keeps the benchmark
    much faster than running the same number of iterations for every batch size.

Energy model:

    active_power_w = runtime_average_power_w - idle_average_power_w
    energy_J_per_image = active_power_w * effective_latency_s_per_image

Alternatively, pass active power directly with --cpu-active-power-w or
--gpu-active-power-w.

Timed region:
    model(data) forward pass only.

Excluded from timing:
    data loading, batch preparation, host-side metric computation, logit
    aggregation, loss, and accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        info["nvidia_smi"] = maybe_run_command(
            ["nvidia-smi", "--query-gpu=name,driver_version,power.limit,power.draw", "--format=csv,noheader"]
        )
    else:
        info["cuda_devices"] = []

    info["lscpu"] = maybe_run_command(["bash", "-lc", "lscpu | head -45"])
    return info


def normalize_device_name(device_name: str) -> str:
    return "cuda" if device_name == "gpu" else device_name


def device_available(device_name: str) -> bool:
    device_name = normalize_device_name(device_name)
    if device_name == "cpu":
        return True
    if device_name.startswith("cuda"):
        return torch.cuda.is_available()
    return False


def parse_batch_sizes(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Batch size must be positive: {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("No valid batch sizes were provided")
    return values


def timed_runs_for_batch(args: argparse.Namespace, batch_size: int) -> int:
    if args.target_timed_samples is None or args.target_timed_samples <= 0:
        return max(int(args.num_runs), 1)
    runs_from_samples = (int(args.target_timed_samples) + int(batch_size) - 1) // int(batch_size)
    runs = max(int(args.min_runs_per_batch), runs_from_samples)
    return max(1, min(int(args.num_runs), runs))


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def active_power_for_device(args: argparse.Namespace, device_name: str) -> Tuple[Optional[float], Dict[str, Any]]:
    device_name = normalize_device_name(device_name)
    if device_name == "cpu":
        direct = args.cpu_active_power_w
        idle = args.cpu_idle_power_w
        runtime = args.cpu_runtime_power_w
    elif device_name.startswith("cuda"):
        direct = args.gpu_active_power_w
        idle = args.gpu_idle_power_w
        runtime = args.gpu_runtime_power_w
    else:
        direct = idle = runtime = None

    meta: Dict[str, Any] = {
        "direct_active_power_w": direct,
        "idle_average_power_w": idle,
        "runtime_average_power_w": runtime,
        "computed_active_power_w": None,
        "method": None,
    }

    if direct is not None:
        meta["computed_active_power_w"] = float(direct)
        meta["method"] = "provided_active_power"
        return float(direct), meta

    if idle is not None and runtime is not None:
        active = max(float(runtime) - float(idle), 0.0)
        meta["computed_active_power_w"] = active
        meta["method"] = "runtime_average_minus_idle_average"
        return active, meta

    return None, meta


def synchronize_if_needed(device_name: str) -> None:
    if normalize_device_name(device_name).startswith("cuda"):
        torch.cuda.synchronize()


def timed_forward(
    model: torch.nn.Module,
    data: torch.Tensor,
    device_name: str,
    cuda_timing: str,
) -> Tuple[Any, float, str]:
    device_name = normalize_device_name(device_name)

    if device_name.startswith("cuda") and cuda_timing == "event":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits_t = model(data)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
        return logits_t, elapsed_s, "cuda_event_forward_only"

    synchronize_if_needed(device_name)
    t0 = time.perf_counter()
    logits_t = model(data)
    synchronize_if_needed(device_name)
    t1 = time.perf_counter()
    return logits_t, t1 - t0, "wall_forward_only"


def summarize_timed_rows(
    rows: List[Dict[str, Any]],
    active_power_w: Optional[float],
) -> Dict[str, Any]:
    timed = [r for r in rows if int(r.get("warmup", 0)) == 0 and r.get("status") == "ok"]
    if not timed:
        return {
            "available": False,
            "num_timed_batches": 0,
            "total_timed_samples": 0,
            "mean_latency_ms_per_image": None,
            "throughput_images_per_s": None,
            "energy_mJ_per_image": None,
        }

    batch_s = [float(r["latency_s"]) for r in timed]
    per_image_s = [float(r["latency_s_per_image"]) for r in timed]
    total_samples = sum(int(r["batch_size"]) for r in timed)
    total_time_s = sum(batch_s)
    effective_latency_s_per_image = total_time_s / max(total_samples, 1)

    summary: Dict[str, Any] = {
        "available": True,
        "num_timed_batches": len(timed),
        "total_timed_samples": total_samples,
        "total_forward_time_s": total_time_s,
        "mean_batch_latency_s": statistics.mean(batch_s),
        "mean_batch_latency_ms": statistics.mean(batch_s) * 1000.0,
        "median_batch_latency_ms": statistics.median(batch_s) * 1000.0,
        "std_batch_latency_ms": statistics.pstdev(batch_s) * 1000.0 if len(batch_s) > 1 else 0.0,
        "p90_batch_latency_ms": percentile(batch_s, 90) * 1000.0,
        "p95_batch_latency_ms": percentile(batch_s, 95) * 1000.0,
        "mean_latency_s_per_image": effective_latency_s_per_image,
        "mean_latency_ms_per_image": effective_latency_s_per_image * 1000.0,
        "mean_of_batch_latency_ms_per_image": statistics.mean(per_image_s) * 1000.0,
        "median_latency_ms_per_image": statistics.median(per_image_s) * 1000.0,
        "p90_latency_ms_per_image": percentile(per_image_s, 90) * 1000.0,
        "p95_latency_ms_per_image": percentile(per_image_s, 95) * 1000.0,
        "throughput_images_per_s": total_samples / total_time_s if total_time_s > 0 else None,
        "active_power_w": active_power_w,
        "energy_J_per_image": None,
        "energy_mJ_per_image": None,
    }

    if active_power_w is not None:
        summary["energy_J_per_image"] = active_power_w * effective_latency_s_per_image
        summary["energy_mJ_per_image"] = active_power_w * effective_latency_s_per_image * 1000.0

    return summary


def next_batch(loader_iter: Any, loader: Iterable[Any], cycle_loader: bool) -> Tuple[Any, Any, Any]:
    try:
        data, target = next(loader_iter)
        return data, target, loader_iter
    except StopIteration:
        if not cycle_loader:
            raise
        loader_iter = iter(loader)
        data, target = next(loader_iter)
        return data, target, loader_iter


@torch.inference_mode()
def benchmark_batch_size(
    *,
    ctx: Any,
    model: torch.nn.Module,
    split: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
    num_warmup: int,
    num_runs: int,
    max_batches: Optional[int],
    non_strict: bool,
    allow_no_split: bool,
    cuda_timing: str,
    cycle_loader: bool,
    active_power_w: Optional[float],
) -> Dict[str, Any]:
    device_name = normalize_device_name(device_name)
    loader = build_loader(
        ctx.config,
        split_name=split,
        batch_size=batch_size,
        num_workers=num_workers,
        allow_no_split=allow_no_split,
        shuffle=False,
    )
    loader_iter = iter(loader)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    agg_mode = logits_aggregation_from_config(ctx.config)

    rows: List[Dict[str, Any]] = []
    total_batches = num_warmup + num_runs
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    total_timed_samples = 0
    correct = 0
    loss_sum = 0.0
    sample_index = 0

    for batch_idx in range(total_batches):
        try:
            data, target, loader_iter = next_batch(loader_iter, loader, cycle_loader)
        except StopIteration:
            break

        reset_snn_state(model)
        data, target = prepare_snn_batch(data, target, ctx.config, ctx.device)
        is_warmup = batch_idx < num_warmup

        try:
            logits_t, elapsed_s, timing_method = timed_forward(model, data, device_name, cuda_timing)
        except RuntimeError as exc:
            message = str(exc)
            if device_name.startswith("cuda") and ("out of memory" in message.lower() or "cuda" in message.lower()):
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "device": device_name,
                        "batch_size": batch_size,
                        "batch_index": batch_idx,
                        "warmup": 1 if is_warmup else 0,
                        "status": "runtime_error",
                        "error": message[:500],
                    }
                )
                break
            raise

        logits = aggregate_time_logits(logits_t, agg_mode)
        loss = criterion(logits, target)
        c, n = accuracy_from_logits(logits, target)
        n_int = int(n)

        row = {
            "device": device_name,
            "batch_size": batch_size,
            "batch_index": batch_idx,
            "sample_index_start": sample_index,
            "num_samples": n_int,
            "latency_s": elapsed_s,
            "latency_ms": elapsed_s * 1000.0,
            "latency_s_per_image": elapsed_s / max(n_int, 1),
            "latency_ms_per_image": elapsed_s * 1000.0 / max(n_int, 1),
            "correct": int(c),
            "loss_sum": float(loss.item()),
            "warmup": 1 if is_warmup else 0,
            "timing_method": timing_method,
            "status": "ok",
        }
        rows.append(row)

        if not is_warmup:
            total_timed_samples += n_int
            correct += int(c)
            loss_sum += float(loss.item())

        sample_index += n_int

    summary = summarize_timed_rows(rows, active_power_w)
    accuracy = 100.0 * correct / max(total_timed_samples, 1)
    avg_loss = loss_sum / max(total_timed_samples, 1)

    return {
        "device": device_name,
        "batch_size": batch_size,
        "available": bool(summary.get("available", False)),
        "accuracy_percent_timed": accuracy,
        "loss_timed": avg_loss,
        "summary": summary,
        "rows": rows,
    }


def choose_best_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    available = [
        r for r in results
        if r.get("available") and r.get("summary", {}).get("mean_latency_ms_per_image") is not None
    ]
    if not available:
        return None
    return min(available, key=lambda r: float(r["summary"]["mean_latency_ms_per_image"]))


def flatten_sweep_rows(device_name: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        s = result.get("summary", {})
        rows.append(
            {
                "device": device_name,
                "batch_size": result.get("batch_size"),
                "available": result.get("available"),
                "accuracy_percent_timed": result.get("accuracy_percent_timed"),
                "mean_batch_latency_ms": s.get("mean_batch_latency_ms"),
                "mean_latency_ms_per_image": s.get("mean_latency_ms_per_image"),
                "throughput_images_per_s": s.get("throughput_images_per_s"),
                "active_power_w": s.get("active_power_w"),
                "energy_mJ_per_image": s.get("energy_mJ_per_image"),
                "num_timed_batches": s.get("num_timed_batches"),
                "total_timed_samples": s.get("total_timed_samples"),
            }
        )
    return rows


def benchmark_device(
    *,
    args: argparse.Namespace,
    device_name: str,
    batch_sizes: Sequence[int],
    active_power_w: Optional[float],
) -> Dict[str, Any]:
    device_name = normalize_device_name(device_name)
    if not device_available(device_name):
        return {
            "device": device_name,
            "available": False,
            "skipped_reason": f"Device not available: {device_name}",
            "batch_sweep": [],
            "samples": [],
        }

    ctx = load_eval_context(
        config_path=args.config,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        output_root=args.output_root,
        eval_name="eval_06_tmp",
        device=device_name,
    )
    model = build_eval_model(ctx, strict=not args.non_strict)
    model.eval()

    if device_name.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    batch_results: List[Dict[str, Any]] = []
    all_samples: List[Dict[str, Any]] = []

    for batch_size in batch_sizes:
        effective_num_runs = timed_runs_for_batch(args, int(batch_size))
        print(
            f"[eval_06] device={device_name}, batch_size={batch_size}, "
            f"timed_runs={effective_num_runs}"
        )
        result = benchmark_batch_size(
            ctx=ctx,
            model=model,
            split=args.split,
            batch_size=batch_size,
            num_workers=args.num_workers,
            device_name=device_name,
            num_warmup=args.num_warmup,
            num_runs=effective_num_runs,
            max_batches=args.max_batches,
            non_strict=args.non_strict,
            allow_no_split=args.allow_no_split,
            cuda_timing=args.cuda_timing,
            cycle_loader=not args.no_cycle_loader,
            active_power_w=active_power_w,
        )
        for row in result["rows"]:
            row["dataset"] = args.dataset
        all_samples.extend(result["rows"])
        result = {k: v for k, v in result.items() if k != "rows"}
        batch_results.append(result)

        s = result.get("summary", {})
        if result.get("available"):
            print(
                "  {:.4f} ms/image, {:.2f} img/s, energy={}".format(
                    float(s.get("mean_latency_ms_per_image") or 0.0),
                    float(s.get("throughput_images_per_s") or 0.0),
                    "NA" if s.get("energy_mJ_per_image") is None else f"{float(s['energy_mJ_per_image']):.4f} mJ/image",
                )
            )
        else:
            print("  unavailable or failed")

    best = choose_best_result(batch_results)
    batch1 = next((r for r in batch_results if int(r.get("batch_size", -1)) == 1 and r.get("available")), None)

    if best is None:
        return {
            "device": device_name,
            "available": False,
            "skipped_reason": "No batch size completed successfully",
            "batch_sweep": batch_results,
            "samples": all_samples,
        }

    selected_summary = dict(best["summary"])
    selected_summary["selected_batch_size"] = int(best["batch_size"])
    selected_summary["selection_rule"] = "minimum mean_latency_ms_per_image over batch sweep"

    device_result = {
        "device": device_name,
        "available": True,
        "dataset": args.dataset,
        "main_metric": "best_throughput_equivalent_latency",
        "selected_batch_size": int(best["batch_size"]),
        "selection_rule": "minimum mean_latency_ms_per_image over batch sweep",
        "accuracy_percent_timed": best.get("accuracy_percent_timed"),
        "loss_timed": best.get("loss_timed"),
        "summary": selected_summary,
        "batch1_summary": batch1.get("summary") if batch1 is not None else None,
        "batch_sweep": batch_results,
        "samples": all_samples,
    }

    del model
    gc.collect()
    if device_name.startswith("cuda"):
        torch.cuda.empty_cache()
    return device_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_06 final CPU/CUDA software baseline")

    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--run-dir", default=None, type=str)
    parser.add_argument("--output-root", default="results/eval_v2", type=str)
    parser.add_argument("--split", default="test", type=str)

    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--batch-sizes", default="1,8,32", type=str)
    parser.add_argument("--cpu-batch-sizes", default=None, type=str)
    parser.add_argument("--gpu-batch-sizes", default=None, type=str)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--num-warmup", default=10, type=int)
    parser.add_argument("--num-runs", default=80, type=int)
    parser.add_argument("--target-timed-samples", default=256, type=int)
    parser.add_argument("--min-runs-per-batch", default=10, type=int)
    parser.add_argument("--max-batches", default=None, type=int)
    parser.add_argument("--no-cycle-loader", action="store_true")
    parser.add_argument("--cuda-timing", choices=["wall", "event"], default="wall")

    parser.add_argument("--cpu-active-power-w", default=None, type=float)
    parser.add_argument("--gpu-active-power-w", default=None, type=float)
    parser.add_argument("--cpu-idle-power-w", default=None, type=float)
    parser.add_argument("--cpu-runtime-power-w", default=None, type=float)
    parser.add_argument("--gpu-idle-power-w", default=None, type=float)
    parser.add_argument("--gpu-runtime-power-w", default=None, type=float)

    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--non-strict", action="store_true")

    return parser.parse_args()


def batch_sizes_for_device(args: argparse.Namespace, device_name: str) -> List[int]:
    device_name = normalize_device_name(device_name)
    if device_name == "cpu" and args.cpu_batch_sizes:
        return parse_batch_sizes(args.cpu_batch_sizes)
    if device_name.startswith("cuda") and args.gpu_batch_sizes:
        return parse_batch_sizes(args.gpu_batch_sizes)
    return parse_batch_sizes(args.batch_sizes)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root) / args.dataset / "eval_06"
    output_dir.mkdir(parents=True, exist_ok=True)

    platform_info = get_platform_info()
    save_json(platform_info, output_dir / "platform_info.json")

    print("=" * 80)
    print("[eval_06] final CPU/CUDA batch-sweep software baseline")
    print(f"dataset    : {args.dataset}")
    print(f"devices    : {args.devices}")
    print(f"batch_sizes: {args.batch_sizes}")
    print(f"timing     : {args.cuda_timing}")
    print("=" * 80)

    power_meta: Dict[str, Any] = {}
    device_results: List[Dict[str, Any]] = []
    all_samples: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []

    for raw_device in args.devices:
        device_name = normalize_device_name(raw_device)
        batch_sizes = batch_sizes_for_device(args, device_name)
        active_power_w, meta = active_power_for_device(args, device_name)
        power_meta[device_name] = meta

        result = benchmark_device(
            args=args,
            device_name=device_name,
            batch_sizes=batch_sizes,
            active_power_w=active_power_w,
        )
        samples = result.pop("samples", [])
        all_samples.extend(samples)
        sweep_rows.extend(flatten_sweep_rows(device_name, result.get("batch_sweep", [])))
        device_results.append(result)

        if result.get("available"):
            s = result["summary"]
            print(
                "[eval_06] selected device={} batch={} latency={:.4f} ms/image throughput={:.2f} img/s".format(
                    device_name,
                    int(result["selected_batch_size"]),
                    float(s.get("mean_latency_ms_per_image") or 0.0),
                    float(s.get("throughput_images_per_s") or 0.0),
                )
            )
        else:
            print(f"[eval_06] selected device={device_name} unavailable: {result.get('skipped_reason')}")

    summary = {
        "eval_name": "eval_06",
        "purpose": "cpu_cuda_software_baseline_batch_sweep",
        "command": " ".join(sys.argv),
        "dataset": args.dataset,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "devices": args.devices,
        "batch_sizes": args.batch_sizes,
        "cpu_batch_sizes": args.cpu_batch_sizes,
        "gpu_batch_sizes": args.gpu_batch_sizes,
        "num_workers": args.num_workers,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "target_timed_samples": args.target_timed_samples,
        "min_runs_per_batch": args.min_runs_per_batch,
        "cycle_loader": not args.no_cycle_loader,
        "cuda_timing": args.cuda_timing,
        "power_measurement": power_meta,
        "device_results": device_results,
        "notes": {
            "main_metric": "For each software device, the main reported latency is the minimum effective per-image latency over the compact batch sweep.",
            "default_batch_points": "1 is the strict single-image reference, 8 is a small-batch latency-throughput point, and 32 is the throughput-oriented GPU point.",
            "run_budget": "Timed iterations are adapted per batch size from target_timed_samples and capped by num_runs, so large batches do not dominate runtime.",
            "batch1": "Batch-1 latency is retained in batch1_summary for latency-oriented discussion, but it is not forced as the main GPU baseline.",
            "effective_latency": "effective per-image latency = total timed forward time / total timed samples",
            "energy": "energy per image = active power * selected effective latency per image",
            "timed_region": "Forward pass only; data loading, batch preparation, logit aggregation, loss, and accuracy are outside the timed region.",
            "paper_wording": "Call the selected metric throughput-equivalent per-image latency, not strict single-image latency.",
        },
    }

    save_json(summary, output_dir / "runtime_summary.json")
    save_csv_rows(all_samples, output_dir / "runtime_samples.csv")
    save_csv_rows(sweep_rows, output_dir / "batch_sweep_summary.csv")
    save_run_manifest(
        output_dir,
        eval_name="eval_06",
        command=" ".join(sys.argv),
        inputs={"config": args.config, "checkpoint": args.checkpoint},
        outputs={
            "runtime_summary": "runtime_summary.json",
            "runtime_samples": "runtime_samples.csv",
            "batch_sweep_summary": "batch_sweep_summary.csv",
            "platform_info": "platform_info.json",
        },
        extra={"dataset": args.dataset, "devices": args.devices},
    )

    print("=" * 80)
    print("[eval_06] complete")
    print(f"output_dir: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
