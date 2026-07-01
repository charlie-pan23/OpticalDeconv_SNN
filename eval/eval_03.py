"""
eval_03.py

Comparator threshold sweep for HIPSA evaluation.

This script reruns inference with an output-threshold proxy applied to
photonic MVM layers. It measures the accuracy / ADC request / latency /
energy tradeoff across comparator thresholds.

Inputs:
  checkpoint + config
  configs/hardware_hipsa.yaml
  configs/device_params.yaml

Outputs:
  results/eval_v2/<dataset>/eval_03/
    summary.json
    threshold_sweep.csv
    layer_threshold_sweep.csv
    config_snapshot.yaml
    run_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from eval.eval_utils import build_eval_model, build_loader, load_eval_context
from eval.eval_02 import (
    derive_architecture_counts,
    estimate_adc_requests,
    estimate_power,
    estimate_timing,
    model_adc_pool,
)
from utils.config_utils import time_steps
from utils.data_utils import (
    accuracy_from_logits,
    aggregate_time_logits,
    logits_aggregation_from_config,
    prepare_snn_batch,
    reset_snn_state,
)
from utils.result_io import (
    load_yaml,
    save_csv_rows,
    save_json,
    save_run_manifest,
    save_yaml,
)
from utils.seed_utils import set_seed


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ratio(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) > 0 else 0.0


def count_nonzero(x: torch.Tensor) -> int:
    return int(torch.count_nonzero(x).item())


def as_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if torch.is_tensor(item):
                return item
    return None


def is_mvm_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Linear))


def is_lif_module(module: nn.Module) -> bool:
    return "lif" in module.__class__.__name__.lower()


def dense_sop_from_module(module: nn.Module, output: torch.Tensor) -> float:
    if isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        in_per_out = (module.in_channels // module.groups) * kh * kw
        return float(output.numel() * in_per_out)

    if isinstance(module, nn.Linear):
        return float(output.numel() * module.in_features)

    return 0.0


def default_photonic_layer_names(model: nn.Module, include_fc2: bool = False) -> List[str]:
    if hasattr(model, "photonic_mvm_layer_names"):
        names = list(getattr(model, "photonic_mvm_layer_names"))
    else:
        names = [
            name
            for name, module in model.named_modules()
            if is_mvm_module(module)
        ]

    if include_fc2 and "fc2" in dict(model.named_modules()) and "fc2" not in names:
        names.append("fc2")

    if not include_fc2:
        names = [name for name in names if name != "fc2"]

    return names


def default_lif_names(model: nn.Module) -> List[str]:
    return [
        name
        for name, module in model.named_modules()
        if is_lif_module(module)
    ]


def map_mvm_to_lif(target_layers: Sequence[str], lif_names: Sequence[str]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    lif_idx = 0

    for layer in target_layers:
        if layer == "fc2":
            mapping[layer] = None
            continue

        if lif_idx < len(lif_names):
            mapping[layer] = lif_names[lif_idx]
            lif_idx += 1
        else:
            mapping[layer] = None

    return mapping


@dataclass
class LayerSweepStats:
    name: str
    module_type: str
    mapped_lif_name: Optional[str] = None

    input_shape_last: Optional[Tuple[int, ...]] = None
    output_shape_last: Optional[Tuple[int, ...]] = None
    lif_output_shape_last: Optional[Tuple[int, ...]] = None

    calls: int = 0
    lif_calls: int = 0

    dense_sop_total: float = 0.0
    active_sop_total: float = 0.0

    mvm_input_active: int = 0
    mvm_input_total: int = 0

    mvm_output_nonzero_active: int = 0
    mvm_output_total: int = 0

    adc_request_active: int = 0
    adc_request_total: int = 0

    threshold_suppressed_active: int = 0
    threshold_suppressed_total: int = 0

    lif_spike_active: int = 0
    lif_spike_total: int = 0

    output_abs_sum: float = 0.0
    output_sq_sum: float = 0.0
    output_max_abs: float = 0.0

    def add_mvm(
        self,
        module: nn.Module,
        x: torch.Tensor,
        y_raw: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> None:
        input_active = count_nonzero(x)
        input_total = int(x.numel())

        out_abs = y_raw.detach().abs()
        output_total = int(out_abs.numel())
        output_nonzero = int((out_abs > 0).sum().item()) if output_total else 0
        adc_request = int(y_mask.sum().item()) if output_total else 0

        dense_sop = dense_sop_from_module(module, y_raw)
        active_sop = dense_sop * ratio(input_active, input_total)

        self.calls += 1
        self.input_shape_last = tuple(int(v) for v in x.shape)
        self.output_shape_last = tuple(int(v) for v in y_raw.shape)

        self.dense_sop_total += dense_sop
        self.active_sop_total += active_sop

        self.mvm_input_active += input_active
        self.mvm_input_total += input_total

        self.mvm_output_nonzero_active += output_nonzero
        self.mvm_output_total += output_total

        self.adc_request_active += adc_request
        self.adc_request_total += output_total

        self.threshold_suppressed_active += max(output_nonzero - adc_request, 0)
        self.threshold_suppressed_total += output_total

        if output_total:
            self.output_abs_sum += float(out_abs.sum().item())
            self.output_sq_sum += float((out_abs * out_abs).sum().item())
            self.output_max_abs = max(self.output_max_abs, float(out_abs.max().item()))

    def add_lif(self, y: torch.Tensor) -> None:
        active = count_nonzero(y)
        total = int(y.numel())

        self.lif_calls += 1
        self.lif_output_shape_last = tuple(int(v) for v in y.shape)
        self.lif_spike_active += active
        self.lif_spike_total += total

    def to_dict(self, dataset: str, threshold_fs: float, threshold_abs: float, num_samples: int) -> Dict[str, Any]:
        return {
            "dataset": dataset,
            "threshold_fs": float(threshold_fs),
            "threshold_abs": float(threshold_abs),

            "layer": self.name,
            "module_type": self.module_type,
            "mapped_lif_name": self.mapped_lif_name,

            "calls": int(self.calls),
            "lif_calls": int(self.lif_calls),

            "input_shape_last": list(self.input_shape_last) if self.input_shape_last else None,
            "output_shape_last": list(self.output_shape_last) if self.output_shape_last else None,
            "lif_output_shape_last": list(self.lif_output_shape_last) if self.lif_output_shape_last else None,

            "dense_sop_total": float(self.dense_sop_total),
            "active_sop_total": float(self.active_sop_total),
            "dense_sop_per_image": float(self.dense_sop_total / max(num_samples, 1)),
            "active_sop_per_image": float(self.active_sop_total / max(num_samples, 1)),
            "active_sop_ratio": ratio(self.active_sop_total, self.dense_sop_total),

            "mvm_input_active": int(self.mvm_input_active),
            "mvm_input_total": int(self.mvm_input_total),
            "mvm_input_activity": ratio(self.mvm_input_active, self.mvm_input_total),

            "mvm_output_nonzero_active": int(self.mvm_output_nonzero_active),
            "mvm_output_total": int(self.mvm_output_total),
            "mvm_output_nonzero_activity": ratio(
                self.mvm_output_nonzero_active,
                self.mvm_output_total,
            ),

            "adc_request_active": int(self.adc_request_active),
            "adc_request_total": int(self.adc_request_total),
            "adc_request_activity": ratio(
                self.adc_request_active,
                self.adc_request_total,
            ),

            "threshold_suppressed_active": int(self.threshold_suppressed_active),
            "threshold_suppressed_total": int(self.threshold_suppressed_total),
            "threshold_suppressed_activity": ratio(
                self.threshold_suppressed_active,
                self.threshold_suppressed_total,
            ),

            "lif_spike_active": int(self.lif_spike_active),
            "lif_spike_total": int(self.lif_spike_total),
            "lif_spike_activity": ratio(self.lif_spike_active, self.lif_spike_total),

            "output_mean_abs": ratio(self.output_abs_sum, self.mvm_output_total),
            "output_rms": math.sqrt(ratio(self.output_sq_sum, self.mvm_output_total)),
            "output_max_abs": float(self.output_max_abs),
        }


class ThresholdSweepCollector:
    def __init__(
        self,
        model: nn.Module,
        target_layers: Sequence[str],
        time_steps: int,
        threshold_fs: float,
        threshold_abs: float,
        apply_threshold_to_model: bool,
    ) -> None:
        self.model = model
        self.modules = dict(model.named_modules())
        self.target_layers = list(target_layers)
        self.time_steps = int(time_steps)
        self.threshold_fs = float(threshold_fs)
        self.threshold_abs = float(threshold_abs)
        self.apply_threshold_to_model = bool(apply_threshold_to_model)

        self.lif_names = default_lif_names(model)
        self.mvm_to_lif = map_mvm_to_lif(self.target_layers, self.lif_names)

        self.stats: Dict[str, LayerSweepStats] = {}
        for layer in self.target_layers:
            module = self.modules[layer]
            self.stats[layer] = LayerSweepStats(
                name=layer,
                module_type=module.__class__.__name__,
                mapped_lif_name=self.mvm_to_lif.get(layer),
            )

        self.handles: List[RemovableHandle] = []

        self.num_batches = 0
        self.num_samples = 0
        self.model_input_active = 0
        self.model_input_total = 0

    def register(self) -> None:
        for layer in self.target_layers:
            module = self.modules[layer]

            def hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any, layer_name: str = layer) -> Any:
                return self._on_mvm(layer_name, mod, inputs, output)

            self.handles.append(module.register_forward_hook(hook))

        lif_to_mvm = {
            lif: mvm
            for mvm, lif in self.mvm_to_lif.items()
            if lif is not None and lif in self.modules
        }

        for lif_name, mvm_name in lif_to_mvm.items():
            module = self.modules[lif_name]

            def lif_hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any, layer_name: str = mvm_name) -> None:
                self._on_lif(layer_name, output)

            self.handles.append(module.register_forward_hook(lif_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def observe_model_input(self, x: torch.Tensor) -> None:
        self.num_batches += 1
        if x.dim() >= 2:
            self.num_samples += int(x.shape[1])
        else:
            self.num_samples += int(x.shape[0])

        self.model_input_active += count_nonzero(x)
        self.model_input_total += int(x.numel())

    def _threshold_mask(self, y: torch.Tensor) -> torch.Tensor:
        y_abs = y.detach().abs()
        if y_abs.numel() == 0:
            return torch.zeros_like(y_abs, dtype=torch.bool)

        max_abs = float(y_abs.max().item())
        threshold = max(float(self.threshold_abs), float(self.threshold_fs) * max_abs)

        if threshold <= 0:
            return y_abs > 0

        return y_abs > threshold

    def _on_mvm(self, layer_name: str, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        x = as_tensor(inputs)
        y = as_tensor(output)

        if x is None or y is None:
            return output

        mask = self._threshold_mask(y)
        self.stats[layer_name].add_mvm(
            module=module,
            x=x.detach(),
            y_raw=y.detach(),
            y_mask=mask.detach(),
        )

        if not self.apply_threshold_to_model:
            return output

        return y * mask.to(dtype=y.dtype)

    def _on_lif(self, layer_name: str, output: Any) -> None:
        y = as_tensor(output)
        if y is None:
            return
        self.stats[layer_name].add_lif(y.detach())

    def make_activity_summary(
        self,
        dataset: str,
        threshold_fs: float,
        threshold_abs: float,
        accuracy_percent: float,
        loss: float,
    ) -> Dict[str, Any]:
        layers = {
            name: stat.to_dict(dataset, threshold_fs, threshold_abs, self.num_samples)
            for name, stat in self.stats.items()
        }

        dense_sop_total = sum(x["dense_sop_total"] for x in layers.values())
        active_sop_total = sum(x["active_sop_total"] for x in layers.values())

        mvm_input_active = sum(x["mvm_input_active"] for x in layers.values())
        mvm_input_total = sum(x["mvm_input_total"] for x in layers.values())

        mvm_output_nonzero_active = sum(x["mvm_output_nonzero_active"] for x in layers.values())
        mvm_output_total = sum(x["mvm_output_total"] for x in layers.values())

        adc_request_active = sum(x["adc_request_active"] for x in layers.values())
        adc_request_total = sum(x["adc_request_total"] for x in layers.values())

        suppressed_active = sum(x["threshold_suppressed_active"] for x in layers.values())
        suppressed_total = sum(x["threshold_suppressed_total"] for x in layers.values())

        lif_spike_active = sum(x["lif_spike_active"] for x in layers.values())
        lif_spike_total = sum(x["lif_spike_total"] for x in layers.values())

        return {
            "dataset": dataset,
            "num_batches": int(self.num_batches),
            "num_samples": int(self.num_samples),
            "time_steps": int(self.time_steps),
            "threshold_fs": float(threshold_fs),
            "threshold_abs": float(threshold_abs),
            "accuracy_percent": float(accuracy_percent),
            "loss": float(loss),

            "model_input_active": int(self.model_input_active),
            "model_input_total": int(self.model_input_total),
            "model_input_activity": ratio(self.model_input_active, self.model_input_total),

            "dense_sop_total": float(dense_sop_total),
            "active_sop_total": float(active_sop_total),
            "dense_sop_per_image": float(dense_sop_total / max(self.num_samples, 1)),
            "active_sop_per_image": float(active_sop_total / max(self.num_samples, 1)),
            "active_sop_ratio": ratio(active_sop_total, dense_sop_total),

            "mvm_input_active": int(mvm_input_active),
            "mvm_input_total": int(mvm_input_total),
            "mvm_input_activity": ratio(mvm_input_active, mvm_input_total),

            "mvm_output_nonzero_active": int(mvm_output_nonzero_active),
            "mvm_output_total": int(mvm_output_total),
            "mvm_output_nonzero_activity": ratio(
                mvm_output_nonzero_active,
                mvm_output_total,
            ),

            "adc_request_active": int(adc_request_active),
            "adc_request_total": int(adc_request_total),
            "adc_request_activity": ratio(adc_request_active, adc_request_total),

            "threshold_suppressed_active": int(suppressed_active),
            "threshold_suppressed_total": int(suppressed_total),
            "threshold_suppressed_activity": ratio(suppressed_active, suppressed_total),

            "lif_spike_active": int(lif_spike_active),
            "lif_spike_total": int(lif_spike_total),
            "lif_spike_activity": ratio(lif_spike_active, lif_spike_total),

            "layers": layers,
        }

    def layer_rows(self, dataset: str, threshold_fs: float, threshold_abs: float) -> List[Dict[str, Any]]:
        return [
            stat.to_dict(dataset, threshold_fs, threshold_abs, self.num_samples)
            for stat in self.stats.values()
        ]


def run_one_threshold(
    *,
    model: nn.Module,
    loader: Any,
    config: Mapping[str, Any],
    device: torch.device,
    dataset: str,
    target_layers: Sequence[str],
    threshold_fs: float,
    threshold_abs: float,
    apply_threshold_to_model: bool,
    max_batches: Optional[int],
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    collector = ThresholdSweepCollector(
        model=model,
        target_layers=target_layers,
        time_steps=time_steps(config),
        threshold_fs=threshold_fs,
        threshold_abs=threshold_abs,
        apply_threshold_to_model=apply_threshold_to_model,
    )

    collector.register()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    agg_mode = logits_aggregation_from_config(config)

    correct = 0
    total = 0
    loss_sum = 0.0

    try:
        for batch_idx, (data, target) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            reset_snn_state(model)
            data, target = prepare_snn_batch(data, target, config, device)

            collector.observe_model_input(data)

            logits_t = model(data)
            logits = aggregate_time_logits(logits_t, agg_mode)

            loss = criterion(logits, target)
            c, n = accuracy_from_logits(logits, target)

            correct += int(c)
            total += int(n)
            loss_sum += float(loss.item())
    finally:
        collector.remove()

    accuracy = 100.0 * correct / max(total, 1)
    loss = loss_sum / max(total, 1)

    activity_summary = collector.make_activity_summary(
        dataset=dataset,
        threshold_fs=threshold_fs,
        threshold_abs=threshold_abs,
        accuracy_percent=accuracy,
        loss=loss,
    )

    counts = derive_architecture_counts(
        hardware_cfg,
        hapr_group_size_override=args.hapr_group_size,
        adc_macros_override=args.adc_macros,
    )

    adc_requests = estimate_adc_requests(
        activity_summary,
        hapr_group_size=counts["hapr_group_size"],
    )

    timing_base = estimate_timing(
        activity_summary=activity_summary,
        hardware_cfg=hardware_cfg,
        adc_pool=None,
    )

    adc_pool = model_adc_pool(
        adc_requests=adc_requests,
        timing_base=timing_base,
        counts=counts,
    )

    timing = estimate_timing(
        activity_summary=activity_summary,
        hardware_cfg=hardware_cfg,
        adc_pool=adc_pool,
    )

    power = estimate_power(
        activity_summary=activity_summary,
        timing=timing,
        adc_pool=adc_pool,
        counts=counts,
        device_cfg=device_cfg,
        modulator_activity_source=args.modulator_activity_source,
        adc_power_mode=args.adc_power_mode,
        mrr_stabilization_mw=args.mrr_stabilization_mw,
    )

    row = {
        "dataset": dataset,
        "threshold_fs": float(threshold_fs),
        "threshold_abs": float(threshold_abs),
        "apply_threshold_to_model": bool(apply_threshold_to_model),

        "num_samples": int(activity_summary["num_samples"]),
        "accuracy_percent": float(activity_summary["accuracy_percent"]),
        "loss": float(activity_summary["loss"]),

        "dense_sop_per_image": float(activity_summary["dense_sop_per_image"]),
        "active_sop_per_image": float(activity_summary["active_sop_per_image"]),
        "active_sop_ratio": float(activity_summary["active_sop_ratio"]),

        "model_input_activity": float(activity_summary["model_input_activity"]),
        "mvm_input_activity": float(activity_summary["mvm_input_activity"]),
        "lif_spike_activity": float(activity_summary["lif_spike_activity"]),
        "adc_element_request_activity": float(activity_summary["adc_request_activity"]),
        "threshold_suppressed_activity": float(activity_summary["threshold_suppressed_activity"]),

        "adc_group_request_activity": float(adc_requests["adc_group_request_activity"]),
        "adc_requests_per_image": float(adc_requests["adc_requests_per_image"]),
        "adc_demand_per_cycle": float(adc_pool["adc_demand_per_cycle"]),
        "adc_macro_utilization": float(adc_pool["adc_macro_utilization"]),
        "adc_stall_cycles_proxy": float(adc_pool["adc_stall_cycles_proxy"]),
        "adc_is_saturated": bool(adc_pool["adc_is_saturated"]),

        "latency_us_per_image": float(timing["latency_us_per_image"]),
        "throughput_images_per_s": float(timing["throughput_images_per_s"]),
        "total_power_w": float(power["total_power_w"]),
        "energy_uJ_per_image": float(power["energy_uJ_per_image"]),
        "active_GOPS_per_W": float(power["active_GOPS_per_W"]),
        "dense_equivalent_GOPS_per_W": float(power["dense_equivalent_GOPS_per_W"]),
    }

    layer_rows = collector.layer_rows(dataset, threshold_fs, threshold_abs)
    return row, layer_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_03: comparator threshold sweep")

    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)

    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml", type=str)
    parser.add_argument("--device-params", default="configs/device_params.yaml", type=str)

    parser.add_argument("--output-root", default="results/eval_v2", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--max-batches", default=None, type=int)

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
    )
    parser.add_argument("--threshold-abs", default=0.0, type=float)

    parser.add_argument("--include-fc2", action="store_true")
    parser.add_argument("--request-only", action="store_true")
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--non-strict", action="store_true")

    parser.add_argument("--hapr-group-size", default=None, type=int)
    parser.add_argument("--adc-macros", default=None, type=int)
    parser.add_argument(
        "--modulator-activity-source",
        default="mvm_input_activity",
        choices=["model_input_activity", "mvm_input_activity", "active_sop_ratio", "always_on"],
    )
    parser.add_argument(
        "--adc-power-mode",
        default="activity_scaled",
        choices=["activity_scaled", "all_biased"],
    )
    parser.add_argument("--mrr-stabilization-mw", default=None, type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hardware_cfg = load_yaml(args.hardware)
    device_cfg = load_yaml(args.device_params)

    ctx = load_eval_context(
        config_path=args.config,
        checkpoint=args.checkpoint,
        hardware_config=args.hardware,
        device_params=args.device_params,
        output_root=args.output_root,
        eval_name="eval_03",
        device=args.device,
    )

    if args.dataset != ctx.dataset:
        print(f"[WARN] --dataset={args.dataset}, config dataset={ctx.dataset}; using --dataset for output labels.")

    seed = int(
        ctx.config.get("experiment", {}).get(
            "seed",
            ctx.config.get("split", {}).get("seed", 42),
        )
    )
    set_seed(seed, deterministic=True, benchmark=False)

    model = build_eval_model(ctx, strict=not args.non_strict)

    loader = build_loader(
        ctx.config,
        split_name=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_no_split=args.allow_no_split,
        shuffle=False,
    )

    target_layers = default_photonic_layer_names(
        model,
        include_fc2=args.include_fc2,
    )

    output_dir = Path(args.output_root) / args.dataset / "eval_03"
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_threshold_to_model = not args.request_only

    sweep_rows: List[Dict[str, Any]] = []
    all_layer_rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("[eval_03] comparator threshold sweep")
    print(f"dataset                  : {args.dataset}")
    print(f"thresholds               : {args.thresholds}")
    print(f"apply_threshold_to_model : {apply_threshold_to_model}")
    print(f"target_layers            : {target_layers}")
    print("=" * 80)

    for threshold_fs in args.thresholds:
        print(f"[eval_03] running threshold_fs={threshold_fs}")

        row, layer_rows = run_one_threshold(
            model=model,
            loader=loader,
            config=ctx.config,
            device=ctx.device,
            dataset=args.dataset,
            target_layers=target_layers,
            threshold_fs=float(threshold_fs),
            threshold_abs=float(args.threshold_abs),
            apply_threshold_to_model=apply_threshold_to_model,
            max_batches=args.max_batches,
            hardware_cfg=hardware_cfg,
            device_cfg=device_cfg,
            args=args,
        )

        sweep_rows.append(row)
        all_layer_rows.extend(layer_rows)

        print(
            "  acc={:.2f}% adc_util={:.2f}% latency={:.2f}us energy={:.2f}uJ".format(
                row["accuracy_percent"],
                row["adc_macro_utilization"] * 100.0,
                row["latency_us_per_image"],
                row["energy_uJ_per_image"],
            )
        )

    summary = {
        "eval_name": "eval_03",
        "purpose": "comparator_threshold_sweep",
        "created_utc": now_utc(),
        "command": " ".join(sys.argv),

        "dataset": args.dataset,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "hardware": args.hardware,
        "device_params": args.device_params,

        "thresholds": args.thresholds,
        "threshold_abs": args.threshold_abs,
        "apply_threshold_to_model": apply_threshold_to_model,
        "target_layers": target_layers,

        "rows": sweep_rows,
        "notes": {
            "threshold_rule": "threshold = max(threshold_abs, threshold_fs * per-call max(abs(MVM output))).",
            "accuracy_proxy": "When request_only is false, Conv/Linear outputs below threshold are zeroed before LIF update.",
            "request_only": "If --request-only is passed, accuracy remains clean while ADC request is swept.",
            "adc_pool": "Latency/energy are estimated using the same eval02 timing/power model.",
        },
    }

    save_json(summary, output_dir / "summary.json")
    save_csv_rows(sweep_rows, output_dir / "threshold_sweep.csv")
    save_csv_rows(all_layer_rows, output_dir / "layer_threshold_sweep.csv")

    save_yaml(
        {
            "eval_03_args": vars(args),
            "hardware": hardware_cfg,
            "device_params": device_cfg,
        },
        output_dir / "config_snapshot.yaml",
    )

    save_run_manifest(
        output_dir,
        eval_name="eval_03",
        command=" ".join(sys.argv),
        inputs={
            "config": args.config,
            "checkpoint": args.checkpoint,
            "hardware": args.hardware,
            "device_params": args.device_params,
        },
        outputs={
            "summary": "summary.json",
            "threshold_sweep": "threshold_sweep.csv",
            "layer_threshold_sweep": "layer_threshold_sweep.csv",
            "config_snapshot": "config_snapshot.yaml",
        },
        extra={
            "dataset": args.dataset,
            "thresholds": args.thresholds,
            "apply_threshold_to_model": apply_threshold_to_model,
        },
    )

    print("=" * 80)
    print("[eval_03] complete")
    print(f"output_dir: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()