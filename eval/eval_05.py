"""
eval_05.py

Device-specific robustness evaluation for HIPSA.

This script runs forward-only hardware-aware inference robustness sweeps.
It does NOT train.
It does NOT generate figures.

Perturbation types:
  clean
  adc_bits
  mrr
  laser
  wdm
  tia
  combined

Outputs:
  results/eval_v2/<dataset>/eval_05/
    summary.json
    robustness_summary.csv
    robustness_detail.csv
    config_snapshot.yaml
    run_manifest.json

Recommended main design context:
  HAPR group size = 16
  ADC macros = 32
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import random
import sys
from dataclasses import dataclass
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


def tensor_channel_mask_shape(y: torch.Tensor) -> Tuple[int, ...]:
    if y.dim() == 4:
        return (1, int(y.shape[1]), 1, 1)
    if y.dim() == 2:
        return (1, int(y.shape[1]))
    if y.dim() >= 1:
        return tuple([1] * y.dim())
    return tuple()


def apply_symmetric_quantization(y: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return y

    # Keep 8-bit and above effectively quantized but low-error.
    y_abs = y.detach().abs()
    max_abs = float(y_abs.max().item()) if y_abs.numel() else 0.0
    if max_abs <= 0:
        return y

    qmax = max((2 ** (bits - 1)) - 1, 1)
    scale = max_abs / qmax

    yq = torch.clamp(torch.round(y / scale), -qmax, qmax) * scale
    return yq


def apply_wdm_crosstalk(y: torch.Tensor, crosstalk_db: float) -> torch.Tensor:
    # Use power ratio for intensity-style crosstalk.
    alpha = 10.0 ** (float(crosstalk_db) / 10.0)
    if alpha <= 0:
        return y

    if y.dim() == 4 and y.shape[1] > 1:
        left = torch.zeros_like(y)
        right = torch.zeros_like(y)
        left[:, 1:, :, :] = y[:, :-1, :, :]
        right[:, :-1, :, :] = y[:, 1:, :, :]
        return y + alpha * (left + right)

    if y.dim() == 2 and y.shape[1] > 1:
        left = torch.zeros_like(y)
        right = torch.zeros_like(y)
        left[:, 1:] = y[:, :-1]
        right[:, :-1] = y[:, 1:]
        return y + alpha * (left + right)

    return y


def rms(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x.detach().float() ** 2)).item())


@dataclass
class PerturbationSpec:
    perturbation_type: str
    level: float
    level_label: str
    adc_bits: Optional[int] = None
    mrr_sigma: float = 0.0
    laser_sigma: float = 0.0
    wdm_crosstalk_db: Optional[float] = None
    tia_noise_sigma: float = 0.0
    combined: bool = False


@dataclass
class LayerRobustStats:
    name: str
    module_type: str
    mapped_lif_name: Optional[str] = None

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

    lif_spike_active: int = 0
    lif_spike_total: int = 0

    output_abs_sum: float = 0.0
    output_sq_sum: float = 0.0
    output_max_abs: float = 0.0

    def add_mvm(
        self,
        module: nn.Module,
        x: torch.Tensor,
        y_after: torch.Tensor,
        adc_threshold_fs: float,
        adc_threshold_abs: float,
    ) -> None:
        input_active = count_nonzero(x)
        input_total = int(x.numel())

        out_abs = y_after.detach().abs()
        output_total = int(out_abs.numel())
        output_nonzero = int((out_abs > 0).sum().item()) if output_total else 0

        max_abs = float(out_abs.max().item()) if output_total else 0.0
        threshold = max(float(adc_threshold_abs), float(adc_threshold_fs) * max_abs)

        if output_total == 0:
            adc_request = 0
        elif threshold <= 0:
            adc_request = output_nonzero
        else:
            adc_request = int((out_abs > threshold).sum().item())

        dense_sop = dense_sop_from_module(module, y_after)
        active_sop = dense_sop * ratio(input_active, input_total)

        self.calls += 1
        self.dense_sop_total += dense_sop
        self.active_sop_total += active_sop

        self.mvm_input_active += input_active
        self.mvm_input_total += input_total

        self.mvm_output_nonzero_active += output_nonzero
        self.mvm_output_total += output_total

        self.adc_request_active += adc_request
        self.adc_request_total += output_total

        if output_total:
            self.output_abs_sum += float(out_abs.sum().item())
            self.output_sq_sum += float((out_abs * out_abs).sum().item())
            self.output_max_abs = max(self.output_max_abs, max_abs)

    def add_lif(self, y: torch.Tensor) -> None:
        active = count_nonzero(y)
        total = int(y.numel())

        self.lif_calls += 1
        self.lif_spike_active += active
        self.lif_spike_total += total

    def to_dict(
            self,
            dataset: str,
            spec: PerturbationSpec,
            seed: int,
            num_samples: int,
    ) -> Dict[str, Any]:
        dense_sop_total = float(self.dense_sop_total)
        active_sop_total = float(self.active_sop_total)

        return {
            "dataset": dataset,
            "seed": int(seed),
            "perturbation_type": spec.perturbation_type,
            "level": spec.level,
            "level_label": spec.level_label,
            "layer": self.name,
            "module_type": self.module_type,
            "mapped_lif_name": self.mapped_lif_name,

            "calls": int(self.calls),
            "lif_calls": int(self.lif_calls),

            # Required by eval_02.estimate_adc_requests()
            "dense_sop_total": dense_sop_total,
            "active_sop_total": active_sop_total,
            "dense_sop_per_image": dense_sop_total / max(num_samples, 1),
            "active_sop_per_image": active_sop_total / max(num_samples, 1),
            "active_sop_ratio": ratio(active_sop_total, dense_sop_total),

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

            "lif_spike_active": int(self.lif_spike_active),
            "lif_spike_total": int(self.lif_spike_total),
            "lif_spike_activity": ratio(self.lif_spike_active, self.lif_spike_total),

            "output_mean_abs": ratio(self.output_abs_sum, self.mvm_output_total),
            "output_rms": math.sqrt(ratio(self.output_sq_sum, self.mvm_output_total)),
            "output_max_abs": float(self.output_max_abs),
        }


class RobustnessCollector:
    def __init__(
        self,
        model: nn.Module,
        target_layers: Sequence[str],
        spec: PerturbationSpec,
        seed: int,
        adc_threshold_fs: float,
        adc_threshold_abs: float,
        hapr_group_size: int,
    ) -> None:
        self.model = model
        self.modules = dict(model.named_modules())
        self.target_layers = list(target_layers)
        self.spec = spec
        self.seed = int(seed)
        self.adc_threshold_fs = float(adc_threshold_fs)
        self.adc_threshold_abs = float(adc_threshold_abs)
        self.hapr_group_size = int(hapr_group_size)

        self.lif_names = default_lif_names(model)
        self.mvm_to_lif = map_mvm_to_lif(self.target_layers, self.lif_names)

        self.stats: Dict[str, LayerRobustStats] = {}
        for layer in self.target_layers:
            module = self.modules[layer]
            self.stats[layer] = LayerRobustStats(
                name=layer,
                module_type=module.__class__.__name__,
                mapped_lif_name=self.mvm_to_lif.get(layer),
            )

        self.handles: List[RemovableHandle] = []
        self.static_masks: Dict[str, torch.Tensor] = {}
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

    def _get_static_mrr_mask(self, layer_name: str, y: torch.Tensor, sigma: float) -> torch.Tensor:
        key = f"{layer_name}:{tuple(y.shape)}:{sigma}:{self.seed}"
        if key in self.static_masks:
            return self.static_masks[key].to(device=y.device, dtype=y.dtype)

        gen = torch.Generator(device="cpu")
        # Stable but layer-dependent seed.
        layer_offset = abs(hash(layer_name)) % 100000
        gen.manual_seed(int(self.seed) + layer_offset + 2027)

        shape = tensor_channel_mask_shape(y)
        eps = torch.randn(shape, generator=gen, dtype=torch.float32) * float(sigma)
        mask = 1.0 + eps
        self.static_masks[key] = mask.cpu()
        return mask.to(device=y.device, dtype=y.dtype)

    def _apply_laser(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return y
        gain = 1.0 + torch.randn((), device=y.device, dtype=y.dtype) * float(sigma)
        return y * gain

    def _apply_tia_noise(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return y

        base = rms(y)
        if base <= 0:
            return y

        # Larger HAPR groups have a slightly stronger analog summing penalty.
        group_factor = math.sqrt(max(self.hapr_group_size, 1) / 8.0)
        noise_std = float(sigma) * group_factor * base
        return y + torch.randn_like(y) * noise_std

    def _apply_perturbation(self, layer_name: str, y: torch.Tensor) -> torch.Tensor:
        spec = self.spec
        out = y

        if spec.perturbation_type == "clean":
            return out

        if spec.perturbation_type == "mrr" or spec.combined:
            if spec.mrr_sigma > 0:
                mask = self._get_static_mrr_mask(layer_name, out, spec.mrr_sigma)
                out = out * mask

        if spec.perturbation_type == "laser" or spec.combined:
            out = self._apply_laser(out, spec.laser_sigma)

        if spec.perturbation_type == "wdm" or spec.combined:
            if spec.wdm_crosstalk_db is not None:
                out = apply_wdm_crosstalk(out, spec.wdm_crosstalk_db)

        if spec.perturbation_type == "tia" or spec.combined:
            out = self._apply_tia_noise(out, spec.tia_noise_sigma)

        if spec.perturbation_type == "adc_bits" or spec.combined:
            if spec.adc_bits is not None:
                out = apply_symmetric_quantization(out, int(spec.adc_bits))

        return out

    def _on_mvm(self, layer_name: str, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        x = as_tensor(inputs)
        y = as_tensor(output)

        if x is None or y is None:
            return output

        y_after = self._apply_perturbation(layer_name, y)

        self.stats[layer_name].add_mvm(
            module=module,
            x=x.detach(),
            y_after=y_after.detach(),
            adc_threshold_fs=self.adc_threshold_fs,
            adc_threshold_abs=self.adc_threshold_abs,
        )

        return y_after

    def _on_lif(self, layer_name: str, output: Any) -> None:
        y = as_tensor(output)
        if y is None:
            return
        self.stats[layer_name].add_lif(y.detach())

    def make_activity_summary(
        self,
        dataset: str,
        accuracy_percent: float,
        loss: float,
    ) -> Dict[str, Any]:
        layers = {
            name: stat.to_dict(dataset, self.spec, self.seed, self.num_samples)
            for name, stat in self.stats.items()
        }

        dense_sop_total = sum(x["dense_sop_per_image"] for x in layers.values()) * max(self.num_samples, 1)
        active_sop_total = sum(x["active_sop_per_image"] for x in layers.values()) * max(self.num_samples, 1)

        mvm_input_active = sum(stat.mvm_input_active for stat in self.stats.values())
        mvm_input_total = sum(stat.mvm_input_total for stat in self.stats.values())
        mvm_output_nonzero_active = sum(stat.mvm_output_nonzero_active for stat in self.stats.values())
        mvm_output_total = sum(stat.mvm_output_total for stat in self.stats.values())
        adc_request_active = sum(stat.adc_request_active for stat in self.stats.values())
        adc_request_total = sum(stat.adc_request_total for stat in self.stats.values())
        lif_spike_active = sum(stat.lif_spike_active for stat in self.stats.values())
        lif_spike_total = sum(stat.lif_spike_total for stat in self.stats.values())

        return {
            "dataset": dataset,
            "seed": int(self.seed),
            "perturbation_type": self.spec.perturbation_type,
            "level": self.spec.level,
            "level_label": self.spec.level_label,
            "num_batches": int(self.num_batches),
            "num_samples": int(self.num_samples),
            "accuracy_percent": float(accuracy_percent),
            "loss": float(loss),

            "model_input_activity": ratio(self.model_input_active, self.model_input_total),

            "dense_sop_total": float(dense_sop_total),
            "active_sop_total": float(active_sop_total),
            "dense_sop_per_image": float(dense_sop_total / max(self.num_samples, 1)),
            "active_sop_per_image": float(active_sop_total / max(self.num_samples, 1)),
            "active_sop_ratio": ratio(active_sop_total, dense_sop_total),

            "mvm_input_activity": ratio(mvm_input_active, mvm_input_total),
            "mvm_output_nonzero_activity": ratio(mvm_output_nonzero_active, mvm_output_total),
            "adc_request_activity": ratio(adc_request_active, adc_request_total),
            "lif_spike_activity": ratio(lif_spike_active, lif_spike_total),

            "layers": layers,
        }

    def layer_rows(self, dataset: str) -> List[Dict[str, Any]]:
        return [
            stat.to_dict(dataset, self.spec, self.seed, self.num_samples)
            for stat in self.stats.values()
        ]


def build_specs(mode: str) -> List[PerturbationSpec]:
    specs: List[PerturbationSpec] = []

    specs.append(
        PerturbationSpec(
            perturbation_type="clean",
            level=0.0,
            level_label="clean",
        )
    )

    if mode == "fast":
        adc_bits = [6, 4]
        mrr = [0.03]
        laser = [0.02]
        wdm = [-20.0]
        tia = [0.02]
    else:
        adc_bits = [4, 5, 6, 8]
        mrr = [0.01, 0.02, 0.03, 0.05]
        laser = [0.01, 0.02, 0.03]
        wdm = [-30.0, -25.0, -20.0, -15.0]
        tia = [0.005, 0.01, 0.02, 0.03]

    for b in adc_bits:
        specs.append(
            PerturbationSpec(
                perturbation_type="adc_bits",
                level=float(b),
                level_label=f"{b}bit",
                adc_bits=int(b),
            )
        )

    for s in mrr:
        specs.append(
            PerturbationSpec(
                perturbation_type="mrr",
                level=float(s),
                level_label=f"{100*s:.1f}pct",
                mrr_sigma=float(s),
            )
        )

    for s in laser:
        specs.append(
            PerturbationSpec(
                perturbation_type="laser",
                level=float(s),
                level_label=f"{100*s:.1f}pct",
                laser_sigma=float(s),
            )
        )

    for db in wdm:
        specs.append(
            PerturbationSpec(
                perturbation_type="wdm",
                level=float(db),
                level_label=f"{db:.0f}dB",
                wdm_crosstalk_db=float(db),
            )
        )

    for s in tia:
        specs.append(
            PerturbationSpec(
                perturbation_type="tia",
                level=float(s),
                level_label=f"{100*s:.1f}pct",
                tia_noise_sigma=float(s),
            )
        )

    specs.append(
        PerturbationSpec(
            perturbation_type="combined",
            level=1.0,
            level_label="mrr2_laser1_wdm25_adc6_tia1",
            adc_bits=6,
            mrr_sigma=0.02,
            laser_sigma=0.01,
            wdm_crosstalk_db=-25.0,
            tia_noise_sigma=0.01,
            combined=True,
        )
    )

    return specs


def compute_power_perf(
    activity_summary: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    hapr_group_size: int,
    adc_macros: int,
    modulator_activity_source: str,
    adc_power_mode: str,
) -> Dict[str, Any]:
    counts = derive_architecture_counts(
        hardware_cfg,
        hapr_group_size_override=hapr_group_size,
        adc_macros_override=adc_macros,
    )

    adc_requests = estimate_adc_requests(
        activity_summary,
        hapr_group_size=hapr_group_size,
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
        modulator_activity_source=modulator_activity_source,
        adc_power_mode=adc_power_mode,
        mrr_stabilization_mw=0.0,
    )

    return {
        "hapr_group_size": int(hapr_group_size),
        "adc_macros": int(adc_macros),
        "adc_group_request_activity": float(adc_requests["adc_group_request_activity"]),
        "adc_requests_per_image": float(adc_requests["adc_requests_per_image"]),
        "adc_macro_utilization": float(adc_pool["adc_macro_utilization"]),
        "adc_is_saturated": int(bool(adc_pool["adc_is_saturated"])),
        "latency_us_per_image": float(timing["latency_us_per_image"]),
        "throughput_images_per_s": float(timing["throughput_images_per_s"]),
        "total_power_w": float(power["total_power_w"]),
        "energy_uJ_per_image": float(power["energy_uJ_per_image"]),
        "active_GOPS_per_W": float(power["active_GOPS_per_W"]),
    }


@torch.inference_mode()
def run_one_spec(
    *,
    model: nn.Module,
    loader: Any,
    config: Mapping[str, Any],
    device: torch.device,
    dataset: str,
    target_layers: Sequence[str],
    spec: PerturbationSpec,
    seed: int,
    max_batches: Optional[int],
    adc_threshold_fs: float,
    adc_threshold_abs: float,
    hapr_group_size: int,
    adc_macros: int,
    hardware_cfg: Mapping[str, Any],
    device_cfg: Mapping[str, Any],
    modulator_activity_source: str,
    adc_power_mode: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(seed, deterministic=True, benchmark=False)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    collector = RobustnessCollector(
        model=model,
        target_layers=target_layers,
        spec=spec,
        seed=seed,
        adc_threshold_fs=adc_threshold_fs,
        adc_threshold_abs=adc_threshold_abs,
        hapr_group_size=hapr_group_size,
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

    activity = collector.make_activity_summary(
        dataset=dataset,
        accuracy_percent=accuracy,
        loss=loss,
    )

    perf = compute_power_perf(
        activity_summary=activity,
        hardware_cfg=hardware_cfg,
        device_cfg=device_cfg,
        hapr_group_size=hapr_group_size,
        adc_macros=adc_macros,
        modulator_activity_source=modulator_activity_source,
        adc_power_mode=adc_power_mode,
    )

    row = {
        "dataset": dataset,
        "seed": int(seed),
        "perturbation_type": spec.perturbation_type,
        "level": spec.level,
        "level_label": spec.level_label,

        "num_samples": int(activity["num_samples"]),
        "accuracy_percent": float(activity["accuracy_percent"]),
        "loss": float(activity["loss"]),

        "model_input_activity": float(activity["model_input_activity"]),
        "mvm_input_activity": float(activity["mvm_input_activity"]),
        "lif_spike_activity": float(activity["lif_spike_activity"]),
        "adc_element_request_activity": float(activity["adc_request_activity"]),
        "active_sop_ratio": float(activity["active_sop_ratio"]),
        "active_sop_per_image": float(activity["active_sop_per_image"]),

        **perf,
    }

    layer_rows = collector.layer_rows(dataset)
    return row, layer_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_05: device-specific robustness")

    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)

    parser.add_argument("--hardware", default="configs/hardware_hipsa.yaml", type=str)
    parser.add_argument("--device-params", default="configs/device_params.yaml", type=str)

    parser.add_argument("--output-root", default="results/eval_v2", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--max-batches", default=None, type=int)

    parser.add_argument("--mode", default="full", choices=["fast", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument("--hapr-group-size", default=16, type=int)
    parser.add_argument("--adc-macros", default=32, type=int)
    parser.add_argument("--adc-threshold-fs", default=0.02, type=float)
    parser.add_argument("--adc-threshold-abs", default=0.0, type=float)

    parser.add_argument("--include-fc2", action="store_true")
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--non-strict", action="store_true")

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
        eval_name="eval_05",
        device=args.device,
    )

    model = build_eval_model(ctx, strict=not args.non_strict)

    loader = build_loader(
        ctx.config,
        split_name=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allow_no_split=args.allow_no_split,
        shuffle=False,
    )

    target_layers = default_photonic_layer_names(model, include_fc2=args.include_fc2)
    specs = build_specs(args.mode)

    output_dir = Path(args.output_root) / args.dataset / "eval_05"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("[eval_05] device-specific robustness")
    print(f"dataset      : {args.dataset}")
    print(f"mode         : {args.mode}")
    print(f"batch_size   : {args.batch_size}")
    print(f"seeds        : {args.seeds}")
    print(f"HAPR / ADC   : {args.hapr_group_size} / {args.adc_macros}")
    print(f"target layers: {target_layers}")
    print("=" * 80)

    for seed in args.seeds:
        for spec in specs:
            print(f"[eval_05] seed={seed} type={spec.perturbation_type} level={spec.level_label}")

            row, layer_rows = run_one_spec(
                model=model,
                loader=loader,
                config=ctx.config,
                device=ctx.device,
                dataset=args.dataset,
                target_layers=target_layers,
                spec=spec,
                seed=seed,
                max_batches=args.max_batches,
                adc_threshold_fs=args.adc_threshold_fs,
                adc_threshold_abs=args.adc_threshold_abs,
                hapr_group_size=args.hapr_group_size,
                adc_macros=args.adc_macros,
                hardware_cfg=hardware_cfg,
                device_cfg=device_cfg,
                modulator_activity_source=args.modulator_activity_source,
                adc_power_mode=args.adc_power_mode,
            )

            summary_rows.append(row)
            detail_rows.extend(layer_rows)

            print(
                "  acc={:.2f}% energy={:.2f}uJ adc_util={:.2f}%".format(
                    row["accuracy_percent"],
                    row["energy_uJ_per_image"],
                    row["adc_macro_utilization"] * 100.0,
                )
            )

    clean_rows = [r for r in summary_rows if r["perturbation_type"] == "clean"]
    clean_acc_by_seed = {int(r["seed"]): float(r["accuracy_percent"]) for r in clean_rows}

    for row in summary_rows:
        seed = int(row["seed"])
        clean_acc = clean_acc_by_seed.get(seed)
        if clean_acc is not None:
            row["clean_accuracy_percent_same_seed"] = clean_acc
            row["accuracy_drop_percent"] = clean_acc - float(row["accuracy_percent"])
        else:
            row["clean_accuracy_percent_same_seed"] = ""
            row["accuracy_drop_percent"] = ""

    summary = {
        "eval_name": "eval_05",
        "purpose": "device_specific_robustness",
        "created_utc": now_utc(),
        "command": " ".join(sys.argv),

        "dataset": args.dataset,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "hardware": args.hardware,
        "device_params": args.device_params,

        "mode": args.mode,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "hapr_group_size": args.hapr_group_size,
        "adc_macros": args.adc_macros,
        "adc_threshold_fs": args.adc_threshold_fs,
        "adc_threshold_abs": args.adc_threshold_abs,
        "target_layers": target_layers,

        "num_rows": len(summary_rows),
        "perturbation_types": sorted(set(r["perturbation_type"] for r in summary_rows)),

        "notes": {
            "mrr": "Static per-layer channel multiplicative perturbation applied to MVM output.",
            "laser": "Common-mode multiplicative gain fluctuation applied per MVM call.",
            "wdm": "Adjacent-channel crosstalk using power-ratio alpha=10^(dB/10).",
            "adc_bits": "Symmetric per-call quantization applied to MVM output before LIF.",
            "tia": "Gaussian output noise scaled by layer RMS and sqrt(HAPR/8).",
            "combined": "MRR 2%, laser 1%, WDM -25 dB, ADC 6-bit, TIA 1%.",
            "scope": "Hardware-aware simulation/proxy, not fabricated silicon measurement.",
        },
    }

    save_json(summary, output_dir / "summary.json")
    save_csv_rows(summary_rows, output_dir / "robustness_summary.csv")
    save_csv_rows(detail_rows, output_dir / "robustness_detail.csv")

    save_yaml(
        {
            "eval_05_args": vars(args),
            "hardware": hardware_cfg,
            "device_params": device_cfg,
        },
        output_dir / "config_snapshot.yaml",
    )

    save_run_manifest(
        output_dir,
        eval_name="eval_05",
        command=" ".join(sys.argv),
        inputs={
            "config": args.config,
            "checkpoint": args.checkpoint,
            "hardware": args.hardware,
            "device_params": args.device_params,
        },
        outputs={
            "summary": "summary.json",
            "robustness_summary": "robustness_summary.csv",
            "robustness_detail": "robustness_detail.csv",
            "config_snapshot": "config_snapshot.yaml",
        },
        extra={
            "dataset": args.dataset,
            "mode": args.mode,
            "seeds": args.seeds,
            "hapr_group_size": args.hapr_group_size,
            "adc_macros": args.adc_macros,
        },
    )

    print("=" * 80)
    print("[eval_05] complete")
    print(f"output_dir: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()