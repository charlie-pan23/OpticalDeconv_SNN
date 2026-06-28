"""
HIPSA eval_03: device-calibrated robustness/sensitivity evaluation on CIFAR10-DVS.

Project-layout version requested by the user:
    - Model is imported directly from:  models/snn_vgg.py
    - Config is loaded directly from:   configs/config_cifar10dvs.yaml
    - Results are saved under:          ./results/eval_03_robustness/

This script is designed for one-click execution in PyCharm from the project root.
It does not embed the SNN architecture in this file; it uses the external
SpikingVGG definition and the external YAML configuration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikingjelly.activation_based import functional as sf
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

# The project guarantees that this file exists.
# Do not replace this with a local fallback; keep model architecture external.
from models.snn_vgg import SpikingVGG


# =============================================================================
# Config
# =============================================================================


@dataclass
class EvalConfig:
    raw_config: Dict[str, Any]

    config_path: str
    dataset_root: str
    checkpoint: str
    output_dir: str

    batch_size: int
    num_workers: int
    time_steps: int
    num_classes: int
    split_ratio: float
    binarize_input: bool
    max_batches: Optional[int]

    seed: int
    trials: int
    amp: bool
    device: str

    include_linear: bool
    include_fc2: bool
    calibration_batches: int
    comparator_threshold_fs: float
    hapr_group_size: int
    hapr_tia_noise_base_fs: float
    default_adc_bits: int

    # External YAML sections, copied so the power model is driven by config.
    dataset_cfg: Dict[str, Any] = field(default_factory=dict)
    network_cfg: Dict[str, Any] = field(default_factory=dict)
    hardware_cfg: Dict[str, Any] = field(default_factory=dict)
    power_cfg: Dict[str, Any] = field(default_factory=dict)

    # Derived from YAML / CLI for architecture-level energy reporting.
    image_latency_ms: float = 0.0717
    reference_adc_activity: float = 0.38
    reference_input_activity: float = 0.15
    reference_hapr_group_size: int = 8
    adc_macros: int = 16

    save_plots: bool = True


def load_yaml_config(path: str) -> Dict[str, Any]:
    # The project guarantees that configs/config_cifar10dvs.yaml exists.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_dataset_root(dataset_cfg: Dict[str, Any]) -> str:
    root_dir = Path(str(dataset_cfg["root_dir"]))
    dataset_name = str(dataset_cfg.get("name", "cifar10dvs")).lower().replace("-", "")
    if dataset_name in {"cifar10dvs", "cifar10_dvs"}:
        # SpikingJelly CIFAR10DVS normally points to the CIFAR10DVS subdirectory.
        return str(root_dir / "CIFAR10DVS")
    return str(root_dir / dataset_cfg.get("name", ""))


def build_config(args: argparse.Namespace) -> EvalConfig:
    raw = load_yaml_config(args.config)
    dataset_cfg = raw["dataset"]
    network_cfg = raw.get("network", {})
    hardware_cfg = raw["hardware"]
    power_cfg = raw["power_model"]
    power_eval_cfg = power_cfg.get("evaluation", {})

    # Use YAML as source of truth. CLI args only override YAML when explicitly passed.
    dataset_root = args.dataset_root or build_dataset_root(dataset_cfg)
    batch_size = args.batch_size if args.batch_size is not None else int(dataset_cfg.get("batch_size_eval", dataset_cfg["batch_size"]))
    time_steps = args.time_steps if args.time_steps is not None else int(dataset_cfg["time_steps"])
    num_classes = int(dataset_cfg["num_classes"])
    default_adc_bits = args.default_adc_bits if args.default_adc_bits is not None else int(hardware_cfg["adc_resolution"])

    # These can be moved into YAML later as hardware.hapr_group_size, hardware.adc_macros,
    # hardware.image_latency_ms, and dataset/input_spike_activity. The current config already
    # provides the main ADC activity through power_model.evaluation.adc_trigger_duty_cycle.
    reference_adc_activity = float(power_eval_cfg["adc_trigger_duty_cycle"])
    image_latency_ms = float(
        args.image_latency_ms
        if args.image_latency_ms is not None
        else hardware_cfg.get("image_latency_ms", power_eval_cfg.get("image_latency_ms", 0.0717))
    )
    hapr_group_size = int(args.hapr_group_size if args.hapr_group_size is not None else hardware_cfg.get("hapr_group_size", 8))
    reference_hapr_group_size = int(hardware_cfg.get("reference_hapr_group_size", hapr_group_size))
    adc_macros = int(hardware_cfg.get("adc_macros", 16))
    reference_input_activity = float(dataset_cfg.get("input_spike_activity", power_eval_cfg.get("input_spike_activity", 0.15)))

    return EvalConfig(
        raw_config=raw,
        config_path=args.config,
        dataset_root=dataset_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        time_steps=time_steps,
        num_classes=num_classes,
        split_ratio=args.split_ratio,
        binarize_input=(not args.no_binarize),
        max_batches=args.max_batches,
        seed=args.seed,
        trials=args.trials,
        amp=args.amp,
        device=args.device,
        include_linear=(not args.conv_only),
        include_fc2=args.include_fc2,
        calibration_batches=args.calibration_batches,
        comparator_threshold_fs=args.comparator_threshold_fs,
        hapr_group_size=hapr_group_size,
        hapr_tia_noise_base_fs=args.hapr_tia_noise_base_fs,
        default_adc_bits=default_adc_bits,
        dataset_cfg=dataset_cfg,
        network_cfg=network_cfg,
        hardware_cfg=hardware_cfg,
        power_cfg=power_cfg,
        image_latency_ms=image_latency_ms,
        reference_adc_activity=reference_adc_activity,
        reference_input_activity=reference_input_activity,
        reference_hapr_group_size=reference_hapr_group_size,
        adc_macros=adc_macros,
        save_plots=(not args.no_plots),
    )


def model_kwargs_from_config(cfg: EvalConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"num_classes": cfg.num_classes}

    # Keep constructor parameters external/configurable when the YAML provides them.
    # The uploaded model accepts num_classes, tau, v_threshold, and v_reset.
    model_kwargs = cfg.network_cfg.get("model_kwargs", {}) or {}
    kwargs.update(model_kwargs)
    for key in ["tau", "v_threshold", "v_reset"]:
        if key in cfg.network_cfg:
            kwargs[key] = cfg.network_cfg[key]
    return kwargs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# =============================================================================
# Dataset / model
# =============================================================================


def build_loader(cfg: EvalConfig) -> DataLoader:
    full_dataset = CIFAR10DVS(
        root=cfg.dataset_root,
        data_type="frame",
        frames_number=cfg.time_steps,
        split_by="number",
    )

    # CIFAR10-DVS has no official train/test split. Keep this deterministic and
    # match the training script's split_ratio/seed for final paper numbers.
    set_seed(cfg.seed)
    try:
        _, test_ds = split_to_train_test_set(cfg.split_ratio, full_dataset, num_classes=cfg.num_classes)
    except TypeError:
        _, test_ds = split_to_train_test_set(cfg.split_ratio, full_dataset, cfg.num_classes)

    return DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def instantiate_model(cfg: EvalConfig, device: torch.device) -> nn.Module:
    model = SpikingVGG(**model_kwargs_from_config(cfg))
    return model.to(device)


def torch_load(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch_load(checkpoint_path, device)
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "net", "network", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in ckpt.items():
        if not torch.is_tensor(value):
            continue
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {len(missing)}. First few: {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)}. First few: {unexpected[:8]}")

    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def target_modules(cfg: EvalConfig, model: nn.Module) -> List[Tuple[str, nn.Module]]:
    modules: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            modules.append((name, module))
        elif cfg.include_linear and isinstance(module, nn.Linear):
            if name == "fc2" and not cfg.include_fc2:
                continue
            modules.append((name, module))
    return modules


# =============================================================================
# Batch / output utilities
# =============================================================================


def prepare_batch(data: torch.Tensor, cfg: EvalConfig, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.dim() != 5:
        raise ValueError(f"Expected CIFAR10-DVS frames [B,T,C,H,W] or [T,B,C,H,W], got {tuple(data.shape)}")

    # The external SpikingVGG forward uses [T, B, C, H, W].
    if data.shape[1] == cfg.time_steps:
        data = data.transpose(0, 1).contiguous()
    elif data.shape[0] == cfg.time_steps:
        data = data.contiguous()
    else:
        raise ValueError(f"Cannot infer time dimension from shape={tuple(data.shape)}, T={cfg.time_steps}")

    data = data.to(device, non_blocking=True).float()
    if cfg.binarize_input:
        data = (data > 0).float()
    return data


def aggregate_logits(output: Any, cfg: EvalConfig, batch_size: int) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        tensors = [x for x in output if torch.is_tensor(x)]
        output = tensors[-1]
    if output.dim() == 2:
        return output
    if output.dim() == 3:
        if output.shape[0] == cfg.time_steps and output.shape[1] == batch_size:
            return output.mean(dim=0)
        if output.shape[0] == batch_size and output.shape[1] == cfg.time_steps:
            return output.mean(dim=1)
        return output.mean(dim=0)
    raise ValueError(f"Unsupported model output shape: {tuple(output.shape)}")


def reset_snn_state(model: nn.Module) -> None:
    try:
        sf.reset_net(model)
    except Exception:
        pass


# =============================================================================
# Calibration / activity / perturbation
# =============================================================================


def symmetric_quantize_fixed_fs(x: torch.Tensor, bits: int, full_scale: float, eps: float = 1e-12) -> torch.Tensor:
    if bits <= 0 or full_scale <= eps:
        return x
    qmax = (2 ** (bits - 1)) - 1
    if qmax <= 0:
        return x
    fs = torch.as_tensor(float(full_scale), dtype=x.dtype, device=x.device)
    y = torch.clamp(x / fs, -1.0, 1.0)
    return torch.round(y * qmax) / qmax * fs


class FullScaleCalibrator:
    def __init__(self, modules: Sequence[Tuple[str, nn.Module]]):
        self.modules = list(modules)
        self.max_abs = {name: 0.0 for name, _ in self.modules}
        self.handles: List[Any] = []

    def __enter__(self) -> "FullScaleCalibrator":
        for name, module in self.modules:
            self.handles.append(module.register_forward_hook(self._hook_for(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook_for(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if torch.is_tensor(output) and output.numel() > 0:
                self.max_abs[name] = max(self.max_abs[name], float(output.detach().abs().amax().item()))
            return None
        return hook

    def full_scales(self) -> Dict[str, float]:
        return {name: max(v, 1e-12) for name, v in self.max_abs.items()}


class ActivityCollector:
    def __init__(self, modules: Sequence[Tuple[str, nn.Module]], full_scales: Dict[str, float], threshold_fs: float):
        self.modules = list(modules)
        self.full_scales = full_scales
        self.threshold_fs = threshold_fs
        self.handles: List[Any] = []
        self.active_count = 0
        self.total_count = 0
        self.layer_active = {name: 0 for name, _ in self.modules}
        self.layer_total = {name: 0 for name, _ in self.modules}

    def __enter__(self) -> "ActivityCollector":
        for name, module in self.modules:
            self.handles.append(module.register_forward_hook(self._hook_for(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook_for(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if not torch.is_tensor(output) or output.numel() == 0:
                return None
            with torch.no_grad():
                threshold = self.threshold_fs * self.full_scales[name]
                active = int((output.detach().abs() > threshold).sum().item())
                total = int(output.numel())
                self.active_count += active
                self.total_count += total
                self.layer_active[name] += active
                self.layer_total[name] += total
            return None
        return hook

    def summary(self) -> Dict[str, Any]:
        by_layer = {
            k: self.layer_active[k] / max(self.layer_total[k], 1)
            for k in self.layer_total
            if self.layer_total[k] > 0
        }
        return {
            "adc_activity_proxy": float(self.active_count / max(self.total_count, 1)),
            "adc_active_count": float(self.active_count),
            "adc_total_count": float(self.total_count),
            "adc_activity_by_layer": by_layer,
        }


class PerturbationContext:
    def __init__(
        self,
        modules: Sequence[Tuple[str, nn.Module]],
        full_scales: Dict[str, float],
        mrr_pct: float = 0.0,
        laser_pct: float = 0.0,
        wdm_db: Optional[float] = None,
        tia_noise_fs: float = 0.0,
        adc_bits: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.modules = list(modules)
        self.full_scales = full_scales
        self.mrr_pct = float(mrr_pct or 0.0)
        self.laser_pct = float(laser_pct or 0.0)
        self.wdm_db = wdm_db
        self.tia_noise_fs = float(tia_noise_fs or 0.0)
        self.adc_bits = adc_bits
        self.seed = seed
        self.handles: List[Any] = []

    def __enter__(self) -> "PerturbationContext":
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        if self.mrr_pct > 0:
            self._apply_static_mrr_weight_error(self.mrr_pct)

        for name, module in self.modules:
            if self.laser_pct > 0:
                self.handles.append(module.register_forward_pre_hook(self._laser_pre_hook))
            if self.wdm_db is not None or self.tia_noise_fs > 0 or self.adc_bits is not None:
                self.handles.append(module.register_forward_hook(self._output_hook_for(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _apply_static_mrr_weight_error(self, pct: float) -> None:
        ratio = pct / 100.0
        with torch.no_grad():
            for _, module in self.modules:
                weight = getattr(module, "weight", None)
                if weight is not None:
                    eps = torch.empty_like(weight).uniform_(-ratio, ratio)
                    weight.mul_(1.0 + eps)

    def _laser_pre_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        x = inputs[0]
        ratio = self.laser_pct / 100.0
        if ratio <= 0:
            return inputs
        if x.dim() == 4:
            shape = (x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 2:
            shape = (x.shape[0], 1)
        else:
            shape = tuple([x.shape[0]] + [1] * (x.dim() - 1))
        scale = torch.clamp(1.0 + torch.randn(shape, device=x.device, dtype=x.dtype) * ratio, min=0.0)
        return (x * scale, *inputs[1:])

    def _output_hook_for(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if not torch.is_tensor(output):
                return output
            y = output
            if self.wdm_db is not None:
                leak = 10.0 ** (float(self.wdm_db) / 10.0)
                y = adjacent_channel_crosstalk(y, leak)
            if self.tia_noise_fs > 0:
                std = (self.tia_noise_fs / 100.0) * self.full_scales[name]
                y = y + torch.randn_like(y) * std
            if self.adc_bits is not None:
                y = symmetric_quantize_fixed_fs(y, int(self.adc_bits), self.full_scales[name])
            return y
        return hook


def adjacent_channel_crosstalk(y: torch.Tensor, leak: float) -> torch.Tensor:
    if leak <= 0 or y.dim() < 2:
        return y
    ch_dim = 1
    if y.shape[ch_dim] < 2:
        return y

    n_ch = y.shape[ch_dim]
    degree = torch.zeros(n_ch, device=y.device, dtype=y.dtype)
    degree[0] = 1.0
    degree[-1] = 1.0
    if n_ch > 2:
        degree[1:-1] = 2.0
    view_shape = [1] * y.dim()
    view_shape[ch_dim] = n_ch

    mixed = y * torch.clamp(1.0 - leak * degree.view(*view_shape), min=0.0)
    left = torch.zeros_like(y)
    right = torch.zeros_like(y)

    src = [slice(None)] * y.dim()
    dst = [slice(None)] * y.dim()
    src[ch_dim] = slice(0, n_ch - 1)
    dst[ch_dim] = slice(1, n_ch)
    left[tuple(dst)] = y[tuple(src)]

    src = [slice(None)] * y.dim()
    dst = [slice(None)] * y.dim()
    src[ch_dim] = slice(1, n_ch)
    dst[ch_dim] = slice(0, n_ch - 1)
    right[tuple(dst)] = y[tuple(src)]
    return mixed + leak * (left + right)


# =============================================================================
# External-config-driven power / energy model
# =============================================================================


def hapr_lanes(cfg: EvalConfig, group_size: int) -> int:
    tiles = int(cfg.hardware_cfg["num_tiles"])
    tile_outputs = int(cfg.hardware_cfg["array_size"][1])
    return tiles * (tile_outputs // int(group_size))


def estimate_power_energy(
    cfg: EvalConfig,
    adc_activity_proxy: float,
    input_spike_activity: float,
    group_size: int,
) -> Dict[str, float]:
    """
    Use configs/config_cifar10dvs.yaml as the source of truth for power numbers.

    The current YAML gives a component-level baseline power model. We scale the
    event-gated components with measured activity rather than embedding a separate
    hard-coded power table in this script.
    """
    p = cfg.power_cfg
    ref_alpha = max(float(cfg.reference_adc_activity), 1e-12)
    alpha = float(np.clip(adc_activity_proxy, 0.0, 1.0))

    lanes = hapr_lanes(cfg, group_size)
    ref_lanes = hapr_lanes(cfg, cfg.reference_hapr_group_size)
    lane_scale = lanes / max(ref_lanes, 1)
    update_scale = (lanes * alpha) / max(ref_lanes * ref_alpha, 1e-12)
    mod_scale = float(input_spike_activity) / max(float(cfg.reference_input_activity), 1e-12)

    static_mw = float(p["cw_laser_source_mw"] + p["global_mrr_stabilization_mw"] + p["leakage_misc_io_mw"])
    modulator_mw = float(p["event_gated_modulator_drivers_mw"]) * mod_scale
    pd_tia_cmp_mw = float(p["pd_tia_comparator_mw"]) * lane_scale
    adc_pool_mw = float(p["shared_adc_pool_mw"]) * min(update_scale, 1.0 / ref_alpha)
    sram_mw = float(p["sram_register_files_mw"]) * update_scale
    noc_mw = float(p["noc_bus_controller_clock_mw"]) * update_scale

    # The uploaded YAML does not expose a separate digital LIF primitive power.
    # If a future YAML adds it, this line will use it automatically.
    lif_mw = float(p.get("digital_lif_update_mw", 0.0)) * update_scale

    total_mw = static_mw + modulator_mw + pd_tia_cmp_mw + adc_pool_mw + sram_mw + noc_mw + lif_mw
    power_w = total_mw / 1000.0
    energy_uJ = power_w * cfg.image_latency_ms * 1000.0

    return {
        "hapr_group_size": float(group_size),
        "hapr_lanes": float(lanes),
        "adc_demand_per_cycle": float(lanes * alpha),
        "adc_macro_utilization": float(min(1.0, (lanes * alpha) / max(cfg.adc_macros, 1))),
        "lane_scale": float(lane_scale),
        "update_scale": float(update_scale),
        "modulator_activity_scale": float(mod_scale),
        "static_power_mw": float(static_mw),
        "modulator_power_mw": float(modulator_mw),
        "pd_tia_comparator_power_mw": float(pd_tia_cmp_mw),
        "adc_pool_power_mw": float(adc_pool_mw),
        "sram_power_mw": float(sram_mw),
        "noc_power_mw": float(noc_mw),
        "lif_power_mw": float(lif_mw),
        "total_power_mw": float(total_mw),
        "power_w": float(power_w),
        "energy_uJ_per_image": float(energy_uJ),
        "config_baseline_total_system_power_w": float(p["total_system_power_w"]),
    }


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def calibrate_full_scales(
    model: nn.Module,
    loader: DataLoader,
    cfg: EvalConfig,
    device: torch.device,
    modules: Sequence[Tuple[str, nn.Module]],
) -> Dict[str, float]:
    model.eval()
    with FullScaleCalibrator(modules) as cal:
        for batch_idx, (data, _) in enumerate(tqdm(loader, desc="Calibrating", leave=False)):
            if batch_idx >= cfg.calibration_batches:
                break
            reset_snn_state(model)
            data = prepare_batch(data, cfg, device)
            _ = model(data)
    reset_snn_state(model)
    return cal.full_scales()


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    cfg: EvalConfig,
    device: torch.device,
    collector: Optional[ActivityCollector] = None,
) -> Dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    input_active = 0
    input_total = 0

    amp_ctx = (
        torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda"))
        if hasattr(torch, "autocast")
        else nullcontext()
    )

    for batch_idx, (data, targets) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
            break
        reset_snn_state(model)
        data = prepare_batch(data, cfg, device)
        targets = targets.to(device, non_blocking=True).view(-1)

        input_active += int((data > 0).sum().item())
        input_total += int(data.numel())

        with amp_ctx:
            logits = aggregate_logits(model(data), cfg, batch_size=targets.numel())

        pred = logits.argmax(dim=1)
        total += int(targets.numel())
        correct += int((pred == targets).sum().item())

    reset_snn_state(model)
    metrics: Dict[str, Any] = {
        "accuracy": float(100.0 * correct / max(total, 1)),
        "num_samples": float(total),
        "input_spike_activity": float(input_active / max(input_total, 1)),
        "input_active_count": float(input_active),
        "input_total_count": float(input_total),
    }
    if collector is not None:
        metrics.update(collector.summary())
    return metrics


def evaluate_condition(
    name: str,
    model: nn.Module,
    clean_state: Dict[str, torch.Tensor],
    loader: DataLoader,
    cfg: EvalConfig,
    device: torch.device,
    modules: Sequence[Tuple[str, nn.Module]],
    full_scales: Dict[str, float],
    perturb_kwargs: Dict[str, Any],
    trials: int,
    threshold_fs: Optional[float] = None,
    hapr_group_size: Optional[int] = None,
) -> Dict[str, Any]:
    threshold_fs = cfg.comparator_threshold_fs if threshold_fs is None else threshold_fs
    hapr_group_size = cfg.hapr_group_size if hapr_group_size is None else hapr_group_size
    trial_metrics: List[Dict[str, Any]] = []

    for t in range(max(int(trials), 1)):
        model.load_state_dict(clean_state, strict=True)
        reset_snn_state(model)
        seed = cfg.seed + 1009 * (t + 1)
        with PerturbationContext(modules, full_scales, seed=seed, **perturb_kwargs):
            with ActivityCollector(modules, full_scales, threshold_fs=threshold_fs) as collector:
                metrics = run_evaluation(model, loader, cfg, device, collector=collector)
        metrics.update(
            estimate_power_energy(
                cfg,
                adc_activity_proxy=metrics.get("adc_activity_proxy", 0.0),
                input_spike_activity=metrics.get("input_spike_activity", cfg.reference_input_activity),
                group_size=hapr_group_size,
            )
        )
        trial_metrics.append(metrics)

    def arr(key: str) -> np.ndarray:
        return np.array([m.get(key, np.nan) for m in trial_metrics], dtype=np.float64)

    return {
        "name": name,
        "perturbation": perturb_kwargs,
        "threshold_fs": float(threshold_fs),
        "hapr_group_size": int(hapr_group_size),
        "num_trials": int(len(trial_metrics)),
        "trials": trial_metrics,
        "accuracy_mean": float(np.nanmean(arr("accuracy"))),
        "accuracy_std": float(np.nanstd(arr("accuracy"))),
        "adc_activity_proxy_mean": float(np.nanmean(arr("adc_activity_proxy"))),
        "adc_activity_proxy_std": float(np.nanstd(arr("adc_activity_proxy"))),
        "input_spike_activity_mean": float(np.nanmean(arr("input_spike_activity"))),
        "power_w_mean": float(np.nanmean(arr("power_w"))),
        "power_w_std": float(np.nanstd(arr("power_w"))),
        "energy_uJ_per_image_mean": float(np.nanmean(arr("energy_uJ_per_image"))),
        "energy_uJ_per_image_std": float(np.nanstd(arr("energy_uJ_per_image"))),
        "adc_macro_utilization_mean": float(np.nanmean(arr("adc_macro_utilization"))),
        "adc_demand_per_cycle_mean": float(np.nanmean(arr("adc_demand_per_cycle"))),
        "hapr_lanes": float(trial_metrics[0].get("hapr_lanes", np.nan)),
    }


# =============================================================================
# Output
# =============================================================================


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "group", "condition", "accuracy_mean", "accuracy_std",
        "adc_activity_proxy_mean", "adc_activity_proxy_std", "input_spike_activity_mean",
        "threshold_fs", "hapr_group_size", "hapr_lanes", "adc_demand_per_cycle_mean",
        "adc_macro_utilization_mean", "power_w_mean", "power_w_std",
        "energy_uJ_per_image_mean", "energy_uJ_per_image_std", "num_trials", "perturbation",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "group": row.get("group", ""),
                "condition": row.get("name", ""),
                "accuracy_mean": row.get("accuracy_mean", ""),
                "accuracy_std": row.get("accuracy_std", ""),
                "adc_activity_proxy_mean": row.get("adc_activity_proxy_mean", ""),
                "adc_activity_proxy_std": row.get("adc_activity_proxy_std", ""),
                "input_spike_activity_mean": row.get("input_spike_activity_mean", ""),
                "threshold_fs": row.get("threshold_fs", ""),
                "hapr_group_size": row.get("hapr_group_size", ""),
                "hapr_lanes": row.get("hapr_lanes", ""),
                "adc_demand_per_cycle_mean": row.get("adc_demand_per_cycle_mean", ""),
                "adc_macro_utilization_mean": row.get("adc_macro_utilization_mean", ""),
                "power_w_mean": row.get("power_w_mean", ""),
                "power_w_std": row.get("power_w_std", ""),
                "energy_uJ_per_image_mean": row.get("energy_uJ_per_image_mean", ""),
                "energy_uJ_per_image_std": row.get("energy_uJ_per_image_std", ""),
                "num_trials": row.get("num_trials", ""),
                "perturbation": json.dumps(row.get("perturbation", {}), ensure_ascii=False),
            })


def write_plots(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib is unavailable. Skip plots.")
        return

    groups = sorted(set(str(r.get("group", "")) for r in rows))
    for group in groups:
        group_rows = [r for r in rows if r.get("group") == group]
        if not group_rows:
            continue
        x = list(range(len(group_rows)))
        labels = [str(r.get("name", "")) for r in group_rows]

        plt.figure(figsize=(max(7, len(labels) * 0.8), 4.2))
        plt.errorbar(
            x,
            [r.get("accuracy_mean", np.nan) for r in group_rows],
            yerr=[r.get("accuracy_std", 0.0) for r in group_rows],
            marker="o",
            capsize=3,
        )
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Accuracy (%)")
        plt.title(group)
        plt.tight_layout()
        plt.savefig(output_dir / f"{group}_accuracy.png", dpi=200)
        plt.close()

        plt.figure(figsize=(max(7, len(labels) * 0.8), 4.2))
        plt.plot(x, [r.get("energy_uJ_per_image_mean", np.nan) for r in group_rows], marker="o")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Energy per image (uJ)")
        plt.title(f"{group} energy")
        plt.tight_layout()
        plt.savefig(output_dir / f"{group}_energy.png", dpi=200)
        plt.close()


def write_readme(path: Path, cfg: EvalConfig) -> None:
    path.write_text(
        "# HIPSA eval_03 robustness results\n\n"
        "Generated by eval_03_robustness_external_config.py.\n\n"
        "Main outputs:\n"
        "- eval_03_robustness_results.json: full per-trial results.\n"
        "- eval_03_robustness_summary.csv: compact table for paper figures.\n"
        "- *_accuracy.png and *_energy.png: quick-look plots if matplotlib is installed.\n\n"
        "Notes:\n"
        "- Model architecture is imported from models/snn_vgg.py.\n"
        "- Workload, hardware, precision, and power entries are loaded from configs/config_cifar10dvs.yaml.\n"
        "- Comparator/request activity uses clean calibrated full scales.\n"
        "- HAPR G sweep changes HAPR/request/power accounting, not CNN channel semantics.\n"
        f"- Config path: {cfg.config_path}\n",
        encoding="utf-8",
    )


def run_and_record(
    all_results: Dict[str, Any],
    flat_rows: List[Dict[str, Any]],
    group_name: str,
    condition_name: str,
    model: nn.Module,
    clean_state: Dict[str, torch.Tensor],
    loader: DataLoader,
    cfg: EvalConfig,
    device: torch.device,
    modules: Sequence[Tuple[str, nn.Module]],
    full_scales: Dict[str, float],
    perturb_kwargs: Dict[str, Any],
    trials: int,
    threshold_fs: Optional[float] = None,
    hapr_group_size: Optional[int] = None,
) -> Dict[str, Any]:
    result = evaluate_condition(
        condition_name, model, clean_state, loader, cfg, device, modules, full_scales,
        perturb_kwargs=perturb_kwargs, trials=trials,
        threshold_fs=threshold_fs, hapr_group_size=hapr_group_size,
    )
    result["group"] = group_name
    all_results.setdefault("groups", {}).setdefault(group_name, []).append(result)
    flat_rows.append(result)
    print(
        f"  acc={result['accuracy_mean']:.2f}±{result['accuracy_std']:.2f}% | "
        f"ADC_act={result['adc_activity_proxy_mean']:.4f} | "
        f"P={result['power_w_mean']:.3f} W | "
        f"E={result['energy_uJ_per_image_mean']:.1f} uJ"
    )
    return result


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="HIPSA eval_03 robustness evaluation")
    parser.add_argument("--config", type=str, default="configs/config_cifar10dvs.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="results/snn_vgg_best.pth")
    parser.add_argument("--output-dir", type=str, default="results/eval_03_robustness")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-binarize", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--conv-only", action="store_true")
    parser.add_argument("--include-fc2", action="store_true")
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--comparator-threshold-fs", type=float, default=0.02)
    parser.add_argument("--default-adc-bits", type=int, default=None)
    parser.add_argument("--hapr-group-size", type=int, default=None)
    parser.add_argument("--hapr-tia-noise-base-fs", type=float, default=0.0)
    parser.add_argument("--image-latency-ms", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = build_config(args)
    set_seed(cfg.seed)
    device = choose_device(cfg.device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== HIPSA eval_03 robustness ===")
    print(f"Config file:  {cfg.config_path}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Checkpoint:   {cfg.checkpoint}")
    print(f"Output dir:   {cfg.output_dir}")
    print(f"Device:       {device}")
    print(f"T / batch:    {cfg.time_steps} / {cfg.batch_size}")

    loader = build_loader(cfg)
    model = instantiate_model(cfg, device)
    clean_state = load_checkpoint(model, cfg.checkpoint, device)
    modules = target_modules(cfg, model)

    print("Target photonic/MVM modules:")
    for name, module in modules:
        print(f"  {name}: {module.__class__.__name__}")

    model.load_state_dict(clean_state, strict=True)
    full_scales = calibrate_full_scales(model, loader, cfg, device, modules)

    all_results: Dict[str, Any] = {
        "config": json_safe(asdict(cfg)),
        "raw_yaml_config": json_safe(cfg.raw_config),
        "model_import": "from models.snn_vgg import SpikingVGG",
        "target_modules": [name for name, _ in modules],
        "full_scales": full_scales,
        "groups": {},
    }
    flat_rows: List[Dict[str, Any]] = []

    print("\n[0] Clean baseline")
    baseline = run_and_record(
        all_results, flat_rows, "baseline", "clean", model, clean_state, loader, cfg, device,
        modules, full_scales, perturb_kwargs={}, trials=1,
        threshold_fs=cfg.comparator_threshold_fs, hapr_group_size=cfg.hapr_group_size,
    )
    all_results["baseline"] = baseline
    if baseline["accuracy_mean"] < 70:
        print("[WARN] Clean accuracy is low. Check split/T/preprocessing/checkpoint. Try --no-binarize if training used raw counts.")

    group = "photonic_single_factor"
    for pct in [0.0, 1.0, 2.0, 3.0, 5.0]:
        print(f"\n[A] MRR static transmission perturbation {pct:.1f}%")
        run_and_record(
            all_results, flat_rows, group, f"MRR_{pct:.1f}pct", model, clean_state, loader,
            cfg, device, modules, full_scales, {"mrr_pct": pct}, cfg.trials if pct > 0 else 1,
        )

    for pct in [0.0, 1.0, 2.0, 3.0]:
        print(f"\n[A] Laser intensity fluctuation {pct:.1f}%")
        run_and_record(
            all_results, flat_rows, group, f"Laser_{pct:.1f}pct", model, clean_state, loader,
            cfg, device, modules, full_scales, {"laser_pct": pct}, cfg.trials if pct > 0 else 1,
        )

    for db in [-30, -25, -20, -15]:
        print(f"\n[A] WDM adjacent-channel crosstalk {db} dB")
        run_and_record(
            all_results, flat_rows, group, f"WDM_{db}dB", model, clean_state, loader,
            cfg, device, modules, full_scales, {"wdm_db": float(db)}, 1,
        )

    group = "photonic_combined_stress"
    for cname, kwargs in [
        ("nominal", {"mrr_pct": 0.0, "laser_pct": 0.0, "wdm_db": -30.0}),
        ("mild", {"mrr_pct": 1.0, "laser_pct": 1.0, "wdm_db": -30.0}),
        ("moderate", {"mrr_pct": 3.0, "laser_pct": 2.0, "wdm_db": -25.0}),
        ("severe", {"mrr_pct": 5.0, "laser_pct": 3.0, "wdm_db": -20.0}),
        ("very_severe", {"mrr_pct": 5.0, "laser_pct": 3.0, "wdm_db": -15.0}),
    ]:
        print(f"\n[B] Combined photonic stress: {cname}")
        run_and_record(
            all_results, flat_rows, group, cname, model, clean_state, loader, cfg, device,
            modules, full_scales, kwargs, cfg.trials if cname != "nominal" else 1,
        )

    group = "conversion_path"
    for bits in [4, 5, 6, 8]:
        print(f"\n[C] ADC quantization {bits}-bit")
        run_and_record(
            all_results, flat_rows, group, f"ADC_{bits}bit", model, clean_state, loader,
            cfg, device, modules, full_scales, {"adc_bits": bits}, 1,
        )

    for thr in [0.01, 0.02, 0.05, 0.10]:
        print(f"\n[C] Comparator threshold {thr:.2f} full scale")
        run_and_record(
            all_results, flat_rows, group, f"THR_{thr:.2f}FS", model, clean_state, loader,
            cfg, device, modules, full_scales, {}, 1, threshold_fs=thr,
        )

    for g in [4, 8, 16]:
        tia_noise = cfg.hapr_tia_noise_base_fs * math.sqrt(g / max(cfg.reference_hapr_group_size, 1))
        print(f"\n[C] HAPR group size G={g}, optional TIA noise={tia_noise:.3f}% FS")
        run_and_record(
            all_results, flat_rows, group, f"HAPR_G{g}", model, clean_state, loader,
            cfg, device, modules, full_scales,
            {"tia_noise_fs": tia_noise} if tia_noise > 0 else {},
            cfg.trials if tia_noise > 0 else 1,
            hapr_group_size=g,
        )

    group = "tia_disturbance"
    for fs in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print(f"\n[D] HAPR/TIA output disturbance {fs:.1f}% full scale")
        run_and_record(
            all_results, flat_rows, group, f"TIA_{fs:.1f}pctFS", model, clean_state, loader,
            cfg, device, modules, full_scales, {"tia_noise_fs": fs}, cfg.trials if fs > 0 else 1,
        )

    json_path = output_dir / "eval_03_robustness_results.json"
    csv_path = output_dir / "eval_03_robustness_summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    write_csv(csv_path, flat_rows)
    write_readme(output_dir / "README.md", cfg)
    if cfg.save_plots:
        write_plots(output_dir, flat_rows)

    print("\nDone.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Dir:  {output_dir}")


if __name__ == "__main__":
    main()
