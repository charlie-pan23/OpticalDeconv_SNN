"""
HIPSA eval_03: device-calibrated robustness/sensitivity evaluation on CIFAR10-DVS.

One-click usage in PyCharm:
    1) Put this file in the project root, next to config_cifar10dvs.yaml, or keep the
       default working directory as the project root.
    2) Make sure your trained checkpoint exists at ./results/snn_vgg_best.pth.
    3) Run this file. Results are saved under ./results/eval_03_robustness/.

What this script evaluates, aligned with the revised Section 4 plan:
    - Clean baseline accuracy and ADC-request activity proxy.
    - Photonic single-factor sweeps:
        MRR static transmission perturbation: 0, 1, 2, 3, 5 %
        laser intensity fluctuation:        0, 1, 2, 3 %
        WDM adjacent-channel crosstalk:     -30, -25, -20, -15 dB
    - Photonic combined stress cases.
    - Conversion/front-end sweeps:
        ADC precision:                      4, 5, 6, 8 bits
        comparator threshold:               0.01, 0.02, 0.05, 0.10 full scale
        HAPR group size:                    G = 4, 8, 16
        optional HAPR/TIA output noise:      0, 0.5, 1, 2, 3 % full scale

Important modeling choices:
    - MRR perturbation is a static weight-bank perturbation applied once before inference.
    - Laser fluctuation is injected on active input/activation carriers with pre-forward hooks.
    - WDM crosstalk, HAPR/TIA disturbance, and ADC quantization are injected at selected
      Conv/Linear layer outputs, representing photonic accumulation outputs.
    - ADC request activity is measured using a fixed clean-calibrated full scale per layer,
      instead of recomputing full scale after every perturbation.
    - Energy/image is computed with a Section-4-style device-calibrated power model, using
      measured ADC activity proxy and HAPR lane count. It is an architecture-level estimate,
      not a fabricated-chip measurement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import inspect
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    from spikingjelly.datasets import split_to_train_test_set
    from spikingjelly.activation_based import functional as sf
except Exception as exc:  # pragma: no cover - clear runtime error for PyCharm users
    raise ImportError(
        "Failed to import SpikingJelly. Install it in the current PyCharm interpreter, "
        "for example: pip install spikingjelly"
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd().resolve()


@dataclass
class EvalConfig:
    # Paths
    config: str = "config_cifar10dvs.yaml"
    dataset_root: str = "./datasets/CIFAR10DVS"
    checkpoint: str = "./results/snn_vgg_best.pth"
    output_dir: str = "./results/eval_03_robustness"

    # Dataset/model
    batch_size: int = 1
    num_workers: int = 4
    time_steps: int = 10
    split_ratio: float = 0.9
    num_classes: int = 10
    binarize_input: bool = True
    max_batches: Optional[int] = None

    # Runtime
    seed: int = 42
    trials: int = 3
    amp: bool = False
    device: str = "auto"

    # Model import/instantiation
    model_module: str = "models.snn_vgg"
    model_class: str = "SpikingVGG"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Which layers are treated as photonic MVM layers
    include_linear: bool = True
    include_final_linear: bool = False
    target_name_keywords: Optional[List[str]] = None
    exclude_name_keywords: List[str] = field(default_factory=lambda: ["lif", "bn", "batchnorm", "pool"])

    # Calibration and front-end settings
    calibration_batches: int = 8
    comparator_threshold_fs: float = 0.02
    default_adc_bits: int = 6
    hapr_group_size: int = 8
    hapr_tia_noise_base_fs: float = 0.0

    # Section-4-style architecture/device constants
    tiles: int = 4
    tile_outputs: int = 64
    input_mod_lanes: int = 256
    adc_macros: int = 16
    image_latency_ms: float = 0.0717
    reference_adc_activity: float = 0.38
    reference_hapr_group_size: int = 8

    pd_mw: float = 1.1
    tia_mw: float = 3.0
    comparator_mw: float = 2.2
    adc_macro_mw: float = 14.8
    switch_proxy_mw: float = 0.1
    modulator_mw: float = 2.25
    input_spike_activity: float = 0.15
    laser_power_mw: float = 1473.0
    sram_power_mw_ref: float = 243.25
    noc_power_mw_ref: float = 77.84
    lif_power_mw_ref: float = 2.43

    save_plots: bool = True


def _resolve_existing_or_default(path_like: Optional[str], default: Optional[str] = None) -> str:
    """Resolve a path robustly for PyCharm: cwd first, then script directory."""
    raw = path_like if path_like not in [None, ""] else default
    if raw is None:
        return ""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    cwd_p = (CWD / p).resolve()
    if cwd_p.exists():
        return str(cwd_p)
    script_p = (SCRIPT_DIR / p).resolve()
    if script_p.exists():
        return str(script_p)
    # Prefer cwd for new output files and common project layout.
    return str(cwd_p)


def _dataset_root_from_yaml(dataset_cfg: Dict[str, Any]) -> str:
    root = (
        dataset_cfg.get("dataset_root")
        or dataset_cfg.get("root")
        or dataset_cfg.get("root_dir")
        or "./datasets"
    )
    p = Path(str(root))
    if p.name.lower() in {"cifar10dvs", "cifar10-dvs", "cifar10_dvs"}:
        return str(p)
    return str(p / "CIFAR10DVS")


def load_yaml_config(path: str) -> Dict[str, Any]:
    resolved = _resolve_existing_or_default(path)
    if resolved and Path(resolved).exists():
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_config(args: argparse.Namespace) -> EvalConfig:
    cfg_dict = load_yaml_config(args.config)
    dataset_cfg = cfg_dict.get("dataset", {}) or {}
    network_cfg = cfg_dict.get("network", {}) or {}
    model_cfg = cfg_dict.get("model", {}) or {}

    dataset_root_yaml = _dataset_root_from_yaml(dataset_cfg)

    model_module = (
        args.model_module
        or model_cfg.get("module")
        or network_cfg.get("module")
        or "models.snn_vgg"
    )
    model_class = (
        args.model_class
        or model_cfg.get("class")
        or network_cfg.get("class")
        or "SpikingVGG"
    )

    # Only pass simple network/model keys as candidate constructor kwargs.
    raw_model_kwargs: Dict[str, Any] = {}
    for d in [network_cfg, model_cfg]:
        for k, v in d.items():
            if k not in {"module", "class", "name", "type"} and isinstance(v, (int, float, str, bool, list, tuple)):
                raw_model_kwargs[k] = v
    raw_model_kwargs.update(_parse_model_kwargs(args.model_kwargs))

    cfg = EvalConfig(
        config=_resolve_existing_or_default(args.config),
        dataset_root=_resolve_existing_or_default(args.dataset_root, dataset_root_yaml),
        checkpoint=_resolve_existing_or_default(args.checkpoint),
        output_dir=_resolve_existing_or_default(args.output_dir),
        batch_size=args.batch_size if args.batch_size is not None else int(dataset_cfg.get("batch_size_eval", dataset_cfg.get("batch_size", 1))),
        num_workers=args.num_workers if args.num_workers is not None else int(dataset_cfg.get("num_workers", 4)),
        time_steps=args.time_steps if args.time_steps is not None else int(dataset_cfg.get("time_steps", dataset_cfg.get("T", 10))),
        split_ratio=args.split_ratio if args.split_ratio is not None else float(dataset_cfg.get("split_ratio", 0.9)),
        num_classes=int(dataset_cfg.get("num_classes", network_cfg.get("num_classes", 10))),
        binarize_input=not args.no_binarize,
        max_batches=args.max_batches,
        seed=args.seed,
        trials=args.trials,
        amp=args.amp,
        device=args.device,
        model_module=model_module,
        model_class=model_class,
        model_kwargs=raw_model_kwargs,
        include_linear=not args.conv_only,
        include_final_linear=args.include_final_linear,
        target_name_keywords=_parse_csv_list(args.target_keywords),
        exclude_name_keywords=_parse_csv_list(args.exclude_keywords) or ["lif", "bn", "batchnorm", "pool"],
        calibration_batches=args.calibration_batches,
        comparator_threshold_fs=args.comparator_threshold_fs,
        default_adc_bits=args.default_adc_bits,
        hapr_group_size=args.hapr_group_size,
        hapr_tia_noise_base_fs=args.hapr_tia_noise_base_fs,
        save_plots=not args.no_plots,
    )
    return cfg


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def _parse_model_kwargs(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as exc:
        raise ValueError("--model-kwargs must be a JSON object, e.g. '{\"channels\":128}'") from exc
    raise ValueError("--model-kwargs must be a JSON object")


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
# Dataset and model
# =============================================================================


def build_loader(cfg: EvalConfig) -> DataLoader:
    if not Path(cfg.dataset_root).exists():
        raise FileNotFoundError(
            f"CIFAR10-DVS dataset root not found: {cfg.dataset_root}\n"
            "Please edit --dataset-root or dataset.root_dir in config_cifar10dvs.yaml."
        )

    full_dataset = CIFAR10DVS(
        root=cfg.dataset_root,
        data_type="frame",
        frames_number=cfg.time_steps,
        split_by="number",
    )

    set_seed(cfg.seed)
    try:
        _, test_ds = split_to_train_test_set(
            cfg.split_ratio,
            full_dataset,
            num_classes=cfg.num_classes,
        )
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


def import_model_class(cfg: EvalConfig) -> type:
    # Make both project root and script directory importable. This fixes the common
    # PyCharm issue where from models.snn_vgg works in terminal but not in the IDE.
    for p in [str(CWD), str(SCRIPT_DIR), str(SCRIPT_DIR.parent)]:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    tried: List[str] = []
    candidate_modules = [cfg.model_module]
    if cfg.model_module != "snn_vgg":
        candidate_modules.append("snn_vgg")
    if cfg.model_module != "models.snn_vgg":
        candidate_modules.append("models.snn_vgg")

    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, cfg.model_class)
            return cls
        except Exception as exc:
            tried.append(f"{module_name}.{cfg.model_class}: {exc}")

    raise ImportError(
        "Cannot import the SNN model class. Tried:\n  " + "\n  ".join(tried) +
        "\nUse --model-module and --model-class if your file/class name differs."
    )


def instantiate_model(cfg: EvalConfig, device: torch.device) -> nn.Module:
    cls = import_model_class(cfg)
    kwargs = dict(cfg.model_kwargs)

    # Common constructor aliases. They are filtered by inspect.signature below.
    kwargs.setdefault("num_classes", cfg.num_classes)
    kwargs.setdefault("T", cfg.time_steps)
    kwargs.setdefault("time_steps", cfg.time_steps)

    sig = inspect.signature(cls)
    accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_var_kw:
        valid = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in valid}

    try:
        model = cls(**kwargs)
    except TypeError as exc:
        # Minimal fallback for older SpikingVGG definitions.
        try:
            model = cls(num_classes=cfg.num_classes)
        except TypeError:
            try:
                model = cls()
            except TypeError as exc2:
                raise TypeError(
                    f"Failed to instantiate {cfg.model_class}. Constructor kwargs tried: {kwargs}. "
                    "Pass explicit JSON via --model-kwargs if needed."
                ) from exc2

    return model.to(device)


def _torch_load(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please set --checkpoint to your trained model checkpoint."
        )

    ckpt = _torch_load(checkpoint_path, device)
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "net", "network", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if not torch.is_tensor(v):
            continue
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {len(missing)} keys. First few: {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)} keys. First few: {unexpected[:8]}")

    # Store a clean state that can be restored before every trial.
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def target_modules(cfg: EvalConfig, model: nn.Module) -> List[Tuple[str, nn.Module]]:
    convs: List[Tuple[str, nn.Module]] = []
    linears: List[Tuple[str, nn.Module]] = []
    target_keywords = [s.lower() for s in cfg.target_name_keywords] if cfg.target_name_keywords else None
    exclude_keywords = [s.lower() for s in cfg.exclude_name_keywords]

    for name, module in model.named_modules():
        lname = name.lower()
        if target_keywords and not any(k in lname for k in target_keywords):
            continue
        if any(k in lname for k in exclude_keywords):
            continue
        if isinstance(module, nn.Conv2d):
            convs.append((name, module))
        elif cfg.include_linear and isinstance(module, nn.Linear):
            linears.append((name, module))

    selected_linears = list(linears)
    if selected_linears and not cfg.include_final_linear:
        # The final classifier/readout is often digital. Excluding the last Linear by
        # default avoids overstating photonic output sensitivity.
        selected_linears = selected_linears[:-1]

    targets = convs + selected_linears
    if not targets:
        raise RuntimeError(
            "No target Conv2d/Linear modules were found. Use --target-keywords or "
            "--include-final-linear if your model naming differs."
        )
    return targets


# =============================================================================
# Batch/model-output utilities
# =============================================================================


def prepare_batch(data: torch.Tensor, cfg: EvalConfig, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.dim() != 5:
        raise ValueError(
            f"Expected CIFAR10-DVS frames with shape [B,T,C,H,W] or [T,B,C,H,W], got {tuple(data.shape)}"
        )

    # SpikingJelly usually returns [B,T,C,H,W]. Most SNN models in this project use [T,B,C,H,W].
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


def aggregate_logits(output: Any, batch_size: int, cfg: EvalConfig) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        # Prefer the last tensor because some models return (spikes, logits) or similar.
        tensors = [x for x in output if torch.is_tensor(x)]
        if not tensors:
            raise TypeError("Model output tuple/list contains no tensor")
        output = tensors[-1]

    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)}")

    if output.dim() == 2:
        return output

    if output.dim() == 3:
        # [T,B,C]
        if output.shape[0] == cfg.time_steps and output.shape[1] == batch_size:
            return output.mean(dim=0)
        # [B,T,C]
        if output.shape[0] == batch_size and output.shape[1] == cfg.time_steps:
            return output.mean(dim=1)
        # Fallback: average the first dimension.
        return output.mean(dim=0)

    if output.dim() > 3:
        # Last dimension should be classes after flattening all temporal/spatial axes.
        if output.shape[0] == cfg.time_steps and output.shape[1] == batch_size:
            return output.flatten(start_dim=2).mean(dim=0)
        if output.shape[0] == batch_size:
            return output.flatten(start_dim=1)

    raise ValueError(f"Cannot aggregate model output with shape={tuple(output.shape)}")


def reset_snn_state(model: nn.Module) -> None:
    try:
        sf.reset_net(model)
    except Exception:
        # Some non-SpikingJelly models may not expose resettable states.
        pass


# =============================================================================
# Calibration, activity collection, perturbation injection
# =============================================================================


def symmetric_quantize_fixed_fs(x: torch.Tensor, bits: int, full_scale: float, eps: float = 1e-12) -> torch.Tensor:
    if bits <= 0 or full_scale <= eps:
        return x
    qmax = (2 ** (bits - 1)) - 1
    if qmax <= 0:
        return x
    fs_tensor = torch.as_tensor(float(full_scale), dtype=x.dtype, device=x.device)
    y = torch.clamp(x / fs_tensor, -1.0, 1.0)
    yq = torch.round(y * qmax) / qmax
    return yq * fs_tensor


class FullScaleCalibrator:
    """Collect fixed clean full scale per target layer for ADC/threshold simulation."""

    def __init__(self, modules: Sequence[Tuple[str, nn.Module]], percentile: Optional[float] = None):
        self.modules = list(modules)
        self.percentile = percentile
        self.handles: List[Any] = []
        self.max_abs: Dict[str, float] = {name: 0.0 for name, _ in self.modules}
        self.samples: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in self.modules}

    def __enter__(self) -> "FullScaleCalibrator":
        for name, module in self.modules:
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if not torch.is_tensor(output) or output.numel() == 0:
                return None
            with torch.no_grad():
                y = output.detach().abs()
                self.max_abs[name] = max(self.max_abs[name], float(y.amax().item()))
                if self.percentile is not None:
                    # Store a small subsample to avoid huge memory use.
                    flat = y.flatten()
                    if flat.numel() > 4096:
                        idx = torch.linspace(0, flat.numel() - 1, 4096, device=flat.device).long()
                        flat = flat[idx]
                    self.samples[name].append(flat.cpu())
            return None
        return hook

    def full_scales(self) -> Dict[str, float]:
        if self.percentile is None:
            return {k: max(v, 1e-12) for k, v in self.max_abs.items()}
        out: Dict[str, float] = {}
        for name in self.max_abs:
            if self.samples[name]:
                values = torch.cat(self.samples[name])
                q = torch.quantile(values, float(self.percentile) / 100.0).item()
                out[name] = max(float(q), 1e-12)
            else:
                out[name] = max(self.max_abs[name], 1e-12)
        return out


class ActivityCollector:
    """
    ADC-request proxy:
        request = |photonic_output| > comparator_threshold_fs * clean_layer_full_scale

    The key improvement over the previous version is that full scale is fixed from clean
    calibration. Otherwise, the threshold would move together with each perturbation and
    hide real comparator/ADC sensitivity.
    """

    def __init__(
        self,
        modules: Sequence[Tuple[str, nn.Module]],
        full_scales: Dict[str, float],
        threshold_fs: float = 0.02,
    ):
        self.modules = list(modules)
        self.full_scales = full_scales
        self.threshold_fs = threshold_fs
        self.handles: List[Any] = []
        self.active_count = 0
        self.total_count = 0
        self.layer_active: Dict[str, int] = {name: 0 for name, _ in self.modules}
        self.layer_total: Dict[str, int] = {name: 0 for name, _ in self.modules}

    def __enter__(self) -> "ActivityCollector":
        for name, module in self.modules:
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if not torch.is_tensor(output) or output.numel() == 0:
                return None
            with torch.no_grad():
                fs = float(self.full_scales.get(name, 0.0))
                if fs <= 0:
                    fs = float(output.detach().abs().amax().item())
                thresh = self.threshold_fs * max(fs, 1e-12)
                active = int((output.detach().abs() > thresh).sum().item())
                total = int(output.numel())
                self.active_count += active
                self.total_count += total
                self.layer_active[name] += active
                self.layer_total[name] += total
            return None
        return hook

    def summary(self) -> Dict[str, Any]:
        ratio = self.active_count / max(self.total_count, 1)
        layer_ratio = {
            k: self.layer_active[k] / max(self.layer_total[k], 1)
            for k in self.layer_total
            if self.layer_total[k] > 0
        }
        return {
            "adc_activity_proxy": float(ratio),
            "adc_active_count": float(self.active_count),
            "adc_total_count": float(self.total_count),
            "adc_activity_by_layer": layer_ratio,
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
                self.handles.append(module.register_forward_pre_hook(self._make_laser_pre_hook(name)))
            if self.wdm_db is not None or self.tia_noise_fs > 0 or self.adc_bits is not None:
                self.handles.append(module.register_forward_hook(self._make_output_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _apply_static_mrr_weight_error(self, pct: float) -> None:
        ratio = pct / 100.0
        with torch.no_grad():
            for name, module in self.modules:
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                eps = torch.empty_like(weight).uniform_(-ratio, ratio)
                weight.mul_(1.0 + eps)

    def _make_laser_pre_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if not inputs or not torch.is_tensor(inputs[0]):
                return inputs
            x = inputs[0]
            ratio = self.laser_pct / 100.0
            if ratio <= 0 or x.numel() == 0:
                return inputs

            # Broadcast multiplicative fluctuation over spatial dimensions. Zero input remains zero.
            if x.dim() == 5:      # [T,B,C,H,W] or similar
                shape = (x.shape[0], x.shape[1], x.shape[2], 1, 1)
            elif x.dim() == 4:    # [B,C,H,W]
                shape = (x.shape[0], x.shape[1], 1, 1)
            elif x.dim() == 3:    # [T/B, B/T, C] or [B,N,features]
                shape = (x.shape[0], x.shape[1], 1)
            elif x.dim() == 2:    # [B,features]
                shape = (x.shape[0], 1)
            else:
                shape = tuple([x.shape[0]] + [1] * (x.dim() - 1))

            scale = 1.0 + torch.randn(shape, device=x.device, dtype=x.dtype) * ratio
            scale = torch.clamp(scale, min=0.0)
            return (x * scale, *inputs[1:])
        return hook

    def _make_output_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if not torch.is_tensor(output):
                return output
            y = output

            if self.wdm_db is not None:
                # dB is interpreted as optical-power / photocurrent leakage ratio.
                leak = 10.0 ** (float(self.wdm_db) / 10.0)
                y = adjacent_channel_crosstalk(y, leak)

            if self.tia_noise_fs > 0:
                fs = float(self.full_scales.get(name, 0.0))
                if fs > 0:
                    std = (self.tia_noise_fs / 100.0) * fs
                    y = y + torch.randn_like(y) * std

            if self.adc_bits is not None:
                fs = float(self.full_scales.get(name, 0.0))
                y = symmetric_quantize_fixed_fs(y, int(self.adc_bits), fs)

            return y
        return hook


def adjacent_channel_crosstalk(y: torch.Tensor, leak: float) -> torch.Tensor:
    if leak <= 0:
        return y

    if y.dim() == 5 and y.shape[2] >= 2:
        ch_dim = 2
    elif y.dim() >= 2 and y.shape[1] >= 2:
        ch_dim = 1
    else:
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

    src_prev = [slice(None)] * y.dim()
    dst_next = [slice(None)] * y.dim()
    src_prev[ch_dim] = slice(0, n_ch - 1)
    dst_next[ch_dim] = slice(1, n_ch)
    left[tuple(dst_next)] = y[tuple(src_prev)]

    src_next = [slice(None)] * y.dim()
    dst_prev = [slice(None)] * y.dim()
    src_next[ch_dim] = slice(1, n_ch)
    dst_prev[ch_dim] = slice(0, n_ch - 1)
    right[tuple(dst_prev)] = y[tuple(src_next)]

    return mixed + leak * (left + right)


# =============================================================================
# Device-calibrated power model
# =============================================================================


def hapr_lanes(cfg: EvalConfig, group_size: int) -> int:
    if cfg.tile_outputs % group_size != 0:
        raise ValueError(f"tile_outputs={cfg.tile_outputs} must be divisible by HAPR group size G={group_size}")
    return cfg.tiles * (cfg.tile_outputs // group_size)


def estimate_power_energy(
    cfg: EvalConfig,
    adc_activity_proxy: float,
    group_size: int,
) -> Dict[str, float]:
    alpha = float(np.clip(adc_activity_proxy, 0.0, 1.0))
    n_pd = cfg.tiles * cfg.tile_outputs
    n_hapr = hapr_lanes(cfg, group_size)
    n_ref = hapr_lanes(cfg, cfg.reference_hapr_group_size)

    demand_per_cycle = n_hapr * alpha
    adc_macro_util = min(1.0, demand_per_cycle / max(cfg.adc_macros, 1))

    pd_power = n_pd * cfg.pd_mw
    tia_power = n_hapr * cfg.tia_mw
    comp_power = n_hapr * cfg.comparator_mw
    adc_power = cfg.adc_macros * cfg.adc_macro_mw * adc_macro_util
    selection_power = n_hapr * cfg.switch_proxy_mw
    mod_power = cfg.input_mod_lanes * cfg.modulator_mw * cfg.input_spike_activity
    laser_power = cfg.laser_power_mw

    # SRAM/NoC/LIF scale with number of ADC-requested membrane updates relative to the Section-4 main point.
    ref_demand = max(n_ref * cfg.reference_adc_activity, 1e-12)
    update_scale = demand_per_cycle / ref_demand
    sram_power = cfg.sram_power_mw_ref * update_scale
    noc_power = cfg.noc_power_mw_ref * update_scale
    lif_power = cfg.lif_power_mw_ref * update_scale

    total_mw = (
        pd_power + tia_power + comp_power + adc_power + selection_power + mod_power + laser_power
        + sram_power + noc_power + lif_power
    )
    power_w = total_mw / 1000.0
    energy_uJ = power_w * cfg.image_latency_ms * 1000.0

    return {
        "hapr_group_size": float(group_size),
        "hapr_lanes": float(n_hapr),
        "adc_demand_per_cycle": float(demand_per_cycle),
        "adc_macro_utilization": float(adc_macro_util),
        "pd_power_mw": float(pd_power),
        "tia_power_mw": float(tia_power),
        "comparator_power_mw": float(comp_power),
        "adc_power_mw": float(adc_power),
        "selection_power_mw": float(selection_power),
        "modulator_power_mw": float(mod_power),
        "laser_power_mw": float(laser_power),
        "sram_power_mw": float(sram_power),
        "noc_power_mw": float(noc_power),
        "lif_power_mw": float(lif_power),
        "total_power_mw": float(total_mw),
        "power_w": float(power_w),
        "energy_uJ_per_image": float(energy_uJ),
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
    print(f"\n[calibration] Collecting clean full scale from {cfg.calibration_batches} batch(es)")
    with FullScaleCalibrator(modules) as cal:
        for batch_idx, (data, targets) in enumerate(tqdm(loader, desc="Calibrating", leave=False)):
            if batch_idx >= cfg.calibration_batches:
                break
            reset_snn_state(model)
            data = prepare_batch(data, cfg, device)
            _ = model(data)
    reset_snn_state(model)
    fs = cal.full_scales()
    print("[calibration] Full-scale summary:")
    for name, value in fs.items():
        print(f"  {name}: {value:.6g}")
    return fs


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
        targets = targets.to(device, non_blocking=True)
        if targets.dim() > 1:
            targets = targets.view(-1)

        input_active += int((data > 0).sum().item())
        input_total += int(data.numel())

        with amp_ctx:
            out = model(data)
            logits = aggregate_logits(out, batch_size=targets.numel(), cfg=cfg)

        pred = logits.argmax(dim=1)
        total += int(targets.numel())
        correct += int((pred == targets).sum().item())

    reset_snn_state(model)
    acc = 100.0 * correct / max(total, 1)
    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
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
    trials = max(int(trials), 1)

    for t in range(trials):
        model.load_state_dict(clean_state, strict=True)
        reset_snn_state(model)
        seed = cfg.seed + 1009 * (t + 1)
        with PerturbationContext(modules, full_scales, seed=seed, **perturb_kwargs):
            with ActivityCollector(modules, full_scales, threshold_fs=threshold_fs) as collector:
                metrics = run_evaluation(model, loader, cfg, device, collector=collector)

        power = estimate_power_energy(cfg, metrics.get("adc_activity_proxy", 0.0), hapr_group_size)
        metrics.update(power)
        trial_metrics.append(metrics)

    def arr(key: str) -> np.ndarray:
        return np.array([m.get(key, np.nan) for m in trial_metrics], dtype=np.float64)

    return {
        "name": name,
        "perturbation": perturb_kwargs,
        "threshold_fs": float(threshold_fs),
        "hapr_group_size": int(hapr_group_size),
        "num_trials": int(trials),
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
# Output helpers
# =============================================================================


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "group",
        "condition",
        "accuracy_mean",
        "accuracy_std",
        "adc_activity_proxy_mean",
        "adc_activity_proxy_std",
        "input_spike_activity_mean",
        "threshold_fs",
        "hapr_group_size",
        "hapr_lanes",
        "adc_demand_per_cycle_mean",
        "adc_macro_utilization_mean",
        "power_w_mean",
        "power_w_std",
        "energy_uJ_per_image_mean",
        "energy_uJ_per_image_std",
        "num_trials",
        "perturbation",
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
        print("[WARN] matplotlib is unavailable. Skipping plots.")
        return

    def select(group: str) -> List[Dict[str, Any]]:
        return [r for r in rows if r.get("group") == group]

    def x_from_condition(row: Dict[str, Any]) -> str:
        return str(row.get("name", ""))

    for group in sorted(set(str(r.get("group", "")) for r in rows)):
        group_rows = select(group)
        if not group_rows:
            continue
        x = list(range(len(group_rows)))
        labels = [x_from_condition(r) for r in group_rows]
        y = [r.get("accuracy_mean", np.nan) for r in group_rows]
        yerr = [r.get("accuracy_std", 0.0) for r in group_rows]

        plt.figure(figsize=(max(7, len(labels) * 0.8), 4.2))
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Accuracy (%)")
        plt.title(group)
        plt.tight_layout()
        plt.savefig(output_dir / f"{group}_accuracy.png", dpi=200)
        plt.close()

        y2 = [r.get("energy_uJ_per_image_mean", np.nan) for r in group_rows]
        plt.figure(figsize=(max(7, len(labels) * 0.8), 4.2))
        plt.plot(x, y2, marker="o")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Energy per image (uJ)")
        plt.title(f"{group} energy")
        plt.tight_layout()
        plt.savefig(output_dir / f"{group}_energy.png", dpi=200)
        plt.close()


def write_readme(path: Path, cfg: EvalConfig) -> None:
    text = f"""# HIPSA eval_03 robustness results

This directory was generated by `eval_03_robustness_fixed.py`.

Main files:
- `eval_03_robustness_results.json`: full per-trial and per-condition results.
- `eval_03_robustness_summary.csv`: compact table for paper figures.
- `*_accuracy.png` and `*_energy.png`: quick-look plots, if matplotlib is installed.

Interpretation notes:
- Accuracy is measured from the PyTorch/SpikingJelly SNN model.
- `adc_activity_proxy` is the fraction of selected photonic-layer outputs above the comparator threshold.
- Full scale is calibrated from the clean model using {cfg.calibration_batches} batch(es).
- Energy/image is computed from the device-calibrated Section-4-style model using the measured ADC activity proxy.
- HAPR group-size sweep changes HAPR/TIA/comparator lane count and ADC demand in the power model; it does not change CNN channel semantics.
- This is an architecture-level hardware-aware inference simulation, not fabricated-chip measurement.
"""
    path.write_text(text, encoding="utf-8")


# =============================================================================
# Main sweep
# =============================================================================


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
    res = evaluate_condition(
        condition_name,
        model,
        clean_state,
        loader,
        cfg,
        device,
        modules,
        full_scales,
        perturb_kwargs=perturb_kwargs,
        trials=trials,
        threshold_fs=threshold_fs,
        hapr_group_size=hapr_group_size,
    )
    res["group"] = group_name
    all_results.setdefault("groups", {}).setdefault(group_name, []).append(res)
    flat_rows.append(res)
    print(
        f"  acc={res['accuracy_mean']:.2f}±{res['accuracy_std']:.2f}% | "
        f"ADC_act={res['adc_activity_proxy_mean']:.4f} | "
        f"P={res['power_w_mean']:.3f} W | E={res['energy_uJ_per_image_mean']:.1f} uJ"
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="HIPSA eval_03 device-calibrated robustness evaluation")
    parser.add_argument("--config", type=str, default="config_cifar10dvs.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="./results/snn_vgg_best.pth")
    parser.add_argument("--output-dir", type=str, default="./results/eval_03_robustness")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--split-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only: evaluate at most this many batches.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA. Disabled by default for stable robustness numbers.")
    parser.add_argument("--no-binarize", action="store_true", help="Use this if training used raw event counts instead of binary frames.")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--model-module", type=str, default=None, help="e.g. models.snn_vgg or snn_vgg")
    parser.add_argument("--model-class", type=str, default=None, help="e.g. SpikingVGG")
    parser.add_argument("--model-kwargs", type=str, default=None, help="JSON dict passed to model constructor.")
    parser.add_argument("--conv-only", action="store_true", help="Perturb only Conv2d layers.")
    parser.add_argument("--include-final-linear", action="store_true", help="Also perturb the final Linear classifier/readout.")
    parser.add_argument("--target-keywords", type=str, default=None, help="Comma-separated layer-name keywords to include.")
    parser.add_argument("--exclude-keywords", type=str, default="lif,bn,batchnorm,pool", help="Comma-separated layer-name keywords to exclude.")

    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--comparator-threshold-fs", type=float, default=0.02)
    parser.add_argument("--default-adc-bits", type=int, default=6)
    parser.add_argument("--hapr-group-size", type=int, default=8)
    parser.add_argument("--hapr-tia-noise-base-fs", type=float, default=0.0,
                        help="Optional TIA noise scaling used in the HAPR G sweep. 0 keeps accuracy unchanged by G.")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)
    device = choose_device(cfg.device)

    print("=== HIPSA eval_03: device-calibrated robustness ===")
    print(f"Working directory: {CWD}")
    print(f"Script directory:  {SCRIPT_DIR}")
    print(f"Device:            {device}")
    print(f"Config:            {json.dumps(json_safe(asdict(cfg)), ensure_ascii=False, indent=2)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(cfg)
    model = instantiate_model(cfg, device)
    clean_state = load_checkpoint(model, cfg.checkpoint, device)
    modules = target_modules(cfg, model)

    print("\nTarget photonic/MVM modules:")
    for name, module in modules:
        print(f"  {name}: {module.__class__.__name__}")

    # Calibrate full scale from the clean checkpoint.
    model.load_state_dict(clean_state, strict=True)
    full_scales = calibrate_full_scales(model, loader, cfg, device, modules)

    all_results: Dict[str, Any] = {
        "config": json_safe(asdict(cfg)),
        "target_modules": [name for name, _ in modules],
        "full_scales": full_scales,
        "notes": {
            "dataset_split": "CIFAR10-DVS has no official train/test split. Use the same split/indices as training for final paper numbers.",
            "full_scale": "ADC quantization and comparator thresholds use clean calibrated layer full scales.",
            "power_model": "Energy/image is computed from Section-4-style device-calibrated power model using measured ADC activity proxy.",
            "scope": "Architecture-level hardware-aware inference simulation, not fabricated-chip measurement.",
        },
        "groups": {},
    }
    flat_rows: List[Dict[str, Any]] = []

    print("\n[0] Clean baseline")
    baseline = run_and_record(
        all_results,
        flat_rows,
        "baseline",
        "clean",
        model,
        clean_state,
        loader,
        cfg,
        device,
        modules,
        full_scales,
        perturb_kwargs={},
        trials=1,
        threshold_fs=cfg.comparator_threshold_fs,
        hapr_group_size=cfg.hapr_group_size,
    )
    all_results["baseline"] = baseline
    if baseline["accuracy_mean"] < 70:
        print(
            "[WARN] Clean accuracy is low. First check checkpoint/model/T/preprocessing/split. "
            "Try --no-binarize if the model was trained on raw event counts."
        )

    # A. Photonic single-factor sweeps.
    group = "photonic_single_factor"
    for pct in [0.0, 1.0, 2.0, 3.0, 5.0]:
        print(f"\n[A] MRR static transmission perturbation {pct:.1f}%")
        run_and_record(
            all_results, flat_rows, group, f"MRR_{pct:.1f}pct",
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs={"mrr_pct": pct},
            trials=cfg.trials if pct > 0 else 1,
        )

    for pct in [0.0, 1.0, 2.0, 3.0]:
        print(f"\n[A] Laser intensity fluctuation {pct:.1f}%")
        run_and_record(
            all_results, flat_rows, group, f"Laser_{pct:.1f}pct",
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs={"laser_pct": pct},
            trials=cfg.trials if pct > 0 else 1,
        )

    for db in [-30, -25, -20, -15]:
        print(f"\n[A] WDM adjacent-channel crosstalk {db} dB")
        run_and_record(
            all_results, flat_rows, group, f"WDM_{db}dB",
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs={"wdm_db": float(db)},
            trials=1,
        )

    # B. Combined photonic stress cases.
    group = "photonic_combined_stress"
    combined = [
        ("nominal", {"mrr_pct": 0.0, "laser_pct": 0.0, "wdm_db": -30.0}),
        ("mild", {"mrr_pct": 1.0, "laser_pct": 1.0, "wdm_db": -30.0}),
        ("moderate", {"mrr_pct": 3.0, "laser_pct": 2.0, "wdm_db": -25.0}),
        ("severe", {"mrr_pct": 5.0, "laser_pct": 3.0, "wdm_db": -20.0}),
        ("very_severe", {"mrr_pct": 5.0, "laser_pct": 3.0, "wdm_db": -15.0}),
    ]
    for cname, kwargs in combined:
        print(f"\n[B] Combined photonic stress: {cname} {kwargs}")
        run_and_record(
            all_results, flat_rows, group, cname,
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs=kwargs,
            trials=cfg.trials if cname != "nominal" else 1,
        )

    # C. Conversion/request front-end sweeps.
    group = "conversion_path"
    for bits in [4, 5, 6, 8]:
        print(f"\n[C] ADC quantization {bits}-bit")
        run_and_record(
            all_results, flat_rows, group, f"ADC_{bits}bit",
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs={"adc_bits": bits},
            trials=1,
        )

    for thr in [0.01, 0.02, 0.05, 0.10]:
        print(f"\n[C] Comparator threshold {thr:.2f} full scale")
        run_and_record(
            all_results, flat_rows, group, f"THR_{thr:.2f}FS",
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs={},
            trials=1,
            threshold_fs=thr,
        )

    for g in [4, 8, 16]:
        tia_noise = cfg.hapr_tia_noise_base_fs * math.sqrt(g / max(cfg.reference_hapr_group_size, 1))
        print(f"\n[C] HAPR group size G={g}, optional TIA noise={tia_noise:.3f}% FS")
        run_and_record(
            all_results, flat_rows, group, f"HAPR_G{g}",
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs={"tia_noise_fs": tia_noise} if tia_noise > 0 else {},
            trials=cfg.trials if tia_noise > 0 else 1,
            threshold_fs=cfg.comparator_threshold_fs,
            hapr_group_size=g,
        )

    # Extra TIA/HAPR disturbance sweep. This is useful for the figure supplement and for reviewers.
    group = "tia_disturbance"
    for fs in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print(f"\n[D] HAPR/TIA output disturbance {fs:.1f}% full scale")
        run_and_record(
            all_results, flat_rows, group, f"TIA_{fs:.1f}pctFS",
            model, clean_state, loader, cfg, device, modules, full_scales,
            perturb_kwargs={"tia_noise_fs": fs},
            trials=cfg.trials if fs > 0 else 1,
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
