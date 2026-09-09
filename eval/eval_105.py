"""Official HIPSA publication replay (eval_105).

This script replays a frozen checkpoint and split under three paired conditions:
clean FP32, W6/ADC6/Vm16 quantized-clean, and the declared combined hardware
proxy.  It records provenance and never relabels architecture estimates as
silicon measurements.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from hardware.activity_trace import default_photonic_layer_names
from hardware.membrane_fixedpoint import quantize_membrane
from hardware.weight_quant import quantize_symmetric
from utils.checkpoint_utils import checkpoint_summary
from utils.config_utils import copy_config_snapshot, dataset_tag, load_eval_config, save_json
from utils.eval_v10 import canonical_sha256, file_sha256, validate_eval105_publication_summary, validate_g4_a8_contract
from utils.result_io import save_csv_rows, save_run_manifest
from utils.seed_utils import set_seed


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_output_dir(output_root: str | Path, dataset: str, seed: int, seed_indexed: bool) -> Path:
    """Return the eval_105 output directory without cross-seed overwrites."""
    base = Path(output_root) / dataset / "eval_105"
    return base / "seeds" / f"seed_{int(seed):03d}" if seed_indexed else base


def configure_publication_runtime(args: argparse.Namespace) -> None:
    """Configure deterministic CUDA execution before a CUDA context is created."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(args.cublas_workspace_config)


def build_runtime_provenance(
    args: argparse.Namespace,
    *,
    resolved_device: Optional[str],
    actual_batch_size: Optional[int],
) -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name: Optional[str] = None
    gpu_capability: Optional[List[int]] = None
    if cuda_available and resolved_device and str(resolved_device).startswith("cuda"):
        parsed = torch.device(resolved_device)
        index = parsed.index if parsed.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(index)
        gpu_capability = list(torch.cuda.get_device_capability(index))
    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    full_replay_requested = args.max_batches is None and args.calibration_batches is None
    deterministic_ready = bool(
        torch.are_deterministic_algorithms_enabled()
        and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark
        and cublas_config in {":4096:8", ":16:8"}
    )
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "device_requested": args.device,
        "device_resolved": resolved_device,
        "gpu_name": gpu_name,
        "gpu_compute_capability": gpu_capability,
        "batch_size_requested": args.batch_size,
        "batch_size_actual": actual_batch_size,
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "seed_type": "paired_replay_perturbation_seed",
        "seed_output_mode": "seed_indexed" if bool(args.seed_indexed_output) else "legacy_single_output",
        "cublas_workspace_config": cublas_config,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "full_replay_requested": full_replay_requested,
        "publication_replay_ready": bool(deterministic_ready and full_replay_requested),
    }


def build_confusion_matrix(targets: Iterable[int], predictions: Iterable[int], classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def save_confusion(matrix: np.ndarray, path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for target in range(matrix.shape[0]):
        row: Dict[str, Any] = {"true_class": target}
        for prediction in range(matrix.shape[1]):
            row[f"pred_{prediction}"] = int(matrix[target, prediction])
        row["total"] = int(matrix[target].sum())
        row["correct"] = int(matrix[target, target])
        row["accuracy_percent"] = 100.0 * row["correct"] / row["total"] if row["total"] else None
        rows.append(row)
    save_csv_rows(rows, path)


def _channel_axis(tensor: torch.Tensor) -> int:
    return 2 if tensor.dim() >= 5 else (1 if tensor.dim() >= 2 else 0)


def _channel_mask_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    shape = [1] * tensor.dim()
    shape[_channel_axis(tensor)] = tensor.shape[_channel_axis(tensor)]
    return tuple(shape)


def _adc_quantize(value: torch.Tensor, bits: int, full_scale: float) -> torch.Tensor:
    """Quantize with a frozen validation-calibrated ADC full scale.

    The reporting/test batch is never used to choose its own range. Values beyond
    the calibrated range saturate, matching a fixed-range ADC rather than an
    optimistic per-batch oracle quantizer.
    """
    if value.numel() == 0:
        return value
    levels = max(2 ** int(bits) - 1, 1)
    eps = torch.finfo(value.dtype).eps if value.dtype.is_floating_point else 1.0e-12
    scale = torch.as_tensor(max(float(full_scale), float(eps)), device=value.device, dtype=value.dtype)
    normalized = torch.clamp(value / scale, -1.0, 1.0)
    code = torch.round((normalized + 1.0) * 0.5 * levels)
    return (code / levels * 2.0 - 1.0) * scale


def _wdm_crosstalk(value: torch.Tensor, crosstalk_db: float) -> torch.Tensor:
    if value.dim() < 2 or value.shape[_channel_axis(value)] <= 1:
        return value
    alpha = 10.0 ** (float(crosstalk_db) / 10.0)
    axis = _channel_axis(value)
    left = torch.zeros_like(value)
    right = torch.zeros_like(value)
    left_slices = [slice(None)] * value.dim()
    source_slices = [slice(None)] * value.dim()
    left_slices[axis] = slice(1, None)
    source_slices[axis] = slice(None, -1)
    left[tuple(left_slices)] = value[tuple(source_slices)]
    right_slices = [slice(None)] * value.dim()
    source_slices = [slice(None)] * value.dim()
    right_slices[axis] = slice(None, -1)
    source_slices[axis] = slice(1, None)
    right[tuple(right_slices)] = value[tuple(source_slices)]
    return value + alpha * (left + right)


@dataclass(frozen=True)
class ReplayCondition:
    name: str
    weight_bits: Optional[int]
    adc_bits: Optional[int]
    membrane_bits: Optional[int]
    membrane_fractional_bits: int = 8
    mrr_sigma: float = 0.0
    laser_sigma: float = 0.0
    wdm_crosstalk_db: Optional[float] = None
    tia_noise_sigma: float = 0.0


MVM_MODULE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


def resolve_photonic_mvm_layer_names(
    model: nn.Module, *, include_final_classifier: bool = False,
) -> List[str]:
    """Resolve and validate layers mapped to the optical accumulation plane.

    The model-owned photonic_mvm_layer_names contract is authoritative when
    available. The final electronic classifier is excluded unless explicitly
    requested; publication replay keeps it in FP32.
    """
    names = default_photonic_layer_names(model, include_fc2=include_final_classifier)
    modules = dict(model.named_modules())
    missing = [name for name in names if name not in modules]
    unsupported = [
        name for name in names
        if name in modules and not isinstance(modules[name], MVM_MODULE_TYPES)
    ]
    if missing or unsupported:
        raise ValueError(
            "Invalid photonic MVM layer contract: "
            f"missing={missing}, unsupported={unsupported}."
        )
    if not names:
        raise ValueError("The model exposes no photonic MVM layers for replay.")
    if not include_final_classifier and "fc2" in names:
        raise ValueError("Electronic final classifier fc2 must not be in publication replay.")
    return names


def build_replay_scope(model: nn.Module, target_layer_names: Iterable[str]) -> Dict[str, Any]:
    target_names = list(target_layer_names)
    modules = dict(model.named_modules())
    electronic_linear_layers = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name not in target_names
    ]
    return {
        "target_policy": "model_photonic_mvm_layer_names_with_safe_fallback",
        "photonic_mvm_layer_names": target_names,
        "include_final_classifier": False,
        "electronic_linear_layers": electronic_linear_layers,
        "final_classifier_present": "fc2" in modules,
        "final_classifier_excluded": "fc2" not in target_names,
        "status": "verified_photonic_only",
    }


CONDITIONS: Dict[str, ReplayCondition] = {
    "clean": ReplayCondition("clean", None, None, None),
    "quantized_clean": ReplayCondition("quantized_clean", 6, 6, 16),
    "combined": ReplayCondition(
        "combined", 6, 6, 16, mrr_sigma=0.02, laser_sigma=0.01,
        wdm_crosstalk_db=-25.0, tia_noise_sigma=0.01,
    ),
}


class HardwareReplayHooks:
    def __init__(
        self,
        model: nn.Module,
        condition: ReplayCondition,
        seed: int,
        hapr_group_size: int,
        adc_full_scales: Optional[Mapping[str, float]] = None,
        target_layer_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.model = model
        self.condition = condition
        self.seed = int(seed)
        self.hapr_group_size = int(hapr_group_size)
        self.adc_full_scales = dict(adc_full_scales or {})
        self.target_layer_names = tuple(
            target_layer_names or resolve_photonic_mvm_layer_names(model)
        )
        self.target_layer_set = set(self.target_layer_names)
        self.handles: List[Any] = []
        self.generators: Dict[str, torch.Generator] = {}
        self.static_masks: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}
        self.mvm_layers: List[str] = []
        self.membrane_layers: List[str] = []

    def _generator(self, name: str) -> torch.Generator:
        if name not in self.generators:
            digest = hashlib.sha256(f"{self.seed}:{name}".encode("utf-8")).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
            self.generators[name] = generator
        return self.generators[name]

    def quantize_weights(self) -> None:
        bits = self.condition.weight_bits
        if bits is None:
            return
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.target_layer_set and getattr(module, "weight", None) is not None:
                    quantized, _ = quantize_symmetric(module.weight, int(bits), axis=0)
                    module.weight.copy_(quantized.to(device=module.weight.device, dtype=module.weight.dtype))
                    self.mvm_layers.append(name)

    def _mvm_hook(self, name: str):
        condition = self.condition

        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            if not torch.is_tensor(output):
                return output
            value = output
            generator = self._generator(name)
            if condition.mrr_sigma > 0.0:
                key = (name, _channel_mask_shape(value))
                if key not in self.static_masks:
                    noise = torch.randn(_channel_mask_shape(value), generator=generator, dtype=torch.float32)
                    self.static_masks[key] = 1.0 + noise * condition.mrr_sigma
                value = value * self.static_masks[key].to(device=value.device, dtype=value.dtype)
            if condition.laser_sigma > 0.0:
                gain = 1.0 + float(torch.randn((), generator=generator).item()) * condition.laser_sigma
                value = value * gain
            if condition.wdm_crosstalk_db is not None:
                value = _wdm_crosstalk(value, condition.wdm_crosstalk_db)
            if condition.tia_noise_sigma > 0.0 and value.numel():
                rms = float(torch.sqrt(torch.mean(value.detach().float() ** 2)).item())
                std = condition.tia_noise_sigma * math.sqrt(max(self.hapr_group_size, 1) / 4.0) * rms
                if std > 0.0:
                    noise = torch.randn(value.shape, generator=generator, dtype=torch.float32)
                    value = value + noise.to(device=value.device, dtype=value.dtype) * std
            if condition.adc_bits is not None:
                if name not in self.adc_full_scales:
                    raise RuntimeError(
                        f"Missing validation-calibrated ADC full scale for layer {name!r}. "
                        "Run eval_105 with a valid calibration split before reporting accuracy."
                    )
                value = _adc_quantize(value, condition.adc_bits, self.adc_full_scales[name])
            return value

        return hook

    def _membrane_hook(self, name: str):
        condition = self.condition

        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            state = getattr(module, "v", None)
            if condition.membrane_bits is None or state is None:
                return None
            if torch.is_tensor(state):
                module.v = quantize_membrane(state, condition.membrane_bits, condition.membrane_fractional_bits)
            elif isinstance(state, (int, float)):
                scale = float(2 ** condition.membrane_fractional_bits)
                limit = float(2 ** (condition.membrane_bits - 1) - 1)
                module.v = max(min(round(float(state) * scale), limit), -limit) / scale
            return None

        return hook

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if name in self.target_layer_set:
                self.handles.append(module.register_forward_hook(self._mvm_hook(name)))
                if name not in self.mvm_layers:
                    self.mvm_layers.append(name)
            class_name = module.__class__.__name__.lower()
            if self.condition.membrane_bits is not None and ("lif" in class_name or "ifnode" in class_name):
                self.handles.append(module.register_forward_hook(self._membrane_hook(name)))
                self.membrane_layers.append(name)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _load_replay_runtime() -> Tuple[Any, ...]:
    """Load model/dataset helpers only when an actual replay starts.

    This keeps ``eval_105.py --help`` and static inspection independent of the
    optional SpikingJelly runtime while preserving a clear failure for replay.
    """
    try:
        from eval.eval_utils import build_eval_model, build_loader, load_eval_context
        from utils.data_utils import (
            aggregate_time_logits,
            logits_aggregation_from_config,
            prepare_snn_batch,
            reset_snn_state,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional evaluation dependency"
        raise RuntimeError(
            f"eval_105 replay requires the training/evaluation environment; missing module: {missing}. "
            "Install the repository's SNN runtime dependencies (including SpikingJelly) "
            "before running checkpoint replay. The architecture-only eval_101/eval_104 stages "
            "do not require this dependency."
        ) from exc
    return (
        build_eval_model,
        build_loader,
        load_eval_context,
        aggregate_time_logits,
        logits_aggregation_from_config,
        prepare_snn_batch,
        reset_snn_state,
    )




@torch.inference_mode()
def calibrate_adc_full_scales(args: argparse.Namespace, output_dir: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Calibrate fixed per-layer ADC ranges on a disjoint validation split.

    Publication runs consume the complete calibration split.  The optional
    batch cap exists only for smoke tests and is marked non-publication.
    """
    if args.calibration_split == args.split:
        raise ValueError(
            "ADC calibration split must differ from the reporting split; "
            f"both were {args.split!r}."
        )
    (
        build_eval_model,
        build_loader,
        load_eval_context,
        aggregate_time_logits,
        logits_aggregation_from_config,
        prepare_snn_batch,
        reset_snn_state,
    ) = _load_replay_runtime()
    set_seed(args.seed, deterministic=True, benchmark=False)
    context = load_eval_context(
        config_path=args.config, checkpoint=args.checkpoint,
        hardware_config=args.hardware, device_params=args.device_params,
        output_dir=output_dir, eval_name="eval_105_adc_calibration", device=args.device,
    )
    validate_g4_a8_contract(context.hardware_cfg)
    model = build_eval_model(context, strict=not args.non_strict)
    target_layer_names = resolve_photonic_mvm_layer_names(model)
    replay_scope = build_replay_scope(model, target_layer_names)
    weight_hooks = HardwareReplayHooks(
        model, CONDITIONS["quantized_clean"], args.seed, 4, adc_full_scales={},
        target_layer_names=target_layer_names,
    )
    weight_hooks.quantize_weights()
    maxima: Dict[str, float] = {}
    handles: List[Any] = []

    def collector(name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if torch.is_tensor(output) and output.numel():
                observed = float(output.detach().abs().amax().item())
                maxima[name] = max(maxima.get(name, 0.0), observed)
            return None
        return hook

    target_layer_set = set(target_layer_names)
    for name, module in model.named_modules():
        if name in target_layer_set:
            handles.append(module.register_forward_hook(collector(name)))

    loader = build_loader(
        context.config, split_name=args.calibration_split, batch_size=args.batch_size,
        num_workers=args.num_workers, allow_no_split=False, shuffle=False,
    )
    expected_samples = len(loader.dataset)
    aggregation = logits_aggregation_from_config(context.config)
    batches = 0
    samples = 0
    try:
        for batch_index, (data, target) in enumerate(loader):
            if args.calibration_batches is not None and batch_index >= args.calibration_batches:
                break
            reset_snn_state(model)
            data, target = prepare_snn_batch(data, target, context.config, context.device)
            aggregate_time_logits(model(data), aggregation)
            batches += 1
            samples += int(target.numel())
    finally:
        for handle in handles:
            handle.remove()

    if batches <= 0 or samples <= 0 or not maxima:
        raise RuntimeError("ADC calibration produced no samples or layer ranges.")
    invalid = [name for name, value in maxima.items() if not math.isfinite(value) or value <= 0.0]
    if invalid:
        raise RuntimeError(f"Invalid ADC calibration full scales for layers: {invalid}")
    full_split_used = samples == expected_samples
    split_file = context.artifacts.get("split_file")
    split_path = Path(split_file) if split_file else None
    manifest = {
        "method": "frozen_per_layer_absmax",
        "calibration_split": args.calibration_split,
        "reporting_split": args.split,
        "calibration_batches_requested": args.calibration_batches,
        "calibration_batches_used": batches,
        "calibration_samples_expected": expected_samples,
        "calibration_samples_used": samples,
        "full_calibration_split_used": full_split_used,
        "status": "verified_full_calibration_split" if full_split_used else "diagnostic_partial_calibration",
        "batch_size_actual": int(loader.batch_size or 1),
        "weight_bits_during_calibration": 6,
        "adc_bits": 6,
        "full_scales": {name: maxima[name] for name in target_layer_names},
        "photonic_mvm_layer_names": target_layer_names,
        "include_final_classifier": False,
        "replay_scope": replay_scope,
        "split_file": str(split_path.resolve()) if split_path and split_path.exists() else None,
        "split_file_sha256": file_sha256(split_path) if split_path and split_path.exists() else None,
        "test_data_used_for_calibration": False,
        "claim_boundary": "software activation-range calibration; fixed for all reporting batches and perturbation conditions",
    }
    save_json(manifest, output_dir / "adc_calibration.json")
    return maxima, manifest


@torch.inference_mode()
def run_condition(
    args: argparse.Namespace,
    condition: ReplayCondition,
    output_dir: Path,
    adc_full_scales: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    (
        build_eval_model,
        build_loader,
        load_eval_context,
        aggregate_time_logits,
        logits_aggregation_from_config,
        prepare_snn_batch,
        reset_snn_state,
    ) = _load_replay_runtime()
    set_seed(args.seed, deterministic=True, benchmark=False)
    context = load_eval_context(
        config_path=args.config, checkpoint=args.checkpoint,
        hardware_config=args.hardware, device_params=args.device_params,
        output_dir=output_dir, eval_name="eval_105", device=args.device,
    )
    contract = validate_g4_a8_contract(context.hardware_cfg)
    model = build_eval_model(context, strict=not args.non_strict)
    target_layer_names = resolve_photonic_mvm_layer_names(model)
    replay_scope = build_replay_scope(model, target_layer_names)
    hooks = HardwareReplayHooks(
        model, condition, args.seed, contract["hapr_group_size"],
        adc_full_scales=adc_full_scales, target_layer_names=target_layer_names,
    )
    hooks.quantize_weights()
    hooks.register()
    loader = build_loader(
        context.config, split_name=args.split, batch_size=args.batch_size,
        num_workers=args.num_workers, allow_no_split=args.allow_no_split, shuffle=False,
    )
    expected_samples = len(loader.dataset)
    aggregation = logits_aggregation_from_config(context.config)
    criterion = nn.CrossEntropyLoss(reduction="none")
    targets: List[int] = []
    predictions: List[int] = []
    logits_parts: List[np.ndarray] = []
    prediction_rows: List[Dict[str, Any]] = []
    loss_sum = 0.0
    try:
        for batch_index, (data, target) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            reset_snn_state(model)
            data, target = prepare_snn_batch(data, target, context.config, context.device)
            logits = aggregate_time_logits(model(data), aggregation)
            losses = criterion(logits, target)
            predicted = logits.argmax(dim=1)
            targets_cpu = target.detach().cpu().numpy().astype(int)
            predictions_cpu = predicted.detach().cpu().numpy().astype(int)
            losses_cpu = losses.detach().cpu().numpy().astype(float)
            if args.save_logits:
                logits_parts.append(logits.detach().cpu().numpy())
            start = len(targets)
            for offset in range(len(targets_cpu)):
                prediction_rows.append({
                    "condition": condition.name,
                    "sample_ordinal": start + offset,
                    "batch_index": batch_index,
                    "target": int(targets_cpu[offset]),
                    "prediction": int(predictions_cpu[offset]),
                    "correct": int(targets_cpu[offset] == predictions_cpu[offset]),
                    "loss": float(losses_cpu[offset]),
                })
            targets.extend(targets_cpu.tolist())
            predictions.extend(predictions_cpu.tolist())
            loss_sum += float(losses.sum().item())
    finally:
        hooks.remove()
    correct = sum(int(a == b) for a, b in zip(targets, predictions))
    total = len(targets)
    if args.save_logits and logits_parts:
        np.save(output_dir / f"{condition.name}_logits.npy", np.concatenate(logits_parts, axis=0))
    return {
        "condition": condition.name,
        "targets": targets,
        "predictions": predictions,
        "prediction_rows": prediction_rows,
        "num_samples": total,
        "expected_samples": expected_samples,
        "full_split_replayed": total == expected_samples,
        "batch_size": int(loader.batch_size or 1),
        "device": str(context.device),
        "correct": correct,
        "accuracy_percent": 100.0 * correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "mvm_layers": hooks.mvm_layers,
        "replay_scope": replay_scope,
        "membrane_layers": hooks.membrane_layers,
        "condition_spec": condition.__dict__,
        "dataset": dataset_tag(context.config),
        "num_classes": int(context.artifacts.get("num_classes", 10)),
        "split_file": str(context.artifacts.get("split_file")) if context.artifacts.get("split_file") else None,
        "contract": contract,
        "config": context.config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIPSA eval_105: paired publication replay for G4/A8")
    parser.add_argument("--dataset", default=None, help="Optional assertion against the dataset in --config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hardware", default="configs/hardware_hipsa_paper.yaml")
    parser.add_argument("--device-params", default="configs/device_params.yaml")
    parser.add_argument("--output-root", default="results/eval_v10")
    parser.add_argument("--split", default="test")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--calibration-batches", type=int, default=None, help="Diagnostic-only cap; omit for full validation-split calibration.")
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), default=["clean", "quantized_clean", "combined"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cublas-workspace-config", default=":4096:8", choices=[":4096:8", ":16:8"], help="Deterministic CUDA BLAS workspace policy recorded in provenance.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-indexed-output", action="store_true", help="Write this replay to eval_105/seeds/seed_NNN instead of replacing the legacy single-seed directory.")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--allow-no-split", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_publication_runtime(args)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.calibration_batches is not None and args.calibration_batches <= 0:
        raise ValueError("--calibration-batches must be positive when provided")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided")
    raw_config = load_eval_config(args.config, args.hardware, args.device_params)
    dataset = dataset_tag(raw_config)
    if args.dataset and args.dataset.lower().replace("-", "") != dataset.lower().replace("-", ""):
        raise ValueError(f"--dataset={args.dataset} does not match config dataset {dataset}")
    output_dir = resolve_output_dir(args.output_root, dataset, args.seed, args.seed_indexed_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    needs_adc = any(CONDITIONS[name].adc_bits is not None for name in args.conditions)
    adc_full_scales: Dict[str, float] = {}
    calibration_manifest: Optional[Dict[str, Any]] = None
    if needs_adc:
        adc_full_scales, calibration_manifest = calibrate_adc_full_scales(args, output_dir)
    runs = [
        run_condition(args, CONDITIONS[name], output_dir, adc_full_scales=adc_full_scales)
        for name in args.conditions
    ]
    reference_targets = runs[0]["targets"] if runs else []
    expected_samples = runs[0]["expected_samples"] if runs else 0
    for run in runs[1:]:
        if run["targets"] != reference_targets:
            raise RuntimeError("Paired replay lost split/order equivalence across conditions.")
        if run["expected_samples"] != expected_samples:
            raise RuntimeError("Paired replay changed expected split size across conditions.")
    replay_scopes = [run["replay_scope"] for run in runs]
    if replay_scopes and any(scope != replay_scopes[0] for scope in replay_scopes[1:]):
        raise RuntimeError("Photonic replay scope changed across paired conditions.")
    replay_scope = replay_scopes[0] if replay_scopes else None
    if calibration_manifest and replay_scope:
        calibrated_names = list(calibration_manifest.get("full_scales", {}).keys())
        if calibrated_names != replay_scope["photonic_mvm_layer_names"]:
            raise RuntimeError(
                "ADC calibration layers do not exactly match the photonic replay scope: "
                f"calibrated={calibrated_names}, "
                f"replay={replay_scope['photonic_mvm_layer_names']}"
            )
    clean = next((run for run in runs if run["condition"] == "clean"), None)
    accuracy_rows: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    for run in runs:
        paired_changes = None
        if clean is not None:
            paired_changes = sum(
                int(target == prediction) - int(target == clean_prediction)
                for target, prediction, clean_prediction in zip(run["targets"], run["predictions"], clean["predictions"])
            )
        accuracy_rows.append({
            "dataset": dataset,
            "condition": run["condition"],
            "seed": args.seed,
            "split": args.split,
            "num_samples": run["num_samples"],
            "accuracy_percent": run["accuracy_percent"],
            "loss": run["loss"],
            "paired_accuracy_delta_percent_vs_clean": (
                run["accuracy_percent"] - clean["accuracy_percent"] if clean is not None else None
            ),
            "paired_net_correct_change_vs_clean": paired_changes,
            "weight_bits": run["condition_spec"].get("weight_bits"),
            "adc_bits": run["condition_spec"].get("adc_bits"),
            "membrane_bits": run["condition_spec"].get("membrane_bits"),
            "evidence_class": "software_checkpoint_evaluation" if run["condition"] == "clean" else "hardware_aware_software_proxy_replay",
        })
        all_predictions.extend(run["prediction_rows"])
        save_confusion(
            build_confusion_matrix(run["targets"], run["predictions"], run["num_classes"]),
            output_dir / f"confusion_matrix_{run['condition']}.csv",
        )
    split_path = Path(runs[0]["split_file"]) if runs and runs[0]["split_file"] else None
    full_split_replayed = bool(runs and all(run["full_split_replayed"] for run in runs))
    runtime_provenance = build_runtime_provenance(
        args,
        resolved_device=runs[0]["device"] if runs else None,
        actual_batch_size=runs[0]["batch_size"] if runs else None,
    )
    split_manifest = {
        "dataset": dataset,
        "split": args.split,
        "split_file": str(split_path.resolve()) if split_path and split_path.exists() else None,
        "split_file_sha256": file_sha256(split_path) if split_path and split_path.exists() else None,
        "split_order_hash": canonical_sha256(reference_targets),
        "num_samples_expected": expected_samples,
        "num_samples_replayed": len(reference_targets),
        "full_split_replayed": full_split_replayed,
        "max_batches": args.max_batches,
        "allow_no_split": bool(args.allow_no_split),
        "status": (
            "verified_full_split_replay"
            if split_path and split_path.exists() and full_split_replayed
            else "diagnostic_partial_or_unverified_split"
        ),
    }
    hashes = {
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "hardware_config_sha256": file_sha256(args.hardware),
        "device_params_sha256": file_sha256(args.device_params),
    }
    by_name = {row["condition"]: row for row in accuracy_rows}
    summary = {
        "eval_name": "eval_105",
        "purpose": "paired_publication_replay",
        "created_utc": now_utc(),
        "dataset": dataset,
        "split": args.split,
        "seed": args.seed,
        "seed_type": "paired_replay_perturbation_seed",
        "seed_output_mode": "seed_indexed" if args.seed_indexed_output else "legacy_single_output",
        "num_samples": len(reference_targets),
        "clean_accuracy_percent": by_name.get("clean", {}).get("accuracy_percent"),
        "quantized_clean_accuracy_percent": by_name.get("quantized_clean", {}).get("accuracy_percent"),
        "combined_condition_accuracy_percent": by_name.get("combined", {}).get("accuracy_percent"),
        "hardware_aware_accuracy_percent": by_name.get("combined", {}).get("accuracy_percent"),
        "hardware_aware_accuracy_status": "software_proxy_replay_not_silicon_measurement" if "combined" in by_name else "not_run",
        "precision_contract": {"weight_bits": 6, "adc_bits": 6, "membrane_bits": 16},
        "g4_a8_contract": runs[0]["contract"] if runs else None,
        "replay_scope": replay_scope,
        "conditions": accuracy_rows,
        "adc_calibration": calibration_manifest,
        "split_manifest": split_manifest,
        "runtime_provenance": runtime_provenance,
        "hashes": hashes,
        "checkpoint_summary": checkpoint_summary(args.checkpoint),
        "claim_boundary": {
            "accuracy": "checkpoint-and-split software replay",
            "combined": "declared perturbation proxy with validation-calibrated fixed ADC ranges; not silicon measurement",
            "adc_calibration": "non-test split, frozen per-layer absmax; no reporting-batch oracle scaling",
            "architecture_performance": "not computed by eval_105; join with eval_101 by matching config hashes",
        },
    }
    save_json(summary, output_dir / "summary.json")
    save_json(summary, output_dir / "metrics.json")
    save_json(validate_eval105_publication_summary(summary), output_dir / "validation.json")
    save_json(split_manifest, output_dir / "split_manifest.json")
    save_json(runtime_provenance, output_dir / "runtime_provenance.json")
    if replay_scope is not None:
        save_json(replay_scope, output_dir / "replay_scope.json")
    save_csv_rows(accuracy_rows, output_dir / "accuracy_summary.csv")
    save_csv_rows(all_predictions, output_dir / "predictions.csv")
    copy_config_snapshot(
        [Path(args.config), Path(args.hardware), Path(args.device_params)], output_dir,
        snapshot_name="config_snapshot.yaml", merged_config=runs[0]["config"] if runs else raw_config,
    )
    save_run_manifest(
        output_dir, eval_name="eval_105", command=" ".join(sys.argv),
        inputs={"config": args.config, "checkpoint": args.checkpoint, "hardware": args.hardware, "device_params": args.device_params},
        outputs={"summary": "summary.json", "metrics": "metrics.json", "validation": "validation.json", "accuracy_summary": "accuracy_summary.csv", "predictions": "predictions.csv", "split_manifest": "split_manifest.json", "runtime_provenance": "runtime_provenance.json", "replay_scope": "replay_scope.json" if replay_scope else None, "adc_calibration": "adc_calibration.json" if calibration_manifest else None, "config_snapshot": "config_snapshot.yaml"},
        extra={"dataset": dataset, "split": args.split, "calibration_split": args.calibration_split, "calibration_batches": args.calibration_batches, "max_batches": args.max_batches, "seed": args.seed, "seed_type": "paired_replay_perturbation_seed", "seed_indexed_output": bool(args.seed_indexed_output), "conditions": args.conditions, "runtime_provenance": runtime_provenance, **hashes},
    )
    print(f"[eval_105] paired publication replay saved to {output_dir}")
    for row in accuracy_rows:
        print(f"  {row['condition']}: {row['accuracy_percent']:.4f}% ({row['num_samples']} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








