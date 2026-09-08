"""Fixed-point membrane update and reference equivalence helpers."""

from __future__ import annotations

from typing import Any, Dict


def quantize_membrane(value: Any, bits: int = 16, fractional_bits: int = 8) -> Any:
    """Round and saturate a membrane value to signed fixed-point format."""

    scale = float(2 ** int(fractional_bits))
    limit = float(2 ** (int(bits) - 1) - 1)
    try:
        import torch

        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        return torch.round(tensor * scale).clamp(-limit, limit) / scale
    except ImportError:
        import numpy as np

        array = np.asarray(value)
        return np.clip(np.rint(array * scale), -limit, limit) / scale


def lazy_decay_update(
    membrane: Any,
    input_current: Any,
    elapsed_steps: Any,
    *,
    decay: float,
    bits: int = 16,
    fractional_bits: int = 8,
    input_scale: float = 0.5,
) -> Any:
    """Apply leakage for elapsed idle steps and then one current update."""

    try:
        import torch

        m = membrane if torch.is_tensor(membrane) else torch.as_tensor(membrane)
        i = input_current if torch.is_tensor(input_current) else torch.as_tensor(input_current)
        e = elapsed_steps if torch.is_tensor(elapsed_steps) else torch.as_tensor(elapsed_steps)
        return quantize_membrane(m * (float(decay) ** e) + i * float(input_scale), bits, fractional_bits)
    except ImportError:
        import numpy as np

        return quantize_membrane(np.asarray(membrane) * (float(decay) ** np.asarray(elapsed_steps)) + np.asarray(input_current) * float(input_scale), bits, fractional_bits)


def fixedpoint_spec(bits: int = 16, fractional_bits: int = 8) -> Dict[str, Any]:
    return {
        "membrane_bits": int(bits),
        "fractional_bits": int(fractional_bits),
        "signed": True,
        "tau": 2,
        "per_step_equation": "Vm_next = Vm/2 + input/2",
        "overflow": "saturate",
        "equivalence_target": "dense_decay_then_update vs timestamp_lazy_decay",
    }
