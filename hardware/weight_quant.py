"""Hardware-aware symmetric weight quantization utilities."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def quantize_symmetric(values: Any, bits: int = 6, *, axis: int | None = None) -> Tuple[Any, Dict[str, Any]]:
    """Quantize a tensor/array with a signed symmetric integer codebook."""

    if int(bits) < 2:
        raise ValueError("signed quantization needs at least two bits")
    try:
        import torch

        tensor = values.detach() if torch.is_tensor(values) else torch.as_tensor(values)
        qmax = 2 ** (int(bits) - 1) - 1
        if axis is None:
            scale = tensor.abs().max() / max(qmax, 1)
        else:
            scale = tensor.abs().amax(dim=axis, keepdim=True) / max(qmax, 1)
        scale = scale.clamp_min(torch.finfo(tensor.dtype).eps if tensor.is_floating_point() else 1e-12)
        integer = torch.round(tensor / scale).clamp(-qmax, qmax)
        return integer * scale, {
            "bits": int(bits),
            "qmin": -qmax - 1,
            "qmax": qmax,
            "scale": scale.detach().cpu().tolist(),
            "axis": axis,
        }
    except ImportError:
        import numpy as np

        array = np.asarray(values)
        qmax = 2 ** (int(bits) - 1) - 1
        scale = np.max(np.abs(array), axis=axis, keepdims=axis is not None) / max(qmax, 1)
        scale = np.maximum(scale, np.finfo(np.float32).eps)
        integer = np.clip(np.rint(array / scale), -qmax, qmax)
        return integer * scale, {"bits": int(bits), "qmin": -qmax - 1, "qmax": qmax, "scale": scale.tolist(), "axis": axis}
