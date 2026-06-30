"""Reproducibility helpers for HIPSA train/eval scripts."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

try:
    from utils.logger import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("HIPSA")


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    For final evaluation and robustness sweeps, use deterministic=True and
    benchmark=False. For training speed, benchmark=True can be enabled.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)

    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:
        pass

    logger.info(
        "Seed set to %d | deterministic=%s | cudnn.benchmark=%s",
        seed,
        deterministic,
        benchmark,
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn compatible with PyTorch generators."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int, device: Optional[str] = None) -> torch.Generator:
    """Create a seeded torch.Generator for DataLoader shuffling/splits."""
    if device is None:
        gen = torch.Generator()
    else:
        gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def configure_torch_backend(training: bool = False, deterministic: Optional[bool] = None) -> None:
    """Set practical backend flags.

    training=True favors speed; training=False favors stable/eval behavior.
    """
    if deterministic is None:
        deterministic = not training
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(training and not deterministic)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(training)
        torch.backends.cudnn.allow_tf32 = bool(training)
    try:
        torch.set_float32_matmul_precision("high" if training else "highest")
    except Exception:
        pass
