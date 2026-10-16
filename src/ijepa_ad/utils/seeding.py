"""Reproducibility helpers for experiments."""

from __future__ import annotations

import os
import random
from importlib import import_module
from typing import Any


def set_seed(seed: int) -> None:
    """Set the project's random seeds and deterministic execution flags.

    PyTorch is imported lazily because utility modules should remain importable in
    documentation and configuration environments that do not install the training
    stack. A full training environment is expected to provide PyTorch.
    """
    _validate_seed(seed)

    random.seed(seed)
    numpy = _load_numpy()
    if numpy is not None:
        numpy.random.seed(seed)

    # CuBLAS requires this setting before CUDA is initialized for deterministic matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch = _load_torch()
    if torch is None:
        return

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seeded_rng(seed: int) -> random.Random:
    """Return an isolated Python RNG for deterministic data operations.

    Unlike ``set_seed``, this leaves global random state and PyTorch flags
    untouched, so library code (e.g. dataset splitting) can be deterministic
    without side effects on the caller's experiment state.
    """
    _validate_seed(seed)
    return random.Random(seed)


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _load_torch() -> Any | None:
    """Return PyTorch when installed, otherwise ``None``."""
    try:
        return import_module("torch")
    except ImportError:
        return None


def _load_numpy() -> Any | None:
    """Return NumPy when installed, otherwise ``None``."""
    try:
        return import_module("numpy")
    except ImportError:
        return None
