"""Runtime helpers shared by training, evaluation, and tests."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch for reproducible dataset splits and runs."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a PyTorch DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve ``auto``/empty to CUDA when available, otherwise CPU."""
    value = (requested or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def prepare_log_dir(log_dir: str | Path, overwrite: bool = False) -> Path:
    """Create a run log directory, deleting only an explicitly allowed old log."""
    path = Path(log_dir)
    resolved = path.resolve(strict=False)
    filesystem_root = Path(resolved.anchor)

    if path.name != "log" or resolved.parent == filesystem_root:
        raise ValueError(f"refusing unsafe log directory: {path}")
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlinked log directory: {path}")

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        if not overwrite:
            raise FileExistsError(
                f"log directory already exists: {path}; pass --overwrite_log=1 to replace it"
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=False)
    return path
