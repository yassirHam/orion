"""Reproducibility helpers (O1.5): seeds, git commit, hardware, config hashing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we use and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Make DataLoader workers deterministic too.
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown (not a git repo)"


def hardware_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["gpu"] = "cpu-only"
    return info


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class Timer:
    """Context manager that records wall-clock seconds."""

    label: str = "elapsed"
    seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start
