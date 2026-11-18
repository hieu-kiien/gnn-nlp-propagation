"""Deterministic random seeding utilities for SBi-FAGAT experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Five mandatory random seeds defined in GEMINI.md
OFFICIAL_SEEDS: tuple[int, ...] = (42, 100, 2024, 777, 999)


def seed_everything(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch RNG states for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
