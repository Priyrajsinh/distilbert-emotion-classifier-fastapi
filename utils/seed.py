"""Reproducibility utilities for the B1 project.

Call set_seed(42) at the top of every training script to ensure
deterministic behaviour across runs.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds to ensure reproducible training runs.

    Sets seeds for Python's built-in random module, NumPy, and PyTorch
    (both CPU and CUDA if available).

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
