"""
device.py
---------
Utility for detecting the best available computation device.

Supports:
- NVIDIA GPU via CUDA
- Apple Silicon GPU via MPS
- Fallback to CPU

Author: Manda Surel
Date: 2025-10-30
"""

import torch
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


def get_device() -> str:
    """
    Detect and return the best available device.

    Returns
    -------
    device : str
        'cuda' if an NVIDIA GPU is available,
        'mps' if running on Apple Silicon (Mac M1/M2/M3),
        otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"[DeviceManager] Using device: {device}")
    return device
