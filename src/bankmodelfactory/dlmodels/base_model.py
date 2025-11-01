"""
base_model.py
=========================
Abstract base class for all neural network models.

Responsibilities:
- Device management (via get_device)
- Model saving & loading
- Automatic model naming
- Unified forward() interface for subclasses

Author: Manda Surel
Date: 2025-10-31
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from datetime import datetime
from bankmodelfactory.utils.logger import get_logger
from bankmodelfactory.utils.device import get_device

logger = get_logger()
device = get_device()


class BaseNN(nn.Module, ABC):
    """Abstract base class for all neural networks."""

    def __init__(self, input_dim: int, output_dim: int, name: str = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.name = name or self.__class__.__name__
        self.to(self.device)

        logger.info(f"Initialized model '{self.name}' on device: {self.device}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to be implemented by subclasses."""
        pass

    def save(self, path: str = None):
        """
        Save model weights to disk.
        If no path is provided, creates a timestamped file under artifacts/models/.
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"artifacts/models/{self.name}_{timestamp}.pt"
        torch.save(self.state_dict(), path)
        logger.success(f"Model '{self.name}' saved successfully at: {path}")

    def load(self, path: str):
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        logger.info(f"Model '{self.name}' loaded from: {path}")
