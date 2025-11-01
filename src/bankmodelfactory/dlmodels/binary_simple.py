"""
binary_simple.py
=========================
A minimal binary classification neural network.

Architecture:
Input → Linear → ReLU → Linear → Output (1 logit)

This model is a baseline for binary classification tasks.

Author: Manda Surel
Date: 2025-10-31
"""

import torch
import torch.nn as nn
from bankmodelfactory.dlmodels.base_model import BaseNN
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


class SimpleNN(BaseNN):
    """A simple feed-forward binary classifier."""

    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 64, name: str = None):
        super().__init__(input_dim, output_dim, name or "SimpleNN")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        logger.info(f"Initialized {self.name} with hidden_dim={hidden_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x
