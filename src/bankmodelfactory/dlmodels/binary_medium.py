"""
binary_medium.py
=========================
A medium-sized binary classification neural network.

Architecture:
Input → (Linear → BN → ReLU → Dropout) × 2 → Linear → Output

Features:
- Batch Normalization for stable training
- Dropout for regularization
- Suitable for more complex tabular datasets

Author: Manda Surel
Date: 2025-10-31
"""

import torch
import torch.nn as nn
from bankmodelfactory.dlmodels.base_model import BaseNN
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


class MediumNN(BaseNN):
    """A deeper binary classifier with dropout and batch normalization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: tuple = (128, 64),
        dropout: float = 0.3,
        name: str = None,
    ):
        super().__init__(input_dim, output_dim, name or "MediumNN")

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.relu = nn.ReLU()

        logger.info(
            f"Initialized {self.name} with hidden_dims={hidden_dims}, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x
