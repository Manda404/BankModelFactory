"""
binary_advanced.py
=========================
An advanced binary classification neural network.

Architecture:
Input → Multiple (Linear → BN → ReLU → Dropout) layers → Linear → Output

Features:
- Flexible depth (configurable hidden_dims)
- Batch Normalization and Dropout
- Designed for complex tabular datasets

Author: Manda Surel
Date: 2025-10-31
"""

import torch
import torch.nn as nn
from bankmodelfactory.dlmodels.base_model import BaseNN
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


class AdvancedNN(BaseNN):
    """An advanced neural network for binary classification."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.4,
        name: str = None,
    ):
        super().__init__(input_dim, output_dim, name or "AdvancedNN")

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

        logger.info(
            f"Initialized {self.name} with {len(hidden_dims)} hidden layers, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
