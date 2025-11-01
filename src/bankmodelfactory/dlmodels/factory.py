"""
factory.py
=========================
Model factory for binary neural networks.

Responsibilities:
- Centralized model instantiation
- Automatic naming and device assignment
- Unified logging for model creation

Author: Manda Surel
Date: 2025-10-31
"""

from bankmodelfactory.dlmodels.binary_simple import SimpleNN
from bankmodelfactory.dlmodels.binary_medium import MediumNN
from bankmodelfactory.dlmodels.binary_advanced import AdvancedNN
from bankmodelfactory.utils.logger import get_logger
from bankmodelfactory.utils.device import get_device

logger = get_logger()
device = get_device()


def get_model(model_name: str, input_dim: int, output_dim: int = 1, **kwargs):
    """
    Factory function to instantiate a neural network model.

    Parameters
    ----------
    model_name : str
        One of {"simple", "medium", "advanced"}.
    input_dim : int
        Number of input features.
    output_dim : int, default=1
        Output dimension (for binary classification = 1).
    **kwargs : dict
        Additional hyperparameters (e.g., hidden_dims, dropout).

    Returns
    -------
    torch.nn.Module
        Instantiated model with automatic naming and device assignment.
    """
    model_name = model_name.lower()

    if model_name == "simple":
        model = SimpleNN(input_dim, output_dim, **kwargs)
    elif model_name == "medium":
        model = MediumNN(input_dim, output_dim, **kwargs)
    elif model_name == "advanced":
        model = AdvancedNN(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)
    logger.success(f"[Factory] Model created: {model.name} ({model_name}) on device: {device}")
    return model
