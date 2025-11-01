"""
dlmodels/__init__.py
=========================
Model package initializer for deep learning models (binary classification).

This module provides:
- Unified imports for all binary network architectures
- Easy access to the BaseNN class and model factory
- Clean naming and organization (simple → advanced)

Usage example
-------------
>>> from bankmodelfactory.dlmodels import get_model
>>> model = get_model("medium", input_dim=52, output_dim=1)
>>> print(model.name)

Author: Manda Surel
Date: 2025-10-31
"""

from bankmodelfactory.dlmodels.base_model import BaseNN
from bankmodelfactory.dlmodels.binary_simple import SimpleNN
from bankmodelfactory.dlmodels.binary_medium import MediumNN
from bankmodelfactory.dlmodels.binary_advanced import AdvancedNN
from bankmodelfactory.dlmodels.factory import get_model

__all__ = [
    "BaseNN",
    "SimpleNN",
    "MediumNN",
    "AdvancedNN",
    "get_model",
]
