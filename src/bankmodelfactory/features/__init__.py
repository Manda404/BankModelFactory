"""
bankmodelfactory.features
-------------------------
Feature engineering subpackage for the BankModelFactory project.

Includes:
- SemanticCleaner: Ensures data consistency and standardization
- BehavioralTransformer: Generates business-driven features
- BusinessEncoder: Encodes variables using business logic
- FeaturePipeline: Orchestrates the full preprocessing workflow

Author: Manda Surel
Date: 2025-10-30
"""

from .semantic_cleaner import SemanticCleaner
from .behavioral_transformer import BehavioralTransformer
from .business_encoder import BusinessEncoder
from .FeaturePipeline import FeaturePipeline

__all__ = [
    "SemanticCleaner",
    "BehavioralTransformer",
    "BusinessEncoder",
    "FeaturePipeline",
]
