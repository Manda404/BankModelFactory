"""
bankmodelfactory.features
-------------------------

Feature engineering subpackage for the BankModelFactory project.

This subpackage provides tools to transform and prepare data 
for predictive modeling in the Bank Marketing domain.

Contents:
- FeatureEngineer class: responsible for preprocessing, 
  crafting business-relevant features, and applying consistent 
  encoding strategies across training and testing datasets.

Author: Manda Surel
Date: 2025-10-30
"""

from .feature_engineering import FeatureEngineer
from  .feature_engineering_simple import FeatureEngineerSimple

__all__ = ["FeatureEngineer","FeatureEngineerSimple"]
