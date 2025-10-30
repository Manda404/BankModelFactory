"""
semantic_cleaner.py
-------------------
Module: Semantic data cleaning with a strict no-NaN policy.

- Keep "unknown" literal for categoricals
- Convert NaN to "unknown"
- Impute numerics with business-friendly defaults

Author: Manda Surel
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


class SemanticCleaner(BaseEstimator, TransformerMixin):
    """Semantic data cleaner with no-NaN policy."""

    def fit(self, X, y=None):
        logger.info("[SemanticCleaner] Fitting stage started.")
        return self

    def transform(self, X, y=None):
        logger.info("[SemanticCleaner] Starting semantic cleaning...")
        X = X.copy()

        # Harmonize strings
        obj_cols = X.select_dtypes(include="object").columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype(str).str.lower().str.strip()

        # Replace NaN in categoricals with "unknown"
        for col in obj_cols:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna("unknown")

        # Convert only true numeric columns
        numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Impute numeric columns (no NaN left)
        if "pdays" in X.columns:
            X["pdays"] = X["pdays"].fillna(-1)
        for col in [c for c in numeric_cols if c != "pdays"]:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # Boolean indicators
        for col in ["housing", "loan", "default"]:
            if col in X.columns:
                X[f"has_{col}"] = X[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

        # Seasonal features
        if "month" in X.columns:
            X["is_summer"] = X["month"].isin(["may", "jun", "jul", "aug"]).astype(int)
            X["is_winter"] = X["month"].isin(["nov", "dec", "jan", "feb"]).astype(int)

        logger.success(f"[SemanticCleaner] Cleaning complete. Shape: {X.shape}")
        return X
