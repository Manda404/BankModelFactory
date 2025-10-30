"""
behavioral_transformer.py
-------------------------
Business-driven feature creation.

- Demographic segmentation
- Engagement & marketing intensity
- Channel modernity
- Seasonality and loyalty indicators

Author: Manda Surel
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


class BehavioralTransformer(BaseEstimator, TransformerMixin):
    """Creates new features with business relevance."""

    def fit(self, X, y=None):
        logger.info("[BehavioralTransformer] Fitting stage started.")
        return self

    def transform(self, X, y=None):
        logger.info("[BehavioralTransformer] Creating business features...")
        X = X.copy()

        # Demographics
        if "age" in X.columns:
            X["age_group"] = pd.cut(
                X["age"],
                bins=[0, 30, 50, 70, np.inf],
                labels=["young", "adult", "senior", "elder"],
                include_lowest=True,
            )

        # Wealth & balance-based insights
        if "balance" in X.columns:
            X["balance_effort_ratio"] = (X["balance"] / (1 + X.get("campaign", 0))).clip(-5000, 50000)
            X["wealth_segment"] = pd.cut(
                X["balance"],
                bins=[-np.inf, 0, 1000, 5000, 20000, np.inf],
                labels=["negative", "low", "medium", "high", "very_high"],
            )

        # Marketing engagement
        if {"campaign", "previous"}.issubset(X.columns):
            X["total_contacts"] = X["campaign"] + X["previous"]
            X["overcontacted_flag"] = (X["total_contacts"] > 5).astype(int)

        # Relationship maturity
        if "pdays" in X.columns:
            X["recent_contact_flag"] = ((X["pdays"] < 30) & (X["pdays"] > 0)).astype(int)

        # Channel quality
        if "contact" in X.columns:
            X["modern_channel"] = X["contact"].map({"cellular": 2, "telephone": 1, "unknown": 0}).fillna(0).astype(int)

        # Seasonality
        if "month" in X.columns:
            X["quarter"] = X["month"].map({
                "jan": "Q1", "feb": "Q1", "mar": "Q1",
                "apr": "Q2", "may": "Q2", "jun": "Q2",
                "jul": "Q3", "aug": "Q3", "sep": "Q3",
                "oct": "Q4", "nov": "Q4", "dec": "Q4",
                "unknown": "Q0",
            })

        logger.success(f"[BehavioralTransformer] Generated features. Shape: {X.shape}")
        return X
