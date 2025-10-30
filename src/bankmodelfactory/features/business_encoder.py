"""
business_encoder.py
-------------------
Business-aware encoding for categorical and numeric features.

- Ordinal mapping (education, job)
- Frequency encoding for nominal variables
- Robust scaling for original numerics only

Author: Manda Surel
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


class BusinessEncoder(BaseEstimator, TransformerMixin):
    """Encodes features using business-relevant encoding methods."""

    def fit(self, X, y=None):
        X = X.copy()

        # Save numeric columns before any new features are created
        self.num_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        # Build frequency maps for all categorical features
        self.freq_maps_ = {
            col: X[col].value_counts(normalize=True).to_dict()
            for col in X.select_dtypes(include="object").columns
        }

        # Fit scaler only on numeric columns existing at fit time
        self.scaler_ = RobustScaler()
        if len(self.num_cols_) > 0:
            self.scaler_.fit(X[self.num_cols_])

        logger.info(f"[BusinessEncoder] Learned scaling for {len(self.num_cols_)} numeric features.")
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # === Ordinal encoding for hierarchical features ===
        edu_order = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
        job_order = {
            "unknown": 0, "student": 1, "blue-collar": 2,
            "admin.": 3, "technician": 4, "management": 5,
            "retired": 6,
        }

        if "education" in X.columns:
            X["education_encoded"] = X["education"].map(edu_order).fillna(0)
        if "job" in X.columns:
            X["job_encoded"] = X["job"].map(job_order).fillna(0)

        # === Frequency encoding for remaining categorical variables ===
        for col, mapping in self.freq_maps_.items():
            if col not in ["education", "job"]:
                X[f"{col}_freq"] = X[col].map(mapping).fillna(0)

        # === Apply scaling only to original numeric columns ===
        if hasattr(self, "num_cols_") and len(self.num_cols_) > 0:
            X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])

        # === Drop redundant categorical columns ===
        to_drop = [col for col in self.freq_maps_.keys() if col in X.columns]
        X.drop(columns=to_drop, inplace=True, errors="ignore")

        logger.success(f"[BusinessEncoder] Encoding complete. Shape: {X.shape}")
        return X


