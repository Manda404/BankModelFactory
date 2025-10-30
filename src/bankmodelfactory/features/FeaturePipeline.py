"""
FeaturePipeline.py
------------------
Professional preprocessing pipeline for bank marketing data.

Combines:
- Semantic cleaning
- Behavioral feature generation
- Business encoding
- Target encoding (binary or categorical)

Fully scikit-learn compatible and visually inspectable in Jupyter notebooks.

Author: Manda Surel
Date: 2025-10-30
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import set_config
import pandas as pd
import numpy as np

from bankmodelfactory.features.semantic_cleaner import SemanticCleaner
from bankmodelfactory.features.behavioral_transformer import BehavioralTransformer
from bankmodelfactory.features.business_encoder import BusinessEncoder
from bankmodelfactory.utils.logger import get_logger

logger = get_logger()


# -------------------------------------------------------------------------
# Target encoder helper
# -------------------------------------------------------------------------
class _TargetEncoder:
    """Encodes the target column (y) robustly."""

    def __init__(self, strategy: str = "auto", mapping: dict | None = None):
        self.strategy = strategy
        self.mapping = mapping
        self.mapping_ = None
        self.le_ = None

    def fit(self, y):
        y = pd.Series(y)
        if self.mapping is not None:
            self.mapping_ = dict(self.mapping)
            return self

        vals = (
            y.dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
            .tolist()
        )

        if set(vals).issubset({"yes", "no"}):
            self.mapping_ = {"no": 0, "yes": 1}
        else:
            self.le_ = LabelEncoder().fit(y.astype(str))
        return self

    def transform(self, y):
        if y is None:
            return None
        y = pd.Series(y)
        if self.mapping_ is not None:
            return (
                y.astype(str)
                .str.lower()
                .str.strip()
                .map(self.mapping_)
                .astype(int)
                .to_numpy()
            )
        return self.le_.transform(y.astype(str))

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        y_encoded = np.asarray(y_encoded)
        if self.mapping_ is not None:
            inv = {v: k for k, v in self.mapping_.items()}
            return np.vectorize(inv.get)(y_encoded)
        return self.le_.inverse_transform(y_encoded)


# -------------------------------------------------------------------------
# FeaturePipeline
# -------------------------------------------------------------------------
class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Professional, modular preprocessing pipeline for ML models.

    Steps:
    - SemanticCleaner: cleans and harmonizes raw columns
    - BehavioralTransformer: adds domain-driven features
    - BusinessEncoder: encodes and scales features
    - _TargetEncoder: encodes the target (yes/no → 1/0 or label-based)
    """

    def __init__(self, target_strategy="auto", target_mapping=None):
        self.target_strategy = target_strategy
        self.target_mapping = target_mapping

        # Internal sklearn pipeline for X
        self.pipeline = Pipeline([
            ("semantic_cleaner", SemanticCleaner()),
            ("behavioral_transformer", BehavioralTransformer()),
            ("business_encoder", BusinessEncoder()),
        ])

        # Target encoder
        self.target_encoder = _TargetEncoder(strategy=target_strategy, mapping=target_mapping)
        self._is_fitted = False

    # ---------------------------------------------------------------------
    def fit_transform(self, X, y=None):
        """Fit and transform both X and y."""
        logger.info("[FeaturePipeline] Starting full preprocessing pipeline...")

        X_ready = self.pipeline.fit_transform(X)
        y_ready = self.target_encoder.fit_transform(y) if y is not None else None

        self._is_fitted = True
        logger.success("[FeaturePipeline] Pipeline complete. Data ready for modeling.")
        return X_ready, y_ready

    # ---------------------------------------------------------------------
    def transform(self, X, y=None):
        """Apply the trained transformations to new data (test/production)."""
        if not self._is_fitted:
            raise RuntimeError("FeaturePipeline must be fit before transform().")

        logger.info("[FeaturePipeline] Applying fitted transformations...")
        X_ready = self.pipeline.transform(X)
        y_ready = self.target_encoder.transform(y) if y is not None else None

        logger.success("[FeaturePipeline] Transformation complete.")
        return (X_ready, y_ready) if y is not None else X_ready

    # ---------------------------------------------------------------------
    def inverse_transform_target(self, y_encoded):
        """Inverse-transform the encoded target back to its original labels."""
        return self.target_encoder.inverse_transform(y_encoded)

    # ---------------------------------------------------------------------
    def diagram(self):
        """
        Display a full sklearn pipeline diagram in Jupyter.
        This is purely visual — for exploration and documentation.
        """
        set_config(display="diagram")
        return self.pipeline