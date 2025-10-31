"""
split.py
=========================
Module responsible for splitting datasets into train/validation/test subsets.

This module ensures:
- Configuration-driven splitting (via YAML for train/test/validation)
- Reproducible stratified splits
- Clean, structured logging

Author: Rostand & Manda Surel
Date: 2025-10-31
"""

from typing import Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from bankmodelfactory.utils.logger import get_logger
from bankmodelfactory.utils.config import Config
from bankmodelfactory.utils.path import get_project_root


logger = get_logger()


# -------------------------------------------------------------------------
# 1. Config-driven train/test split (YAML)
# -------------------------------------------------------------------------
def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a full dataset into training and testing sets based on YAML configuration.

    YAML example:
    -------------
    data:
      label: "TARGET_PRODUCT_FINAL"
      test_size: 0.2
      val_size: 0.25
      stratify: true
      random_seed: 42

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    # Load YAML configuration
    root = get_project_root()
    config_path = Path(root / "configs" / "data.yaml")
    config = Config.load({"data": str(config_path)})

    # Extract parameters
    test_size = config.data.get("test_size", 0.2)
    stratify = config.data.get("stratify", True)
    random_state = config.data.get("random_seed", 42)
    target_column = config.data.get("label", "y")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(
        f"[Dataset Split] YAML config → test_size={test_size}, stratify={stratify}, random_state={random_state}"
    )

    # Perform split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
        shuffle=True,
    )

    logger.success(
        f"[Dataset Split] Completed. Train shape={X_train.shape}, Test shape={X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------------------
# 2. Programmatic train/validation split (val_size read from YAML)
# -------------------------------------------------------------------------
def split_train_valid(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    **kwargs,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split dataset into train and validation sets with stratification on y.
    The validation size is automatically loaded from the YAML configuration.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    **kwargs : dict
        Additional parameters passed to sklearn.model_selection.train_test_split
        (e.g. shuffle=True, random_state=42).

    Returns
    -------
    X_train, y_train, X_valid, y_valid
    """

    # Load YAML configuration
    root = get_project_root()
    config_path = Path(root / "configs" / "data.yaml")
    config = Config.load({"data": str(config_path)})

    val_size = config.data.get("val_size", 0.2)
    stratify = config.data.get("stratify", True)

    # Validate inputs
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be a pandas DataFrame or numpy array.")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be a pandas Series or numpy array.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")
    if not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1.")

    # Ensure consistent types
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Perform stratified split
    stratify_col = y if stratify else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=val_size,
        stratify=stratify_col,
        **kwargs,
    )

    logger.info(
        f"[split_train_valid] Completed. Total={len(X)}, "
        f"Train={len(X_train)} ({(1 - val_size) * 100:.1f}%), "
        f"Valid={len(X_valid)} ({val_size * 100:.1f}%), "
        f"Stratify={stratify}, Params={kwargs}"
    )

    return X_train, y_train, X_valid, y_valid
