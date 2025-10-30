"""
split.py
=========================
Module responsible for splitting datasets into train and test subsets.

This module ensures:
- Configuration-driven data splitting (from YAML)
- Clean logging (via Loguru)
- Reproducibility and optional stratification

Author: Rostand Surel
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from bankmodelfactory.utils.logger import get_logger
from bankmodelfactory.utils.config import Config
from bankmodelfactory.utils.path import get_project_root

logger = get_logger()


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataset into training and testing sets based on YAML configuration.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to split.
        Path to the YAML configuration file.
        Defaults to `<project_root>/configs/data.yaml`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """

    # --- Load configuration ---
    root = get_project_root()
    config_path = Path(root / "configs" / "data.yaml")
    config = Config.load({"data": str(config_path)})

    # --- Extract split parameters (now at YAML root) ---
    test_size = config.data.get("test_size", 0.2)
    stratify = config.data.get("stratify", True)
    random_state = config.data.get("random_seed", 42)

    # --- Identify target column ---
    target_column = config.data.get("label", "y")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame columns: {list(df.columns)}"
        )

    logger.info(
        f"[Dataset Split] Using YAML config → test_size={test_size}, stratify={stratify}, random_state={random_state}"
    )

    # --- Perform split ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    logger.success(
        f"[Dataset Split] - Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test
