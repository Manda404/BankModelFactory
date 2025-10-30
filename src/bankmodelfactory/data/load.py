"""
loader.py
=========================
Module responsible for loading and inspecting datasets.

This module centralizes all dataset loading logic, ensuring:
- Consistent configuration management (via YAML)
- Clean and unified logging (via Loguru)
- Robust error handling
- Clear data inspection support

Author: Rostand Surel
"""

import pandas as pd
from pathlib import Path

from bankmodelfactory.utils.config import Config
from bankmodelfactory.utils.path import get_project_root
from bankmodelfactory.utils.logger import get_logger


logger = get_logger()


class SourceDataLoader:
    """
    Centralized class for loading and inspecting CSV datasets.

    Attributes
    ----------
    config : Config
        Loaded configuration object (from YAML).
    data : pd.DataFrame | None
        Loaded dataset, initialized as None.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the SourceDataLoader by loading configuration.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to the YAML configuration file.
            If not provided, defaults to `<project_root>/configs/data.yaml`.
        """
        root = get_project_root()
        self.config_path = Path(config_path or root / "configs" / "data.yaml")
        self.config = Config.load({"data": str(self.config_path)})
        self.data: pd.DataFrame | None = None

        logger.info(f"[SourceDataLoader] Configuration loaded from: {self.config_path}")

    def load_dataset(self, dataset_name: str = "Bank Marketing Dataset") -> pd.DataFrame:
        """
        Load the dataset (CSV format only) into a pandas DataFrame.

        Parameters
        ----------
        dataset_name : str, optional
            A descriptive name for the dataset (for logging).

        Returns
        -------
        pd.DataFrame
            The loaded dataset.
        """
        file_path = Path(self.config.data.get("local_raw", ""))

        if not file_path.exists():
            logger.error(f"[SourceDataLoader] Dataset file not found: {file_path}")
            raise FileNotFoundError(f"Dataset not found at: {file_path}")

        if file_path.suffix.lower() != ".csv":
            logger.error(f"[SourceDataLoader] Unsupported file format: {file_path.suffix}. Expected '.csv'.")
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Only '.csv' files are supported.")

        try:
            self.data = pd.read_csv(
                file_path,
                sep=self.config.data.get("sep", ";"),
                encoding=self.config.data.get("encoding", "utf-8"),
            )
            logger.success(f"[SourceDataLoader] '{dataset_name}' loaded successfully from: {file_path}")
            logger.info(f"[SourceDataLoader] Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")

            return self.data

        except pd.errors.EmptyDataError:
            logger.error("[SourceDataLoader] The CSV file is empty.")
            raise

        except pd.errors.ParserError as e:
            logger.error(f"[SourceDataLoader] CSV parsing error: {e}")
            raise

        except Exception as e:
            logger.exception(f"[SourceDataLoader] Unexpected error while loading dataset: {e}")
            raise

    # ==========================================================
    # Nouvelle méthode : inspection détaillée du dataset
    # ==========================================================
    def inspect_dataset(self) -> pd.DataFrame:
        """
        Provide a detailed inspection of the dataset columns.

        Returns
        -------
        pd.DataFrame
            A summary DataFrame containing:
            - Column name
            - Data type
            - Number of missing values
            - Percentage of missing values
            - Cardinality (number of unique values)
            - Example values (unique or representative)
        """
        if self.data is None:
            logger.warning("[SourceDataLoader] No dataset loaded yet. Call `load_dataset()` first.")
            return pd.DataFrame(columns=["Column", "Type", "Missing", "% Missing", "Cardinality", "Examples"])

        logger.info("[SourceDataLoader] Inspecting dataset structure...")

        total_rows = len(self.data)
        column_details = []

        for col in self.data.columns:
            col_type = self.data[col].dtype

            # Compute missing values
            missing_count = self.data[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100

            # Compute cardinality
            cardinality = self.data[col].nunique(dropna=True)

            # Representative examples
            if col_type == "object":
                examples = self.data[col].dropna().unique()[:10]
            else:
                examples = self.data[col].dropna().unique()[:5]

            column_details.append([
                col,
                col_type,
                missing_count,
                round(missing_pct, 2),
                cardinality,
                examples
            ])

        columns_df = pd.DataFrame(
            column_details,
            columns=["Column", "Type", "Missing", "% Missing", "Cardinality", "Examples"]
        )

        # Sort by missing percentage (descending)
        columns_df.sort_values(by="% Missing", ascending=False, inplace=True)

        # logger.info(f"[SourceDataLoader] Dataset contains {len(columns_df)} columns and {total_rows} rows.")
        # logger.info(f"[SourceDataLoader] Missing values overview:\n{columns_df[['Column', 'Missing', '% Missing']].head(10)}")

        return columns_df

