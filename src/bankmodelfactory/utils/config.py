"""
===========================================================
BankModelFactory - Configuration Loader
-----------------------------------------------------------
This module centralizes the loading of configuration files
(YAML) used across the ML pipeline.

It allows you to define all parameters (data, model, training, etc.)
in separate YAML files under the 'configs/' directory.

By separating configuration from code, the project remains:
    - modular
    - reproducible
    - easy to maintain
===========================================================
"""

import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    """
    Main configuration class that holds all loaded YAML sections.

    Attributes
    ----------
    data : Dict[str, Any]
        Parameters related to data ingestion, preprocessing and paths.
    (You can later add attributes such as 'features', 'model', 'train', etc.)
    """

    data: Dict[str, Any]

    @staticmethod
    def load(paths: Dict[str, str]) -> "Config":
        """
        Load multiple YAML configuration files and return a unified Config object.

        Parameters
        ----------
        paths : dict
            Dictionary mapping configuration section names to file paths.
            Example:
                {
                    "data": "configs/data.yaml",
                    "model": "configs/model.yaml",
                    "train": "configs/train.yaml"
                }

        Returns
        -------
        Config
            A dataclass instance containing all loaded YAML content.

        Example
        -------
        >>> cfg = Config.load({"data": "configs/data.yaml"})
        >>> print(cfg.data["local_raw"])
        'data/raw/bank-additional/bank-additional-full.csv'
        """
        cfg = {}  # Dictionary to store loaded YAML sections

        # Iterate over each config file provided
        for key, path in paths.items():
            with open(path, "r") as f:
                # Parse YAML file and store it under its key (e.g., 'data', 'model', etc.)
                cfg[key] = yaml.safe_load(f)

        # Instantiate the dataclass dynamically
        return Config(**cfg)
