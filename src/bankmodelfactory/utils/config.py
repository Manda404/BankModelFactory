"""
===========================================================
BankModelFactory - Configuration Loader
-----------------------------------------------------------
This module centralizes the loading of configuration files (YAML)
used across the ML pipeline.

It supports multiple configuration sections (data, model, train)
and dynamically merges them into a unified Config object.

By separating configuration from code, the project remains:
    - modular
    - reproducible
    - easy to maintain
===========================================================
"""

import yaml
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    """
    Main configuration class that holds all loaded YAML sections.

    Attributes
    ----------
    data : Dict[str, Any]
        Parameters related to data ingestion and preprocessing.
    train : Dict[str, Any]
        Parameters for training (epochs, learning rate, etc.)
    model : Dict[str, Any]
        Model-specific configuration.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(paths: Dict[str, str]) -> "Config":
        """
        Load multiple YAML configuration files and return a unified Config object.

        Parameters
        ----------
        paths : dict
            Mapping of configuration section names to YAML file paths.
            Example:
                {
                    "data": "configs/data.yaml",
                    "train": "configs/train.yaml",
                    "model": "configs/model.yaml"
                }

        Returns
        -------
        Config
            Instance containing all loaded configuration sections.

        Example
        -------
        >>> cfg = Config.load({"train": "configs/train.yaml"})
        >>> print(cfg.train["num_epochs"])
        20
        """
        cfg = {}

        for key, path in paths.items():
            with open(path, "r") as f:
                cfg[key] = yaml.safe_load(f)

        # Merge dynamically: only known sections + loaded ones
        return Config(**{**{k: {} for k in ["data", "train", "model"]}, **cfg})
