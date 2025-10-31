"""
bank_dataset.py
----------------
PyTorch-ready Dataset for the Bank Marketing project.

Features:
- Converts pandas DataFrame to torch.Tensor
- Automatically handles device placement (CPU, CUDA, or MPS)
- Includes strong input validation
- Compatible with PyTorch DataLoader

Author: Manda Surel
Date: 2025-10-31
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from bankmodelfactory.utils.logger import get_logger
from bankmodelfactory.utils.device import get_device

logger = get_logger()


class BankMarketingDataset(Dataset):
    """
    Custom PyTorch Dataset for the Bank Marketing data.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing feature columns.
    y : np.ndarray or pd.Series
        Target values (numeric or convertible to float).
    device : str, optional
        Device to send tensors to ('cuda', 'mps', or 'cpu').
        Automatically detects the best available device if not specified.
    dtype : torch.dtype, default=torch.float32
        Data type for feature tensors.
    as_tensor : bool, default=True
        If True, returns torch tensors; otherwise returns numpy arrays.

    Example
    -------
    >>> dataset = BankMarketingDataset(X_train_ready, y_train_ready)
    >>> len(dataset)
    40689
    >>> X_sample, y_sample = dataset[0]
    >>> X_sample.shape, y_sample
    (52,), tensor(0.)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        as_tensor: bool = True,
    ):
        # ---------- Validation ----------
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy array or pandas Series.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if X.empty:
            raise ValueError("X DataFrame is empty.")

        # Ensure all features are numeric
        if not np.all([np.issubdtype(dt, np.number) for dt in X.dtypes]):
            raise ValueError("All feature columns in X must be numeric to create PyTorch tensors.")

        # ---------- Configuration ----------
        self.device = device or get_device()
        self.dtype = dtype
        self.as_tensor = as_tensor

        # ---------- Conversion ----------
        self.X = X.to_numpy(copy=False)
        self.y = np.asarray(y, dtype=np.float32)

        logger.info(
            f"[BankMarketingDataset] Initialized with {len(self)} samples "
            f"and {self.X.shape[1]} features on device='{self.device}'."
        )

    # ---------- Required Methods ----------
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """Return one sample (features, target)."""
        X_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.as_tensor:
            X_sample = torch.tensor(X_sample, dtype=self.dtype, device=self.device)
            y_sample = torch.tensor(y_sample, dtype=torch.float32, device=self.device)

        return X_sample, y_sample
