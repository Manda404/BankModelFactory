"""
bankmodelfactory.data
---------------------
Data management and preparation module for the BankModelFactory project.

Includes:
- Data loading utilities (load.py)
- Data splitting utilities (split.py)
- PyTorch-ready dataset and dataloader (bank_dataset.py, bank_dataloader.py)

Main Components
---------------
BankMarketingDataset : torch.utils.data.Dataset
    Converts a pandas DataFrame into torch tensors and handles device placement
    (CPU, CUDA, or MPS depending on hardware).

BankMarketingDataLoader : DataLoader wrapper
    Provides batched and shuffled access to BankMarketingDataset for efficient
    deep learning training and evaluation.

load_dataset : function
    Loads raw or preprocessed datasets as pandas DataFrame from configured paths.

split_data : function
    Splits dataset into train/validation/test subsets with configurable ratios.

Author: Manda Surel
Date: 2025-10-30
"""

# ---------------------------------------------------------------------
# Import key classes and functions for easy access
# ---------------------------------------------------------------------
from bankmodelfactory.data.bank_dataset import BankMarketingDataset
#from bankmodelfactory.data.bank_dataloader import BankMarketingDataLoader

# Optional (if they exist in your project):
#from bankmodelfactory.data.load import load_dataset
#from bankmodelfactory.data.split import split_data

# ---------------------------------------------------------------------
# Define what is publicly accessible when importing the module
# ---------------------------------------------------------------------
__all__ = [
#    "BankMarketingDataset",
    "BankMarketingDataLoader",
#    "load_dataset",
#    "split_data",
]
