"""
preview.py
-----------
Utility module for pretty-printing and debugging DataLoaders in BankModelFactory.

Responsibilities:
- Provide colorized console output for better readability
- Allow quick preview of batches (features + labels)
- Integrate with the global project logger

Author: Manda Surel
Date: 2025-10-30
"""

from bankmodelfactory.utils.logger import get_logger as setup_logger

logger = setup_logger()


class clr:
    """Simple ANSI color helper for clean console output."""

    R = "\033[91m"   # Rouge
    G = "\033[92m"   # Vert
    Y = "\033[93m"   # Jaune
    B = "\033[94m"   # Bleu
    M = "\033[95m"   # Magenta
    C = "\033[96m"   # Cyan
    E = "\033[0m"    # Reset (fin de couleur)

    # Alias pour compatibilité
    S = R  # S = Start color → rouge


def preview_dataloader(name: str, loader, n_batches: int = 3):
    """
    Display a colored preview of DataLoader batches in the console.

    Parameters
    ----------
    name : str
        Name or identifier of the DataLoader (e.g., 'Train', 'Valid', etc.).
    loader : torch.utils.data.DataLoader
        A DataLoader instance (can be wrapped in a custom class).
    n_batches : int, default=3
        Number of batches to preview.
    """
    print("\n" + "=" * 60)
    print(f"{clr.B}{name} DataLoader Preview{clr.E}")
    print("=" * 60 + "\n")

    for k, (features, labels) in enumerate(loader):
        print(
            f"{clr.Y}Batch {k}{clr.E}\n"
            f"{clr.C}Features:{clr.E} shape = {tuple(features.shape)}\n"
            f"{clr.M}Labels:{clr.E} {labels[:10].tolist()} ...\n"
            + "-" * 50
        )

        if k >= n_batches - 1:
            break

    print(f"\n{clr.G}✔ Preview completed for {name} DataLoader{clr.E}\n")
    logger.info(f"[preview_dataloader] Displayed {n_batches} batches from '{name}'.")
