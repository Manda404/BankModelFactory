"""
trainer.py
=========================
Training module for binary classification neural networks.

Responsibilities:
- Full training loop (train + validation)
- Compute loss, accuracy, and AUC
- Save best model checkpoint (with timestamp)
- Visualize and/or save training curves (with timestamp)
- Automatically ensures the model is on the correct device

Author: Manda Surel
Date: 2025-10-31
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from bankmodelfactory.utils.config import Config
from bankmodelfactory.utils.path import get_project_root
from bankmodelfactory.utils.logger import get_logger
from bankmodelfactory.utils.device import get_device

logger = get_logger()
device = get_device()


class Trainer:
    """Training manager for binary neural networks."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        scheduler=None,
        num_epochs: int = None,
    ):
        # ----------------------------------------------------
        # Safety check: ensure model is on correct device
        # ----------------------------------------------------
        self.model = model
        try:
            sample_param = next(model.parameters())
            if sample_param.device != torch.device(device):
                logger.warning(
                    f"Model detected on {sample_param.device}, moving to {device}."
                )
                self.model = model.to(device)
            else:
                logger.info(f"Model already on correct device: {device}.")
        except StopIteration:
            logger.warning("Model has no parameters — skipping device check.")

        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.scheduler = scheduler

        # ----------------------------------------------------
        # Load YAML configuration for training parameters
        # ----------------------------------------------------
        root = get_project_root()
        config_path = Path(root / "configs" / "train.yaml")
        config = Config.load({"train": str(config_path)})

        train_cfg = config.train.get("train")

        self.num_epochs = num_epochs or train_cfg.get("num_epochs", 10)
        self.visualize = train_cfg.get("visualize", True)
        self.save_plot = train_cfg.get("save_plot", True)
        self.plot_path = train_cfg.get("plot_path", None)
        self.checkpoint_path = train_cfg.get("checkpoint_path", None)

        # ----------------------------------------------------
        # Naming setup
        # ----------------------------------------------------
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name = getattr(model, "name", "UnnamedModel")

        # ----------------------------------------------------
        # Initialize history tracking
        # ----------------------------------------------------
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
        }

        logger.info(
            f"Trainer initialized for model '{self.model_name}' "
            f"on {device} for {self.num_epochs} epochs. "
            f"Visualization={self.visualize}, SavePlot={self.save_plot}, "
            f"Checkpoint={self.checkpoint_path}"
        )

    # ---------------------------------------------------------
    # 1. Run one epoch
    # ---------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool = True):
        """Run one epoch (training or validation)."""
        mode = "Train" if train else "Valid"
        total_loss = 0.0
        y_true, y_pred = [], []

        self.model.train(train)

        for X_batch, y_batch in tqdm(loader, desc=f"{mode} Epoch", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

            with torch.set_grad_enabled(train):
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                probs = torch.sigmoid(outputs)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(probs.detach().cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float("nan")

        logger.info(f"[{mode}] Loss: {avg_loss:.4f} | AUC: {auc:.4f}")
        return avg_loss, auc

    # ---------------------------------------------------------
    # 2. Full training loop
    # ---------------------------------------------------------
    def fit(self, train_loader: DataLoader, valid_loader: DataLoader = None):
        """Train the model and optionally visualize/save learning curves."""
        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"Epoch [{epoch}/{self.num_epochs}] -------------------")

            train_loss, train_auc = self._run_epoch(train_loader, train=True)
            self.history["train_loss"].append(train_loss)
            self.history["train_auc"].append(train_auc)

            val_loss, val_auc = (None, None)
            if valid_loader is not None:
                val_loss, val_auc = self._run_epoch(valid_loader, train=False)
                self.history["val_loss"].append(val_loss)
                self.history["val_auc"].append(val_auc)

                # Save best model (only if path is defined)
                if self.checkpoint_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

                    model_filename = f"{self.model_name}_{self.timestamp}_best.pt"
                    save_path = os.path.join(
                        os.path.dirname(self.checkpoint_path), model_filename
                    )

                    torch.save(self.model.state_dict(), save_path)
                    logger.success(
                        f"New best model saved at epoch {epoch} → {save_path} "
                        f"(val_loss={val_loss:.4f}, val_auc={val_auc:.4f})"
                    )
                elif not self.checkpoint_path:
                    logger.warning(
                        "Checkpoint path not defined — skipping model saving."
                    )

            # Scheduler step
            if self.scheduler:
                if "ReduceLROnPlateau" in self.scheduler.__class__.__name__:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        logger.success("Training complete.")
        if valid_loader is not None:
            logger.info(f"Best validation loss: {best_val_loss:.4f}")

        # --- Plot visualization if configured ---
        if self.visualize or self.save_plot:
            self._plot_training_curves(save=self.save_plot, path=self.plot_path)

    # ---------------------------------------------------------
    # 3. Plot training curves
    # ---------------------------------------------------------
    def _plot_training_curves(self, save: bool = True, path: str = None):
        """Visualize and/or save training and validation curves."""
        if len(self.history["val_loss"]) == 0:
            logger.warning("No validation data provided — skipping visualization.")
            return

        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(12, 5))

        # Left: Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Right: AUC
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_auc"], label="Train AUC", linewidth=2)
        plt.plot(epochs, self.history["val_auc"], label="Validation AUC", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.title("Training vs Validation AUC")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save or show only if path is defined
        if save and path:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(
                path,
                f"{self.model_name}_training_curves_{self.timestamp}.png"
            )
            plt.savefig(file_path, bbox_inches="tight")
            logger.success(f"Training curves saved to: {file_path}")
        elif save and not path:
            logger.warning("Plot path not defined — skipping plot saving.")

        plt.show()

    # ---------------------------------------------------------
    # 4. Evaluation
    # ---------------------------------------------------------
    def evaluate(self, loader: DataLoader):
        """Evaluate the trained model on a given DataLoader."""
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                probs = torch.sigmoid(outputs)

                total_loss += loss.item() * X_batch.size(0)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float("nan")

        logger.info(f"[Evaluation] Loss: {avg_loss:.4f} | AUC: {auc:.4f}")
        return avg_loss, auc

    # ---------------------------------------------------------
    # 5. Get trained model
    # ---------------------------------------------------------
    def get_model(self) -> nn.Module:
        """Return the trained model."""
        return self.model