"""
===========================================================
Evaluator Module - BankModelFactory
-----------------------------------------------------------
Evaluate trained binary classification models on test data.

Responsibilities:
- Compute technical and business metrics
- Visualize Confusion Matrix, ROC, Lift & Gain Curves
- Optional saving of plots with automatic model naming

Author: Manda Surel
Date: 2025-11-01
===========================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy.stats import ks_2samp

from bankmodelfactory.utils.logger import get_logger
from bankmodelfactory.utils.device import get_device
from bankmodelfactory.utils.path import get_project_root

logger = get_logger()
device = get_device()


class Evaluator:
    """Comprehensive evaluator for binary classification models."""

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        save: bool = True,
        save_path: str = None,
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model to evaluate.
        test_loader : DataLoader
            Loader for the test dataset.
        save : bool, default=True
            Whether to save evaluation plots.
        save_path : str, optional
            Custom path to save plots. Defaults to `<project_root>/reports`.
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.save = save

        # Default save path = /reports
        root = get_project_root()
        default_report_path = Path(root / "reports")

        self.save_path = Path(save_path) if save_path else default_report_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.model_name = getattr(model, "name", model.__class__.__name__)
        self.y_true, self.y_pred, self.y_prob = [], [], []

        logger.info(
            f"Evaluator initialized for model '{self.model_name}' — "
            f"save={'ON' if self.save else 'OFF'} → {self.save_path}"
        )

    # ---------------------------------------------------------
    # Evaluation Loop
    # ---------------------------------------------------------
    def evaluate(self):
        """Run inference on the test dataset."""
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = self.model(X_batch)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs >= 0.5).float()

                self.y_true.extend(y_batch.cpu().numpy())
                self.y_pred.extend(preds.cpu().numpy())
                self.y_prob.extend(probs.cpu().numpy())

        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(self.y_pred)
        self.y_prob = np.array(self.y_prob)

        logger.success("Evaluation completed — predictions generated.")
        return self.y_true, self.y_pred, self.y_prob

    # ---------------------------------------------------------
    # Technical Metrics
    # ---------------------------------------------------------
    def compute_metrics(self):
        """Compute accuracy, precision, recall, F1, and ROC-AUC."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average="binary", zero_division=0
        )
        acc = np.mean(self.y_true == self.y_pred)
        auc_score = roc_auc_score(self.y_true, self.y_prob)

        metrics = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc_score,
        }

        logger.info("Technical Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k:<12}: {v:.4f}")
        return metrics

    # ---------------------------------------------------------
    # Business Metrics
    # ---------------------------------------------------------
    def compute_ks_statistic(self):
        """Kolmogorov–Smirnov (KS) statistic."""
        positives = self.y_prob[self.y_true == 1]
        negatives = self.y_prob[self.y_true == 0]
        ks_value = ks_2samp(positives, negatives).statistic
        logger.info(f"KS Statistic: {ks_value:.4f}")
        return ks_value

    def compute_lift_at_k(self, k: float = 0.1):
        """Compute Lift@K."""
        n = int(len(self.y_true) * k)
        order = np.argsort(self.y_prob)[::-1]
        top_k = order[:n]
        lift = (np.mean(self.y_true[top_k]) / np.mean(self.y_true))
        logger.info(f"Lift@{int(k*100)}%: {lift:.3f}")
        return lift

    def compute_capture_rate_at_k(self, k: float = 0.1):
        """Compute Capture Rate@K."""
        n = int(len(self.y_true) * k)
        order = np.argsort(self.y_prob)[::-1]
        top_k = order[:n]
        capture_rate = np.sum(self.y_true[top_k]) / np.sum(self.y_true)
        logger.info(f"Capture Rate@{int(k*100)}%: {capture_rate:.3f}")
        return capture_rate

    def compute_expected_profit(self, revenue_per_tp=100, cost_per_fp=10):
        """Estimate expected profit."""
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        profit = (tp * revenue_per_tp) - (fp * cost_per_fp)
        logger.info(f"Expected Profit: {profit:.2f}")
        return profit

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    def plot_confusion_roc(self):
        """Plot confusion matrix and ROC curve."""
        cm = confusion_matrix(self.y_true, self.y_pred, normalize="true")
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(12, 5))

        # Confusion Matrix
        plt.subplot(1, 2, 1)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", colorbar=False, ax=plt.gca())
        plt.title(f"{self.model_name} - Normalized Confusion Matrix")

        # ROC Curve
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{self.model_name} - ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if self.save:
            path = self.save_path / f"confusion_roc_{self.model_name}.png"
            plt.savefig(path, bbox_inches="tight")
            logger.success(f"Confusion matrix & ROC saved → {path}")

        plt.show()

    def plot_lift_curve(self):
        """Plot lift and cumulative gain curves."""
        order = np.argsort(self.y_prob)[::-1]
        y_true_sorted = self.y_true[order]

        cum_positives = np.cumsum(y_true_sorted)
        total_positives = np.sum(y_true_sorted)
        total_samples = len(y_true_sorted)

        perc_samples = np.arange(1, total_samples + 1) / total_samples
        gains = cum_positives / total_positives
        lift = gains / perc_samples

        plt.figure(figsize=(12, 5))

        # Cumulative Gain
        plt.subplot(1, 2, 1)
        plt.plot(perc_samples, gains, label="Cumulative Gains", color="blue", lw=2)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Proportion of Population")
        plt.ylabel("Proportion of Positives Captured")
        plt.title(f"{self.model_name} - Cumulative Gain Curve")
        plt.legend()
        plt.grid(alpha=0.3)

        # Lift Curve
        plt.subplot(1, 2, 2)
        plt.plot(perc_samples, lift, label="Lift", color="green", lw=2)
        plt.axhline(1, color="gray", linestyle="--")
        plt.xlabel("Proportion of Population")
        plt.ylabel("Lift")
        plt.title(f"{self.model_name} - Lift Curve")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if self.save:
            path = self.save_path / f"lift_curve_{self.model_name}.png"
            plt.savefig(path, bbox_inches="tight")
            logger.success(f"Lift and Gains curves saved → {path}")

        plt.show()
