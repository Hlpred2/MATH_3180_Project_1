"""Shared plotting utilities for model comparison visuals."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def save_metric_bar_plot(summary_df: pd.DataFrame, metrics: Iterable[str], output_path: Path) -> None:
    """Save grouped bar chart for selected model metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))
    summary_df[list(metrics)].plot(kind="bar", ax=ax, rot=45)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison: key metrics")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_grid(cms: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save confusion matrices side by side with shared color scale."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    vmax = max(cm.max() for cm in cms.values())
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_plot(model_probs: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """Save ROC curve comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_score) in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pr_plot(model_probs: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """Save precision-recall curve comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_score) in model_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
