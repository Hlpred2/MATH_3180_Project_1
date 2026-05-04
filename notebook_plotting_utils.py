"""
Shared plotting utilities for notebook visualizations.

Provides reusable functions for confusion matrix heatmaps and metrics bar charts
that are duplicated across model training notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", figsize: tuple = (7, 5)) -> None:
    """
    Plot confusion matrix as annotated heatmap with standard formatting.
    
    Displays True Negatives, False Positives, False Negatives, True Positives.
    
    Args:
        cm: 2x2 confusion matrix (shape must be (2, 2))
        title: Plot title
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_with_labels(cm: np.ndarray, title: str = "Confusion Matrix", figsize: tuple = (5, 4)) -> None:
    """
    Plot confusion matrix with TN/FP/FN/TP cell labels.
    
    Annotates each cell with both count and label (TN, FP, FN, TP).
    
    Args:
        cm: 2x2 confusion matrix (shape must be (2, 2))
        title: Plot title
        figsize: Figure size as (width, height)
    """
    cell_labels = np.array([["TN", "FP"], ["FN", "TP"]])
    annot_labels = np.array([
        [f"{cm[i, j]}\n({cell_labels[i, j]})" for j in range(2)]
        for i in range(2)
    ], dtype=object)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_metrics_bar_chart(metrics_long, title: str = "Precision and Recall by Class", figsize: tuple = (8, 5)) -> None:
    """
    Plot precision and recall as grouped bar chart by class.
    
    Args:
        metrics_long: DataFrame with columns [class, metric, score]
                      where metric is 'precision' or 'recall'
        title: Plot title
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=metrics_long,
        x="class",
        y="score",
        hue="metric",
        palette=["#1f77b4", "#ff7f0e"],
        ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Class (0: <=50K, 1: >50K)")
    ax.set_ylabel("Score")
    ax.legend(title="Metric", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()
