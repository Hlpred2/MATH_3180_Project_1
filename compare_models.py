"""
compare_models.py

Loads saved confusion matrices (CSV) from the `exports/` folder for the models
Naive Bayes, Logistic Regression, MLP, and Decision Tree, computes standard
classification metrics (accuracy, precision, recall, f1, balanced accuracy, MCC)
from the confusion matrices, saves a summary CSV, and creates a comparison plot
and side-by-side confusion matrix figure saved to `exports/`.

Usage: python compare_models.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

EXPORT_DIR = Path("exports")
MODEL_CMS = {
    "Naive Bayes": EXPORT_DIR / "naive_bayes_confusion_matrix.csv",
    "Logistic Regression": EXPORT_DIR / "logistic_regression_confusion_matrix.csv",
    "MLP": EXPORT_DIR / "mlp_confusion_matrix.csv",
    "Decision Tree": EXPORT_DIR / "decision_tree_confusion_matrix.csv",
}

MODEL_PROBS = {
    "Naive Bayes": EXPORT_DIR / "naive_bayes_probs.csv",
    "Logistic Regression": EXPORT_DIR / "logistic_regression_probs.csv",
    "MLP": EXPORT_DIR / "mlp_probs.csv",
    "Decision Tree": EXPORT_DIR / "decision_tree_probs.csv",
}

os.makedirs(EXPORT_DIR, exist_ok=True)

def load_cm(path: Path):
    if not path.exists():
        return None
    cm = np.loadtxt(path, delimiter=",", dtype=int)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 matrix at {path}, got {cm.shape}")
    return cm

rows = []
cms = {}
for name, path in MODEL_CMS.items():
    cm = load_cm(path)
    if cm is None:
        print(f"Warning: missing confusion matrix for {name} at {path}")
        continue
    cms[name] = cm
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    balanced = ((tp / (tp + fn) if (tp + fn) else 0.0) + (tn / (tn + fp) if (tn + fp) else 0.0)) / 2
    # Matthews correlation coefficient computed from cm
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    rows.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(total),
    })

if not rows:
    print("No confusion matrices found. Exiting.")
    raise SystemExit(1)

summary_df = pd.DataFrame(rows).set_index("model")
summary_csv = EXPORT_DIR / "model_comparison_metrics.csv"
summary_df.to_csv(summary_csv)
print(f"Saved summary metrics to: {summary_csv}")

# Plot selected metrics as grouped bars
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
fig, ax = plt.subplots(figsize=(10, 5))
summary_df[metrics_to_plot].plot(kind="bar", ax=ax, rot=45)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model comparison: key metrics")
plt.tight_layout()
plot_path = EXPORT_DIR / "model_comparison_metrics.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved metrics plot to: {plot_path}")

# Side-by-side confusion matrices with shared color scale
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

cm_plot_path = EXPORT_DIR / "confusion_matrices_comparison.png"
fig.savefig(cm_plot_path, dpi=150)
print(f"Saved confusion matrix comparison to: {cm_plot_path}")

print("\nSummary metrics:\n")
print(summary_df)

model_probs = {}
for name, path in MODEL_PROBS.items():
    if not path.exists():
        print(f"Warning: missing probability file for {name} at {path}")
        continue

    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1 or arr.shape[1] < 2:
        print(f"Warning: invalid probability file format for {name} at {path}")
        continue

    y_true = arr[:, 0].astype(int)
    y_score = arr[:, 1]
    model_probs[name] = (y_true, y_score)
    print(f"Loaded probability file for {name}: {path}")

if not model_probs:
    print("\nNo probability files found for ROC/PR curves. Run each model notebook to export *_probs.csv files.")
    raise SystemExit(1)

# ROC plot
plt.figure(figsize=(8, 6))
for name, (y_true, y_score) in model_probs.items():
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = EXPORT_DIR / "model_comparison_roc.png"
plt.savefig(roc_path, dpi=150)
print(f"Saved ROC curves to: {roc_path}")

# Precision-Recall plot
plt.figure(figsize=(8, 6))
for name, (y_true, y_score) in model_probs.items():
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc="lower left")
plt.tight_layout()
pr_path = EXPORT_DIR / "model_comparison_pr.png"
plt.savefig(pr_path, dpi=150)
print(f"Saved PR curves to: {pr_path}")
