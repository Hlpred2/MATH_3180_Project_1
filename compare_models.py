"""
compare_models.py

Loads saved confusion matrices (CSV) from the `exports/` folder for the models
Naive Bayes, Logistic Regression, MLP, and Decision Tree, computes standard
classification metrics (accuracy, precision, recall, f1, balanced accuracy, MCC)
from the confusion matrices, saves a summary CSV, and creates a comparison plot
and side-by-side confusion matrix figure saved to `exports/`.

Usage: python compare_models.py
"""

from pathlib import Path
import pandas as pd

from evaluation_utils import (
    ensure_dir,
    load_confusion_matrix,
    load_probability_file,
    metrics_from_confusion_matrix,
)
from plotting_utils import (
    save_confusion_grid,
    save_metric_bar_plot,
    save_pr_plot,
    save_roc_plot,
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

ensure_dir(EXPORT_DIR)

rows = []
cms = {}
for name, path in MODEL_CMS.items():
    cm = load_confusion_matrix(path)
    if cm is None:
        print(f"Warning: missing confusion matrix for {name} at {path}")
        continue
    cms[name] = cm

    metrics = metrics_from_confusion_matrix(cm)
    metrics["model"] = name
    rows.append(metrics)

if not rows:
    print("No confusion matrices found. Exiting.")
    raise SystemExit(1)

summary_df = pd.DataFrame(rows).set_index("model")
summary_csv = EXPORT_DIR / "model_comparison_metrics.csv"
summary_df.to_csv(summary_csv)
print(f"Saved summary metrics to: {summary_csv}")

# Plot selected metrics as grouped bars
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
plot_path = EXPORT_DIR / "model_comparison_metrics.png"
save_metric_bar_plot(summary_df, metrics_to_plot, plot_path)
print(f"Saved metrics plot to: {plot_path}")

# Side-by-side confusion matrices with shared color scale
cm_plot_path = EXPORT_DIR / "confusion_matrices_comparison.png"
save_confusion_grid(cms, cm_plot_path)
print(f"Saved confusion matrix comparison to: {cm_plot_path}")

print("\nSummary metrics:\n")
print(summary_df)

model_probs = {}
for name, path in MODEL_PROBS.items():
    loaded = load_probability_file(path)
    if loaded is None:
        print(f"Warning: missing probability file for {name} at {path}")
        continue

    y_true, y_score = loaded
    model_probs[name] = (y_true, y_score)
    print(f"Loaded probability file for {name}: {path}")

if not model_probs:
    print("\nNo probability files found for ROC/PR curves. Run each model notebook to export *_probs.csv files.")
    raise SystemExit(1)

roc_path = EXPORT_DIR / "model_comparison_roc.png"
save_roc_plot(model_probs, roc_path)
print(f"Saved ROC curves to: {roc_path}")

pr_path = EXPORT_DIR / "model_comparison_pr.png"
save_pr_plot(model_probs, pr_path)
print(f"Saved PR curves to: {pr_path}")
