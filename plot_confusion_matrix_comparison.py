import os
import numpy as np
import matplotlib.pyplot as plt


NAIVE_BAYES_CM_PATH = os.path.join("exports", "naive_bayes_confusion_matrix.csv")
LOGISTIC_CM_PATH = os.path.join("exports", "logistic_regression_confusion_matrix.csv")
OUTPUT_PATH = os.path.join("exports", "confusion_matrix_comparison.png")

CELL_LABELS = [["TN", "FP"], ["FN", "TP"]]


def load_confusion_matrix(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing confusion matrix export: {path}")

    cm = np.loadtxt(path, delimiter=",", dtype=int)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix at {path}, got shape {cm.shape}")
    return cm


def annotate_confusion_matrix(ax: plt.Axes, cm: np.ndarray) -> None:
    for i in range(2):
        for j in range(2):
            label = CELL_LABELS[i][j]
            ax.text(j, i, f"{cm[i, j]}\n({label})", ha="center", va="center")


def main() -> None:
    cm_nb = load_confusion_matrix(NAIVE_BAYES_CM_PATH)
    cm_lr = load_confusion_matrix(LOGISTIC_CM_PATH)

    # Shared scale makes the color intensity directly comparable across plots.
    shared_vmax = int(max(cm_nb.max(), cm_lr.max()))

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True
    )

    im_nb = axes[0].imshow(cm_nb, cmap="Blues", vmin=0, vmax=shared_vmax)
    annotate_confusion_matrix(axes[0], cm_nb)
    axes[0].set_xticks([0, 1], labels=["<=50K", ">50K"])
    axes[0].set_yticks([0, 1], labels=["<=50K", ">50K"])
    axes[0].set_title("Naive Bayes")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    axes[1].imshow(cm_lr, cmap="Blues", vmin=0, vmax=shared_vmax)
    annotate_confusion_matrix(axes[1], cm_lr)
    axes[1].set_xticks([0, 1], labels=["<=50K", ">50K"])
    axes[1].set_yticks([0, 1], labels=["<=50K", ">50K"])
    axes[1].set_title("Logistic Regression")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")

    fig.colorbar(im_nb, ax=axes.ravel().tolist(), shrink=0.9, label="Count")

    fig.suptitle("Confusion Matrix Comparison (Shared Color Scale)")

    os.makedirs("exports", exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved comparison plot to: {OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
