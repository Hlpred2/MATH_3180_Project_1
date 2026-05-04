"""Shared evaluation and export helpers for model outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def load_confusion_matrix(path: Path) -> Optional[np.ndarray]:
    """Load a 2x2 confusion matrix CSV. Returns None if path is missing."""
    if not path.exists():
        return None
    cm = np.loadtxt(path, delimiter=",", dtype=int)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix at {path}, got {cm.shape}")
    return cm


def load_probability_file(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a (y_true, y_score) file. Returns None if missing or malformed."""
    if not path.exists():
        return None

    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1 or arr.shape[1] < 2:
        return None

    y_true = arr[:, 0].astype(int)
    y_score = arr[:, 1]
    return y_true, y_score


def metrics_from_confusion_matrix(cm: np.ndarray) -> Dict[str, float]:
    """Compute standard binary metrics from a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    balanced = (
        (tp / (tp + fn) if (tp + fn) else 0.0)
        + (tn / (tn + fp) if (tn + fp) else 0.0)
    ) / 2

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    return {
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
    }
