# MATH 3180 Project 1: Income Classification

## Overview
This project predicts whether a person earns more than $50K using the Adult Census Income dataset.

Two modeling approaches are implemented and compared:
- Naive Bayes with engineered text-style tokens from structured tabular features.
- Logistic Regression with mixed-type preprocessing (numeric scaling + one-hot encoding for categoricals).

The project also exports confusion matrices and produces a shared-scale comparison plot.

## Project Structure
- `Naive_Bayes.ipynb`
  - Loads and preprocesses the dataset via `adult_preprocessing.py`.
  - Engineers column-aware tokens (supports raw, binned, or both for continuous features).
  - Trains a Bernoulli Naive Bayes model with `CountVectorizer`.
  - Reports accuracy, confusion matrix, and classification report.
  - Exports Naive Bayes confusion matrix to `exports/naive_bayes_confusion_matrix.csv`.

- `Logistic_Regression.ipynb`
  - Loads the same dataset with explicit schema.
  - Drops `fnlwgt` from features.
  - Uses `ColumnTransformer`:
    - Numeric pipeline: median imputation + standard scaling.
    - Categorical pipeline: most-frequent imputation + one-hot encoding.
  - Trains a logistic regression classifier.
  - Reports accuracy, confusion matrix, and classification report.
  - Exports Logistic Regression confusion matrix to `exports/logistic_regression_confusion_matrix.csv`.

- `adult_preprocessing.py`
  - Central preprocessing module used by the Naive Bayes workflow.
  - Handles field mapping, token normalization, continuous-feature binning, and mode control (`raw`, `binned`, `both`).
  - Provides bin-frequency summaries to inspect whether bin definitions capture the data distribution well.

- `plot_confusion_matrix_comparison.py`
  - Reads both exported confusion matrices.
  - Plots side-by-side confusion matrices with:
    - shared color scale,
    - TN/FP/FN/TP annotations,
    - responsive font scaling for resize readability.
  - Saves the final figure to `exports/confusion_matrix_comparison.png`.

- `census+income/`
  - Raw dataset and metadata files used by both modeling notebooks.

- `exports/`
  - Generated confusion matrix CSV files and comparison figure.

## How to Regenerate Comparison Plot
From the project root:

```powershell
python plot_confusion_matrix_comparison.py
```

This script expects these files to exist:
- `exports/naive_bayes_confusion_matrix.csv`
- `exports/logistic_regression_confusion_matrix.csv`

## Notes
- The target is binary: `<=50K` vs `>50K`.
- `fnlwgt` is intentionally excluded as an input feature in the current model setup.
- Binning support is included to stabilize sparse count-based features in the Naive Bayes pipeline.
