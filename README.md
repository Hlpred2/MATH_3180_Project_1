# MATH 3180 Project 1: Income Classification

## Team Members
- Harrison Perone
- Maggie Mannella
- Brendan Hurley
- Kyan Potenciano

## Overview
This project predicts whether a person earns more than $50K using the [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income).

The repository contains two modeling workflows and a shared comparison utility:
- Naive Bayes on tokenized, text-style features derived from tabular columns.
- Logistic Regression on mixed numeric/categorical features via sklearn pipelines.
- A comparison script that plots both confusion matrices on a shared color scale.

## Project Structure
- `Naive_Bayes.ipynb`
  - Uses `adult_preprocessing.py` to load and tokenize features.
  - Supports continuous-feature handling modes: `raw`, `binned`, `both`
  - Trains `BernoulliNB` with `CountVectorizer`
  - Produces:
    - accuracy, confusion matrix, classification report
    - precision/recall bar chart
    - confusion matrix heat map
    - feature-importance plots/tables
    - accuracy vs. maximum feature count plot
  - Exports `exports/naive_bayes_confusion_matrix.csv`

- `Logistic_Regression.ipynb`
  - Builds a `ColumnTransformer` pipeline:
    - numeric: median imputation + `StandardScaler`
    - categorical: most-frequent imputation + `OneHotEncoder`
  - Trains `LogisticRegression`
  - Produces:
    - accuracy, confusion matrix, classification report
    - precision/recall bar chart
    - confusion matrix heat map
    - feature-importance plots/tables
  - Exports `exports/logistic_regression_confusion_matrix.csv`

- `adult_preprocessing.py`
  - Preprocessing module for the Naive Bayes workflow
  - Maps raw Adult rows to named fields and normalized tokens
  - Bins continuous variables and reports bin-frequency summaries

- `plot_confusion_matrix_comparison.py`
  - Loads both exported confusion matrices
  - Generates side-by-side confusion matrices with:
    - shared color scale
    - TN/FP/FN/TP cell annotations
    - class labels (`<=50K`, `>50K`)
  - Saves `exports/confusion_matrix_comparison.png`.

- `census+income/`
  - Dataset files (`adult.data`, `adult.test`) and metadata

- `exports/`
  - Confusion matrix CSV exports and generated visual outputs

## Reproducible Workflow
Run from the project root in this order:

1. Open and run all cells in `Naive_Bayes.ipynb`
2. Open and run all cells in `Logistic_Regression.ipynb`
3. Generate the side-by-side comparison image:

```powershell
python plot_confusion_matrix_comparison.py
```

Expected confusion matrix exports before step 3:
- `exports/naive_bayes_confusion_matrix.csv`
- `exports/logistic_regression_confusion_matrix.csv`

## Notes
- Target classes are binary: `<=50K` and `>50K`
- `fnlwgt` is intentionally excluded from model features
