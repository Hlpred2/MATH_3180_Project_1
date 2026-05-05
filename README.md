# MATH 3180 Project 1: Income Classification

## Team Members
- Harrison Perone
- Maggie Mannella
- Brendan Hurley
- Kyan Potenciano

## Overview
This project predicts whether a person earns more than $50K using the [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income).

The repository contains four modeling workflows, exploratory analysis, and a model comparison utility:
- **Data Exploration**: Initial analysis of dataset characteristics and distributions
- **Naive Bayes**: Text-tokenized features with `BernoulliNB` classifier
- **Logistic Regression**: Mixed numeric/categorical features via sklearn pipelines
- **Decision Tree**: Tree-based classification on preprocessed features
- **Neural Network**: Multi-layer perceptron on standardized features
- **Model Comparison**: Compares performance metrics across all four models

## Project Structure
- `Data_Exploration.ipynb`
  - Exploratory data analysis of the Adult Census Income dataset
  - Investigates feature distributions, data types, and class balance
  - Provides statistical summaries and initial visualizations
  - Helps inform feature engineering decisions for modeling

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
  - Exports `exports/naive_bayes_confusion_matrix.csv` and `exports/naive_bayes_probs.csv`

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
  - Exports `exports/logistic_regression_confusion_matrix.csv` and `exports/logistic_regression_probs.csv`

- `Decision_Tree.ipynb`
  - Preprocesses features using `ColumnTransformer` pipeline
  - Trains `DecisionTreeClassifier` with optimized depth and parameters
  - Produces:
    - accuracy, confusion matrix, classification report
    - feature-importance bar chart
    - tree visualization
    - confusion matrix heat map
  - Exports `exports/decision_tree_confusion_matrix.csv` and `exports/decision_tree_probs.csv`

- `Neural_Network.ipynb`
  - Preprocesses features with normalization and scaling
  - Trains `MLPClassifier` (multi-layer perceptron)
  - Produces:
    - accuracy, confusion matrix, classification report
    - learning curves and loss plots
    - confusion matrix heat map
    - feature-importance analysis
  - Exports `exports/mlp_confusion_matrix.csv` and `exports/mlp_probs.csv`

- `adult_preprocessing.py`
  - Preprocessing module for the Naive Bayes workflow
  - Maps raw Adult rows to named fields and normalized tokens
  - Bins continuous variables and reports bin-frequency summaries

- `compare_models.py`
  - Compares performance metrics across all four models
  - Generates a consolidated metrics report
  - Creates visualizations comparing model performance
  - Exports `exports/model_comparison_metrics.csv`

- `census+income/`
  - Dataset files and metadata

- `exports/`
  - Confusion matrix CSV exports, probability predictions, and generated visual outputs

## Reproducible Workflow
Run from the project root in this order:

1. Open and run all cells in `Data_Exploration.ipynb` for initial analysis
2. Open and run all cells in `Naive_Bayes.ipynb`
3. Open and run all cells in `Logistic_Regression.ipynb`
4. Open and run all cells in `Decision_Tree.ipynb`
5. Open and run all cells in `Neural_Network.ipynb`
6. Generate the model comparison report:

```powershell
python compare_models.py
```

Expected exports after steps 2-5:
- `exports/naive_bayes_confusion_matrix.csv` and `exports/naive_bayes_probs.csv`
- `exports/logistic_regression_confusion_matrix.csv` and `exports/logistic_regression_probs.csv`
- `exports/decision_tree_confusion_matrix.csv` and `exports/decision_tree_probs.csv`
- `exports/mlp_confusion_matrix.csv` and `exports/mlp_probs.csv`

After step 6:
- `exports/model_comparison_metrics.csv`

## Notes
- Target classes are binary: `<=50K` and `>50K`
- `fnlwgt` is intentionally excluded from model features
