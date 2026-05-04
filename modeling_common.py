"""Shared data loading and preprocessing helpers for Adult Income models."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ADULT_COL_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def load_adult_dataframe(data_path: str = "census+income/adult.data") -> pd.DataFrame:
    """Load and minimally clean the Adult dataset used by model notebooks."""
    df = pd.read_csv(
        data_path,
        header=None,
        names=ADULT_COL_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )
    df = df.drop(columns=["fnlwgt"])
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split cleaned dataframe into X and binary target y."""
    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income"])
    return X, y


def build_tabular_preprocess() -> ColumnTransformer:
    """Build the standard numeric+categorical preprocessing pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 11,
):
    """Standardized train/test split used across notebooks and scripts."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
