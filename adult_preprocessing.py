"""Preprocessing utilities for the Adult Census Income dataset.

This module centralizes the feature engineering used by the notebook so Cell 2
stays short and readable.

What this file does:
1. Reads raw rows from ``census+income/adult.data``.
2. Selects model features and intentionally excludes ``fnlwgt`` as an input
    feature.
3. Maps each row into named fields (for example ``age``, ``education_num``).
4. Converts fields into text tokens for a Naive Bayes + CountVectorizer
    pipeline.
5. Applies configurable handling for continuous fields:
    - ``raw``: keep only column-tagged raw numeric tokens.
    - ``binned``: keep only column-tagged bin tokens.
    - ``both``: include both token types.
6. Returns ``demographics`` text records plus binary income labels.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass
class AdultPreprocessConfig:
    """Runtime options for data path and continuous-feature encoding mode."""

    data_path: str = os.path.join("census+income", "adult.data")
    continuous_mode: str = "both"  # one of: raw, binned, both


class AdultIncomePreprocessor:
    """Converts Adult rows into tokenized features and binary income labels."""

    feature_cols: List[str] = [
        "age",
        "workclass",
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
    ]

    def __init__(self, config: AdultPreprocessConfig | None = None) -> None:
        self.config = config or AdultPreprocessConfig()
        if self.config.continuous_mode not in {"raw", "binned", "both"}:
            raise ValueError("continuous_mode must be one of: 'raw', 'binned', 'both'")

        self.continuous_bin_map: Dict[str, Callable[[str], str]] = {
            "age": self.bin_age,
            "education_num": self.bin_education_num,
            "capital_gain": self.bin_capital_gain,
            "capital_loss": self.bin_capital_loss,
            "hours_per_week": self.bin_hours_per_week,
        }

    @staticmethod
    def normalize_token(value: str) -> str:
        value = value.strip()
        if value == "?":
            value = "UNKNOWN"
        return re.sub(r"[^A-Za-z0-9]+", "_", value)

    @staticmethod
    def bin_age(value: str) -> str:
        x = int(value)
        if x < 25:
            return "lt25"
        if x < 35:
            return "25_34"
        if x < 45:
            return "35_44"
        if x < 55:
            return "45_54"
        if x < 65:
            return "55_64"
        return "65_plus"

    @staticmethod
    def bin_education_num(value: str) -> str:
        x = int(value)
        if x <= 8:
            return "low"
        if x <= 12:
            return "mid"
        if x <= 14:
            return "high"
        return "very_high"

    @staticmethod
    def bin_capital_gain(value: str) -> str:
        x = int(value)
        if x == 0:
            return "zero"
        if x < 5000:
            return "1_4999"
        if x < 10000:
            return "5000_9999"
        return "10000_plus"

    @staticmethod
    def bin_capital_loss(value: str) -> str:
        x = int(value)
        if x == 0:
            return "zero"
        if x < 1000:
            return "1_999"
        if x < 2000:
            return "1000_1999"
        return "2000_plus"

    @staticmethod
    def bin_hours_per_week(value: str) -> str:
        x = int(value)
        if x < 20:
            return "lt20"
        if x < 35:
            return "20_34"
        if x < 45:
            return "35_44"
        if x <= 50:
            return "45_50"
        return "gt50"

    @staticmethod
    def parse_row(parts: List[str]) -> Dict[str, str]:
        """Map the 15-column Adult row into the selected named feature fields."""
        return {
            "age": parts[0],
            "workclass": parts[1],
            "education": parts[3],
            "education_num": parts[4],
            "marital_status": parts[5],
            "occupation": parts[6],
            "relationship": parts[7],
            "race": parts[8],
            "sex": parts[9],
            "capital_gain": parts[10],
            "capital_loss": parts[11],
            "hours_per_week": parts[12],
            "native_country": parts[13],
        }

    def row_to_tokens(self, row: Dict[str, str]) -> List[str]:
        """Create CountVectorizer-ready tokens from one parsed row."""
        tokens: List[str] = []
        for col in self.feature_cols:
            raw_val = row[col]
            if col in self.continuous_bin_map:
                if self.config.continuous_mode in {"binned", "both"}:
                    binned = self.continuous_bin_map[col](raw_val)
                    tokens.append(f"{col}_BIN_{binned}")
                if self.config.continuous_mode in {"raw", "both"}:
                    tokens.append(f"{col}_RAW_{self.normalize_token(raw_val)}")
            else:
                tokens.append(f"{col}_{self.normalize_token(raw_val)}")
        return tokens

    def load(self) -> Tuple[List[str], List[int]]:
        """Load the dataset and return (tokenized_demographics, income_labels)."""
        demographics: List[str] = []
        income: List[int] = []

        with open(self.config.data_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue

                parts = [value.strip() for value in line.split(",")]
                if len(parts) != 15:
                    continue

                row = self.parse_row(parts)
                demographics.append(" ".join(self.row_to_tokens(row)))

                label = parts[-1].rstrip(".")
                income.append(1 if label == ">50K" else 0)

        return demographics, income

    def get_bin_counts(self) -> Dict[str, Dict[str, int]]:
        """Return counts for each bin category of each continuous field."""
        counts = {col: Counter() for col in self.continuous_bin_map}

        with open(self.config.data_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue

                parts = [value.strip() for value in line.split(",")]
                if len(parts) != 15:
                    continue

                row = self.parse_row(parts)
                for col, bin_func in self.continuous_bin_map.items():
                    counts[col][bin_func(row[col])] += 1

        return {col: dict(counter) for col, counter in counts.items()}

    def format_bin_frequency_summary(self) -> List[str]:
        """Create printable summary lines of bin counts and percentages."""
        bin_counts = self.get_bin_counts()
        lines: List[str] = []

        for col in sorted(bin_counts):
            total = sum(bin_counts[col].values())
            lines.append(f"{col} bin frequencies (n={total}):")

            for bin_name, count in sorted(
                bin_counts[col].items(), key=lambda item: item[1], reverse=True
            ):
                pct = 100.0 * count / total if total else 0.0
                lines.append(f"  {bin_name:>12}: {count:>6} ({pct:5.2f}%)")

        return lines
