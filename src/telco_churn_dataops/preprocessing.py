"""Pre-processing: type fixes, data-quality reporting, leakage-free split and sklearn transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES: list[str] = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class SplitData:
    """Train/test partitions; IDs are kept only for traceability, never as features."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing count / percentage table (blank strings count as missing)."""
    blank = df.apply(lambda s: s.astype(str).str.strip().eq("") if not _is_numeric(s) else s.isna())
    missing = df.isna() | blank
    table = pd.DataFrame(
        {
            "missing_count": missing.sum().astype(int),
            "missing_pct": (missing.mean() * 100).round(3),
        }
    )
    table.index.name = "column"
    return table


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise shape, dtypes, duplicates and missingness of ``df``."""
    mv = missing_value_table(df)
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df else None,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_total": int(mv["missing_count"].sum()),
        "missing_by_column": {k: int(v) for k, v in mv["missing_count"].items() if v > 0},
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, row-level type fixes that do not learn from the data.

    * ``TotalCharges`` arrives as text with blank strings for new customers; blanks
      become ``NaN`` and the column becomes float (imputed later inside the pipeline).
    * ``Churn`` Yes/No becomes 1/0.
    * Exact duplicate rows are dropped.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce"
    )
    for col in ("SeniorCitizen", "tenure", "MonthlyCharges"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[TARGET_COLUMN] = (out[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(
        int
    )
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("object").where(out[col].notna(), np.nan)
    return out.drop_duplicates().reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``describe()`` statistics for numeric model features."""
    return df[NUMERIC_FEATURES].describe().T.round(4)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return cardinality and the most frequent value for categorical features."""
    rows = []
    for col in CATEGORICAL_FEATURES:
        vc = df[col].value_counts(dropna=True)
        rows.append(
            {
                "column": col,
                "n_unique": int(df[col].nunique()),
                "top": str(vc.index[0]) if len(vc) else None,
                "top_freq": int(vc.iloc[0]) if len(vc) else 0,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def split(df: pd.DataFrame, test_size: float, seed: int) -> SplitData:
    """Stratified train/test split with a fixed seed. ``customerID`` is excluded from X."""
    x = df[FEATURES]
    y = df[TARGET_COLUMN]
    ids = df[ID_COLUMN]
    x_tr, x_te, y_tr, y_te, id_tr, id_te = train_test_split(
        x, y, ids, test_size=test_size, random_state=seed, stratify=y
    )
    return SplitData(x_tr, x_te, y_tr, y_te, id_tr, id_te)


def build_preprocessor() -> ColumnTransformer:
    """Build the leakage-safe transformer (fitted on training data only inside a Pipeline).

    Numeric: median imputation -> StandardScaler.
    Categorical: most-frequent imputation -> OneHotEncoder(handle_unknown="ignore").
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=True,
    )


def transformed_preview(
    preprocessor: ColumnTransformer, x_train: pd.DataFrame, n: int = 5
) -> dict[str, Any]:
    """Fit a copy of the preprocessor on training data; return encoding/scaling evidence."""
    fitted = clone(preprocessor).fit(x_train)
    names = list(fitted.get_feature_names_out())
    matrix = fitted.transform(x_train)
    num_idx = [i for i, n_ in enumerate(names) if n_.startswith("num__")]
    return {
        "n_input_features": len(FEATURES),
        "n_output_features": len(names),
        "output_feature_names": names,
        "numeric_after_scaling": {
            names[i]: {
                "mean": round(float(matrix[:, i].mean()), 6),
                "std": round(float(matrix[:, i].std()), 6),
            }
            for i in num_idx
        },
        "imputation_values": {
            "numeric_median": dict(
                zip(
                    NUMERIC_FEATURES,
                    [float(v) for v in fitted.named_transformers_["num"]["impute"].statistics_],
                    strict=True,
                )
            ),
            "categorical_most_frequent": dict(
                zip(
                    CATEGORICAL_FEATURES,
                    [str(v) for v in fitted.named_transformers_["cat"]["impute"].statistics_],
                    strict=True,
                )
            ),
        },
        "sample_rows": [[round(float(v), 4) for v in row] for row in matrix[:n]],
    }
