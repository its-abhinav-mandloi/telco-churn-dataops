"""Exploratory data analysis: univariate/bivariate plots, correlations, Cramer's V and binning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from telco_churn_dataops.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from telco_churn_dataops.storage import relative_path, write_json  # noqa: E402

TENURE_BINS = [0, 12, 24, 48, 60, 72]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"]
CHARGE_BIN_LABELS = ["Low", "Medium", "High", "Very high"]

sns.set_theme(style="whitegrid", context="notebook")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013) between two categorical series."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``tenure_bin`` (fixed-width business bins) and ``monthly_charges_bin`` (quartiles)."""
    out = df.copy()
    out["tenure_bin"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    )
    out["monthly_charges_bin"] = pd.qcut(out["MonthlyCharges"], q=4, labels=CHARGE_BIN_LABELS)
    return out


def categorical_churn_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Long table of churn rate and count per category level for each categorical feature."""
    frames = []
    for col in CATEGORICAL_FEATURES:
        g = df.groupby(col, observed=True)[TARGET_COLUMN].agg(["mean", "count"]).reset_index()
        g.columns = ["level", "churn_rate", "count"]
        g.insert(0, "feature", col)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["level"] = out["level"].astype(str)
    out["churn_rate"] = out["churn_rate"].round(4)
    return out


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return relative_path(path)


def run_eda(df: pd.DataFrame, out_dir: Path, run_id: str) -> dict[str, Any]:
    """Generate all EDA artifacts for the cleaned frame and return a JSON-serialisable summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = add_bins(df)
    data["Churn_label"] = data[TARGET_COLUMN].map({0: "No", 1: "Yes"})
    p = lambda name: out_dir / f"{run_id}_{name}"  # noqa: E731
    plots: dict[str, str] = {}
    tables: dict[str, str] = {}

    # Univariate: target distribution.
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = data["Churn_label"].value_counts().reindex(["No", "Yes"])
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(title="Target distribution (Churn)", xlabel="Churn", ylabel="Customers")
    plots["target_distribution"] = _save(fig, p("target_distribution.png"))

    # Univariate + bivariate: numeric distributions split by churn.
    cont = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, cont, strict=True):
        sns.histplot(data=data, x=col, hue="Churn_label", bins=30, kde=True, ax=ax, element="step")
        ax.set_title(f"{col} distribution by churn")
    plots["numeric_distributions"] = _save(fig, p("numeric_distributions.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, cont, strict=True):
        sns.boxplot(data=data, x="Churn_label", y=col, hue="Churn_label", ax=ax, legend=False)
        ax.set_title(f"{col} vs churn")
    plots["numeric_boxplots"] = _save(fig, p("numeric_boxplots.png"))

    # Bivariate: categorical churn rates.
    rates = categorical_churn_rates(data)
    rates.to_csv(p("categorical_churn_rates.csv"), index=False)
    tables["categorical_churn_rates"] = relative_path(p("categorical_churn_rates.csv"))
    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES, strict=True):
        sub = rates[rates["feature"] == col]
        sns.barplot(data=sub, x="level", y="churn_rate", hue="level", ax=ax, legend=False)
        ax.set(title=f"Churn rate by {col}", xlabel="", ylabel="Churn rate", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    plots["categorical_churn_rates"] = _save(fig, p("categorical_churn_rates.png"))

    # Correlations: Pearson among numeric features + point-biserial with the binary target.
    corr = data[NUMERIC_FEATURES + [TARGET_COLUMN]].corr(method="pearson").round(4)
    corr.to_csv(p("correlation_matrix.csv"))
    tables["correlation_matrix"] = relative_path(p("correlation_matrix.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlation (numeric features + Churn)")
    plots["correlation_heatmap"] = _save(fig, p("correlation_heatmap.png"))

    # Categorical association with the target: bias-corrected Cramer's V.
    cv = (
        pd.Series(
            {c: cramers_v(data[c], data[TARGET_COLUMN]) for c in CATEGORICAL_FEATURES},
            name="cramers_v",
        )
        .sort_values(ascending=False)
        .round(4)
    )
    cv.to_csv(p("cramers_v.csv"), index_label="feature")
    tables["cramers_v"] = relative_path(p("cramers_v.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=cv.values, y=cv.index, hue=cv.index, ax=ax, legend=False)
    ax.set(title="Cramer's V with Churn (categorical features)", xlabel="Cramer's V", ylabel="")
    plots["cramers_v"] = _save(fig, p("cramers_v.png"))

    # Binning: tenure bins and monthly-charge quartiles vs churn.
    tenure_tbl = (
        data.groupby("tenure_bin", observed=False)[TARGET_COLUMN]
        .agg(customers="count", churn_rate="mean")
        .round(4)
    )
    tenure_tbl.to_csv(p("tenure_bins.csv"))
    tables["tenure_bins"] = relative_path(p("tenure_bins.csv"))
    charge_tbl = (
        data.groupby("monthly_charges_bin", observed=False)[TARGET_COLUMN]
        .agg(customers="count", churn_rate="mean")
        .round(4)
    )
    charge_tbl.to_csv(p("monthly_charge_bins.csv"))
    tables["monthly_charge_bins"] = relative_path(p("monthly_charge_bins.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tbl, title in (
        (axes[0], tenure_tbl, "Churn rate by tenure bin"),
        (axes[1], charge_tbl, "Churn rate by monthly-charge quartile"),
    ):
        labels = [str(i) for i in tbl.index]
        sns.barplot(x=labels, y=tbl["churn_rate"].values, hue=labels, ax=ax, legend=False)
        ax.set(title=title, xlabel="", ylabel="Churn rate", ylim=(0, 1))
    plots["binning_churn_rates"] = _save(fig, p("binning_churn_rates.png"))

    # Contract x tenure bin heatmap (bivariate interaction).
    pivot = data.pivot_table(
        index="Contract", columns="tenure_bin", values=TARGET_COLUMN, aggfunc="mean", observed=False
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax)
    ax.set_title("Churn rate: contract x tenure bin")
    plots["contract_tenure_heatmap"] = _save(fig, p("contract_tenure_heatmap.png"))

    target_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(key=np.abs, ascending=False)
    summary = {
        "run_id": run_id,
        "n_rows": int(len(data)),
        "churn_rate": round(float(data[TARGET_COLUMN].mean()), 4),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "numeric_target_correlation": target_corr.to_dict(),
        "numeric_correlation_matrix": corr.to_dict(),
        "cramers_v": cv.to_dict(),
        "top_categorical_associations": cv.head(5).to_dict(),
        "tenure_bins": {
            str(k): {"customers": int(v["customers"]), "churn_rate": float(v["churn_rate"])}
            for k, v in tenure_tbl.iterrows()
        },
        "monthly_charge_bins": {
            str(k): {"customers": int(v["customers"]), "churn_rate": float(v["churn_rate"])}
            for k, v in charge_tbl.iterrows()
        },
        "binning_method": {
            "tenure_bin": f"fixed edges {TENURE_BINS} months",
            "monthly_charges_bin": "quartiles (pd.qcut, q=4)",
        },
        "association_method": {
            "numeric": "Pearson correlation (point-biserial for the binary target)",
            "categorical": "Bias-corrected Cramer's V from a chi-squared contingency table",
        },
        "plots": plots,
        "tables": tables,
    }
    write_json(out_dir / f"{run_id}_eda_summary.json", summary)
    return summary
