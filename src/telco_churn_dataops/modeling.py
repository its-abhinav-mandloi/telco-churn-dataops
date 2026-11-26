"""Model training, evaluation, champion selection, persistence and model card generation."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.base import ClassifierMixin  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from telco_churn_dataops.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SplitData,
    build_preprocessor,
)
from telco_churn_dataops.storage import relative_path, write_json  # noqa: E402

MODEL_FILENAME = "champion.joblib"
METADATA_FILENAME = "champion_metadata.json"
SELECTION_METRIC = "roc_auc"
DECISION_THRESHOLD = 0.5
TRACKED_PACKAGES = ("scikit-learn", "pandas", "numpy", "prefect", "joblib")


def candidate_models(seed: int) -> dict[str, ClassifierMixin]:
    """Return the candidate estimators compared under the same preprocessing."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compute classification metrics on a held-out set."""
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def package_versions() -> dict[str, str]:
    """Return versions of key packages plus the Python version."""
    versions = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def train_and_compare(data: SplitData, seed: int, cv_folds: int = 5) -> dict[str, Any]:
    """Fit every candidate inside an identical Pipeline and evaluate on the test split.

    Returns fitted pipelines, metrics per model and the champion name (highest test ROC-AUC).
    """
    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for name, estimator in candidate_models(seed).items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        cv_auc = cross_val_score(pipe, data.x_train, data.y_train, cv=cv, scoring="roc_auc")
        start = time.perf_counter()
        pipe.fit(data.x_train, data.y_train)
        train_seconds = time.perf_counter() - start
        metrics = evaluate(pipe, data.x_test, data.y_test)
        metrics["cv_roc_auc_mean"] = round(float(cv_auc.mean()), 4)
        metrics["cv_roc_auc_std"] = round(float(cv_auc.std()), 4)
        metrics["train_seconds"] = round(train_seconds, 3)
        results[name] = metrics
        fitted[name] = pipe
    champion = max(results, key=lambda n: results[n][SELECTION_METRIC])
    return {"metrics": results, "champion": champion, "fitted": fitted}


def compute_permutation_importance(
    model: Pipeline, data: SplitData, seed: int, n_repeats: int = 5
) -> pd.DataFrame:
    """Permutation importance (ROC-AUC drop) of the *original* input features on the test set."""
    result = permutation_importance(
        model,
        data.x_test,
        data.y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance_mean": result.importances_mean.round(5),
                "importance_std": result.importances_std.round(5),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(fi: pd.DataFrame, path: Path, champion: str) -> str:
    """Save a horizontal bar chart of permutation importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi, x="importance_mean", y="feature", hue="feature", ax=ax, legend=False)
    ax.errorbar(fi["importance_mean"], range(len(fi)), xerr=fi["importance_std"], fmt="none", c="k")
    ax.set(
        title=f"Permutation importance - {champion} (ROC-AUC drop, test set)",
        xlabel="Mean decrease in ROC-AUC",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return relative_path(path)


def plot_confusion_matrices(metrics: dict[str, Any], path: Path) -> str:
    """Save side-by-side confusion matrices for every candidate."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (name, m) in zip(axes, metrics.items(), strict=True):
        cm = m["confusion_matrix"]
        grid = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        sns.heatmap(grid, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(
            title=f"{name}\nROC-AUC={m['roc_auc']}",
            xlabel="Predicted",
            ylabel="Actual",
            xticklabels=["No churn", "Churn"],
            yticklabels=["No churn", "Churn"],
        )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return relative_path(path)


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def persist_champion(
    model: Pipeline, model_metadata: dict[str, Any], run_model_dir: Path, promoted_dir: Path
) -> dict[str, str]:
    """Save the champion in the run folder and atomically promote it for serving."""
    run_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_model_dir / MODEL_FILENAME
    meta_path = run_model_dir / METADATA_FILENAME
    joblib.dump(model, model_path)
    write_json(meta_path, model_metadata)
    _atomic_copy(model_path, promoted_dir / MODEL_FILENAME)
    _atomic_copy(meta_path, promoted_dir / METADATA_FILENAME)
    return {
        "run_model": relative_path(model_path),
        "run_metadata": relative_path(meta_path),
        "serving_model": relative_path(promoted_dir / MODEL_FILENAME),
        "serving_metadata": relative_path(promoted_dir / METADATA_FILENAME),
    }


def build_model_metadata(
    run_id: str,
    champion: str,
    metrics: dict[str, Any],
    dataset_sha256: str,
    seed: int,
    test_size: float,
    n_train: int,
    n_test: int,
    trained_at: str,
) -> dict[str, Any]:
    """Assemble the champion model metadata document."""
    return {
        "model_version": f"{champion}-{run_id}",
        "run_id": run_id,
        "champion": champion,
        "selection_metric": f"test {SELECTION_METRIC}",
        "decision_threshold": DECISION_THRESHOLD,
        "trained_at": trained_at,
        "training_seconds": metrics[champion]["train_seconds"],
        "metrics": metrics,
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "excluded_columns": ["customerID (identifier, traceability only)"],
        "target": "Churn (1 = customer left within the last month, 0 = stayed)",
        "dataset_sha256": dataset_sha256,
        "split": {
            "test_size": test_size,
            "random_seed": seed,
            "n_train": n_train,
            "n_test": n_test,
        },
        "package_versions": package_versions(),
    }


def write_model_card(path: Path, meta: dict[str, Any], top_features: list[str]) -> str:
    """Write a concise Markdown model card derived from real metadata."""
    champ = meta["champion"]
    rows = "\n".join(
        f"| {name} | {m['roc_auc']} | {m['f1']} | {m['precision']} | {m['recall']} | "
        f"{m['accuracy']} | {m['cv_roc_auc_mean']} +/- {m['cv_roc_auc_std']} |"
        for name, m in meta["metrics"].items()
    )
    cm = meta["metrics"][champ]["confusion_matrix"]
    split_ = meta["split"]
    text = f"""# Model card - Telco churn classifier

**Model version:** `{meta["model_version"]}`  |  **Run ID:** `{meta["run_id"]}`
**Trained at (UTC):** {meta["trained_at"]}  |  **Dataset SHA-256:** `{meta["dataset_sha256"]}`

## Intended use
Rank existing telecom customers by estimated probability of churning so that a retention
team can prioritise outreach. It is a decision-support signal for humans, not an automated
decision system, and must not be used to deny service or set prices.

## Target definition
`Churn = 1` if the customer left within the last month according to the IBM Telco Customer
Churn sample dataset, else `0`. Decision threshold: {meta["decision_threshold"]}.

## Data and features
Public IBM Telco Customer Churn sample (single snapshot, fictional California telco).
Numeric: {", ".join(meta["features"]["numeric"])}.
Categorical (one-hot): {", ".join(meta["features"]["categorical"])}.
`customerID` is excluded from features. Split: stratified,
test_size={split_["test_size"]}, seed={split_["random_seed"]}
(train={split_["n_train"]}, test={split_["n_test"]}).

## Metrics (held-out test set; CV = 5-fold stratified ROC-AUC on the training set)
| Model | ROC-AUC | F1 | Precision | Recall | Accuracy | CV ROC-AUC |
|---|---|---|---|---|---|---|
{rows}

Champion: **{champ}** (highest test ROC-AUC). Confusion matrix: TN={cm["tn"]}, FP={cm["fp"]},
FN={cm["fn"]}, TP={cm["tp"]}.

Most important features (permutation importance, ROC-AUC drop on test set):
{", ".join(top_features)}.

## Limitations
- Single, static, sample dataset (~7k rows); no temporal validation and no real-world drift.
- Metrics come from one random split; expect variance of a few points on new data.
- Class weights are balanced, which favours recall over precision; tune the threshold to the
  cost of a retention offer before operational use.

## Bias and ethical considerations
- Inputs include `gender`, `SeniorCitizen`, `Partner` and `Dependents`. No fairness audit
  across these groups has been performed; no claim of fairness is made.
- Predictions must not be used to discriminate against protected groups. A subgroup error
  analysis and human review are required before any production use.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return relative_path(path)


def load_serving_model(promoted_dir: Path) -> tuple[Pipeline, dict[str, Any]] | None:
    """Load the promoted champion model and metadata, or ``None`` if not trained yet."""
    model_path = promoted_dir / MODEL_FILENAME
    meta_path = promoted_dir / METADATA_FILENAME
    if not model_path.exists() or not meta_path.exists():
        return None
    return joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8"))
