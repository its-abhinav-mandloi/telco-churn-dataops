"""Streamlit dashboard reading persisted run manifests, JSONL logs and artifacts.

Run with: ``streamlit run src/telco_churn_dataops/dashboard/app.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from telco_churn_dataops import __version__
from telco_churn_dataops.config import PROJECT_ROOT, get_settings
from telco_churn_dataops.logging_utils import read_jsonl
from telco_churn_dataops.monitoring import THRESHOLDS
from telco_churn_dataops.storage import list_manifests

STATUS_COLOURS = {"PASS": "#d4edda", "WARN": "#fff3cd", "FAIL": "#f8d7da"}

st.set_page_config(page_title="Telco Churn DataOps", page_icon="📉", layout="wide")


def _abs(rel: str | None) -> Path | None:
    if not rel:
        return None
    path = PROJECT_ROOT / rel
    return path if path.exists() else None


def _status_badge(value: str | None) -> str:
    icon = {"success": "🟢", "running": "🔵", "failed": "🔴", "PASS": "🟢", "WARN": "🟠"}.get(
        value or "", "🔴" if value == "FAIL" else "⚪"
    )
    return f"{icon} {value or 'n/a'}"


def _history_frame(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in manifests:
        model = m.get("model") or {}
        champ = model.get("champion")
        rows.append(
            {
                "run_id": m.get("run_id"),
                "status": m.get("status"),
                "started_at": m.get("started_at"),
                "duration_s": m.get("duration_seconds"),
                "prefect_flow_run": (m.get("prefect") or {}).get("flow_run_name"),
                "rows": (m.get("ingestion") or {}).get("n_rows"),
                "monitoring": (m.get("monitoring") or {}).get("overall_status"),
                "champion": champ,
                "roc_auc": model.get("metrics", {}).get(champ, {}).get("roc_auc")
                if champ
                else None,
                "error": "; ".join(e.get("message", "") for e in m.get("errors", [])) or None,
            }
        )
    return pd.DataFrame(rows)


def render_prefect_panel() -> None:
    """Link to the authoritative Prefect dashboard."""
    settings = get_settings()
    url = settings.prefect_dashboard_url
    with st.sidebar:
        st.header("Prefect dashboard")
        if url:
            st.markdown(f"[Open Prefect UI]({url}) - authoritative flow, deployment and log view.")
        else:
            st.info(
                "Set `PREFECT_API_URL` (local: `http://127.0.0.1:4200/api`, or your Prefect Cloud "
                "workspace URL) to show a link. Local UI: http://127.0.0.1:4200"
            )
        st.caption(
            f"Deployment: `{settings.prefect_flow_name}/{settings.prefect_deployment_name}` "
            f"(interval 120 s) on work pool `{settings.prefect_work_pool}`."
        )
        st.caption(f"Dashboard v{__version__}. Auto-refreshes every 30 s.")


@st.fragment(run_every="30s")
def render_body() -> None:
    """Main dashboard body (re-rendered periodically)."""
    settings = get_settings()
    manifests = list_manifests(settings)
    st.title("📉 Telco Churn DataOps - pipeline dashboard")
    if not manifests:
        st.warning(
            "No pipeline runs found yet. Run `make run-once` or start the Prefect deployment."
        )
        return
    latest = manifests[0]
    success = next((m for m in manifests if m.get("status") == "success"), None)

    st.subheader("Latest run")
    c = st.columns(5)
    c[0].metric("Status", _status_badge(latest.get("status")))
    c[1].metric("Duration (s)", latest.get("duration_seconds"))
    c[2].metric("Monitoring", _status_badge((latest.get("monitoring") or {}).get("overall_status")))
    c[3].metric("Runs on record", len(manifests))
    c[4].metric("Flow run", (latest.get("prefect") or {}).get("flow_run_name") or "-")
    st.caption(
        f"Run `{latest['run_id']}` | started {latest.get('started_at')} | finished "
        f"{latest.get('finished_at')} | deployment id "
        f"`{(latest.get('prefect') or {}).get('deployment_id') or 'ad-hoc run (no deployment)'}`"
    )
    for err in latest.get("errors", []):
        st.error(f"{err.get('type')}: {err.get('message')}")

    tabs = st.tabs(
        ["Data quality", "Monitoring & drift", "Model metrics", "EDA", "Run history", "Run log"]
    )

    with tabs[0]:
        ing = latest.get("ingestion") or {}
        prep = latest.get("preprocessing") or {}
        q = st.columns(4)
        q[0].metric("Rows ingested", ing.get("n_rows"))
        q[1].metric("Columns", ing.get("n_columns"))
        q[2].metric("Duplicate rows", prep.get("duplicate_rows"))
        q[3].metric("Blank TotalCharges -> NaN", prep.get("total_charges_blank_converted_to_nan"))
        st.markdown(
            f"**Source:** {ing.get('source_url')}  \n**SHA-256:** `{ing.get('sha256')}`  \n"
            f"**HTTP status / content type:** {ing.get('http_status')} / {ing.get('content_type')}"
        )
        st.markdown("**Missing values before / after cleaning (imputed in-pipeline afterwards)**")
        st.json(
            {
                "before": prep.get("missing_before"),
                "after_cleaning": prep.get("missing_after_cleaning"),
            }
        )
        profile = latest.get("data_profile") or {}
        if profile.get("numeric"):
            st.markdown("**Numeric profile**")
            st.dataframe(pd.DataFrame(profile["numeric"]).T, width="stretch")

    with tabs[1]:
        mon = latest.get("monitoring") or {}
        st.markdown(
            f"Overall **{mon.get('overall_status')}** - quality {mon.get('quality_status')}, drift "
            f"{mon.get('drift_status')} - baseline run `{mon.get('baseline_run_id')}`"
        )
        checks = pd.DataFrame(mon.get("checks", []))
        if not checks.empty:
            checks = checks.astype({c: str for c in checks.columns if c != "status"})
            st.dataframe(
                checks.style.apply(
                    lambda row: [
                        f"background-color: {STATUS_COLOURS.get(row['status'], '')}" for _ in row
                    ],
                    axis=1,
                ),
                width="stretch",
            )
        st.markdown("**Thresholds** (WARN if value > warn, FAIL if value > fail)")
        st.dataframe(pd.DataFrame(THRESHOLDS).T, width="stretch")

    with tabs[2]:
        if not success:
            st.info("No successful run yet.")
        else:
            model = success["model"]
            st.markdown(
                f"Champion **{model['champion']}** (`{model['model_version']}`), selected by "
                f"{model['selection_metric']}. Source run `{success['run_id']}`."
            )
            frame = pd.DataFrame(model["metrics"]).T.drop(columns=["confusion_matrix"])
            st.dataframe(frame, width="stretch")
            arts = model.get("artifacts", {})
            i = st.columns(2)
            for col, key in zip(i, ("confusion_matrices", "feature_importance_plot"), strict=True):
                path = _abs(arts.get(key))
                if path:
                    col.image(str(path), width="stretch")
            st.caption(model.get("feature_importance", {}).get("method", ""))

    with tabs[3]:
        if not success or "eda" not in success:
            st.info("No EDA available yet.")
        else:
            eda = success["eda"]
            e = st.columns(2)
            e[0].metric("Churn rate", f"{eda['churn_rate']:.1%}")
            e[0].markdown("**Numeric correlation with Churn**")
            e[0].json(eda["numeric_target_correlation"])
            e[1].markdown("**Top Cramer's V with Churn**")
            e[1].json(eda["top_categorical_associations"])
            for name, rel in eda["plots"].items():
                path = _abs(rel)
                if path:
                    st.markdown(f"**{name.replace('_', ' ').title()}**")
                    st.image(str(path), width="stretch")

    with tabs[4]:
        hist = _history_frame(manifests)
        st.dataframe(hist, width="stretch")
        chart = hist.dropna(subset=["duration_s"]).set_index("started_at")[["duration_s"]]
        if not chart.empty:
            st.markdown("**Run duration (s)**")
            st.line_chart(chart.sort_index())

    with tabs[5]:
        events = read_jsonl(PROJECT_ROOT / latest.get("run_log", ""))
        if events:
            st.dataframe(pd.DataFrame(events), width="stretch")
        else:
            st.info("No run log found.")


render_prefect_panel()
render_body()
