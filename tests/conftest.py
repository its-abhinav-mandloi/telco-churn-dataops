"""Shared fixtures. All data here is SYNTHETIC and generated in-memory for tests only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from telco_churn_dataops.config import Settings


def make_synthetic_telco(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Create a SYNTHETIC frame with the Telco schema (not real customers, never production)."""
    rng = np.random.default_rng(seed)
    yn = np.array(["Yes", "No"])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    phone = rng.choice(yn, n)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # mimic the blank TotalCharges values of the real dataset

    def svc() -> np.ndarray:
        return np.where(internet == "No", "No internet service", rng.choice(yn, n))

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    logit = (
        -1.0
        + 1.5 * (contract == "Month-to-month")
        - 0.03 * tenure
        + 0.8 * (internet == "Fiber optic")
    )
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(yn, n),
            "Dependents": rng.choice(yn, n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(yn, n)),
            "InternetService": internet,
            "OnlineSecurity": svc(),
            "OnlineBackup": svc(),
            "DeviceProtection": svc(),
            "TechSupport": svc(),
            "StreamingTV": svc(),
            "StreamingMovies": svc(),
            "Contract": contract,
            "PaperlessBilling": rng.choice(yn, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        }
    )


@pytest.fixture()
def synthetic_raw() -> pd.DataFrame:
    """SYNTHETIC raw Telco-shaped frame."""
    return make_synthetic_telco()


@pytest.fixture()
def tmp_settings(tmp_path) -> Settings:
    """Settings isolated to a temporary directory, with no Prefect API configured."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        prefect_api_url=None,
        prefect_api_key=None,
        min_expected_rows=100,
    )
