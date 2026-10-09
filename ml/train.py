"""
Training entrypoint for PlantPulse's two production models:

  1. expected_power  - GradientBoostingRegressor, predicts "healthy" power
                       output given weather/sensor inputs. Used to compute
                       the residual that drives anomaly detection.
  2. anomaly_detector - RandomForestClassifier (class_weight=balanced) on
                       residual + drift/vibration features. Compared
                       against a naive static-threshold baseline so the
                       lift from the model is an actual measured number,
                       not an assumption.
  3. forecaster       - GradientBoostingRegressor, 1h-ahead power forecast
                       from lag/rolling features.

All three runs are logged to MLflow (params, metrics, artifacts). Each
trained artifact is also written into a lightweight content-addressed
model registry under ml/artifacts/<model_name>/<version>/ with a
metadata.json capturing data hash, git commit, features, and metrics -
the same provenance contract DVC enforces, plus the raw data file itself
is tracked with `dvc add` for dataset-level versioning.

Usage:
    python ml/train.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.features import (
    ANOMALY_FEATURES,
    EXPECTED_POWER_FEATURES,
    FORECAST_FEATURES,
    add_anomaly_features,
    add_expected_power_features,
    add_forecast_features,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telemetry.csv"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
MLFLOW_DIR = ROOT / "mlruns"


def _git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):  # set by CI / Docker build, where .git isn't present
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def register_model(model_name: str, model_obj, feature_list: list[str], metrics: dict, params: dict) -> str:
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    out_dir = ARTIFACTS_DIR / model_name / version
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_obj, out_dir / "model.joblib")
    metadata = {
        "model_name": model_name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "data_file": str(DATA_PATH.relative_to(ROOT)),
        "data_sha256_16": _sha256(DATA_PATH),
        "features": feature_list,
        "metrics": metrics,
        "params": params,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    current_pointer = ARTIFACTS_DIR / model_name / "current.json"
    with open(current_pointer, "w") as f:
        json.dump({"version": version, "path": str(out_dir.relative_to(ROOT))}, f, indent=2)

    print(f"[registry] {model_name} -> {version}  metrics={metrics}")
    return version


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    cutoff = df["timestamp"].quantile(train_frac)
    return df[df["timestamp"] < cutoff].copy(), df[df["timestamp"] >= cutoff].copy()


def train_expected_power(train_df: pd.DataFrame, test_df: pd.DataFrame):
    healthy_train = train_df[(train_df.is_anomaly == 0) & (train_df.is_curtailed == 0)]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
    )
    model.fit(healthy_train[EXPECTED_POWER_FEATURES], healthy_train["power_kw"])

    healthy_test = test_df[(test_df.is_anomaly == 0) & (test_df.is_curtailed == 0)]
    pred = model.predict(healthy_test[EXPECTED_POWER_FEATURES])
    mae = mean_absolute_error(healthy_test["power_kw"], pred)

    with mlflow.start_run(run_name="expected_power"):
        mlflow.log_params({"n_estimators": 200, "max_depth": 3, "learning_rate": 0.08})
        mlflow.log_metric("mae_kw", mae)
        mlflow.sklearn.log_model(model, "model")

    metrics = {"mae_kw": round(float(mae), 3)}
    register_model("expected_power", model, EXPECTED_POWER_FEATURES, metrics,
                    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.08})
    return model, metrics


def _naive_anomaly_baseline(df: pd.DataFrame, threshold: float = -0.2) -> np.ndarray:
    """Flag anomaly if residual is >20% below expected and not a known curtailment."""
    return ((df["residual_pct"] < threshold) & (df["is_curtailed"] == 0)).astype(int).values


def train_anomaly_detector(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train, y_train = train_df[ANOMALY_FEATURES], train_df["is_anomaly"]
    X_test, y_test = test_df[ANOMALY_FEATURES], test_df["is_anomaly"]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    model_metrics = {
        "precision": round(float(precision_score(y_test, pred)), 3),
        "recall": round(float(recall_score(y_test, pred)), 3),
        "f1": round(float(f1_score(y_test, pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
    }

    naive_pred = _naive_anomaly_baseline(test_df)
    baseline_metrics = {
        "precision": round(float(precision_score(y_test, naive_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, naive_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, naive_pred, zero_division=0)), 3),
    }

    with mlflow.start_run(run_name="anomaly_detector"):
        mlflow.log_params({"n_estimators": 300, "max_depth": 8, "class_weight": "balanced"})
        for k, v in model_metrics.items():
            mlflow.log_metric(k, v)
        for k, v in baseline_metrics.items():
            mlflow.log_metric(f"baseline_{k}", v)
        mlflow.sklearn.log_model(model, "model")

    print(f"[eval] anomaly_detector  model={model_metrics}  naive_baseline={baseline_metrics}")
    register_model(
        "anomaly_detector", model, ANOMALY_FEATURES, model_metrics,
        {"n_estimators": 300, "max_depth": 8, "class_weight": "balanced",
         "baseline_comparison": baseline_metrics},
    )
    return model, model_metrics, baseline_metrics


def train_forecaster(train_fc: pd.DataFrame, test_fc: pd.DataFrame):
    train_fc = train_fc.dropna(subset=FORECAST_FEATURES + ["target_power_kw"])
    test_fc = test_fc.dropna(subset=FORECAST_FEATURES + ["target_power_kw"])

    model = GradientBoostingRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.06, random_state=42
    )
    model.fit(train_fc[FORECAST_FEATURES], train_fc["target_power_kw"])

    pred = model.predict(test_fc[FORECAST_FEATURES])
    mae = mean_absolute_error(test_fc["target_power_kw"], pred)
    naive_pred = test_fc["power_lag_1"]  # persistence baseline: "next = current"
    naive_mae = mean_absolute_error(test_fc["target_power_kw"], naive_pred)

    with mlflow.start_run(run_name="forecaster_1h"):
        mlflow.log_params({"n_estimators": 250, "max_depth": 4, "learning_rate": 0.06, "horizon_steps": 4})
        mlflow.log_metric("mae_kw", mae)
        mlflow.log_metric("naive_persistence_mae_kw", naive_mae)
        mlflow.sklearn.log_model(model, "model")

    metrics = {"mae_kw": round(float(mae), 3), "naive_persistence_mae_kw": round(float(naive_mae), 3)}
    print(f"[eval] forecaster_1h  {metrics}")
    register_model("forecaster_1h", model, FORECAST_FEATURES, metrics,
                    {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.06, "horizon_steps": 4})
    return model, metrics


def main():
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("plantpulse")

    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = add_expected_power_features(raw)

    train_df, test_df = time_split(df)
    expected_power_model, ep_metrics = train_expected_power(train_df, test_df)

    df["expected_power_kw"] = expected_power_model.predict(df[EXPECTED_POWER_FEATURES])
    df_feat = add_anomaly_features(df, df["expected_power_kw"].values)
    train_a, test_a = time_split(df_feat)
    train_anomaly_detector(train_a, test_a)

    df_fc = add_forecast_features(df)
    train_fc, test_fc = time_split(df_fc)
    train_forecaster(train_fc, test_fc)

    print("\nAll models trained, evaluated, and registered under ml/artifacts/.")
    print("MLflow run data at:", MLFLOW_DIR)


if __name__ == "__main__":
    main()
