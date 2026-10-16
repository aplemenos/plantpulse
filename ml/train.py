"""
Training entrypoint for PlantPulse's three production models:

  1. expected_power  - GradientBoostingRegressor, predicts "healthy" power
                       output given weather/sensor inputs. Used to compute
                       the residual that drives anomaly detection.
  2. anomaly_detector - RandomForestClassifier on residual + drift/vibration
                       features, wrapped in sigmoid (Platt) calibration so
                       anomaly_probability is a real probability. Compared
                       against a naive static-threshold baseline so the
                       lift from the model is an actual measured number,
                       not an assumption.
  3. forecaster_1h    - GradientBoostingRegressor, 1h-ahead power forecast
                       from lag/rolling features.

Each run is logged to MLflow (params, metrics, provenance tags) and the
model is logged with a signature and input example, then registered in the
MLflow Model Registry. A new version becomes `@champion` - the version the
service loads - only if its primary metric is at least as good as the
current champion's (see ml/promote.py). The raw data file is tracked with
DVC for dataset-level versioning.

Usage:
    python -m ml.train
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
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
    physical_expected_power,
)
from ml.mlflow_config import REGISTERED_NAMES, TRAINING_EXPERIMENT, tracking_uri, use_experiment
from ml.promote import promote_if_not_worse

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telemetry.csv"

# Types skops won't load unless told to. Trusted because we create these files
# ourselves in this pipeline: tree node storage shared by the forests and
# boosted trees, plus the two private classes a sigmoid-calibrated classifier
# holds (isotonic would need one fewer, but lost F1 at the 0.5 threshold).
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
]


def _git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):  # set by CI / Docker build, where .git isn't present
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):  # git missing, or not a git checkout
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def log_and_register(
    model_key: str,
    model: GradientBoostingRegressor | CalibratedClassifierCV,
    X_example: pd.DataFrame,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> str:
    """Log one training run to MLflow and register the model; returns the new version."""
    with mlflow.start_run(run_name=model_key):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "git_commit": _git_commit(),
            "data_file": str(DATA_PATH.relative_to(ROOT)),
            "data_sha256_16": _sha256(DATA_PATH),
        })
        # The signature records the exact input columns and dtypes; the service
        # checks it against the features it computes before it will start.
        # Declared as doubles: integer columns (hour, flags) can't hold NaN, and
        # scikit-learn converts every input to float64 anyway.
        X_example = X_example.astype("float64")
        signature = infer_signature(X_example, model.predict(X_example))
        info = log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_example.head(3),
            registered_model_name=REGISTERED_NAMES[model_key],
            # MLflow saves scikit-learn models with skops rather than pickle, so
            # loading can't run arbitrary code; only these reviewed types are allowed.
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    version = str(info.registered_model_version)
    promote_if_not_worse(MlflowClient(), model_key, version)
    return version


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["timestamp"].quantile(train_frac)
    return df[df["timestamp"] < cutoff].copy(), df[df["timestamp"] >= cutoff].copy()


def train_expected_power(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[GradientBoostingRegressor, dict[str, float]]:
    healthy_train = train_df[(train_df.is_anomaly == 0) & (train_df.is_curtailed == 0)]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
    )
    model.fit(healthy_train[EXPECTED_POWER_FEATURES], healthy_train["power_kw"])

    healthy_test = test_df[(test_df.is_anomaly == 0) & (test_df.is_curtailed == 0)]
    pred = physical_expected_power(healthy_test, model.predict(healthy_test[EXPECTED_POWER_FEATURES]))
    mae = mean_absolute_error(healthy_test["power_kw"], pred)

    metrics = {"mae_kw": round(float(mae), 3)}
    print(f"[eval] expected_power  {metrics}")
    log_and_register(
        "expected_power", model, healthy_train[EXPECTED_POWER_FEATURES].head(200),
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.08}, metrics,
    )
    return model, metrics


def _naive_anomaly_baseline(df: pd.DataFrame, threshold: float = -0.2) -> NDArray[np.int64]:
    """Flag anomaly if residual is >20% below expected and not a known curtailment."""
    flagged = (df["residual_pct"] < threshold) & (df["is_curtailed"] == 0)
    return flagged.to_numpy(dtype=np.int64)


def expected_calibration_error(y_true: pd.Series[Any], proba: NDArray[np.float64], n_bins: int = 10) -> float:
    """Mean gap between predicted probability and observed anomaly rate, over equal-count bins."""
    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return float(np.mean(np.abs(observed - predicted)))


def train_anomaly_detector(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[CalibratedClassifierCV, dict[str, float], dict[str, float]]:
    X_train, y_train = train_df[ANOMALY_FEATURES], train_df["is_anomaly"]
    X_test, y_test = test_df[ANOMALY_FEATURES], test_df["is_anomaly"]

    # No class weighting: class_weight="balanced" pushed healthy rows to ~0.3, so
    # 0.4 read as alarming when nothing was wrong. Sigmoid calibration on
    # out-of-fold predictions (ensemble=False: one forest fitted on all the
    # training data, plus a calibrator) makes the score a probability, and
    # 0.5 means "more likely a fault than not". Chosen on an inner time split
    # of the training window: better F1 and Brier than the balanced forest.
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1)
    model = CalibratedClassifierCV(forest, method="sigmoid", cv=5, ensemble=False)
    model.fit(X_train, y_train)
    # The service scores one row at a time; the fitted forest lives inside the
    # wrapper, where the registry's set_params(n_jobs=1) doesn't reach it.
    for calibrated in model.calibrated_classifiers_:
        calibrated.estimator.set_params(n_jobs=1)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    model_metrics = {
        "precision": round(float(precision_score(y_test, pred)), 3),
        "recall": round(float(recall_score(y_test, pred)), 3),
        "f1": round(float(f1_score(y_test, pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "ece": round(expected_calibration_error(y_test, proba), 4),
        "healthy_mean_proba": round(float(proba[(y_test == 0).to_numpy()].mean()), 4),
    }

    naive_pred = _naive_anomaly_baseline(test_df)
    baseline_metrics = {
        "precision": round(float(precision_score(y_test, naive_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, naive_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, naive_pred, zero_division=0)), 3),
    }

    print(f"[eval] anomaly_detector  model={model_metrics}  naive_baseline={baseline_metrics}")
    log_and_register(
        "anomaly_detector", model, X_train.head(200),
        {"n_estimators": 300, "min_samples_leaf": 20, "calibration": "sigmoid", "calibration_cv": 5},
        {**model_metrics, **{f"baseline_{k}": v for k, v in baseline_metrics.items()}},
    )
    return model, model_metrics, baseline_metrics


def train_forecaster(
    train_fc: pd.DataFrame, test_fc: pd.DataFrame
) -> tuple[GradientBoostingRegressor, dict[str, float]]:
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

    metrics = {"mae_kw": round(float(mae), 3), "naive_persistence_mae_kw": round(float(naive_mae), 3)}
    print(f"[eval] forecaster_1h  {metrics}")
    log_and_register(
        "forecaster_1h", model, train_fc[FORECAST_FEATURES].head(200),
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.06, "horizon_steps": 4}, metrics,
    )
    return model, metrics


def main() -> None:
    use_experiment(TRAINING_EXPERIMENT)

    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = add_expected_power_features(raw)

    train_df, test_df = time_split(df)
    expected_power_model, _ = train_expected_power(train_df, test_df)

    expected_power = physical_expected_power(df, expected_power_model.predict(df[EXPECTED_POWER_FEATURES]))
    df["expected_power_kw"] = expected_power
    df_feat = add_anomaly_features(df, expected_power)
    train_a, test_a = time_split(df_feat)
    train_anomaly_detector(train_a, test_a)

    df_fc = add_forecast_features(df)
    train_fc, test_fc = time_split(df_fc)
    train_forecaster(train_fc, test_fc)

    print(f"\nAll models trained, evaluated and registered. MLflow store: {tracking_uri()}")
    print("Browse runs and the registry with: make mlflow-ui")


if __name__ == "__main__":
    main()
