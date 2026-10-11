"""
Single place for MLflow settings shared by training, serving and the agent.

Storage defaults to a local SQLite tracking store under ml/artifacts/mlflow/
(DVC-tracked with the rest of ml/artifacts). Set MLFLOW_TRACKING_URI to point
at a shared MLflow server instead - nothing else needs to change.

View runs, registered models and agent traces with:
    make mlflow-ui      # http://127.0.0.1:5050
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DIR = ROOT / "ml" / "artifacts" / "mlflow"
ARTIFACT_ROOT = MLFLOW_DIR / "artifacts"

TRAINING_EXPERIMENT = "plantpulse-training"
AGENT_EXPERIMENT = "plantpulse-agent"

# Registry name for each model the service serves.
REGISTERED_NAMES: dict[str, str] = {
    "expected_power": "plantpulse-expected-power",
    "anomaly_detector": "plantpulse-anomaly-detector",
    "forecaster_1h": "plantpulse-forecaster-1h",
}
CHAMPION_ALIAS = "champion"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"


def configure_tracking() -> None:
    """Point this process at the tracking store (and model registry)."""
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())


def use_experiment(name: str) -> None:
    """Select an experiment, creating it with a fixed artifact root on first use.

    The artifact root is set explicitly because a SQLite store would otherwise
    default to ./mlruns relative to whatever directory the process started in.
    """
    configure_tracking()
    if mlflow.get_experiment_by_name(name) is None:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(name)


def agent_experiment() -> str:
    """Experiment that receives agent traces (tests point it elsewhere)."""
    return os.environ.get("PLANTPULSE_AGENT_EXPERIMENT", AGENT_EXPERIMENT)

