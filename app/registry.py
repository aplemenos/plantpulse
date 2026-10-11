"""
Model loading from the MLflow Model Registry.

At process startup the service resolves each model's `@champion` alias to a
concrete registry version, checks the version's logged signature against the
feature columns this code computes, and loads it once - so a request never
pays model-load cost, and a model trained on different columns can't be
served by accident.

Rollback is moving the alias (`python -m ml.promote <model> <version>`) and
restarting; no files are copied around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

import mlflow.artifacts
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from mlflow.sklearn import load_model
from numpy.typing import NDArray

from ml.features import ANOMALY_FEATURES, EXPECTED_POWER_FEATURES, FORECAST_FEATURES
from ml.mlflow_config import CHAMPION_ALIAS, REGISTERED_NAMES, configure_tracking

ROOT = Path(__file__).resolve().parent.parent

# The input columns each model must have been trained on, in order.
EXPECTED_INPUTS: dict[str, list[str]] = {
    "expected_power": EXPECTED_POWER_FEATURES,
    "anomaly_detector": ANOMALY_FEATURES,
    "forecaster_1h": FORECAST_FEATURES,
}


class Estimator(Protocol):
    """The scikit-learn estimator surface the service relies on."""

    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Any: ...


class Regressor(Estimator, Protocol):
    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


class Classifier(Estimator, Protocol):
    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]: ...
    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


class ModelContractError(RuntimeError):
    """The registered model can't be served by this version of the code."""


M = TypeVar("M", bound=Estimator)


class LoadedModel(Generic[M]):
    def __init__(
        self,
        key: str,
        model: M,
        registry_version: str,
        run_id: str,
        features: list[str],
        metrics: dict[str, float],
        tags: dict[str, str],
    ) -> None:
        self.key = key
        self.name = REGISTERED_NAMES[key]
        self.model = model
        self.registry_version = registry_version
        self.run_id = run_id
        self.features = features
        self.metrics = metrics
        self.tags = tags

    @property
    def version(self) -> str:
        return f"v{self.registry_version}"

    @property
    def uri(self) -> str:
        """Pinned to the resolved version, so it stays valid if the alias moves."""
        return f"models:/{self.name}/{self.registry_version}"


def _signature_inputs(model_uri: str) -> list[str]:
    signature = get_model_info(model_uri).signature
    if signature is None:
        raise ModelContractError(f"{model_uri} was logged without a signature")
    return list(signature.inputs.input_names())


def load_champion(model_key: str, kind: type[M]) -> LoadedModel[M]:
    """Load the version `@champion` points at, after checking its input contract."""
    configure_tracking()
    client = MlflowClient()
    name = REGISTERED_NAMES[model_key]
    try:
        mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException as exc:
        raise ModelContractError(
            f"no {name}@{CHAMPION_ALIAS} in the registry - run `make train` first"
        ) from exc

    model_uri = f"models:/{name}/{mv.version}"
    expected = EXPECTED_INPUTS[model_key]
    logged = _signature_inputs(model_uri)
    if logged != expected:
        raise ModelContractError(
            f"{name} v{mv.version} expects inputs {logged}, but the service computes {expected}. "
            "Retrain, or point @champion at a compatible version."
        )

    model = cast(Estimator, load_model(model_uri))
    # Online inference scores one row at a time; fanning a single row out across
    # threads (n_jobs=-1, the training setting) costs far more than it saves.
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)

    run_id = mv.run_id or ""
    run_data = client.get_run(run_id).data if run_id else None
    return LoadedModel(
        model_key,
        cast(M, model),
        registry_version=str(mv.version),
        run_id=run_id,
        features=logged,
        metrics=dict(run_data.metrics) if run_data else {},
        tags={k: v for k, v in run_data.tags.items() if not k.startswith("mlflow.")} if run_data else {},
    )


def champion_local_path(model_key: str) -> str:
    """Local directory of the champion's MLflow model, for loading in worker processes."""
    configure_tracking()
    name = REGISTERED_NAMES[model_key]
    mv = MlflowClient().get_model_version_by_alias(name, CHAMPION_ALIAS)
    return mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{mv.version}")


class ModelRegistry:
    """Loads every production model once; holds them for the app's lifetime."""

    def __init__(self) -> None:
        self.expected_power = load_champion("expected_power", Regressor)
        self.anomaly_detector = load_champion("anomaly_detector", Classifier)
        self.forecaster = load_champion("forecaster_1h", Regressor)

    def _all(self) -> list[LoadedModel[Any]]:
        return [self.expected_power, self.anomaly_detector, self.forecaster]

    def versions(self) -> dict[str, str]:
        return {m.key: m.version for m in self._all()}

    def describe(self) -> dict[str, dict[str, Any]]:
        """Provenance for /health: which registry version, run and data each model came from."""
        return {
            m.key: {
                "registered_model": m.name,
                "version": m.version,
                "alias": CHAMPION_ALIAS,
                "run_id": m.run_id,
                "git_commit": m.tags.get("git_commit"),
                "data_sha256_16": m.tags.get("data_sha256_16"),
                "metrics": m.metrics,
            }
            for m in self._all()
        }
