"""
Minimal model registry reader: resolves the "current" version of each
model (as published by ml/train.py's registry writer) and loads it once,
at process startup, so a request never pays model-load cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypedDict  # pydantic needs this on Python < 3.12

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"


class Estimator(Protocol):
    """The scikit-learn estimator surface the service relies on."""

    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Any: ...


class Regressor(Estimator, Protocol):
    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


class Classifier(Estimator, Protocol):
    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]: ...
    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


class ModelMetadata(TypedDict):
    model_name: str
    version: str
    trained_at: str
    git_commit: str
    data_file: str
    data_sha256_16: str
    features: list[str]
    metrics: dict[str, float]
    params: dict[str, Any]


class RegistryPointer(TypedDict):
    version: str
    path: str


M = TypeVar("M", bound=Estimator)


class LoadedModel(Generic[M]):
    def __init__(self, name: str, model: M, metadata: ModelMetadata) -> None:
        self.name = name
        self.model = model
        self.metadata = metadata

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]


def resolve_current(model_name: str) -> Path:
    """Directory of the version that `current.json` marks as live."""
    pointer = cast(RegistryPointer, json.loads((ARTIFACTS_DIR / model_name / "current.json").read_text()))
    return ROOT / pointer["path"]


def load_current(model_name: str, kind: type[M]) -> LoadedModel[M]:
    version_dir = resolve_current(model_name)

    model = cast(Estimator, joblib.load(version_dir / "model.joblib"))
    # Online inference scores one row at a time; fanning a single row out across
    # threads (n_jobs=-1, the training setting) costs far more than it saves.
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    metadata = cast(ModelMetadata, json.loads((version_dir / "metadata.json").read_text()))
    return LoadedModel(model_name, cast(M, model), metadata)


class ModelRegistry:
    """Loads every production model once; holds them for the app's lifetime."""

    def __init__(self) -> None:
        self.expected_power = load_current("expected_power", Regressor)
        self.anomaly_detector = load_current("anomaly_detector", Classifier)
        self.forecaster = load_current("forecaster_1h", Regressor)

    def versions(self) -> dict[str, str]:
        return {
            "expected_power": self.expected_power.version,
            "anomaly_detector": self.anomaly_detector.version,
            "forecaster_1h": self.forecaster.version,
        }
