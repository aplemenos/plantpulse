"""
Minimal model registry reader: resolves the "current" version of each
model (as published by ml/train.py's registry writer) and loads it once,
at process startup, so a request never pays model-load cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"


class LoadedModel:
    def __init__(self, name: str, model, metadata: dict):
        self.name = name
        self.model = model
        self.metadata = metadata

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]


def load_current(model_name: str) -> LoadedModel:
    pointer_path = ARTIFACTS_DIR / model_name / "current.json"
    pointer = json.loads(pointer_path.read_text())
    version_dir = ROOT / pointer["path"]

    model = joblib.load(version_dir / "model.joblib")
    metadata = json.loads((version_dir / "metadata.json").read_text())
    return LoadedModel(model_name, model, metadata)


class ModelRegistry:
    """Loads every production model once; holds them for the app's lifetime."""

    def __init__(self):
        self.expected_power = load_current("expected_power")
        self.anomaly_detector = load_current("anomaly_detector")
        self.forecaster = load_current("forecaster_1h")

    def versions(self) -> dict:
        return {
            "expected_power": self.expected_power.version,
            "anomaly_detector": self.anomaly_detector.version,
            "forecaster_1h": self.forecaster.version,
        }
