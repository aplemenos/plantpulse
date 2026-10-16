from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from app import registry
from app.registry import EXPECTED_INPUTS, ModelContractError, Regressor, load_champion
from ml.mlflow_config import CHAMPION_ALIAS, REGISTERED_NAMES
from ml.promote import promote_if_not_worse


def test_service_serves_the_champion_versions(client: TestClient) -> None:
    models = client.get("/health").json()["models"]
    mlflow_client = MlflowClient()
    for key, name in REGISTERED_NAMES.items():
        champion = mlflow_client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        assert models[key]["version"] == f"v{champion.version}"
        assert models[key]["run_id"] == champion.run_id


def test_every_champion_signature_matches_the_service_features(client: TestClient) -> None:
    for key, name in REGISTERED_NAMES.items():
        version = MlflowClient().get_model_version_by_alias(name, CHAMPION_ALIAS).version
        assert registry._signature_inputs(f"models:/{name}/{version}") == EXPECTED_INPUTS[key]  # pyright: ignore[reportPrivateUsage]


def test_service_refuses_a_model_trained_on_other_columns(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    reordered = list(reversed(EXPECTED_INPUTS["forecaster_1h"]))
    monkeypatch.setitem(EXPECTED_INPUTS, "forecaster_1h", reordered)
    with pytest.raises(ModelContractError, match="expects inputs"):
        load_champion("forecaster_1h", Regressor)


class _FakeRegistryClient:
    """Just enough of MlflowClient for the promotion rule: version -> metric."""

    def __init__(self, f1_by_version: dict[str, float], champion: str | None) -> None:
        self.f1 = f1_by_version
        self.champion = champion
        self.tags: dict[str, str] = {}

    def get_model_version(self, name: str, version: str) -> Any:
        return SimpleNamespace(version=version, run_id=version)

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        from mlflow.exceptions import MlflowException

        if self.champion is None:
            raise MlflowException("no alias")
        return self.get_model_version(name, self.champion)

    def get_run(self, run_id: str) -> Any:
        return SimpleNamespace(data=SimpleNamespace(metrics={"f1": self.f1[run_id]}))

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.champion = version

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags[version] = value


@pytest.mark.parametrize(
    ("new_f1", "promoted"),
    [(0.70, True), (0.666, True), (0.60, False)],
)
def test_challenger_is_promoted_only_if_not_worse(new_f1: float, promoted: bool) -> None:
    fake = _FakeRegistryClient({"1": 0.666, "2": new_f1}, champion="1")
    assert promote_if_not_worse(cast(MlflowClient, fake), "anomaly_detector", "2") is promoted
    assert fake.champion == ("2" if promoted else "1")
    assert fake.tags["2"].startswith("promoted" if promoted else "rejected")


def test_first_version_becomes_champion() -> None:
    fake = _FakeRegistryClient({"1": 0.5}, champion=None)
    assert promote_if_not_worse(cast(MlflowClient, fake), "anomaly_detector", "1")
    assert fake.champion == "1"


def test_anomaly_probabilities_are_calibrated(client: TestClient) -> None:
    # Healthy rows used to average ~0.3 (class_weight="balanced"), so 0.4 read as
    # alarming when nothing was wrong. Calibrated, they sit near the anomaly rate.
    from app.main import get_state
    from ml.features import ANOMALY_FEATURES

    df = get_state().df
    proba = get_state().registry.anomaly_detector.model.predict_proba(df[ANOMALY_FEATURES])[:, 1]
    healthy = ((df["is_anomaly"] == 0) & (df["is_curtailed"] == 0)).to_numpy()
    assert proba[healthy].mean() < 0.1
    assert proba[~healthy & (df["is_anomaly"] == 1).to_numpy()].mean() > 0.3
