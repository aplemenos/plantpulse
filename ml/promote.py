"""
Champion/challenger promotion for the MLflow Model Registry.

Every training run registers a new version of each model. A new version
only becomes the `@champion` the service loads if its primary metric is
at least as good as the current champion's; otherwise it stays registered
but unpromoted, tagged with the reason.

CLI:
    python -m ml.promote                        # list versions; * marks the champion
    python -m ml.promote anomaly_detector 2     # point @champion at version 2 (rollback)
"""

from __future__ import annotations

import argparse
import sys

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from ml.mlflow_config import CHAMPION_ALIAS, REGISTERED_NAMES, configure_tracking

# model key -> (primary metric, higher is better)
PRIMARY_METRICS: dict[str, tuple[str, bool]] = {
    "expected_power": ("mae_kw", False),
    "anomaly_detector": ("f1", True),
    "forecaster_1h": ("mae_kw", False),
}


def current_champion(client: MlflowClient, model_key: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(REGISTERED_NAMES[model_key], CHAMPION_ALIAS)
    except MlflowException:  # no champion yet
        return None


def _metric(client: MlflowClient, version: ModelVersion, metric: str) -> float:
    run_id = version.run_id
    if run_id is None:
        raise RuntimeError(f"model version {version.version} has no source run")
    return float(client.get_run(run_id).data.metrics[metric])


def promote_if_not_worse(client: MlflowClient, model_key: str, new_version: str) -> bool:
    name = REGISTERED_NAMES[model_key]
    metric, higher_is_better = PRIMARY_METRICS[model_key]
    candidate = client.get_model_version(name, new_version)
    candidate_value = _metric(client, candidate, metric)

    champion = current_champion(client, model_key)
    if champion is None:
        promote, reason = True, "first version"
    else:
        champion_value = _metric(client, champion, metric)
        promote = (
            candidate_value >= champion_value if higher_is_better else candidate_value <= champion_value
        )
        reason = f"{metric} {candidate_value:g} vs champion v{champion.version} {champion_value:g}"

    if promote:
        client.set_registered_model_alias(name, CHAMPION_ALIAS, new_version)
    client.set_model_version_tag(name, new_version, "promotion", ("promoted: " if promote else "rejected: ") + reason)
    print(f"[registry] {name} v{new_version}: {'promoted to @champion' if promote else 'kept as challenger'} ({reason})")
    return promote


def list_versions(client: MlflowClient) -> None:
    for model_key, name in REGISTERED_NAMES.items():
        champion = current_champion(client, model_key)
        champion_version = champion.version if champion else None
        metric, _ = PRIMARY_METRICS[model_key]
        print(f"{model_key}  ({name}, primary metric: {metric})")
        versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
        for v in versions:
            marker = "*" if v.version == champion_version else " "
            value = _metric(client, v, metric)
            print(f"  {marker} v{v.version}  {metric}={value:g}  {v.tags.get('promotion', '')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", choices=sorted(REGISTERED_NAMES), help="model to re-point")
    parser.add_argument("version", nargs="?", help="registry version to make @champion")
    args = parser.parse_args(argv)

    configure_tracking()
    client = MlflowClient()
    if args.model is None:
        list_versions(client)
        return 0
    if args.version is None:
        parser.error("give a version to promote, e.g. `python -m ml.promote anomaly_detector 2`")
    name = REGISTERED_NAMES[args.model]
    client.get_model_version(name, args.version)  # fails clearly if the version doesn't exist
    client.set_registered_model_alias(name, CHAMPION_ALIAS, args.version)
    client.set_model_version_tag(name, args.version, "promotion", "promoted manually")
    print(f"{name}@{CHAMPION_ALIAS} -> v{args.version}. Restart the service to load it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
