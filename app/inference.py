"""
Core prediction logic, shared by the REST endpoints, the SSE stream, and
the agent's tools - one implementation, multiple entry points.
"""

from __future__ import annotations

from typing import Any

from app.state import AppState
from ml.features import ANOMALY_FEATURES, FORECAST_FEATURES
from ml.vector_store import SearchHit

Prediction = dict[str, Any]


class NotEnoughData(ValueError):
    pass


def predict_anomaly(state: AppState, site_id: str, as_of: str | None = None) -> Prediction:
    cache_key = f"anomaly:{site_id}:{as_of or 'latest'}"
    cached = state.cache.get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    site_df = state.site_frame(site_id, as_of)
    if site_df.empty:
        raise NotEnoughData(f"no telemetry for site_id={site_id!r} as_of={as_of!r}")
    row = site_df.iloc[-1]

    proba = float(
        state.registry.anomaly_detector.model.predict_proba(site_df[ANOMALY_FEATURES].iloc[[-1]])[0, 1]
    )
    result: Prediction = {
        "site_id": site_id,
        "timestamp": str(row["timestamp"]),
        "is_anomaly": bool(proba >= 0.5),
        "anomaly_probability": round(proba, 4),
        "actual_power_kw": round(float(row["power_kw"]), 2),
        "expected_power_kw": round(float(row["expected_power_kw"]), 2),
        "residual_pct": round(float(row["residual_pct"]), 4),
        "is_curtailed": bool(row["is_curtailed"]),
        "model_version": state.registry.anomaly_detector.version,
        "cached": False,
    }
    state.cache.set(cache_key, result, ttl_seconds=30)
    return result


def predict_forecast(state: AppState, site_id: str, as_of: str | None = None) -> Prediction:
    cache_key = f"forecast:{site_id}:{as_of or 'latest'}"
    cached = state.cache.get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    site_df = state.site_frame(site_id, as_of)
    if site_df.empty:
        raise NotEnoughData(f"no telemetry for site_id={site_id!r} as_of={as_of!r}")
    row = site_df.iloc[-1]
    feat_row = site_df[FORECAST_FEATURES].iloc[[-1]]
    if bool(feat_row.isna().to_numpy().any()):
        raise NotEnoughData(
            f"not enough history yet for site_id={site_id!r} as_of={as_of!r} "
            "(forecaster needs up to 1 day of prior readings)"
        )

    forecast_kw = float(state.registry.forecaster.model.predict(feat_row)[0])
    result: Prediction = {
        "site_id": site_id,
        "as_of": str(row["timestamp"]),
        "forecast_horizon_minutes": 60,
        "forecast_power_kw": round(forecast_kw, 2),
        "current_power_kw": round(float(row["power_kw"]), 2),
        "model_version": state.registry.forecaster.version,
        "cached": False,
    }
    state.cache.set(cache_key, result, ttl_seconds=30)
    return result


class UnknownSite(ValueError):
    pass


def search_logs(state: AppState, query: str, k: int = 3, site_id: str | None = None) -> list[SearchHit]:
    """Hybrid (semantic + keyword) search over maintenance logs, optionally for one site."""
    if site_id is not None and site_id not in {s["site_id"] for s in state.sites}:
        raise UnknownSite(f"unknown site_id={site_id!r}")
    return state.vector_store.search(query, k=k, site_id=site_id)


def batch_predict_anomaly(state: AppState, site_ids: list[str], as_of: str | None = None) -> list[Prediction]:
    """Serial baseline - see ml/benchmark_parallel.py for the concurrent version
    and the measured throughput comparison."""
    out: list[Prediction] = []
    for site_id in site_ids:
        try:
            out.append(predict_anomaly(state, site_id, as_of))
        except NotEnoughData as exc:
            out.append({"site_id": site_id, "error": str(exc)})
    return out
