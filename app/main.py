"""
PlantPulse API - model-serving layer for the agentic analytics platform.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.inference import NotEnoughData, batch_predict_anomaly, predict_anomaly, predict_forecast, search_logs
from app.schemas import AgentQueryRequest, AnomalyRequest, BatchRequest, ForecastRequest, SearchRequest
from app.state import AppState
from ml.features import ANOMALY_FEATURES

_state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = AppState()
    print(f"[startup] models loaded: {_state.registry.versions()}")
    print(f"[startup] sites: {[s['site_id'] for s in _state.sites]}")
    yield


app = FastAPI(title="PlantPulse", version="0.1.0", lifespan=lifespan)


def get_state() -> AppState:
    if _state is None:
        raise HTTPException(503, "service still starting up")
    return _state


@app.get("/health")
def health():
    state = get_state()
    return {"status": "ok", "model_versions": state.registry.versions(), "cache": state.cache.stats()}


@app.get("/sites")
def sites():
    return get_state().sites


@app.post("/predict/anomaly")
def anomaly(req: AnomalyRequest):
    try:
        return predict_anomaly(get_state(), req.site_id, req.as_of)
    except NotEnoughData as exc:
        raise HTTPException(404, str(exc))


@app.post("/predict/forecast")
def forecast(req: ForecastRequest):
    try:
        return predict_forecast(get_state(), req.site_id, req.as_of)
    except NotEnoughData as exc:
        raise HTTPException(404, str(exc))


@app.post("/predict/batch")
def batch(req: BatchRequest):
    return batch_predict_anomaly(get_state(), req.site_ids, req.as_of)


@app.post("/search/logs")
def logs(req: SearchRequest):
    return search_logs(get_state(), req.query, req.k)


@app.post("/agent/query")
def agent_query(req: AgentQueryRequest):
    from agent.orchestrator import answer_question  # local import avoids a hard startup dependency

    return answer_question(get_state(), req.question)


@app.get("/debug/cache-stats")
def cache_stats():
    return get_state().cache.stats()


@app.get("/stream/telemetry/{site_id}")
async def stream_telemetry(site_id: str, speed: float = 20.0, tail: int = 200):
    """Server-Sent Events: replays the tail of a site's telemetry at an
    accelerated rate (`speed`x real time), running the anomaly model live
    on each event as it's emitted - the real-time inference path.
    """
    state = get_state()
    site_df = state.df[state.df["site_id"] == site_id].tail(tail).reset_index(drop=True)
    if site_df.empty:
        raise HTTPException(404, f"no data for site_id={site_id!r}")

    real_interval_s = 15 * 60  # data is at 15-minute resolution
    sleep_s = max(real_interval_s / speed, 0.05)

    async def event_generator():
        for _, row in site_df.iterrows():
            proba = float(
                state.registry.anomaly_detector.model.predict_proba(
                    row[ANOMALY_FEATURES].to_frame().T
                )[0, 1]
            )
            payload = {
                "site_id": site_id,
                "timestamp": str(row["timestamp"]),
                "power_kw": round(float(row["power_kw"]), 2),
                "expected_power_kw": round(float(row["expected_power_kw"]), 2),
                "anomaly_probability": round(proba, 4),
                "is_anomaly": bool(proba >= 0.5),
                "is_curtailed": bool(row["is_curtailed"]),
            }
            yield {"event": "telemetry", "data": json.dumps(payload)}
            await asyncio.sleep(sleep_s)

    return EventSourceResponse(event_generator())
