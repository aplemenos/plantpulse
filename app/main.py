"""
PlantPulse API - model-serving layer for the agentic analytics platform.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from agent.sessions import NothingToResume, SessionStore, UnknownSession, build_context, investigate_in_session, resume
from agent.tracing import ensure_tracing, flush_traces, tracing_enabled
from app.inference import (
    NotEnoughData,
    Prediction,
    UnknownSite,
    batch_predict_anomaly,
    predict_anomaly,
    predict_forecast,
    search_logs,
)
from app.schemas import (
    AgentQueryRequest,
    AnomalyRequest,
    BatchRequest,
    ForecastRequest,
    InvestigateRequest,
    SearchRequest,
)
from app.state import AppState, FlaggedMoment, SiteInfo
from ml.features import ANOMALY_FEATURES
from ml.vector_store import SearchHit

_state: AppState | None = None
_sessions: SessionStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global _state, _sessions
    _state = AppState()
    _sessions = SessionStore()
    print(f"[startup] models loaded: {_state.registry.versions()}")
    print(f"[startup] sites: {[s['site_id'] for s in _state.sites]}")
    ensure_tracing()
    print(f"[startup] agent tracing: {'on' if tracing_enabled() else 'off'}")
    yield
    flush_traces()


app = FastAPI(title="PlantPulse", version="0.1.0", lifespan=lifespan)


def get_state() -> AppState:
    if _state is None:
        raise HTTPException(503, "service still starting up")
    return _state


INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict[str, Any]:
    state = get_state()
    return {
        "status": "ok",
        "model_versions": state.registry.versions(),
        "models": state.registry.describe(),
        "retrieval": {**state.vector_store.info, "mode": "hybrid (vector + BM25, RRF)"},
        "cache": state.cache.stats(),
    }


@app.get("/sites")
def sites() -> list[SiteInfo]:
    return get_state().sites


@app.post("/predict/anomaly")
def anomaly(req: AnomalyRequest) -> Prediction:
    try:
        return predict_anomaly(get_state(), req.site_id, req.as_of)
    except NotEnoughData as exc:
        raise HTTPException(404, str(exc))


@app.post("/predict/forecast")
def forecast(req: ForecastRequest) -> Prediction:
    try:
        return predict_forecast(get_state(), req.site_id, req.as_of)
    except NotEnoughData as exc:
        raise HTTPException(404, str(exc))


@app.post("/predict/batch")
def batch(req: BatchRequest) -> list[Prediction]:
    return batch_predict_anomaly(get_state(), req.site_ids, req.as_of)


@app.post("/search/logs")
def logs(req: SearchRequest) -> list[SearchHit]:
    try:
        return search_logs(get_state(), req.query, req.k, req.site_id)
    except UnknownSite as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/agent/query")
def agent_query(req: AgentQueryRequest) -> dict[str, Any]:
    from agent.orchestrator import (
        answer_question,  # local import avoids a hard startup dependency
    )

    return answer_question(get_state(), req.question)


@app.post("/agent/investigate")
def agent_investigate(req: InvestigateRequest) -> dict[str, Any]:
    """Plan -> execute -> independently verify. See agent/investigate.py."""
    from agent.investigate import investigate

    return investigate(get_state(), req.question, req.as_of)


def get_sessions() -> SessionStore:
    if _sessions is None:
        raise HTTPException(503, "service still starting up")
    return _sessions


@app.post("/sessions")
def create_session() -> dict[str, str]:
    """Start a conversation. Follow-up questions in it see the earlier turns."""
    return {"session_id": get_sessions().create()}


@app.post("/sessions/{session_id}/investigate")
def session_investigate(session_id: str, req: InvestigateRequest) -> dict[str, Any]:
    try:
        return investigate_in_session(get_state(), get_sessions(), session_id, req.question, req.as_of)
    except UnknownSession as exc:
        raise HTTPException(404, f"unknown session {session_id!r}") from exc


@app.post("/sessions/{session_id}/resume")
def session_resume(session_id: str) -> dict[str, Any]:
    """Finish an interrupted turn from the event log, reusing tool outputs already recorded."""
    try:
        return resume(get_state(), get_sessions(), session_id)
    except UnknownSession as exc:
        raise HTTPException(404, f"unknown session {session_id!r}") from exc
    except NothingToResume as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/sessions/{session_id}/events")
def session_events(session_id: str, start: int = 0, end: int | None = None) -> list[dict[str, Any]]:
    """The raw, append-only log; any slice can be reread."""
    try:
        return [dict(e) for e in get_sessions().events(session_id, start, end)]
    except UnknownSession as exc:
        raise HTTPException(404, f"unknown session {session_id!r}") from exc


@app.get("/sessions/{session_id}/context")
def session_context(session_id: str) -> dict[str, Any]:
    """What the agent will see on the next turn: compacted summary plus recent turns."""
    try:
        return dict(build_context(get_sessions().events(session_id)))
    except UnknownSession as exc:
        raise HTTPException(404, f"unknown session {session_id!r}") from exc


@app.get("/debug/cache-stats")
def cache_stats() -> dict[str, Any]:
    return get_state().cache.stats()


@app.get("/sites/{site_id}/top-flagged")
def top_flagged(site_id: str) -> FlaggedMoment:
    moment = get_state().top_flagged_moment(site_id)
    if moment is None:
        raise HTTPException(404, f"no data for site_id={site_id!r}")
    return moment


@app.get("/stream/telemetry/{site_id}")
async def stream_telemetry(
    site_id: str, speed: float = 20.0, tail: int = 200, start: str | None = None
) -> EventSourceResponse:
    """Server-Sent Events: replays a site's telemetry at an accelerated rate
    (`speed`x real time), running the anomaly model live on each event as
    it's emitted - the real-time inference path. Replays the last `tail`
    readings, or `tail` readings from `start` if given.
    """
    state = get_state()
    site_all = state.site_frame(site_id)
    if start is not None:
        site_all = site_all[site_all["timestamp"] >= pd.Timestamp(start)]
        site_df = site_all.head(tail).reset_index(drop=True)
    else:
        site_df = site_all.tail(tail).reset_index(drop=True)
    if site_df.empty:
        raise HTTPException(404, f"no data for site_id={site_id!r}")

    real_interval_s = 15 * 60  # data is at 15-minute resolution
    sleep_s = max(real_interval_s / speed, 0.05)

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        features = site_df[ANOMALY_FEATURES]
        for i in range(len(site_df)):
            row = site_df.iloc[i]
            proba = float(state.registry.anomaly_detector.model.predict_proba(features.iloc[[i]])[0, 1])
            payload: dict[str, Any] = {
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
