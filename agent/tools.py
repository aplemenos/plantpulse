"""
Tool definitions shared by both orchestration modes:
  - the Anthropic tool-use (function-calling) path, as Claude-style tool specs
  - the rule-based fallback path, as a plain Python dispatch table

One set of underlying functions, two ways of deciding when to call them.
"""

from __future__ import annotations

from typing import Any, cast

import mlflow
from anthropic.types import ToolParam
from mlflow.entities import Document, SpanType

from app.inference import NotEnoughData, predict_anomaly, predict_forecast, search_logs
from app.state import AppState
from ml.vector_store import SearchHit

TOOL_SPECS: list[ToolParam] = [
    {
        "name": "check_anomaly",
        "description": (
            "Check whether a given site is currently showing anomalous behavior - "
            "i.e. power output that deviates from what's expected given current "
            "weather/sensor inputs, and isn't explained by intentional curtailment. "
            "Returns the anomaly probability, actual vs expected power, and whether "
            "the site is under a known curtailment order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {"type": "string", "description": "e.g. 'solar-01', 'wind-02'"},
                "as_of": {"type": "string", "description": "Optional ISO timestamp; defaults to latest"},
            },
            "required": ["site_id"],
        },
    },
    {
        "name": "forecast_power",
        "description": "Forecast a site's power output 1 hour ahead from current conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {"type": "string"},
                "as_of": {"type": "string", "description": "Optional ISO timestamp; defaults to latest"},
            },
            "required": ["site_id"],
        },
    },
    {
        "name": "search_maintenance_logs",
        "description": (
            "Semantic search over past maintenance/incident logs. Use this to find "
            "whether a symptom has been seen before and what fixed it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Describe the symptom in plain language"},
                "k": {"type": "integer", "description": "Number of results", "default": 3},
            },
            "required": ["query"],
        },
    },
]


def _run_tool(state: AppState, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    try:
        if tool_name == "check_anomaly":
            return predict_anomaly(state, tool_input["site_id"], tool_input.get("as_of"))
        if tool_name == "forecast_power":
            return predict_forecast(state, tool_input["site_id"], tool_input.get("as_of"))
        if tool_name == "search_maintenance_logs":
            return {"results": search_logs(state, tool_input["query"], tool_input.get("k", 3))}
        return {"error": f"unknown tool {tool_name!r}"}
    except NotEnoughData as exc:
        return {"error": str(exc)}


def _as_documents(hits: list[SearchHit]) -> list[dict[str, Any]]:
    """Retrieved logs in MLflow's Document shape, so the trace UI renders them as sources."""
    return [
        Document(
            page_content=h["text"],
            metadata={"site_id": h["site_id"], "event_type": h["event_type"], "date": h["date"], "score": h["score"]},
            id=h["log_id"],
        ).to_dict()
        for h in hits
    ]


def dispatch(state: AppState, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Run one tool inside its own trace span, recording the full input and output."""
    is_retrieval = tool_name == "search_maintenance_logs"
    with mlflow.start_span(name=tool_name, span_type=SpanType.RETRIEVER if is_retrieval else SpanType.TOOL) as span:
        span.set_inputs(tool_input)
        result = _run_tool(state, tool_name, tool_input)
        if is_retrieval and "results" in result:
            span.set_outputs(_as_documents(cast(list[SearchHit], result["results"])))
        else:
            span.set_outputs(result)
        if "error" in result:
            span.set_attribute("tool_error", result["error"])
    return result
