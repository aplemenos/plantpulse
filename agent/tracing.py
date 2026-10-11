"""
MLflow Tracing for the agent.

Every /agent/query becomes one trace in the `plantpulse-agent` experiment:

    agent_query (AGENT)                 question in, answer + mode out
    ├── route_question (CHAIN)          rule-based mode: sites + intents picked
    ├── Messages.create (CHAT_MODEL)    LLM mode: one span per Claude call, from
    │                                   mlflow.anthropic.autolog (prompt, tool
    │                                   choices, token usage, errors)
    └── check_anomaly / forecast_power (TOOL), search_maintenance_logs (RETRIEVER)
                                        full tool inputs and outputs

so "why did the agent answer that?" is answered by opening the trace in the
MLflow UI (`make mlflow-ui`, Traces tab) instead of re-running the question.

Traces are written asynchronously, so a request doesn't wait on the database.
Set PLANTPULSE_TRACING=0 to turn tracing off.
"""

from __future__ import annotations

import os
import threading

import mlflow
from mlflow.anthropic import autolog as anthropic_autolog
from mlflow.config import enable_async_logging
from mlflow.tracing import disable as disable_tracing
from mlflow.tracing import enable as enable_tracing

from ml.mlflow_config import agent_experiment, use_experiment

_lock = threading.Lock()
_configured = False


def tracing_enabled() -> bool:
    return os.environ.get("PLANTPULSE_TRACING", "1") != "0"


def ensure_tracing() -> None:
    """Point traces at the agent experiment and patch the Anthropic client (idempotent)."""
    global _configured
    if _configured:
        return
    with _lock:
        if _configured:
            return
        if tracing_enabled():
            use_experiment(agent_experiment())
            enable_async_logging(True)
            anthropic_autolog(silent=True)
            enable_tracing()
        else:
            disable_tracing()
        _configured = True


def flush_traces() -> None:
    """Write out traces still queued by async logging (call on shutdown)."""
    if _configured and tracing_enabled():
        mlflow.flush_trace_async_logging()
