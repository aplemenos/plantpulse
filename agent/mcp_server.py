"""
MCP server: PlantPulse's tools for any MCP client (Claude Desktop, Claude Code, MCP Inspector).

    check_anomaly            is a site behaving abnormally right now (or at `as_of`)?
    forecast_power           1-hour-ahead power forecast
    search_maintenance_logs  hybrid search over past incidents, with citable log ids
    investigate              the full plan -> execute -> verify pipeline; every call
                             runs in a session, so passing back `session_id` turns a
                             follow-up ("and wind-03?") into the next turn
    plantpulse://sites       resource listing the sites and their capacity

The three model tools go through agent/tools.dispatch and investigate through
agent/sessions, the same code paths as the HTTP API, so answers, verification
and MLflow traces are identical whichever way a question arrives.

Transports:
    python -m agent.mcp_server                  stdio (Claude Desktop, Inspector)
    python -m agent.mcp_server --http [--port]  streamable HTTP at /mcp

Loading the models and data takes a few seconds, so it starts in the background
when a client connects instead of delaying the handshake; the first tool call
waits for it if it is not done yet.

No `from __future__ import annotations` here: FastMCP builds each tool's input
schema from its signature at runtime and needs real types, not strings.
"""

import argparse
import threading
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, cast

import anyio.to_thread
import mlflow
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from mlflow.entities import SpanType
from pydantic import Field

from agent.sessions import SessionStore, UnknownSession, investigate_in_session
from agent.tools import TOOL_SPECS, dispatch
from agent.tracing import ensure_tracing, tracing_enabled
from app.state import AppState

INSTRUCTIONS = (
    "PlantPulse monitors solar and wind sites (ids like solar-01, wind-02). Use check_anomaly and "
    "forecast_power for one site's current state, search_maintenance_logs for history, and investigate "
    "for open questions: it plans, runs the tools and returns an independently verified answer. Pass "
    "the session_id from a previous investigate call to ask a follow-up in the same conversation."
)

SPECS = {spec["name"]: spec for spec in TOOL_SPECS}
DESCRIPTIONS = {name: spec.get("description", "") for name, spec in SPECS.items()}


def _param(tool: str, name: str) -> str:
    """A parameter's description from the shared tool spec, so MCP and the Claude tool loop agree."""
    props = cast(dict[str, dict[str, Any]], SPECS[tool]["input_schema"].get("properties", {}))
    return str(props[name].get("description", ""))


SiteId = Annotated[str, Field(description=_param("check_anomaly", "site_id"))]
AsOf = Annotated[str | None, Field(description=_param("check_anomaly", "as_of"))]
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


class Backend:
    """AppState and SessionStore, built once on first use and shared by every client."""

    def __init__(self, make_state: Callable[[], AppState] = AppState,
                 make_sessions: Callable[[], SessionStore] = SessionStore) -> None:
        self._make_state = make_state
        self._make_sessions = make_sessions
        self._lock = threading.Lock()
        self._state: AppState | None = None
        self._sessions: SessionStore | None = None

    def state(self) -> AppState:
        with self._lock:
            if self._state is None:
                self._state = self._make_state()
            return self._state

    def sessions(self) -> SessionStore:
        with self._lock:
            if self._sessions is None:
                self._sessions = self._make_sessions()
            return self._sessions

    def warm_up(self) -> None:
        threading.Thread(target=self.state, name="plantpulse-warm-up", daemon=True).start()


def _traced_tool(name: str, inputs: dict[str, Any], run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """One trace per MCP call, tagged so MCP traffic can be told apart from the HTTP API's."""
    ensure_tracing()
    with mlflow.start_span(name=f"mcp.{name}", span_type=SpanType.AGENT) as span:
        span.set_inputs(inputs)
        result = run()
        span.set_outputs(result)
        if tracing_enabled():
            mlflow.update_current_trace(tags={"agent.entrypoint": "mcp", "mcp.tool": name})
    return result


def _model_tool(backend: Backend, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    result = _traced_tool(name, inputs, lambda: dispatch(backend.state(), name, inputs))
    if "error" in result:
        raise ToolError(str(result["error"]))
    return result


def _investigate(backend: Backend, question: str, as_of: str | None, session_id: str | None) -> dict[str, Any]:
    sessions = backend.sessions()
    sid = session_id or sessions.create()
    try:
        result = investigate_in_session(backend.state(), sessions, sid, question, as_of)
    except UnknownSession as exc:
        raise ToolError(f"unknown session {sid!r}; omit session_id to start a new one") from exc
    verification: dict[str, Any] = result["verification"]
    return {
        "answer": result["answer"],
        "verified": verification["passed"],
        "issues": verification["issues"],
        "revisions": verification["revisions"],
        "mode": result["mode"],
        "session_id": sid,
        "tool_calls": [{"tool": s["tool"], "input": s["input"]} for s in result["steps"] if not s["skipped"]],
        "trace_id": result.get("trace_id"),
    }


def build_server(backend: Backend | None = None, host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    backend = backend or Backend()

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncGenerator[None]:
        backend.warm_up()
        yield

    server = FastMCP("plantpulse", instructions=INSTRUCTIONS, lifespan=lifespan, host=host, port=port)

    # Tools run in a worker thread: model inference and retrieval are blocking calls.
    @server.tool(description=DESCRIPTIONS["check_anomaly"], annotations=READ_ONLY)
    async def check_anomaly(site_id: SiteId, as_of: AsOf = None) -> dict[str, Any]:
        inputs = {"site_id": site_id, **({"as_of": as_of} if as_of else {})}
        return await anyio.to_thread.run_sync(_model_tool, backend, "check_anomaly", inputs)

    @server.tool(description=DESCRIPTIONS["forecast_power"], annotations=READ_ONLY)
    async def forecast_power(site_id: SiteId, as_of: AsOf = None) -> dict[str, Any]:
        inputs = {"site_id": site_id, **({"as_of": as_of} if as_of else {})}
        return await anyio.to_thread.run_sync(_model_tool, backend, "forecast_power", inputs)

    @server.tool(description=DESCRIPTIONS["search_maintenance_logs"], annotations=READ_ONLY)
    async def search_maintenance_logs(
        query: Annotated[str, Field(description=_param("search_maintenance_logs", "query"))],
        k: Annotated[int, Field(ge=1, le=20, description=_param("search_maintenance_logs", "k"))] = 3,
        site_id: Annotated[str | None, Field(description=_param("search_maintenance_logs", "site_id"))] = None,
    ) -> dict[str, Any]:
        inputs = {"query": query, "k": k, **({"site_id": site_id} if site_id else {})}
        return await anyio.to_thread.run_sync(_model_tool, backend, "search_maintenance_logs", inputs)

    @server.tool(
        description=(
            "Investigate an open question about one or more sites, e.g. 'is anything wrong at wind-02, "
            "and have we seen it before?'. Plans which tools to run, runs them, drafts an answer and has "
            "it checked by an independent verifier. `verified` is false when the answer failed "
            "verification; `issues` says why. Returns a session_id: pass it back to ask a follow-up "
            "in the same conversation."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, openWorldHint=False),
    )
    async def investigate(
        question: Annotated[str, Field(description="Plain-English question about one or more sites")],
        as_of: Annotated[str | None, Field(description="Optional ISO timestamp to investigate a past moment")] = None,
        session_id: Annotated[str | None, Field(description="From an earlier investigate call, to ask a follow-up")]
        = None,
    ) -> dict[str, Any]:
        inputs = {"question": question, "as_of": as_of, "session_id": session_id}
        return await anyio.to_thread.run_sync(
            _traced_tool, "investigate", inputs, lambda: _investigate(backend, question, as_of, session_id))

    @server.resource("plantpulse://sites", name="sites", mime_type="application/json",
                     description="Monitored sites with type and capacity in kW.")
    async def sites() -> list[dict[str, Any]]:
        state = await anyio.to_thread.run_sync(backend.state)
        return [dict(s) for s in state.sites]

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PlantPulse MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    transport: Literal["stdio", "streamable-http"] = "streamable-http" if args.http else "stdio"
    build_server(host=args.host, port=args.port).run(transport)


if __name__ == "__main__":
    main()
