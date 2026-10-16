import json
import os
import sys
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from fastapi.testclient import TestClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.message import SessionMessage
from mcp.types import TextContent, TextResourceContents
from pydantic import AnyUrl

import app.main
from agent.mcp_server import Backend, build_server
from agent.tools import TOOL_SPECS

ROOT = Path(__file__).resolve().parent.parent

# mcp's stdio_client has no return annotation, and editors infer it differently (Pylance sees a plain
# Generator), so the test states the type it relies on.
StdioStreams = tuple[MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]]


@pytest.fixture(scope="module")
def server(client: TestClient) -> FastMCP:
    # Reuse the state the API test client already loaded instead of loading it again.
    return build_server(Backend(make_state=app.main.get_state, make_sessions=app.main.get_sessions))


def call(server: FastMCP, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    async def run() -> Any:
        return await server.call_tool(tool, args)

    _, structured = cast(tuple[Any, dict[str, Any]], anyio.run(run))
    return structured


def test_tools_mirror_the_shared_specs(server: FastMCP) -> None:
    tools = {t.name: t for t in anyio.run(server.list_tools)}
    assert set(tools) == {spec["name"] for spec in TOOL_SPECS} | {"investigate"}
    for spec in TOOL_SPECS:
        tool = tools[spec["name"]]
        assert tool.description == spec.get("description")
        assert tool.inputSchema.get("required") == spec["input_schema"].get("required")
        assert tool.annotations is not None and tool.annotations.readOnlyHint
    assert tools["investigate"].annotations is not None and not tools["investigate"].annotations.readOnlyHint


def test_model_tool_returns_the_same_result_as_dispatch(server: FastMCP) -> None:
    result = call(server, "check_anomaly", {"site_id": "wind-02"})
    assert result["site_id"] == "wind-02" and {"is_anomaly", "anomaly_probability", "is_curtailed"} <= set(result)
    hits = call(server, "search_maintenance_logs", {"query": "inverter overheating", "k": 2})["results"]
    assert len(hits) == 2 and all(h["log_id"].startswith("log-") for h in hits)


def test_tool_errors_are_reported_as_mcp_errors(server: FastMCP) -> None:
    with pytest.raises(ToolError, match="no telemetry"):
        call(server, "forecast_power", {"site_id": "nope-99"})


def test_investigate_starts_a_session_and_follow_ups_continue_it(server: FastMCP) -> None:
    first = call(server, "investigate", {"question": "is anything wrong at wind-02?"})
    assert first["verified"] and first["mode"] == "rule_based" and first["session_id"]
    assert any(c["input"].get("site_id") == "wind-02" for c in first["tool_calls"])

    follow_up = call(server, "investigate", {"question": "and wind-03?", "session_id": first["session_id"]})
    assert follow_up["session_id"] == first["session_id"]
    assert any(c["input"].get("site_id") == "wind-03" for c in follow_up["tool_calls"])
    turns = [e for e in app.main.get_sessions().events(first["session_id"]) if e["type"] == "user_message"]
    assert len(turns) == 2


def test_unknown_session_is_an_error_not_a_new_conversation(server: FastMCP) -> None:
    with pytest.raises(ToolError, match="unknown session"):
        call(server, "investigate", {"question": "and wind-03?", "session_id": "does-not-exist"})


def test_stdio_end_to_end(tmp_path: Path) -> None:
    """The real thing: a client launches the server as a subprocess and talks MCP over stdin/stdout.
    Any stray print to stdout would corrupt the protocol and fail this test."""
    env = {**os.environ, "PLANTPULSE_TRACING": "0", "PLANTPULSE_SESSIONS_DB": str(tmp_path / "s.db")}
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent.mcp_server"], env=env, cwd=str(ROOT))

    async def run() -> None:
        transport = cast(AbstractAsyncContextManager[StdioStreams], stdio_client(params))
        async with transport as (read, write), ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.serverInfo.name == "plantpulse" and init.instructions

            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {"check_anomaly", "forecast_power",
                                                     "search_maintenance_logs", "investigate"}

            ok = await session.call_tool("check_anomaly", {"site_id": "solar-01"})
            assert not ok.isError and ok.structuredContent and ok.structuredContent["site_id"] == "solar-01"

            bad = await session.call_tool("check_anomaly", {"site_id": "nope-99"})
            assert bad.isError and isinstance(bad.content[0], TextContent) and "no telemetry" in bad.content[0].text

            sites = await session.read_resource(AnyUrl("plantpulse://sites"))
            first = sites.contents[0]
            assert isinstance(first, TextResourceContents)
            assert {s["site_id"] for s in json.loads(first.text)} >= {"solar-01", "wind-02"}

    anyio.run(run)
