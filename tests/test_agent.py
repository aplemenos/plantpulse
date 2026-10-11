import pytest
from fastapi.testclient import TestClient

from agent.orchestrator import extract_intents, extract_site_ids

KNOWN = ["solar-01", "solar-02", "solar-03", "wind-01", "wind-02", "wind-03"]


def test_site_ids_are_normalized() -> None:
    assert extract_site_ids("check wind-2 and Solar-03", KNOWN) == ["solar-03", "wind-02"]


def test_unknown_site_ids_are_ignored() -> None:
    assert extract_site_ids("what about wind-07?", KNOWN) == []


def test_history_intent_tolerates_words_in_between() -> None:
    # Regression test for a routing bug the eval suite caught:
    # "seen this vibration before" used to miss the log-search tool.
    assert "search_maintenance_logs" in extract_intents("have we seen this vibration before?")


def test_multiple_intents_in_one_question() -> None:
    intents = extract_intents("forecast solar-01 and tell me if there's a fault")
    assert set(intents) == {"forecast_power", "check_anomaly"}


def test_agent_answers_with_tool_trace(client: TestClient) -> None:
    body = client.post("/agent/query", json={"question": "is anything wrong with wind-02?"}).json()
    assert body["mode"] == "rule_based"
    assert body["tool_calls"][0]["tool"] == "check_anomaly"
    assert "wind-02" in body["answer"]


def test_agent_explains_curtailment_instead_of_flagging_a_fault(client: TestClient) -> None:
    # Pick a moment we know is curtailed from the data, and make sure the agent says so.
    from app.main import get_state

    df = get_state().df
    row = df[df["is_curtailed"] == 1].iloc[0]
    from agent.tools import dispatch

    result = dispatch(get_state(), "check_anomaly", {"site_id": str(row["site_id"]), "as_of": str(row["timestamp"])})
    assert result["is_curtailed"] is True


def test_agent_asks_for_a_site_when_none_named(client: TestClient) -> None:
    body = client.post("/agent/query", json={"question": "how's everything?"}).json()
    assert body["tool_calls"] == []
    assert "which site" in body["answer"]


def test_agent_query_is_traced_end_to_end(client: TestClient) -> None:
    import mlflow

    body = client.post(
        "/agent/query", json={"question": "Is anything wrong at wind-02, and have we seen this before?"}
    ).json()
    assert body["trace_id"].startswith("tr-")

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(body["trace_id"])
    assert trace is not None
    spans = {s.name: s for s in trace.data.spans}

    root = spans["agent_query"]
    assert root.parent_id is None and root.span_type == "AGENT"
    assert root.inputs == {"question": "Is anything wrong at wind-02, and have we seen this before?"}
    assert root.outputs["answer"] == body["answer"]

    assert spans["route_question"].outputs == {
        "site_ids": ["wind-02"],
        "intents": ["check_anomaly", "search_maintenance_logs"],
    }
    assert spans["check_anomaly"].span_type == "TOOL"
    assert spans["check_anomaly"].outputs == body["tool_calls"][0]["output"]
    assert spans["search_maintenance_logs"].span_type == "RETRIEVER"
    assert spans["search_maintenance_logs"].outputs[0]["page_content"]
    assert trace.info.tags["agent.mode"] == "rule_based"


def test_citations_are_checked_against_what_was_retrieved() -> None:
    from agent.orchestrator import check_citations

    tool_calls = [{"tool": "search_maintenance_logs", "input": {}, "output": {"results": [{"log_id": "log-0009"}]}}]
    supported, unsupported = check_citations("Seen before [log-0009], also [log-0042]; see [log-0009].", tool_calls)
    assert supported == ["log-0009"] and unsupported == ["log-0042"]


def test_rule_based_answer_cites_the_logs_it_used(client: TestClient) -> None:
    body = client.post("/agent/query", json={"question": "have we seen this vibration before on wind-3"}).json()
    search = next(c for c in body["tool_calls"] if c["tool"] == "search_maintenance_logs")
    assert search["input"]["site_id"] == "wind-03"
    retrieved = [h["log_id"] for h in search["output"]["results"]]
    assert body["citations"] == retrieved and body["unsupported_citations"] == []


def test_llm_loop_formats_tool_results_and_checks_citations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the real tool-use loop with a scripted stand-in for Claude."""
    from types import SimpleNamespace
    from typing import Any

    import anthropic
    from anthropic.types import TextBlock, ToolUseBlock

    import app.main
    from agent.orchestrator import answer_question

    sent: list[dict[str, Any]] = []
    tool_results: list[str] = []  # captured per call: the loop keeps appending to `messages`

    class FakeMessages:
        def create(self, **kwargs: Any) -> Any:
            sent.append(kwargs)
            if len(sent) == 1:  # first turn: Claude asks for a log search
                return SimpleNamespace(content=[ToolUseBlock(
                    type="tool_use", id="t1", name="search_maintenance_logs",
                    input={"query": "inverter overheating", "k": 2},
                )])
            # second turn: answer citing the first log it was shown, plus one it never saw
            tool_results.append(kwargs["messages"][-1]["content"][0]["content"])
            first_id = tool_results[0].split("[", 1)[1].split("]", 1)[0]
            return SimpleNamespace(content=[TextBlock(type="text", text=f"Yes - see [{first_id}]. Also [log-9999].")])

    class FakeAnthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    body = answer_question(app.main.get_state(), "has an inverter overheated before?")

    assert body["mode"] == "llm" and "cite" in sent[0]["system"]
    assert tool_results[0].startswith("1. [log-")  # numbered, citable snippets - not a raw dict
    first_retrieved = body["tool_calls"][0]["output"]["results"][0]["log_id"]
    assert body["citations"] == [first_retrieved]
    assert body["unsupported_citations"] == ["log-9999"]
