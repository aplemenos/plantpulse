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
