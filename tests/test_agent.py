from agent.orchestrator import _extract_intents, _extract_site_ids

KNOWN = ["solar-01", "solar-02", "solar-03", "wind-01", "wind-02", "wind-03"]


def test_site_ids_are_normalized():
    assert _extract_site_ids("check wind-2 and Solar-03", KNOWN) == ["solar-03", "wind-02"]


def test_unknown_site_ids_are_ignored():
    assert _extract_site_ids("what about wind-07?", KNOWN) == []


def test_history_intent_tolerates_words_in_between():
    # Regression test for a routing bug the eval suite caught:
    # "seen this vibration before" used to miss the log-search tool.
    assert "search_maintenance_logs" in _extract_intents("have we seen this vibration before?")


def test_multiple_intents_in_one_question():
    intents = _extract_intents("forecast solar-01 and tell me if there's a fault")
    assert set(intents) == {"forecast_power", "check_anomaly"}


def test_agent_answers_with_tool_trace(client):
    body = client.post("/agent/query", json={"question": "is anything wrong with wind-02?"}).json()
    assert body["mode"] == "rule_based"
    assert body["tool_calls"][0]["tool"] == "check_anomaly"
    assert "wind-02" in body["answer"]


def test_agent_explains_curtailment_instead_of_flagging_a_fault(client):
    # Pick a moment we know is curtailed from the data, and make sure the agent says so.
    from app.main import get_state

    df = get_state().df
    row = df[df["is_curtailed"] == 1].iloc[0]
    from agent.tools import dispatch

    result = dispatch(get_state(), "check_anomaly", {"site_id": row["site_id"], "as_of": str(row["timestamp"])})
    assert result["is_curtailed"] is True


def test_agent_asks_for_a_site_when_none_named(client):
    body = client.post("/agent/query", json={"question": "how's everything?"}).json()
    assert body["tool_calls"] == []
    assert "which site" in body["answer"]
