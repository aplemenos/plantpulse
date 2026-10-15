from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main
from agent.investigate import execute, investigate, validate_plan
from agent.verifier import check_citations, check_numbers, check_status, verify

HEALTHY = {"site_id": "wind-02", "is_anomaly": False, "is_curtailed": False, "anomaly_probability": 0.03,
           "actual_power_kw": 450.42, "expected_power_kw": 405.78, "residual_pct": 0.11}
ANOMALOUS = {**HEALTHY, "is_anomaly": True, "anomaly_probability": 0.9955, "residual_pct": -0.3415}
CURTAILED = {**HEALTHY, "is_curtailed": True}
LOGS = {"results": [{"log_id": "log-0021", "text": "2026-02-11 — Site wind-02: Vibration on turbine 4 for 9h."}]}


def evidence(check: dict[str, Any], logs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    steps = [{"tool": "check_anomaly", "input": {"site_id": "wind-02"}, "output": check}]
    if logs:
        steps.append({"tool": "search_maintenance_logs", "input": {"query": "x"}, "output": logs})
    return steps


# --- verifier -----------------------------------------------------------------


def test_numbers_from_the_evidence_pass_allowing_rounding_and_percentages() -> None:
    answer = "wind-02 makes 450 kW vs 405.8 kW expected, 34.2% below, p=0.9955, see log-0021 (9h, turbine 4)."
    assert check_numbers(answer, evidence(ANOMALOUS, LOGS)) == []


def test_invented_numbers_are_caught() -> None:
    issues = check_numbers("wind-02 makes 612.5 kW.", evidence(HEALTHY))
    assert [i["detail"] for i in issues] == ["612.5 does not appear in any tool output"]


def test_ids_dates_and_small_counts_are_not_treated_as_claims() -> None:
    answer = "solar-01 and wind-03 on 2026-04-18 11:00 had 2 incidents [log-0021]."
    assert check_numbers(answer, evidence(HEALTHY)) == []


def test_citations_must_have_been_retrieved() -> None:
    issues = check_citations("See [log-0021] and [log-0099].", evidence(HEALTHY, LOGS))
    assert [i["detail"] for i in issues] == ["log-0099 is cited but was never retrieved"]


@pytest.mark.parametrize(
    ("result", "answer", "ok"),
    [
        (HEALTHY, "wind-02 looks healthy.", True),
        (HEALTHY, "wind-02 is anomalous.", False),
        (HEALTHY, "wind-02 shows no anomaly.", True),
        (ANOMALOUS, "wind-02 looks anomalous.", True),
        (ANOMALOUS, "wind-02 looks healthy.", False),
        (CURTAILED, "wind-02 is low because of a curtailment order.", True),
        (CURTAILED, "wind-02 output is low.", False),
    ],
)
def test_status_claims_must_match_the_model(result: dict[str, Any], answer: str, ok: bool) -> None:
    assert (check_status(answer, evidence(result)) == []) is ok


def test_quoted_logs_do_not_count_as_status_claims() -> None:
    # Regression: with the bge embedder, solar-02's search returned a curtailment log ending in
    # "equipment healthy", and the verifier read that as the answer calling the site healthy.
    log = {"log_id": "log-0007", "text": "2026-03-01 — Site wind-02: Output cap applied; equipment healthy."}
    curtailment_log = {"log_id": "log-0008", "text": "2026-03-02 — Site wind-02: Grid curtailment order for 2h."}
    answer = f"wind-02 looks anomalous. Past incidents: [log-0007] {log['text']} [log-0008] {curtailment_log['text']}"
    logs = {"results": [log, curtailment_log]}
    assert check_status(answer, evidence(ANOMALOUS, logs)) == []
    # ...and a quoted curtailment log doesn't satisfy "mention today's curtailment" either
    issues = check_status("wind-02 output is low. [log-0008] " + curtailment_log["text"], evidence(CURTAILED, logs))
    assert [i["check"] for i in issues] == ["status"]


def test_verify_reports_each_check() -> None:
    v = verify("wind-02 is anomalous at 999 kW [log-0042].", evidence(HEALTHY))
    assert not v["passed"]
    assert v["checks"] == {"numbers": False, "citations": False, "status": False}


# --- rule-based pipeline ------------------------------------------------------


def test_healthy_site_skips_the_conditional_history_search(client: TestClient) -> None:
    body = client.post("/agent/investigate", json={"question": "investigate wind-02"}).json()
    assert [(s["tool"], s["skipped"]) for s in body["steps"]] == [
        ("check_anomaly", False), ("search_maintenance_logs", True)]
    assert body["verification"]["passed"] and body["mode"] == "rule_based"


def test_anomalous_moment_triggers_the_history_search_and_verifies(client: TestClient) -> None:
    moment = client.get("/sites/wind-03/top-flagged").json()
    body = client.post("/agent/investigate",
                       json={"question": "investigate wind-03", "as_of": moment["timestamp"]}).json()
    assert body["steps"][0]["output"]["is_anomaly"] is True
    assert body["steps"][1]["tool"] == "search_maintenance_logs" and not body["steps"][1]["skipped"]
    assert "[log-" in body["answer"]
    assert body["verification"]["passed"]


def test_plan_validation_drops_what_cannot_run(client: TestClient) -> None:
    steps, dropped = validate_plan(app.main.get_state(), [
        {"tool": "check_anomaly", "input": {"site_id": "wind-02"}, "condition": "always"},
        {"tool": "delete_site", "input": {}, "condition": "always"},
        {"tool": "forecast_power", "input": {"site_id": "wind-99"}, "condition": "always"},
        {"tool": "search_maintenance_logs", "input": {"query": "x"}, "condition": "if it feels right"},
    ])
    assert [s["tool"] for s in steps] == ["check_anomaly"]
    assert len(dropped) == 3


def test_conditions_only_follow_anomalies_not_curtailment(client: TestClient) -> None:
    results = execute(app.main.get_state(), [
        {"tool": "check_anomaly", "input": {"site_id": "solar-01"}, "reason": "", "condition": "always"},
        {"tool": "search_maintenance_logs", "input": {"query": "x"}, "reason": "", "condition": "anomalous:solar-01"},
    ])
    first = results[0]["output"]
    assert results[1]["skipped"] is not (first["is_anomaly"] and not first["is_curtailed"])


def test_investigation_is_one_trace_with_an_evaluator_span(client: TestClient) -> None:
    import mlflow

    body = client.post("/agent/investigate", json={"question": "investigate wind-02"}).json()
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(body["trace_id"])
    assert trace is not None
    spans = {s.name: s for s in trace.data.spans}
    assert spans["agent_investigate"].parent_id is None
    assert spans["verify"].span_type == "EVALUATOR"
    assert {"plan", "execute", "check_anomaly"} <= spans.keys()
    assert trace.info.tags["agent.pipeline"] == "investigate"


# --- LLM pipeline, with Claude scripted ---------------------------------------


def _scripted_claude(monkeypatch: pytest.MonkeyPatch, drafts: list[str], verdicts: list[list[str]]) -> list[dict[str, Any]]:
    import anthropic
    from anthropic.types import TextBlock, ToolUseBlock

    calls: list[dict[str, Any]] = []
    plan = [
        {"tool": "check_anomaly", "input": {"site_id": "wind-02"}, "reason": "health first", "condition": "always"},
        {"tool": "search_maintenance_logs", "input": {"query": "wind-02 faults", "site_id": "wind-02"},
         "reason": "precedents", "condition": "anomalous:wind-02"},
        {"tool": "restart_turbine", "input": {"site_id": "wind-02"}, "reason": "?", "condition": "always"},
    ]

    class FakeMessages:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            forced = kwargs.get("tool_choice", {}).get("name")
            if forced == "submit_plan":
                return SimpleNamespace(content=[ToolUseBlock(type="tool_use", id="p", name=forced, input={"steps": plan})])
            if forced == "submit_verdict":
                return SimpleNamespace(content=[ToolUseBlock(type="tool_use", id="v", name=forced,
                                                             input={"issues": verdicts.pop(0)})])
            return SimpleNamespace(content=[TextBlock(type="text", text=drafts.pop(0))])

    class FakeAnthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return calls


def test_llm_pipeline_rejects_an_invented_number_and_accepts_the_revision(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _scripted_claude(monkeypatch, drafts=["wind-02 looks healthy at 999.9 kW.", "wind-02 looks healthy."],
                             verdicts=[[], []])
    body = investigate(app.main.get_state(), "investigate wind-02")

    assert body["mode"] == "llm" and body["answer"] == "wind-02 looks healthy."
    v = body["verification"]
    assert v["passed"] and v["revisions"] == 1
    assert v["rejected_drafts"][0][0]["check"] == "numbers"
    assert body["plan_dropped"] == ["unknown tool 'restart_turbine'"]

    roles = {c.get("tool_choice", {}).get("name", "draft"): c["model"] for c in calls}
    assert roles["draft"] != roles["submit_verdict"]  # no self-verification
    revision_prompt = [c for c in calls if "tool_choice" not in c][1]["messages"][0]["content"]
    assert "999.9 does not appear" in revision_prompt


def test_llm_pipeline_marks_the_answer_unverified_when_the_judge_keeps_objecting(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_claude(monkeypatch, drafts=["wind-02 looks healthy.", "wind-02 looks healthy."],
                     verdicts=[["claims a cause the evidence doesn't show"], ["still unsupported"]])
    v = investigate(app.main.get_state(), "investigate wind-02")["verification"]
    assert not v["passed"] and v["revisions"] == 1
    assert v["checks"]["judge"] is False
    assert v["issues"] == [{"check": "judge", "detail": "still unsupported"}]


def test_regression_gate_can_force_offline_mode(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "would-cost-money")
    assert investigate(app.main.get_state(), "investigate wind-02", allow_llm=False)["mode"] == "rule_based"
