from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import agent.investigate
import app.main
from agent.sessions import RECENT_TURNS, SessionStore, build_context, investigate_in_session


def ask(client: TestClient, session_id: str, question: str) -> dict[str, Any]:
    r = client.post(f"/sessions/{session_id}/investigate", json={"question": question})
    assert r.status_code == 200
    return r.json()


def new_session(client: TestClient) -> str:
    return client.post("/sessions").json()["session_id"]


def test_follow_ups_inherit_sites_and_intents(client: TestClient) -> None:
    sid = new_session(client)
    first = ask(client, sid, "is anything wrong at wind-02, and forecast it")
    assert {s["tool"] for s in first["plan"]} >= {"check_anomaly", "forecast_power"}

    second = ask(client, sid, "and wind-03?")  # new site, same question
    assert second["turn"] == 2
    assert {(s["tool"], s["input"]["site_id"]) for s in second["plan"] if s["condition"] == "always"} == {
        ("check_anomaly", "wind-03"), ("forecast_power", "wind-03")}

    third = ask(client, sid, "have we seen problems there before?")  # same site, new question
    assert {s["input"].get("site_id") for s in third["plan"]} == {"wind-03"}
    assert any(s["tool"] == "search_maintenance_logs" and s["condition"] == "always" for s in third["plan"])
    assert third["verification"]["passed"]


def test_the_log_is_append_only_and_readable_in_slices(client: TestClient) -> None:
    sid = new_session(client)
    ask(client, sid, "investigate wind-02")
    events = client.get(f"/sessions/{sid}/events").json()
    assert [e["type"] for e in events] == ["user_message", "plan", "tool_call", "answer"]
    assert [e["seq"] for e in events] == [0, 1, 2, 3]
    assert client.get(f"/sessions/{sid}/events", params={"start": 1, "end": 3}).json() == events[1:3]


def test_long_conversations_are_compacted_without_losing_events(client: TestClient) -> None:
    sid = new_session(client)
    questions = ["investigate wind-02", "and wind-03?", "and solar-01?", "and solar-02?", "and wind-01?"]
    for q in questions:
        ask(client, sid, q)
    context = client.get(f"/sessions/{sid}/context").json()
    assert [t["turn"] for t in context["turns"]] == [3, 4, 5][-RECENT_TURNS:]
    assert "Turn 1: investigate wind-02" in context["summary"]
    assert "Turn 2: and wind-03?" in context["summary"]
    # compaction is itself an event; the original turns are all still in the log
    events = client.get(f"/sessions/{sid}/events").json()
    assert sum(e["type"] == "user_message" for e in events) == len(questions)
    assert any(e["type"] == "compaction" for e in events)


def test_an_interrupted_turn_resumes_without_repeating_tool_calls(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, store = app.main.get_state(), app.main.get_sessions()
    sid = store.create()
    moment = state.top_flagged_moment("wind-03")
    assert moment is not None

    def crash(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("process killed mid-turn")

    with monkeypatch.context() as m:
        m.setattr(agent.investigate, "template_draft", crash)
        with pytest.raises(RuntimeError):
            investigate_in_session(state, store, sid, "investigate wind-03", as_of=moment["timestamp"])
    logged = [e["type"] for e in store.events(sid)]
    assert logged == ["user_message", "plan", "tool_call", "tool_call"]  # work done, no answer yet

    calls: list[str] = []
    real_dispatch = agent.investigate.dispatch
    monkeypatch.setattr(agent.investigate, "dispatch",
                        lambda s, tool, inp: calls.append(tool) or real_dispatch(s, tool, inp))
    body = client.post(f"/sessions/{sid}/resume").json()

    assert body["resumed"] and body["turn"] == 1 and body["verification"]["passed"]
    assert calls == []  # both tool outputs came from the log
    assert all(s["reused"] for s in body["steps"])
    assert client.post(f"/sessions/{sid}/resume").status_code == 409  # nothing left to resume


def test_a_restarted_process_continues_the_conversation(client: TestClient, tmp_path: Path) -> None:
    db = str(tmp_path / "sessions.db")
    state = app.main.get_state()
    before = SessionStore(db)
    sid = before.create()
    investigate_in_session(state, before, sid, "is anything wrong at wind-02, and forecast it")

    after = SessionStore(db)  # a fresh process: nothing in memory, only the log
    assert build_context(after.events(sid))["sites"] == ["wind-02"]
    follow_up = investigate_in_session(state, after, sid, "and wind-03?")
    assert follow_up["turn"] == 2 and any(s["tool"] == "forecast_power" for s in follow_up["plan"])


def test_unknown_sessions_are_404(client: TestClient) -> None:
    response = client.post("/sessions/nope/investigate", json={"question": "investigate wind-02"})
    assert response.status_code == 404
    assert client.get("/sessions/nope/events").status_code == 404
    assert client.post("/sessions/nope/resume").status_code == 404


def test_turns_are_grouped_as_one_session_in_mlflow(client: TestClient) -> None:
    import mlflow

    sid = new_session(client)
    body = ask(client, sid, "investigate wind-02")
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(body["trace_id"])
    assert trace is not None
    assert trace.info.trace_metadata.get("mlflow.trace.session") == sid
