"""
Sessions: multi-turn investigations backed by an append-only event log.

This follows the "decouple the brain from the hands" design: the session
log lives outside both the agent loop and the tools, so any process can
pick a conversation up from it.

    store     SQLite table of events, append-only. Nothing is ever updated
              or deleted. Interface: create(), emit(), events(start, end).
    context   built from the log by the harness, not stored: a summary of
              older turns plus the recent ones verbatim. Compaction writes a
              new summary *event*; the raw events stay, so nothing the agent
              might need later is irreversibly thrown away.
    resume    a turn that was interrupted (process crash, deploy, timeout)
              is the last question without an answer. Resuming reuses its
              logged plan and tool outputs and only does what's left.

Event types: user_message, plan, tool_call, answer, compaction.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from typing_extensions import TypedDict

from agent.investigate import Context, PlanStep, Turn, empty_context, investigate, step_key
from app.state import AppState

ROOT = Path(__file__).resolve().parent.parent
RECENT_TURNS = 3       # turns kept verbatim in the context
SUMMARY_MAX_CHARS = 1500


class Event(TypedDict):
    session_id: str
    seq: int
    type: str
    payload: dict[str, Any]
    created_at: str


class UnknownSession(KeyError):
    pass


class NothingToResume(RuntimeError):
    pass


def default_db_path() -> str:
    return os.environ.get("PLANTPULSE_SESSIONS_DB") or str(ROOT / "var" / "sessions.db")


class SessionStore:
    """Append-only event log in SQLite. Safe to share across request threads."""

    def __init__(self, path: str | None = None) -> None:
        path = path or default_db_path()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                session_id TEXT NOT NULL REFERENCES sessions(id),
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, seq)
            );
        """)
        self._db.commit()

    def create(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._db.execute("INSERT INTO sessions VALUES (?, ?)", (session_id, _now()))
            self._db.commit()
        return session_id

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None

    def emit(self, session_id: str, kind: str, payload: dict[str, Any]) -> int:
        """Append one event; it is committed before this returns, so it survives a crash."""
        with self._lock:
            row = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE session_id = ?",
                                   (session_id,)).fetchone()
            seq = int(row[0])
            self._db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                             (session_id, seq, kind, json.dumps(payload, default=str), _now()))
            self._db.commit()
        return seq

    def events(self, session_id: str, start: int = 0, end: int | None = None) -> list[Event]:
        """Events with start <= seq < end: a positional slice, so callers can reread any part of the log."""
        if not self.exists(session_id):
            raise UnknownSession(session_id)
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, seq, type, payload, created_at FROM events "
                "WHERE session_id = ? AND seq >= ? AND seq < ? ORDER BY seq",
                (session_id, start, end if end is not None else 2**62),
            ).fetchall()
        return [{"session_id": r[0], "seq": r[1], "type": r[2], "payload": json.loads(r[3]), "created_at": r[4]}
                for r in rows]


def _now() -> str:
    return datetime.now(UTC).isoformat()


# --- reading the log ------------------------------------------------------------


class PendingTurn(TypedDict):
    turn: int
    question: str
    as_of: str | None
    plan: list[PlanStep] | None
    done: dict[str, dict[str, Any]]


def _turns(events: list[Event]) -> tuple[list[Turn], PendingTurn | None, Context]:
    """Replay the log: completed turns, the unfinished one (if any), and the focus of the last answered turn."""
    turns: list[Turn] = []
    current: PendingTurn | None = None
    focus = empty_context()
    plan_focus: tuple[list[str], list[str]] = ([], [])
    for e in events:
        p = e["payload"]
        if e["type"] == "user_message":
            current = {"turn": int(p["turn"]), "question": str(p["question"]), "as_of": p.get("as_of"),
                       "plan": None, "done": {}}
        elif current is None:
            continue
        elif e["type"] == "plan":
            current["plan"] = cast(list[PlanStep], p["plan"])
            plan_focus = (cast(list[str], p["sites"]), cast(list[str], p["intents"]))
        elif e["type"] == "tool_call":
            current["done"][step_key(str(p["tool"]), cast(dict[str, Any], p["input"]))] = cast(dict[str, Any], p["output"])
        elif e["type"] == "answer":
            turns.append({"turn": current["turn"], "question": current["question"], "answer": str(p["answer"])})
            focus["sites"], focus["intents"] = plan_focus
            current = None
    return turns, current, focus


def build_context(events: list[Event]) -> Context:
    """What the agent sees: the latest compaction summary, then the turns it doesn't cover."""
    turns, _, context = _turns(events)
    summary, covered = "", 0
    for e in events:
        if e["type"] == "compaction":
            summary, covered = str(e["payload"]["summary"]), int(e["payload"]["covers_turns_up_to"])
    context["summary"] = summary
    context["turns"] = [t for t in turns if t["turn"] > covered]
    return context


def _first_sentence(text: str, limit: int = 160) -> str:
    sentence = text.split(". ")[0].strip()
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"


def compact_if_needed(store: SessionStore, session_id: str) -> bool:
    """Fold all but the most recent turns into the running summary, as a new event."""
    context = build_context(store.events(session_id))
    if len(context["turns"]) <= RECENT_TURNS:
        return False
    older = context["turns"][:-RECENT_TURNS]
    lines = [context["summary"]] if context["summary"] else []
    lines += [f"Turn {t['turn']}: {t['question']} -> {_first_sentence(t['answer'])}" for t in older]
    summary = "\n".join(lines)[-SUMMARY_MAX_CHARS:]
    store.emit(session_id, "compaction", {"summary": summary, "covers_turns_up_to": older[-1]["turn"]})
    return True


# --- running turns ----------------------------------------------------------------


def investigate_in_session(
    state: AppState, store: SessionStore, session_id: str, question: str, as_of: str | None = None,
    allow_llm: bool = True,
) -> dict[str, Any]:
    """Answer a new question in the conversation, using earlier turns as context."""
    events = store.events(session_id)
    turns, pending, _ = _turns(events)
    turn = max([t["turn"] for t in turns] + ([pending["turn"]] if pending else []), default=0) + 1
    context = build_context(events)
    store.emit(session_id, "user_message", {"turn": turn, "question": question, "as_of": as_of})
    return _run_turn(state, store, session_id, turn, question, as_of, context, None, None, allow_llm)


def resume(state: AppState, store: SessionStore, session_id: str, allow_llm: bool = True) -> dict[str, Any]:
    """Finish an interrupted turn: reuse its logged plan and tool outputs, do only what's left."""
    events = store.events(session_id)
    _, pending, _ = _turns(events)
    if pending is None:
        raise NothingToResume(f"session {session_id} has no unfinished turn")
    context = build_context(events)
    return _run_turn(state, store, session_id, pending["turn"], pending["question"], pending["as_of"], context,
                     pending["plan"], pending["done"], allow_llm)


def _run_turn(
    state: AppState, store: SessionStore, session_id: str, turn: int, question: str, as_of: str | None,
    context: Context, prior_plan: list[PlanStep] | None, done: dict[str, dict[str, Any]] | None, allow_llm: bool,
) -> dict[str, Any]:
    result = investigate(
        state, question, as_of, allow_llm,
        context=context,
        sink=lambda kind, payload: store.emit(session_id, kind, payload),
        prior_plan=prior_plan, done=done, session_id=session_id,
    )
    compacted = compact_if_needed(store, session_id)
    return {**result, "session_id": session_id, "turn": turn, "resumed": prior_plan is not None,
            "context_compacted": compacted}
