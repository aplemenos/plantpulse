"""
Agent orchestration: turns a natural-language analytics question into
tool calls against the ML serving layer, then a synthesized answer.

Two modes, same underlying tools (agent/tools.py):

  - LLM mode (real agentic tool-use): used when ANTHROPIC_API_KEY is set.
    Calls Claude's Messages API with the tool specs and lets the model
    decide what to call and how to compose the answer. This is the
    genuinely agentic path, and the one worth describing as "the" design
    in an interview - but it needs a real API key to run, which this
    sandbox was not given, so it is implemented but not exercised by the
    test suite here.

  - Rule-based mode (fallback, default, no credentials needed): a small
    keyword/entity planner that still calls the *same* tool functions,
    so the service and its tests are fully runnable with zero external
    dependencies. This is the mode exercised below and in CI.

Both paths return the same response shape:
{"answer": str, "tool_calls": [...], "mode": "llm" | "rule_based",
 "citations": [log ids], "unsupported_citations": [log ids], "trace_id": str}.

Answers that use maintenance logs cite them by id ([log-0009]). Citations
are checked against what the search tool actually returned in this
conversation: a cited id that was never retrieved lands in
unsupported_citations, a cheap signal of an ungrounded answer.

Each question is recorded as one MLflow trace (see agent/tracing.py); the
response's trace_id opens it in the MLflow UI.
"""

from __future__ import annotations

import os
import re
from typing import Any, cast

import mlflow
from mlflow.entities import SpanType

from agent.tools import TOOL_SPECS, dispatch, format_tool_result
from agent.tracing import ensure_tracing, tracing_enabled
from app.state import AppState

AgentAnswer = dict[str, Any]

_SITE_ID_RE = re.compile(r"\b((?:solar|wind)-0?\d+)\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\blog-\d+\b")

SYSTEM_PROMPT = """You are PlantPulse, an analytics assistant for a fleet of solar and wind sites.
Answer operators' questions using the tools; don't guess numbers you can fetch.
Low output under a curtailment order is intentional, not a fault - say so when it applies.
When your answer relies on maintenance logs, cite each one inline by its id in square
brackets, e.g. [log-0009]. Only cite logs returned by search_maintenance_logs, and say so
plainly if no similar incident was found. Keep answers short and concrete."""

_HISTORY_RE = re.compile(
    r"\bseen\b.*\bbefore\b|\bsimilar\b|\bhistory\b|\blogs?\b|\bmaintenance\b|\bpast incidents?\b|\bhappened before\b",
    re.IGNORECASE,
)


def extract_site_ids(question: str, known_sites: list[str]) -> list[str]:
    found = {m.group(1).lower() for m in _SITE_ID_RE.finditer(question)}
    # normalize "wind-2" -> "wind-02" to match generated ids
    normalized: set[str] = set()
    for site_id in found:
        kind, num = site_id.split("-")
        normalized.add(f"{kind}-{int(num):02d}")
    matched = [s for s in known_sites if s in normalized]
    return matched


def extract_intents(question: str, default: bool = True) -> list[str]:
    """Tools the question asks for. With default=False, a question that names no intent returns []."""
    q = question.lower()
    intents: list[str] = []
    if any(w in q for w in ("anomal", "wrong", "issue", "problem", "fault", "fail", "drop")):
        intents.append("check_anomaly")
    if any(w in q for w in ("forecast", "predict", "expect", "next hour", "going to be")):
        intents.append("forecast_power")
    if _HISTORY_RE.search(q):
        intents.append("search_maintenance_logs")
    if not intents and default:
        intents.append("check_anomaly")  # sensible default for an analytics question
    return intents


def _rule_based_answer(state: AppState, question: str) -> AgentAnswer:
    known_sites = [s["site_id"] for s in state.sites]
    with mlflow.start_span(name="route_question", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"question": question, "known_sites": known_sites})
        site_ids = extract_site_ids(question, known_sites)
        intents = extract_intents(question)
        span.set_outputs({"site_ids": site_ids, "intents": intents})

    tool_calls: list[dict[str, Any]] = []
    parts: list[str] = []

    if not site_ids:
        return {
            "answer": (
                "I couldn't tell which site you mean - try naming one explicitly, "
                f"e.g. {known_sites[0]} (known sites: {', '.join(known_sites)})."
            ),
            "tool_calls": [],
            "mode": "rule_based",
        }

    for site_id in site_ids:
        if "check_anomaly" in intents:
            result = dispatch(state, "check_anomaly", {"site_id": site_id})
            tool_calls.append({"tool": "check_anomaly", "input": {"site_id": site_id}, "output": result})
            if "error" in result:
                parts.append(f"{site_id}: {result['error']}")
            elif result["is_curtailed"]:
                parts.append(
                    f"{site_id} is producing {result['actual_power_kw']} kW against an expected "
                    f"{result['expected_power_kw']} kW, but it's under a known curtailment order, "
                    "so the deviation is intentional, not a fault."
                )
            elif result["is_anomaly"]:
                parts.append(
                    f"{site_id} looks anomalous (probability {result['anomaly_probability']}): "
                    f"{result['actual_power_kw']} kW actual vs {result['expected_power_kw']} kW expected "
                    f"({result['residual_pct']*100:.1f}% residual)."
                )
            else:
                parts.append(
                    f"{site_id} looks healthy: {result['actual_power_kw']} kW actual vs "
                    f"{result['expected_power_kw']} kW expected."
                )

        if "forecast_power" in intents:
            result = dispatch(state, "forecast_power", {"site_id": site_id})
            tool_calls.append({"tool": "forecast_power", "input": {"site_id": site_id}, "output": result})
            if "error" in result:
                parts.append(f"{site_id} forecast: {result['error']}")
            else:
                parts.append(
                    f"{site_id} is forecast at {result['forecast_power_kw']} kW in the next hour "
                    f"(currently {result['current_power_kw']} kW)."
                )

        if "search_maintenance_logs" in intents:
            # Without an LLM to turn the question into a symptom description, the most
            # useful fallback is the named site's own history, ranked by the question.
            tool_input: dict[str, Any] = {"query": question, "k": 2, "site_id": site_id}
            result = dispatch(state, "search_maintenance_logs", tool_input)
            tool_calls.append({"tool": "search_maintenance_logs", "input": tool_input, "output": result})
            hits = cast(list[dict[str, Any]], result.get("results", []))
            if hits:
                examples = " ".join(f"[{h['log_id']}] {h['text']}" for h in hits)
                parts.append(f"Past incidents at {site_id}: {examples}")
            else:
                parts.append(f"No maintenance history found for {site_id}.")

    return {"answer": " ".join(parts), "tool_calls": tool_calls, "mode": "rule_based"}


def _llm_answer(state: AppState, question: str, api_key: str) -> AgentAnswer:
    """Real agentic tool-use via Anthropic's Messages API. Requires network
    access to api.anthropic.com and a valid key - not exercised in this
    sandbox's test suite, but this is the path to describe as the design
    in an interview: the model, not keyword matching, decides what to call.
    """
    import anthropic
    from anthropic.types import (
        MessageParam,
        TextBlock,
        ToolResultBlockParam,
        ToolUseBlock,
    )

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[MessageParam] = [{"role": "user", "content": question}]
    tool_calls: list[dict[str, Any]] = []

    for _ in range(4):  # bounded agent loop
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        if not tool_use_blocks:
            final_text = "".join(b.text for b in response.content if isinstance(b, TextBlock))
            return {"answer": final_text, "tool_calls": tool_calls, "mode": "llm"}

        tool_results: list[ToolResultBlockParam] = []
        for block in tool_use_blocks:
            tool_input = cast(dict[str, Any], block.input)  # validated by the tool's input_schema
            result = dispatch(state, block.name, tool_input)
            tool_calls.append({"tool": block.name, "input": tool_input, "output": result})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": format_tool_result(block.name, result)}
            )
        messages.append({"role": "user", "content": tool_results})

    return {"answer": "(agent loop exceeded max turns without a final answer)", "tool_calls": tool_calls, "mode": "llm"}


def _answer(state: AppState, question: str) -> AgentAnswer:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import anthropic

        try:
            return _llm_answer(state, question, api_key)
        except anthropic.APIError as exc:  # network, auth, rate-limit or server errors
            fallback = _rule_based_answer(state, question)
            fallback["llm_error"] = str(exc)
            return fallback
    return _rule_based_answer(state, question)


def check_citations(answer: str, tool_calls: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Split the log ids cited in `answer` into (retrieved in this conversation, not retrieved)."""
    retrieved = {
        hit["log_id"]
        for call in tool_calls
        if call["tool"] == "search_maintenance_logs"
        for hit in cast(list[dict[str, Any]], call["output"].get("results", []))
    }
    cited = list(dict.fromkeys(_CITATION_RE.findall(answer)))  # unique, in order of appearance
    return [c for c in cited if c in retrieved], [c for c in cited if c not in retrieved]


def answer_question(state: AppState, question: str) -> AgentAnswer:
    ensure_tracing()
    with mlflow.start_span(name="agent_query", span_type=SpanType.AGENT) as span:
        span.set_inputs({"question": question})
        result = _answer(state, question)
        result["citations"], result["unsupported_citations"] = check_citations(result["answer"], result["tool_calls"])
        span.set_outputs({"answer": result["answer"], "mode": result["mode"], "citations": result["citations"]})
        if result["unsupported_citations"]:
            span.set_attribute("unsupported_citations", result["unsupported_citations"])
        span.set_attribute("tool_call_count", len(result["tool_calls"]))
        if "llm_error" in result:
            span.set_attribute("llm_error", result["llm_error"])
        trace_id = span.trace_id
        if tracing_enabled():
            # Searchable in the MLflow UI: which mode answered, with which model versions.
            mlflow.update_current_trace(
                tags={
                    "agent.mode": result["mode"],
                    **{f"model.{k}": v for k, v in state.registry.versions().items()},
                }
            )
    if tracing_enabled():
        result["trace_id"] = trace_id
    return result
