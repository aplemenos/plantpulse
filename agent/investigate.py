"""
Investigation pipeline: plan -> execute -> verify, with no self-verification.

    planner   decides which tools to run, and in what order, before anything
              runs. Steps can be conditional ("search this site's history
              only if it is anomalous").
    executor  runs the plan's tool calls, then drafts the answer from the
              results alone.
    verifier  checks the draft against the tool outputs (agent/verifier.py),
              plus an LLM judge in LLM mode. It is never the model that wrote
              the draft. If it finds problems, the executor gets one chance
              to revise; if the revision still fails, the answer is returned
              marked as unverified, with the issues, rather than hidden.

LLM mode (ANTHROPIC_API_KEY set) uses three model roles, each configurable:
    PLANTPULSE_PLANNER_MODEL   reasoning model that writes the plan
    PLANTPULSE_EXECUTOR_MODEL  fast model that drafts the answer
    PLANTPULSE_VERIFIER_MODEL  judge model that checks it
Without a key, the same pipeline runs with a rule-based planner and a
template drafter, so it is fully testable offline; the deterministic
verifier runs in both modes.

Every run is one MLflow trace: plan, each tool call, draft, and each
verification (EVALUATOR spans), so a wrong answer can be traced to the step
that caused it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

import mlflow
from mlflow.entities import SpanType
from typing_extensions import TypedDict

from agent.orchestrator import extract_intents, extract_site_ids
from agent.tools import TOOL_SPECS, dispatch, format_tool_result
from agent.tracing import ensure_tracing, tracing_enabled
from agent.verifier import Evidence, Issue, Verification, verify
from app.state import AppState

MAX_STEPS = 8
MAX_REVISIONS = 1
TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


def planner_model() -> str:
    return os.environ.get("PLANTPULSE_PLANNER_MODEL", "claude-sonnet-4-5")


def executor_model() -> str:
    return os.environ.get("PLANTPULSE_EXECUTOR_MODEL", "claude-haiku-4-5")


def verifier_model() -> str:
    return os.environ.get("PLANTPULSE_VERIFIER_MODEL", "claude-sonnet-4-5")


class PlanStep(TypedDict):
    tool: str
    input: dict[str, Any]
    reason: str
    condition: str  # "always" or "anomalous:<site_id>"


class StepResult(TypedDict):
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]
    skipped: bool
    note: str


Planner = Callable[[AppState, str], list[dict[str, Any]]]
Drafter = Callable[[str, list[StepResult], list[Issue]], str]
Judge = Callable[[str, list[StepResult], str], list[Issue]]


# --- planning -----------------------------------------------------------------


def rule_based_plan(state: AppState, question: str) -> list[dict[str, Any]]:
    """Check health first; forecast if asked; search history if asked, or if the site turns out anomalous."""
    sites = extract_site_ids(question, [s["site_id"] for s in state.sites])
    intents = extract_intents(question)
    plan: list[dict[str, Any]] = []
    for site_id in sites:
        plan.append({"tool": "check_anomaly", "input": {"site_id": site_id},
                     "reason": "establish current health first", "condition": "always"})
        if "forecast_power" in intents:
            plan.append({"tool": "forecast_power", "input": {"site_id": site_id},
                         "reason": "the question asks about the next hour", "condition": "always"})
        history: dict[str, Any] = {"query": question, "k": 2, "site_id": site_id}
        if "search_maintenance_logs" in intents:
            plan.append({"tool": "search_maintenance_logs", "input": history,
                         "reason": "the question asks about past incidents", "condition": "always"})
        else:
            plan.append({"tool": "search_maintenance_logs", "input": history,
                         "reason": "look for precedents if something is wrong", "condition": f"anomalous:{site_id}"})
    return plan


def validate_plan(state: AppState, raw: list[dict[str, Any]]) -> tuple[list[PlanStep], list[str]]:
    """Keep only executable steps; report why anything was dropped."""
    known_sites = {s["site_id"] for s in state.sites}
    steps: list[PlanStep] = []
    dropped: list[str] = []
    for item in raw[:MAX_STEPS]:
        tool = str(item.get("tool", ""))
        tool_input = cast(dict[str, Any], item.get("input") or {})
        condition = str(item.get("condition") or "always")
        site = tool_input.get("site_id")
        if tool not in TOOL_NAMES:
            dropped.append(f"unknown tool {tool!r}")
        elif site is not None and site not in known_sites:
            dropped.append(f"{tool}: unknown site {site!r}")
        elif tool == "search_maintenance_logs" and not tool_input.get("query"):
            dropped.append("search_maintenance_logs without a query")
        elif condition != "always" and not (
            condition.startswith("anomalous:") and condition.split(":", 1)[1] in known_sites
        ):
            dropped.append(f"{tool}: unsupported condition {condition!r}")
        else:
            steps.append({"tool": tool, "input": tool_input, "reason": str(item.get("reason", "")),
                          "condition": condition})
    if len(raw) > MAX_STEPS:
        dropped.append(f"plan truncated to {MAX_STEPS} steps")
    return steps, dropped


# --- execution ----------------------------------------------------------------


def execute(state: AppState, plan: list[PlanStep]) -> list[StepResult]:
    anomalous: dict[str, bool] = {}
    results: list[StepResult] = []
    for step in plan:
        if step["condition"].startswith("anomalous:"):
            site = step["condition"].split(":", 1)[1]
            if not anomalous.get(site, False):
                results.append({"tool": step["tool"], "input": step["input"], "output": {},
                                "skipped": True, "note": f"skipped: {site} is not anomalous"})
                continue
        output = dispatch(state, step["tool"], step["input"])
        if step["tool"] == "check_anomaly" and "error" not in output:
            anomalous[step["input"]["site_id"]] = bool(output["is_anomaly"]) and not output["is_curtailed"]
        results.append({"tool": step["tool"], "input": step["input"], "output": output, "skipped": False, "note": ""})
    return results


def evidence_of(results: list[StepResult]) -> Evidence:
    return [{"tool": r["tool"], "input": r["input"], "output": r["output"]} for r in results if not r["skipped"]]


def format_results(results: list[StepResult]) -> str:
    blocks: list[str] = []
    for n, r in enumerate(results, start=1):
        header = f"Step {n}: {r['tool']}({', '.join(f'{k}={v!r}' for k, v in r['input'].items())})"
        body = r["note"] if r["skipped"] else format_tool_result(r["tool"], r["output"])
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def template_draft(question: str, results: list[StepResult], issues: list[Issue]) -> str:
    """Offline drafter: states only what the tools returned."""
    parts: list[str] = []
    for r in results:
        out, site = r["output"], r["input"].get("site_id")
        if r["skipped"]:
            if r["tool"] == "search_maintenance_logs":
                parts.append(f"{site} isn't anomalous, so I didn't search its maintenance history.")
            continue
        if "error" in out:
            parts.append(f"{r['tool']} for {site}: {out['error']}")
        elif r["tool"] == "check_anomaly":
            if out["is_curtailed"]:
                parts.append(f"{site} is producing {out['actual_power_kw']} kW against an expected "
                             f"{out['expected_power_kw']} kW, but it is under a curtailment order, "
                             "so the shortfall is intentional, not a fault.")
            elif out["is_anomaly"]:
                parts.append(f"{site} looks anomalous (probability {out['anomaly_probability']}): "
                             f"{out['actual_power_kw']} kW actual vs {out['expected_power_kw']} kW expected.")
            else:
                parts.append(f"{site} looks healthy: {out['actual_power_kw']} kW actual vs "
                             f"{out['expected_power_kw']} kW expected.")
        elif r["tool"] == "forecast_power":
            parts.append(f"{site} is forecast at {out['forecast_power_kw']} kW in the next hour "
                         f"(currently {out['current_power_kw']} kW).")
        elif r["tool"] == "search_maintenance_logs":
            hits = cast(list[dict[str, Any]], out.get("results", []))
            if hits:
                parts.append(f"Past incidents at {site}: " + " ".join(f"[{h['log_id']}] {h['text']}" for h in hits))
            else:
                parts.append(f"No maintenance history found for {site}.")
    return " ".join(parts) or "I couldn't tell which site you mean - try naming one, e.g. wind-02."


# --- LLM roles ----------------------------------------------------------------

PLANNER_PROMPT = """You plan investigations for PlantPulse, an analytics service for solar and wind sites.
Decide which tool calls answer the question, in order, before anything runs. Rules:
- Check a site's health (check_anomaly) before drawing conclusions about it.
- Use condition "anomalous:<site_id>" for steps that only make sense if that site is anomalous,
  e.g. searching its maintenance history; otherwise use "always".
- Use only the sites listed. At most 8 steps. Do not answer the question yourself."""

EXECUTOR_PROMPT = """You write the answer to an operator's question from tool results only.
- Use only facts and numbers that appear in the results; do not compute new numbers.
- Low output under a curtailment order is intentional, not a fault - say so when it applies.
- Cite maintenance logs inline by id, e.g. [log-0009], and only logs that appear in the results.
- If a step was skipped, you may say why. Keep it short and concrete."""

JUDGE_PROMPT = """You verify an answer written by another model. You get the question, the tool results
(the only evidence), and the answer. Report every claim in the answer that the evidence does not
support, or that contradicts it. Do not judge style. If everything is supported, report no issues."""

SUBMIT_PLAN: dict[str, Any] = {
    "name": "submit_plan",
    "description": "Submit the investigation plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": sorted(TOOL_NAMES)},
                        "input": {"type": "object"},
                        "reason": {"type": "string"},
                        "condition": {"type": "string", "description": "'always' or 'anomalous:<site_id>'"},
                    },
                    "required": ["tool", "input", "reason", "condition"],
                },
            }
        },
        "required": ["steps"],
    },
}

SUBMIT_VERDICT: dict[str, Any] = {
    "name": "submit_verdict",
    "description": "Submit the verification result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "issues": {"type": "array", "items": {"type": "string"},
                       "description": "Unsupported or contradicted claims; empty if none"},
        },
        "required": ["issues"],
    },
}


def _forced_tool_input(client: Any, model: str, system: str, tool: dict[str, Any], content: str) -> dict[str, Any]:
    from anthropic.types import ToolUseBlock

    response = client.messages.create(
        model=model, max_tokens=1024, system=system, tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            return cast(dict[str, Any], block.input)
    return {}


def llm_roles(api_key: str) -> tuple[Planner, Drafter, Judge]:
    import anthropic
    from anthropic.types import TextBlock

    client = anthropic.Anthropic(api_key=api_key)
    tool_list = "\n".join(f"- {s['name']}: {s.get('description', '')}" for s in TOOL_SPECS)

    def plan(state: AppState, question: str) -> list[dict[str, Any]]:
        sites = ", ".join(s["site_id"] for s in state.sites)
        content = f"Question: {question}\n\nKnown sites: {sites}\n\nTools:\n{tool_list}"
        return cast(list[dict[str, Any]],
                    _forced_tool_input(client, planner_model(), PLANNER_PROMPT, SUBMIT_PLAN, content).get("steps", []))

    def draft(question: str, results: list[StepResult], issues: list[Issue]) -> str:
        content = f"Question: {question}\n\nTool results:\n{format_results(results)}"
        if issues:
            content += "\n\nA reviewer rejected your previous answer. Fix these problems:\n" + "\n".join(
                f"- {i['detail']}" for i in issues)
        response = client.messages.create(
            model=executor_model(), max_tokens=1024, system=EXECUTOR_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in response.content if isinstance(b, TextBlock))

    def judge(question: str, results: list[StepResult], answer: str) -> list[Issue]:
        content = f"Question: {question}\n\nTool results:\n{format_results(results)}\n\nAnswer to verify:\n{answer}"
        verdict = _forced_tool_input(client, verifier_model(), JUDGE_PROMPT, SUBMIT_VERDICT, content)
        return [{"check": "judge", "detail": str(d)} for d in cast(list[Any], verdict.get("issues", []))]

    return plan, draft, judge


# --- pipeline -----------------------------------------------------------------


def _verify(question: str, results: list[StepResult], answer: str, judge: Judge | None) -> Verification:
    with mlflow.start_span(name="verify", span_type=SpanType.EVALUATOR) as span:
        span.set_inputs({"answer": answer})
        verification = verify(answer, evidence_of(results))
        if judge is not None:
            judge_issues = judge(question, results, answer)
            verification["checks"]["judge"] = not judge_issues
            verification["issues"] += judge_issues
            verification["passed"] = verification["passed"] and not judge_issues
        span.set_outputs(verification)
    return verification


def run_pipeline(
    state: AppState, question: str, planner: Planner, drafter: Drafter, judge: Judge | None,
    as_of: str | None = None,
) -> dict[str, Any]:
    with mlflow.start_span(name="plan", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"question": question, "as_of": as_of})
        plan, dropped = validate_plan(state, planner(state, question))
        if as_of:  # investigate a past moment: pin every point-in-time tool to it
            for step in plan:
                if step["tool"] in ("check_anomaly", "forecast_power"):
                    step["input"] = {**step["input"], "as_of": as_of}
        span.set_outputs({"plan": plan, "dropped": dropped})

    with mlflow.start_span(name="execute", span_type=SpanType.CHAIN) as span:
        results = execute(state, plan)
        span.set_outputs({"executed": sum(not r["skipped"] for r in results),
                          "skipped": sum(r["skipped"] for r in results)})

    answer = drafter(question, results, [])
    history: list[Verification] = []
    for attempt in range(MAX_REVISIONS + 1):
        verification = _verify(question, results, answer, judge)
        history.append(verification)
        if verification["passed"] or attempt == MAX_REVISIONS:
            break
        with mlflow.start_span(name="revise", span_type=SpanType.CHAIN) as span:
            span.set_inputs({"issues": verification["issues"]})
            answer = drafter(question, results, verification["issues"])
            span.set_outputs({"answer": answer})

    final = history[-1]
    return {
        "answer": answer,
        "plan": plan,
        "plan_dropped": dropped,
        "steps": results,
        "verification": {**final, "revisions": len(history) - 1,
                         "rejected_drafts": [h["issues"] for h in history[:-1]]},
    }


def investigate(state: AppState, question: str, as_of: str | None = None, allow_llm: bool = True) -> dict[str, Any]:
    """Run the pipeline. allow_llm=False forces the offline roles (used by the regression gate)."""
    ensure_tracing()
    with mlflow.start_span(name="agent_investigate", span_type=SpanType.AGENT) as span:
        span.set_inputs({"question": question, "as_of": as_of})
        api_key = os.environ.get("ANTHROPIC_API_KEY") if allow_llm else None
        result: dict[str, Any]
        if api_key:
            import anthropic

            try:
                plan, draft, judge = llm_roles(api_key)
                result = run_pipeline(state, question, plan, draft, judge, as_of)
                result["mode"] = "llm"
                result["models"] = {"planner": planner_model(), "executor": executor_model(),
                                    "verifier": verifier_model()}
            except anthropic.APIError as exc:
                result = run_pipeline(state, question, rule_based_plan, template_draft, None, as_of)
                result["mode"] = "rule_based"
                result["llm_error"] = str(exc)
        else:
            result = run_pipeline(state, question, rule_based_plan, template_draft, None, as_of)
            result["mode"] = "rule_based"

        verification = cast(dict[str, Any], result["verification"])
        span.set_outputs({"answer": result["answer"], "verified": verification["passed"]})
        span.set_attributes({"mode": result["mode"], "verified": verification["passed"],
                             "revisions": verification["revisions"]})
        trace_id = span.trace_id
        if tracing_enabled():
            mlflow.update_current_trace(tags={"agent.mode": result["mode"], "agent.pipeline": "investigate",
                                              "verified": str(verification["passed"]).lower()})
    if tracing_enabled():
        result["trace_id"] = trace_id
    return result
