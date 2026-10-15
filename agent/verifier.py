"""
Independent verification of an agent's draft answer against the tool outputs
it was given. The verifier never sees the drafter's reasoning, only the
evidence and the answer, and it does not trust the drafter's own judgement:
no self-verification.

Deterministic checks (always run, no LLM needed):
  numbers    every number in the answer appears in some tool output
             (allowing for rounding, and for fractions written as percentages)
  citations  every cited log id was actually retrieved
  status     what the answer says about a site's health matches the model:
             no "anomalous" for a healthy site, no "healthy" for a flagged one,
             and curtailment is mentioned when the site is curtailed. Quoted
             maintenance logs are ignored here: they describe the past.

An LLM judge (agent/investigate.py) can add a fourth, semantic check on top.
"""

from __future__ import annotations

import re
from typing import Any, cast

from typing_extensions import TypedDict

_IGNORED_SPANS = re.compile(
    r"\b(?:solar|wind)-\d+\b"            # site ids
    r"|\blog-\d+\b"                       # log ids
    r"|\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"  # dates and timestamps
    r"|\b\d{1,2}:\d{2}\b",                # clock times
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NEGATED_ANOMALY = re.compile(r"\b(?:no|not|isn't|is not|without|nothing)\b[^.]{0,30}anomal", re.IGNORECASE)
_CLAIMS_HEALTHY = re.compile(r"\bhealthy\b|\boperating normally\b|\bno (?:sign of )?(?:issue|problem|anomal)", re.IGNORECASE)

SMALL_COUNT = 10  # bare integers up to this are treated as counts ("2 incidents", "inverter 3")


class Issue(TypedDict):
    check: str
    detail: str


class Verification(TypedDict):
    passed: bool
    checks: dict[str, bool]
    issues: list[Issue]


Evidence = list[dict[str, Any]]  # executed steps: {"tool", "input", "output"}


def _numbers_in(text: str) -> list[tuple[float, int]]:
    """(value, decimals written) for each number in text, ignoring ids, dates and times."""
    cleaned = _IGNORED_SPANS.sub(" ", text)
    out: list[tuple[float, int]] = []
    for m in _NUMBER.finditer(cleaned):
        raw = m.group(0).replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((float(raw), decimals))
    return out


def _evidence_numbers(obj: Any) -> list[float]:
    """Every number in the tool outputs: numeric fields plus numbers inside text fields (e.g. log texts)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, str):
        return [v for v, _ in _numbers_in(obj)]
    if isinstance(obj, dict):
        return [n for v in cast(dict[str, Any], obj).values() for n in _evidence_numbers(v)]
    if isinstance(obj, list):
        return [n for v in cast(list[Any], obj) for n in _evidence_numbers(v)]
    return []


def _is_supported(value: float, decimals: int, pool: list[float]) -> bool:
    tolerance = 0.5 * 10 ** -decimals + 1e-9  # "450 kW" is supported by 450.42
    candidates = (b for x in pool for b in (x, abs(x), x * 100, abs(x) * 100))  # 0.215 may be written as 21.5%
    return any(abs(abs(value) - abs(c)) <= tolerance for c in candidates)


def check_numbers(answer: str, evidence: Evidence) -> list[Issue]:
    pool = _evidence_numbers([step["output"] for step in evidence])
    issues: list[Issue] = []
    for value, decimals in _numbers_in(answer):
        if decimals == 0 and abs(value) <= SMALL_COUNT:
            continue
        if not _is_supported(value, decimals, pool):
            issues.append({"check": "numbers", "detail": f"{value:g} does not appear in any tool output"})
    return issues


def check_citations(answer: str, evidence: Evidence) -> list[Issue]:
    retrieved = {
        hit["log_id"]
        for step in evidence
        if step["tool"] == "search_maintenance_logs"
        for hit in cast(list[dict[str, Any]], step["output"].get("results", []))
    }
    cited = dict.fromkeys(re.findall(r"\blog-\d+\b", answer))
    return [
        {"check": "citations", "detail": f"{log_id} is cited but was never retrieved"}
        for log_id in cited
        if log_id not in retrieved
    ]


def _without_quoted_logs(answer: str, evidence: Evidence) -> str:
    """Drop maintenance-log text from the answer: it describes past events, not the site's current state.

    A quoted log saying "equipment healthy" must not count as the answer calling the site healthy,
    and a quoted curtailment log must not count as the answer mentioning today's curtailment.
    """
    for step in evidence:
        if step["tool"] == "search_maintenance_logs":
            for hit in cast(list[dict[str, Any]], step["output"].get("results", [])):
                answer = answer.replace(hit["text"], " ")
    return answer


def check_status(answer: str, evidence: Evidence) -> list[Issue]:
    results = {
        step["input"]["site_id"]: step["output"]
        for step in evidence
        if step["tool"] == "check_anomaly" and "error" not in step["output"]
    }
    answer = _without_quoted_logs(answer, evidence)
    sentences = _SENTENCE_SPLIT.split(answer)
    issues: list[Issue] = []
    for site_id, result in results.items():
        about_site = [s for s in sentences if site_id in s.lower()]
        # With one site, every sentence is about it ("It's under a curtailment order").
        text = answer if len(results) == 1 else " ".join(about_site)
        if not text:
            continue
        if result["is_curtailed"]:
            if "curtail" not in text.lower():
                issues.append({"check": "status", "detail": f"{site_id} is curtailed, but the answer doesn't say so"})
        elif result["is_anomaly"]:
            if _CLAIMS_HEALTHY.search(text):
                issues.append({"check": "status", "detail": f"answer calls {site_id} healthy, but the model flagged it"})
        else:
            mentions = re.findall(r"anomal\w*", text, re.IGNORECASE)
            if mentions and not _NEGATED_ANOMALY.search(text):
                issues.append({"check": "status", "detail": f"answer calls {site_id} anomalous, but the model says healthy"})
    return issues


def verify(answer: str, evidence: Evidence) -> Verification:
    found = {
        "numbers": check_numbers(answer, evidence),
        "citations": check_citations(answer, evidence),
        "status": check_status(answer, evidence),
    }
    issues = [issue for group in found.values() for issue in group]
    return {"passed": not issues, "checks": {name: not group for name, group in found.items()}, "issues": issues}
