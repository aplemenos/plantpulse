"""
Retrieval evaluation for the maintenance-log search (the R in the agent's RAG).

Golden queries are written the way an operator would describe a symptom,
deliberately *not* reusing the logs' wording, so they test meaning rather
than copy-paste. A hit is relevant when it has the expected event type (and
the requested site, when the query is site-filtered). Relevance by event
type rather than by log id keeps the set valid when the seeded corpus is
regenerated.

Metrics (k = 3, the agent's default):
  hit_at_3        share of queries with at least one relevant log in the top 3
  precision_at_3  share of the top 3 that is relevant
  mrr             mean reciprocal rank of the first relevant log

Usage:
    python -m ml.retrieval_eval     # compare vector, bm25 and hybrid on the current index
"""

from __future__ import annotations

import sys

from typing_extensions import TypedDict

from ml.vector_store import SearchMode, VectorStore

K = 3


class GoldenQuery(TypedDict):
    query: str
    event_type: str
    site_id: str | None


GOLDEN_RETRIEVAL: list[GoldenQuery] = [
    {"query": "inverter shut itself down because it was running too hot", "event_type": "inverter_fault",
     "site_id": None},
    {"query": "power keeps cutting in and out, maybe a bad cable connection", "event_type": "inverter_fault",
     "site_id": None},
    {"query": "output limited even though the sun is strong", "event_type": "inverter_fault",
     "site_id": None},
    {"query": "failed power electronics module in the inverter", "event_type": "inverter_fault",
     "site_id": None},
    {"query": "grid operator asked us to reduce output", "event_type": "curtailment",
     "site_id": None},
    {"query": "we capped production because electricity prices were low", "event_type": "curtailment",
     "site_id": None},
    {"query": "low output was intentional, not a fault", "event_type": "curtailment",
     "site_id": None},
    {"query": "temperature reading is a few degrees off", "event_type": "sensor_drift",
     "site_id": None},
    {"query": "readings slowly wandering away from the true value", "event_type": "sensor_drift",
     "site_id": None},
    {"query": "pyranometer disagrees with our irradiance sensor", "event_type": "sensor_drift",
     "site_id": None},
    {"query": "turbine shaking more than usual", "event_type": "mechanical_wear",
     "site_id": None},
    {"query": "gearbox bearing problems and rising vibration", "event_type": "mechanical_wear",
     "site_id": None},
    {"query": "output falling while the nacelle vibrates", "event_type": "mechanical_wear",
     "site_id": None},
    {"query": "has an inverter overheated here before?", "event_type": "inverter_fault",
     "site_id": "wind-03"},
    {"query": "sensor problems at this site", "event_type": "sensor_drift",
     "site_id": "solar-03"},
    {"query": "was there any grid curtailment?", "event_type": "curtailment",
     "site_id": "wind-03"},
]


def evaluate(store: VectorStore, mode: SearchMode = "hybrid") -> dict[str, float]:
    hits = precision = rr = 0.0
    for g in GOLDEN_RETRIEVAL:
        results = store.search(g["query"], k=K, site_id=g["site_id"], mode=mode)
        relevant = [
            r["event_type"] == g["event_type"] and (g["site_id"] is None or r["site_id"] == g["site_id"])
            for r in results
        ]
        hits += any(relevant)
        precision += sum(relevant) / K
        rr += next((1 / rank for rank, ok in enumerate(relevant, start=1) if ok), 0.0)
    n = len(GOLDEN_RETRIEVAL)
    return {
        "hit_at_3": round(hits / n, 4),
        "precision_at_3": round(precision / n, 4),
        "mrr": round(rr / n, 4),
    }


def failures(store: VectorStore, mode: SearchMode = "hybrid") -> list[str]:
    """Golden queries whose top result is not relevant - for debugging retrieval."""
    out: list[str] = []
    for g in GOLDEN_RETRIEVAL:
        top = store.search(g["query"], k=1, site_id=g["site_id"], mode=mode)
        if not top or top[0]["event_type"] != g["event_type"]:
            got = top[0]["event_type"] if top else "nothing"
            out.append(f"{g['query']!r}: wanted {g['event_type']}, got {got}")
    return out


def main() -> int:
    store = VectorStore.load()
    info = store.info
    print(f"embedder: {info['embedder']}  "
          f"({info['n_records']} logs, {len(GOLDEN_RETRIEVAL)} golden queries)\n")
    print(f"{'mode':<8}{'hit@3':>8}{'P@3':>8}{'MRR':>8}")
    for mode in ("vector", "bm25", "hybrid"):
        m = evaluate(store, mode)
        print(f"{mode:<8}{m['hit_at_3']:>8.3f}{m['precision_at_3']:>8.3f}{m['mrr']:>8.3f}")
    misses = failures(store, "hybrid")
    if misses:
        print("\nhybrid top-1 misses:")
        for line in misses:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
