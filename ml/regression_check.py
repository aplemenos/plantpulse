"""
Regression-testing facility for the analytics engine.

Re-evaluates the *currently registered* models on the frozen held-out
window and checks four kinds of behaviour against a committed baseline
(ml/regression_baseline.json):

  - model quality   : anomaly precision/recall/F1, forecaster + expected-power MAE
  - serving latency : p50/p95 of the real predict_anomaly() path, cache bypassed
  - agent routing   : a golden set of questions -> the tools the agent should call
  - data contract   : schema/row-count/null-rate checks on the input data

Every run is appended to ml/regression_history.jsonl so performance can be
tracked over time, not just pass/failed once. Exit code 1 on any regression,
so CI blocks a model or code change that quietly makes things worse.

Usage:
    python -m ml.regression_check                    # check vs baseline
    python -m ml.regression_check --update-baseline  # accept current numbers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score

from agent.orchestrator import extract_intents, extract_site_ids
from app.inference import predict_anomaly
from app.state import AppState
from ml.features import ANOMALY_FEATURES, EXPECTED_POWER_FEATURES, FORECAST_FEATURES
from ml.train import time_split

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "ml" / "regression_baseline.json"
HISTORY_PATH = ROOT / "ml" / "regression_history.jsonl"

# Allowed slack before a change counts as a regression.
TOLERANCES: dict[str, float] = {
    "anomaly_f1": -0.02,           # may drop at most 0.02 absolute
    "anomaly_recall": -0.03,
    "anomaly_precision": -0.03,
    "forecast_mae_kw": 0.05,        # may rise at most 5% relative
    "expected_power_mae_kw": 0.05,
    "latency_p95_ms": 0.50,         # may rise at most 50% relative (noisy on shared CI runners)
    "agent_routing_accuracy": 0.0,  # must not drop at all
}

GOLDEN_AGENT_QUERIES: list[tuple[str, list[str], list[str]]] = [
    ("is anything wrong with wind-02?", ["wind-02"], ["check_anomaly"]),
    ("what's the forecast for solar-01 next hour", ["solar-01"], ["forecast_power"]),
    ("have we seen this vibration before on wind-3", ["wind-03"], ["search_maintenance_logs"]),
    ("solar-02 output drop - any similar past incident?", ["solar-02"], ["check_anomaly", "search_maintenance_logs"]),
    ("predict solar-03 and check if wind-01 has a fault", ["solar-03", "wind-01"], ["check_anomaly", "forecast_power"]),
    ("how is solar-01 doing", ["solar-01"], ["check_anomaly"]),
]

EXPECTED_COLUMNS = {
    "timestamp", "site_id", "site_type", "capacity_kw", "irradiance_wm2", "wind_speed_ms",
    "ambient_temp_c", "panel_temp_c", "vibration_mm_s", "power_kw", "is_anomaly", "is_curtailed",
}


def check_data_contract(state: AppState) -> dict[str, Any]:
    raw_cols = set(pd.read_csv(ROOT / "data" / "telemetry.csv", nrows=1).columns)
    df = state.df
    return {
        "missing_columns": sorted(EXPECTED_COLUMNS - raw_cols),
        "row_count": len(df),
        "power_null_rate": float(df["power_kw"].isna().mean()),
        "negative_power_rows": int((df["power_kw"] < 0).sum()),
    }


def eval_quality(state: AppState) -> dict[str, float]:
    df = state.df
    _, test = time_split(df)

    pred = state.registry.anomaly_detector.model.predict(test[ANOMALY_FEATURES])
    y = test["is_anomaly"]

    healthy = test[(test.is_anomaly == 0) & (test.is_curtailed == 0)]
    ep_pred = state.registry.expected_power.model.predict(healthy[EXPECTED_POWER_FEATURES])

    fc = test.dropna(subset=FORECAST_FEATURES + ["target_power_kw"])
    fc_pred = state.registry.forecaster.model.predict(fc[FORECAST_FEATURES])

    return {
        "anomaly_precision": round(float(precision_score(y, pred)), 4),
        "anomaly_recall": round(float(recall_score(y, pred)), 4),
        "anomaly_f1": round(float(f1_score(y, pred)), 4),
        "expected_power_mae_kw": round(float(mean_absolute_error(healthy["power_kw"], ep_pred)), 3),
        "forecast_mae_kw": round(float(mean_absolute_error(fc["target_power_kw"], fc_pred)), 3),
    }


def eval_latency(state: AppState, n: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(0)
    sample = state.df.iloc[rng.choice(len(state.df), size=n, replace=False)]
    timings_ms = []
    for _, row in sample.iterrows():
        t0 = time.perf_counter()
        predict_anomaly(state, row["site_id"], str(row["timestamp"]))  # distinct keys -> no cache hits
        timings_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_p50_ms": round(float(np.percentile(timings_ms, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(timings_ms, 95)), 2),
    }


def eval_agent_routing(state: AppState) -> dict[str, Any]:
    known = [s["site_id"] for s in state.sites]
    correct, failures = 0, []
    for question, want_sites, want_tools in GOLDEN_AGENT_QUERIES:
        got_sites = extract_site_ids(question, known)
        got_tools = extract_intents(question)
        if sorted(got_sites) == sorted(want_sites) and sorted(got_tools) == sorted(want_tools):
            correct += 1
        else:
            failures.append({"question": question, "got_sites": got_sites, "got_tools": got_tools})
    return {
        "agent_routing_accuracy": round(correct / len(GOLDEN_AGENT_QUERIES), 4),
        "agent_routing_failures": failures,
    }


def compare(current: dict[str, Any], baseline: dict[str, float]) -> list[str]:
    problems = []
    for key, tol in TOLERANCES.items():
        if key not in baseline:
            continue
        cur, base = current[key], baseline[key]
        lower_is_better = key.endswith(("_kw", "_ms"))
        if lower_is_better:
            if cur > base * (1 + tol):
                problems.append(f"{key}: {cur} vs baseline {base} (allowed +{tol:.0%})")
        else:
            if cur < base + tol:
                problems.append(f"{key}: {cur} vs baseline {base} (allowed {tol:+})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--latency-tolerance", type=float, default=None,
        help="Override allowed relative p95 latency increase (e.g. 2.0 = 3x) for "
             "runners whose hardware differs from where the baseline was recorded. "
             "Quality and routing gates are never loosened.",
    )
    args = parser.parse_args(argv)
    if args.latency_tolerance is not None:
        TOLERANCES["latency_p95_ms"] = args.latency_tolerance

    state = AppState()
    contract = check_data_contract(state)
    current = {
        **eval_quality(state),
        **eval_latency(state),
        **eval_agent_routing(state),
    }

    record = {
        "ran_at": datetime.now(UTC).isoformat(),
        "model_versions": state.registry.versions(),
        "metrics": current,
        "data_contract": contract,
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(json.dumps(record, indent=2))

    if args.update_baseline:
        baseline = {k: v for k, v in current.items() if k in TOLERANCES}
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
        print(f"\nBaseline updated -> {BASELINE_PATH.relative_to(ROOT)}")
        return 0

    problems = []
    if contract["missing_columns"]:
        problems.append(f"data contract: missing columns {contract['missing_columns']}")
    if contract["power_null_rate"] > 0 or contract["negative_power_rows"] > 0:
        problems.append("data contract: null or negative power values present")

    if not BASELINE_PATH.exists():
        print("\nNo baseline yet - run with --update-baseline to create one.")
        return 1
    problems += compare(current, json.loads(BASELINE_PATH.read_text()))

    if problems:
        print("\nREGRESSION DETECTED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nNo regressions vs baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
