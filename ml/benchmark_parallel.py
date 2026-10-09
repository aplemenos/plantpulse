"""
Batch-scoring benchmark: score anomaly probability for every historical
telemetry row (a realistic nightly backfill / re-scoring workload) three ways:

  1. naive_serial  - per-row Python loop calling predict_proba on one row at a
                     time. What a first implementation often looks like.
  2. vectorized    - one predict_proba call on the whole feature matrix.
  3. parallel      - the matrix split into chunks scored across a process pool
                     (model loaded once per worker via an initializer, so the
                     large forest is never re-pickled per task).

The point isn't "parallel always wins". It's measuring which lever actually
matters: vectorizing is usually the first and biggest win; process-level
parallelism only pays off once a single vectorized core is the bottleneck,
and its gain is bounded by available cores and IPC overhead.

Usage:  python -m ml.benchmark_parallel
Writes: ml/benchmark_report.json
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.registry import ARTIFACTS_DIR, ROOT, load_current
from ml.features import (
    ANOMALY_FEATURES,
    EXPECTED_POWER_FEATURES,
    add_anomaly_features,
    add_expected_power_features,
)

DATA_PATH = ROOT / "data" / "telemetry.csv"
REPORT_PATH = ROOT / "ml" / "benchmark_report.json"
NAIVE_SAMPLE = 2000

_worker_model = None


def _init_worker(model_path: str):
    global _worker_model
    _worker_model = joblib.load(model_path)
    _worker_model.set_params(n_jobs=1)  # one core per worker; the pool provides the parallelism


def _score_chunk(chunk: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(chunk, columns=ANOMALY_FEATURES)
    return _worker_model.predict_proba(frame)[:, 1]


def naive_serial(feat: pd.DataFrame, model) -> np.ndarray:
    out = np.empty(len(feat))
    for i in range(len(feat)):
        out[i] = model.predict_proba(feat.iloc[[i]])[0, 1]
    return out


def vectorized(feat: pd.DataFrame, model) -> np.ndarray:
    return model.predict_proba(feat)[:, 1]


def parallel(feat: pd.DataFrame, model_path: str, n_workers: int) -> np.ndarray:
    chunks = np.array_split(feat.to_numpy(), n_workers * 4)
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(model_path,)
    ) as ex:
        return np.concatenate(list(ex.map(_score_chunk, chunks)))


def _timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = add_expected_power_features(raw)
    expected = load_current("expected_power").model.predict(df[EXPECTED_POWER_FEATURES])
    df = add_anomaly_features(df, expected)
    feat = df[ANOMALY_FEATURES].astype("float64")

    detector = load_current("anomaly_detector")
    pointer = json.loads((ARTIFACTS_DIR / "anomaly_detector" / "current.json").read_text())
    model_path = str(ROOT / pointer["path"] / "model.joblib")

    # Fix n_jobs=1 so "vectorized" is honestly single-core and the parallel
    # comparison measures process-level scaling, not the forest's own threading.
    detector.model.set_params(n_jobs=1)

    _, t_naive = _timed(naive_serial, feat.iloc[:NAIVE_SAMPLE], detector.model)
    vec_scores, t_vec = _timed(vectorized, feat, detector.model)

    cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, cores})
    parallel_times = {}
    for n in worker_counts:
        par_scores, t_par = _timed(parallel, feat, model_path, n)
        assert np.allclose(par_scores, vec_scores), "parallel result diverged from vectorized"
        parallel_times[n] = t_par

    rows = len(feat)
    naive_rps = NAIVE_SAMPLE / t_naive
    report = {
        "rows_scored": rows,
        "cpu_cores_available": cores,
        "naive_serial": {
            "rows_per_sec": round(naive_rps, 1),
            "projected_full_run_sec": round(rows / naive_rps, 2),
        },
        "vectorized_single_core": {
            "seconds": round(t_vec, 3),
            "rows_per_sec": round(rows / t_vec, 1),
            "speedup_vs_naive": round((rows / t_vec) / naive_rps, 1),
        },
        "process_pool": {
            str(n): {
                "seconds": round(t, 3),
                "rows_per_sec": round(rows / t, 1),
                "speedup_vs_vectorized": round(t_vec / t, 2),
            }
            for n, t in parallel_times.items()
        },
        "results_identical_across_methods": True,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
