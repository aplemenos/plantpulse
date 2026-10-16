"""
Process-wide application state, built once at startup:
  - loaded models (via the registry)
  - the precomputed feature frame for every site (stand-in for what a
    streaming feature pipeline would maintain continuously in a real
    deployment - see README for the honest caveat)
  - the maintenance-log vector store
  - the prediction cache
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypedDict  # pydantic needs this on Python < 3.12

from app.cache import build_cache
from app.registry import ModelRegistry
from ml.features import (
    ANOMALY_FEATURES,
    EXPECTED_POWER_FEATURES,
    add_anomaly_features,
    add_expected_power_features,
    add_forecast_features,
    physical_expected_power,
)
from ml.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telemetry.csv"


class SiteInfo(TypedDict):
    site_id: str
    site_type: str
    capacity_kw: float


class FlaggedMoment(TypedDict):
    site_id: str
    timestamp: str
    anomaly_probability: float


class AppState:
    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.cache = build_cache()

        raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        df = add_expected_power_features(raw)
        raw_expected = self.registry.expected_power.model.predict(df[EXPECTED_POWER_FEATURES])
        expected = physical_expected_power(df, raw_expected)
        df = add_anomaly_features(df, expected)
        df = add_forecast_features(df)
        self.df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)

        self.vector_store = VectorStore.load()

        # One vectorized scoring pass at startup, used only for "where did the
        # model flag something" highlights in the UI - never for live answers.
        self.df["model_anomaly_proba"] = self.registry.anomaly_detector.model.predict_proba(
            self.df[ANOMALY_FEATURES]
        )[:, 1]

        self.sites = cast(
            list[SiteInfo],
            self.df[["site_id", "site_type", "capacity_kw"]]
            .drop_duplicates()
            .sort_values("site_id")
            .to_dict(orient="records"),
        )

        # Per-site index (sorted by time) so a lookup is a dict hit + binary
        # search instead of a boolean scan over every row in the fleet.
        self._site_frames: dict[str, pd.DataFrame] = {
            str(site_id): frame.reset_index(drop=True)
            for site_id, frame in self.df.groupby("site_id", sort=False)
        }
        self._site_ts: dict[str, NDArray[np.datetime64]] = {
            site_id: frame["timestamp"].to_numpy(dtype="datetime64[ns]")
            for site_id, frame in self._site_frames.items()
        }

    def site_frame(self, site_id: str, as_of: str | None = None) -> pd.DataFrame:
        frame = self._site_frames.get(site_id)
        if frame is None:
            return self.df.iloc[0:0]
        if as_of is None:
            return frame
        cutoff = int(
            np.searchsorted(self._site_ts[site_id], np.datetime64(pd.Timestamp(as_of)), side="right")
        )
        return frame.iloc[:cutoff]

    def top_flagged_moment(self, site_id: str) -> FlaggedMoment | None:
        frame = self._site_frames.get(site_id)
        if frame is None:
            return None
        # exclude known curtailment: the model already treats it as benign
        candidates = frame[frame["is_curtailed"] == 0]
        position = int(np.argmax(candidates["model_anomaly_proba"].to_numpy(dtype=np.float64)))
        row = candidates.iloc[position]
        return {
            "site_id": site_id,
            "timestamp": str(row["timestamp"]),
            "anomaly_probability": round(float(row["model_anomaly_proba"]), 4),
        }
