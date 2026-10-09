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

import pandas as pd

from app.cache import build_cache
from app.registry import ModelRegistry
from ml.features import (
    EXPECTED_POWER_FEATURES,
    add_anomaly_features,
    add_expected_power_features,
    add_forecast_features,
)
from ml.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telemetry.csv"


class AppState:
    def __init__(self):
        self.registry = ModelRegistry()
        self.cache = build_cache()

        raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        df = add_expected_power_features(raw)
        expected = self.registry.expected_power.model.predict(df[EXPECTED_POWER_FEATURES])
        df = add_anomaly_features(df, expected)
        df = add_forecast_features(df)
        self.df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)

        self.vector_store = VectorStore.load()

        self.sites = (
            self.df[["site_id", "site_type", "capacity_kw"]]
            .drop_duplicates()
            .sort_values("site_id")
            .to_dict(orient="records")
        )

    def site_frame(self, site_id: str, as_of: str | None = None) -> pd.DataFrame:
        site_df = self.df[self.df["site_id"] == site_id]
        if as_of is not None:
            site_df = site_df[site_df["timestamp"] <= pd.Timestamp(as_of)]
        return site_df
