"""
Shared feature engineering for PlantPulse, used identically by training
and serving so there's no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ROLL_WINDOW = 8  # 8 * 15min = 2h rolling window for drift/residual smoothing


def add_expected_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-site-type input columns the 'expected power' regressor consumes."""
    df = df.copy()
    df["irradiance_wm2"] = df["irradiance_wm2"].fillna(0.0)
    df["wind_speed_ms"] = df["wind_speed_ms"].fillna(0.0)
    df["panel_temp_c"] = df["panel_temp_c"].fillna(df["ambient_temp_c"])
    df["vibration_mm_s"] = df["vibration_mm_s"].fillna(0.0)
    df["is_solar"] = (df["site_type"] == "solar").astype(int)
    return df


EXPECTED_POWER_FEATURES = [
    "irradiance_wm2",
    "wind_speed_ms",
    "ambient_temp_c",
    "panel_temp_c",
    "capacity_kw",
    "is_solar",
]


def add_anomaly_features(df: pd.DataFrame, expected_power: NDArray[np.float64]) -> pd.DataFrame:
    """Residual + drift/vibration features used by the anomaly classifier.

    `expected_power` must already be computed (model output or ground truth)
    and aligned to df's row order.
    """
    df = df.copy()
    df["expected_power_kw"] = expected_power
    df["residual_kw"] = df["power_kw"] - df["expected_power_kw"]
    df["residual_pct"] = df["residual_kw"] / df["capacity_kw"].clip(lower=1.0)

    df = df.sort_values(["site_id", "timestamp"])
    grp = df.groupby("site_id", group_keys=False)

    df["vibration_roll_med"] = grp["vibration_mm_s"].transform(
        lambda s: s.rolling(ROLL_WINDOW, min_periods=1).median()
    )
    df["vibration_dev"] = df["vibration_mm_s"] - df["vibration_roll_med"]

    df["ambient_roll_med"] = grp["ambient_temp_c"].transform(
        lambda s: s.rolling(ROLL_WINDOW * 4, min_periods=1).median()
    )
    df["ambient_dev"] = df["ambient_temp_c"] - df["ambient_roll_med"]

    df["irr_roll_med"] = grp["irradiance_wm2"].transform(
        lambda s: s.rolling(ROLL_WINDOW, min_periods=1).median()
    )
    df["irr_dev"] = df["irradiance_wm2"] - df["irr_roll_med"]

    return df.sort_index()


ANOMALY_FEATURES = [
    "residual_pct",
    "vibration_dev",
    "ambient_dev",
    "irr_dev",
    "is_curtailed",  # external dispatch/curtailment flag - known-good explanation, not a fault
]


def add_forecast_features(df: pd.DataFrame, horizon_steps: int = 4) -> pd.DataFrame:
    """Lag/rolling features for the N-step-ahead power forecaster.

    `horizon_steps=4` at 15-min resolution = 1 hour ahead.
    Adds a `target_power_kw` column shifted -horizon_steps per site.
    """
    df = df.sort_values(["site_id", "timestamp"]).copy()
    grp = df.groupby("site_id", group_keys=False)

    for lag in (1, 2, 4, 8, 96):  # 15m,30m,1h,2h,1 day ago
        df[f"power_lag_{lag}"] = grp["power_kw"].shift(lag)

    df["power_roll_mean_1h"] = grp["power_kw"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["is_solar"] = (df["site_type"] == "solar").astype(int)

    df["target_power_kw"] = grp["power_kw"].shift(-horizon_steps)
    return df.sort_index()


FORECAST_FEATURES = [
    "power_lag_1",
    "power_lag_2",
    "power_lag_4",
    "power_lag_8",
    "power_lag_96",
    "power_roll_mean_1h",
    "hour_of_day",
    "is_solar",
    "capacity_kw",
]
