import numpy as np
import pandas as pd

from ml.data_generation import Event, build_maintenance_logs
from ml.features import (
    ANOMALY_FEATURES,
    FORECAST_FEATURES,
    add_anomaly_features,
    add_expected_power_features,
    add_forecast_features,
    physical_expected_power,
)


def _toy_frame(n: int = 200, site_id: str = "wind-01", site_type: str = "wind") -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "site_id": site_id,
            "site_type": site_type,
            "capacity_kw": 2000.0,
            "irradiance_wm2": np.nan,
            "wind_speed_ms": rng.uniform(4, 12, n),
            "ambient_temp_c": rng.normal(15, 2, n),
            "panel_temp_c": np.nan,
            "vibration_mm_s": rng.normal(2.2, 0.1, n),
            "power_kw": rng.uniform(200, 1500, n),
            "is_anomaly": 0,
            "is_curtailed": 0,
        }
    )


def test_expected_power_features_fill_type_specific_nulls() -> None:
    df = add_expected_power_features(_toy_frame())
    assert df["irradiance_wm2"].isna().sum() == 0
    assert df["panel_temp_c"].isna().sum() == 0
    assert (df["is_solar"] == 0).all()


def test_residual_is_actual_minus_expected() -> None:
    df = add_expected_power_features(_toy_frame())
    expected = np.full(len(df), 1000.0)
    out = add_anomaly_features(df, expected)
    np.testing.assert_allclose(out["residual_kw"], out["power_kw"] - 1000.0)
    np.testing.assert_allclose(out["residual_pct"], out["residual_kw"] / 2000.0)
    assert set(ANOMALY_FEATURES).issubset(out.columns)
    assert out[ANOMALY_FEATURES].isna().sum().sum() == 0


def test_feature_engineering_is_deterministic() -> None:
    """Same input -> same features: the guarantee that prevents train/serve skew."""
    df = add_expected_power_features(_toy_frame())
    a = add_anomaly_features(df, np.full(len(df), 900.0))
    b = add_anomaly_features(df, np.full(len(df), 900.0))
    pd.testing.assert_frame_equal(a, b)


def test_forecast_target_is_shifted_per_site_not_across_sites() -> None:
    df = pd.concat([_toy_frame(site_id="wind-01"), _toy_frame(site_id="wind-02")], ignore_index=True)
    out = add_forecast_features(df, horizon_steps=4)
    for _, site in out.groupby("site_id"):
        site = site.sort_values("timestamp")
        # target at t equals power at t+4 within the same site
        np.testing.assert_allclose(
            site["target_power_kw"].iloc[:-4].to_numpy(dtype=np.float64),
            site["power_kw"].iloc[4:].to_numpy(dtype=np.float64),
        )
        # and the last 4 rows of each site have no target (no leakage from the next site)
        assert site["target_power_kw"].iloc[-4:].isna().all()
    assert set(FORECAST_FEATURES).issubset(out.columns)


def test_expected_power_is_zero_for_solar_without_sun_and_within_capacity() -> None:
    df = pd.DataFrame({"is_solar": [1, 1, 0, 0], "irradiance_wm2": [0.0, 500.0, 0.0, 0.0],
                       "capacity_kw": [2500.0] * 4})
    out = physical_expected_power(df, np.array([13.5, 1200.0, -4.0, 9000.0]))
    # night solar -> 0; daytime solar untouched; wind never negative, never above 102% of capacity
    assert out.tolist() == [0.0, 1200.0, 0.0, 2550.0]


def test_maintenance_logs_describe_the_sites_own_equipment_and_event() -> None:
    t0 = pd.Timestamp("2026-02-01")
    events: list[Event] = [
        {"site_id": f"wind-0{i}", "site_type": "wind", "event_type": etype, "start": t0,
         "end": t0 + pd.Timedelta(hours=10), "sensor": sensor, "magnitude": 5.5}
        for i, (etype, sensor) in enumerate([("sensor_drift", "wind_speed_ms"), ("inverter_fault", ""),
                                             ("sensor_drift", "ambient_temp_c")] * 4)
    ]
    logs = build_maintenance_logs(events, np.random.default_rng(0))
    texts = [log["text"].lower() for log in logs]
    assert not any("irradiance" in t or "pyranometer" in t for t in texts)
    assert all("anemometer" in t or "wind speed" in t for t in texts[0::3])
    assert all("temperature" in t for t in texts[2::3])
