import numpy as np
import pandas as pd

from ml.features import (
    ANOMALY_FEATURES,
    FORECAST_FEATURES,
    add_anomaly_features,
    add_expected_power_features,
    add_forecast_features,
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
