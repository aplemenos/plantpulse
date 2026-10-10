"""
Synthetic renewable-energy plant telemetry generator for PlantPulse.

Simulates 15-minute telemetry for a fleet of solar and wind sites, with
physically-motivated power curves, realistic noise, and injected
"events" of four kinds:

  - inverter_fault   : true anomaly. Power drops well below expected given
                       the weather inputs (irradiance / wind speed).
  - sensor_drift     : true anomaly. A sensor (temperature or irradiance)
                       drifts away from ground truth, making downstream
                       power look inconsistent with its own inputs.
  - mechanical_wear  : true anomaly (wind only). Vibration rises and
                       efficiency degrades gradually over days.
  - curtailment      : NOT a fault. Grid operator intentionally caps
                       output; inputs look normal, power is low on purpose.
                       Included so a good analytics agent has to tell real
                       faults apart from benign, intentional curtailment.

Also emits a small synthetic maintenance-log corpus, each entry tied to a
real anomaly event, for the vector-search ("find similar past incidents")
tool.

Usage:
    python ml/data_generation.py
writes:
    data/telemetry.csv
    data/anomaly_events.csv
    data/maintenance_logs.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypedDict  # pydantic needs this on Python < 3.12

RNG_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FREQ_MINUTES = 15
DAYS = 120
STEPS_PER_DAY = (24 * 60) // FREQ_MINUTES

FloatArray = NDArray[np.float64]


class Event(TypedDict):
    site_id: str
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp


class MaintenanceLog(TypedDict):
    log_id: str
    site_id: str
    event_type: str
    date: str
    text: str


@dataclass
class Site:
    site_id: str
    site_type: str  # "solar" | "wind"
    capacity_kw: float
    seasonal_phase_days: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(RNG_SEED), repr=False)


def _solar_irradiance(t_days: FloatArray, rng: np.random.Generator) -> FloatArray:
    """W/m^2 irradiance: diurnal sine clipped at 0, seasonal amplitude swing, cloud noise."""
    hour_of_day = (t_days % 1.0) * 24
    diurnal = np.clip(np.sin((hour_of_day - 6) / 12 * np.pi), 0, None)
    seasonal = 0.75 + 0.25 * np.sin(2 * np.pi * (t_days / 365.0))
    base = 950 * diurnal * seasonal
    cloud_noise = rng.normal(1.0, 0.08, size=t_days.shape)
    cloud_dropout = rng.random(size=t_days.shape) < 0.015  # passing cloud banks
    cloud_factor = np.where(cloud_dropout, rng.uniform(0.2, 0.6, size=t_days.shape), 1.0)
    irr = base * cloud_noise * cloud_factor
    return np.clip(irr, 0, None)


def _wind_speed(t_days: FloatArray, rng: np.random.Generator) -> FloatArray:
    """m/s wind speed via a mean-reverting (Ornstein-Uhlenbeck) process with diurnal lift."""
    n = len(t_days)
    speed = np.zeros(n)
    mean_speed = 8.0
    theta, sigma = 0.08, 0.6
    speed[0] = mean_speed
    for i in range(1, n):
        diurnal_lift = 0.6 * math.sin((float(t_days[i]) % 1.0) * 2 * math.pi)
        speed[i] = (
            speed[i - 1]
            + theta * (mean_speed + diurnal_lift - speed[i - 1])
            + sigma * rng.normal() * math.sqrt(1 / STEPS_PER_DAY)
        )
    return np.clip(speed, 0, None)


def _solar_power(irradiance: FloatArray, ambient_temp: FloatArray, capacity_kw: float) -> FloatArray:
    """Simple PV model: linear in irradiance, derated above 25C by ~0.4%/C."""
    derate = 1 - 0.004 * np.clip(ambient_temp - 25, 0, None)
    return capacity_kw * (irradiance / 1000.0) * derate


def _wind_power(wind_speed: FloatArray, capacity_kw: float) -> FloatArray:
    """Cubic power curve with cut-in (3 m/s), rated (12 m/s), cut-out (25 m/s)."""
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    power = np.zeros_like(wind_speed)
    ramp = (wind_speed >= cut_in) & (wind_speed < rated)
    full = (wind_speed >= rated) & (wind_speed < cut_out)
    power[ramp] = capacity_kw * ((wind_speed[ramp] - cut_in) / (rated - cut_in)) ** 3
    power[full] = capacity_kw
    return power


def _ambient_temp(t_days: FloatArray, rng: np.random.Generator) -> FloatArray:
    hour_of_day = (t_days % 1.0) * 24
    seasonal = 18 + 10 * np.sin(2 * np.pi * (t_days / 365.0 - 0.2))
    diurnal = 6 * np.sin((hour_of_day - 9) / 24 * 2 * np.pi)
    noise = rng.normal(0, 1.2, size=t_days.shape)
    return seasonal + diurnal + noise


def _inject_events(df: pd.DataFrame, site: Site, rng: np.random.Generator) -> list[Event]:
    """Mutate df in place with anomaly/curtailment windows; return event metadata."""
    n = len(df)
    events: list[Event] = []
    n_events = rng.integers(4, 8)
    event_types = (
        ["inverter_fault", "sensor_drift", "curtailment"]
        if site.site_type == "solar"
        else ["inverter_fault", "sensor_drift", "curtailment", "mechanical_wear"]
    )

    for _ in range(n_events):
        etype = str(rng.choice(event_types))
        start_idx = int(rng.integers(STEPS_PER_DAY * 2, n - STEPS_PER_DAY * 6))
        duration_steps = int(rng.integers(STEPS_PER_DAY // 2, STEPS_PER_DAY * 3))
        end_idx = min(start_idx + duration_steps, n - 1)
        sl = slice(start_idx, end_idx)

        if etype == "inverter_fault":
            severity = rng.uniform(0.35, 0.75)
            df.loc[df.index[sl], "power_kw"] *= 1 - severity
            df.loc[df.index[sl], "is_anomaly"] = 1
        elif etype == "sensor_drift":
            drift_col = "ambient_temp_c" if rng.random() < 0.5 else "irradiance_wm2" if site.site_type == "solar" else "wind_speed_ms"
            ramp = np.linspace(0, rng.uniform(4, 9), end_idx - start_idx)
            df.loc[df.index[sl], drift_col] = df.loc[df.index[sl], drift_col].to_numpy(dtype=np.float64) + ramp
            df.loc[df.index[sl], "is_anomaly"] = 1
        elif etype == "curtailment":
            cap_fraction = rng.uniform(0.3, 0.6)
            df.loc[df.index[sl], "power_kw"] = np.minimum(
                df.loc[df.index[sl], "power_kw"].to_numpy(dtype=np.float64), site.capacity_kw * cap_fraction
            )
            df.loc[df.index[sl], "is_curtailed"] = 1
            # curtailment is NOT labeled an anomaly - it's intentional
        elif etype == "mechanical_wear":
            ramp = np.linspace(0, rng.uniform(3, 6), end_idx - start_idx)
            df.loc[df.index[sl], "vibration_mm_s"] = df.loc[df.index[sl], "vibration_mm_s"].to_numpy(dtype=np.float64) + ramp
            df.loc[df.index[sl], "power_kw"] *= np.linspace(1.0, 0.85, end_idx - start_idx)
            df.loc[df.index[sl], "is_anomaly"] = 1

        events.append(
            {
                "site_id": site.site_id,
                "event_type": etype,
                "start": pd.Timestamp(df["timestamp"].iloc[start_idx]),
                "end": pd.Timestamp(df["timestamp"].iloc[end_idx]),
            }
        )
    return events


def generate_site_telemetry(site: Site) -> tuple[pd.DataFrame, list[Event]]:
    n = DAYS * STEPS_PER_DAY
    t_days = np.arange(n, dtype=np.float64) / STEPS_PER_DAY + site.seasonal_phase_days
    timestamps = pd.date_range("2026-01-01", periods=n, freq=f"{FREQ_MINUTES}min")

    ambient_temp = _ambient_temp(t_days, site.rng)

    if site.site_type == "solar":
        irradiance = _solar_irradiance(t_days, site.rng)
        power = _solar_power(irradiance, ambient_temp, site.capacity_kw)
        wind_speed = np.full(n, np.nan)
        vibration = np.full(n, np.nan)
        panel_temp = ambient_temp + 0.03 * irradiance
    else:
        wind_speed = _wind_speed(t_days, site.rng)
        power = _wind_power(wind_speed, site.capacity_kw)
        irradiance = np.full(n, np.nan)
        vibration = site.rng.normal(2.2, 0.15, size=n)
        panel_temp = np.full(n, np.nan)

    power = power + site.rng.normal(0, site.capacity_kw * 0.01, size=n)
    power = np.clip(power, 0, site.capacity_kw * 1.02)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": site.site_id,
            "site_type": site.site_type,
            "capacity_kw": site.capacity_kw,
            "irradiance_wm2": irradiance,
            "wind_speed_ms": wind_speed,
            "ambient_temp_c": ambient_temp,
            "panel_temp_c": panel_temp,
            "vibration_mm_s": vibration,
            "power_kw": power,
            "is_anomaly": 0,
            "is_curtailed": 0,
        }
    )

    events = _inject_events(df, site, site.rng)
    return df, events


MAINTENANCE_TEMPLATES = {
    "inverter_fault": [
        "Found loose DC connector on inverter {inv}; intermittent output drop resolved after reseating and torque check.",
        "Inverter {inv} tripped on internal over-temperature fault; cleaned heatsink and replaced cooling fan.",
        "Replaced failed IGBT module on inverter {inv} after repeated output derating under normal irradiance.",
    ],
    "sensor_drift": [
        "Recalibrated ambient temperature sensor at site; reading had drifted ~6C high over two weeks.",
        "Irradiance sensor showed drift vs. reference pyranometer; cleaned lens and recalibrated.",
        "Replaced corroded sensor cable causing gradual drift in reported measurement.",
    ],
    "mechanical_wear": [
        "Vibration trend on turbine {inv} gearbox exceeded threshold; borescope inspection found early bearing wear, scheduled replacement.",
        "Increasing nacelle vibration correlated with falling output; greased main bearing and tightened mounting bolts.",
    ],
    "curtailment": [
        "Grid operator issued curtailment order for {hours}h due to local grid congestion; no equipment fault.",
        "Voluntary output cap applied during low-price market window per dispatch instructions; equipment healthy.",
    ],
}


def build_maintenance_logs(all_events: list[Event], rng: np.random.Generator) -> list[MaintenanceLog]:
    logs: list[MaintenanceLog] = []
    for i, ev in enumerate(all_events):
        templates = MAINTENANCE_TEMPLATES[ev["event_type"]]
        template = templates[int(rng.integers(0, len(templates)))]
        text = template.format(inv=f"{rng.integers(1, 5)}", hours=int(rng.integers(2, 10)))
        logs.append(
            {
                "log_id": f"log-{i:04d}",
                "site_id": ev["site_id"],
                "event_type": ev["event_type"],
                "date": str(ev["end"].date()),
                "text": f"{ev['end'].date()} — Site {ev['site_id']}: {text}",
            }
        )
    return logs


def _child_rng(master: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(int(master.integers(1_000_000_000)))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(RNG_SEED)

    sites = [
        Site("solar-01", "solar", 2500, rng=_child_rng(master_rng)),
        Site("solar-02", "solar", 1800, seasonal_phase_days=40, rng=_child_rng(master_rng)),
        Site("solar-03", "solar", 3200, seasonal_phase_days=10, rng=_child_rng(master_rng)),
        Site("wind-01", "wind", 2000, rng=_child_rng(master_rng)),
        Site("wind-02", "wind", 2600, seasonal_phase_days=25, rng=_child_rng(master_rng)),
        Site("wind-03", "wind", 1500, seasonal_phase_days=60, rng=_child_rng(master_rng)),
    ]

    all_frames: list[pd.DataFrame] = []
    all_events: list[Event] = []
    for site in sites:
        df, events = generate_site_telemetry(site)
        all_frames.append(df)
        all_events.extend(events)

    telemetry = pd.concat(all_frames, ignore_index=True)
    telemetry.to_csv(DATA_DIR / "telemetry.csv", index=False)

    events_df = pd.DataFrame(all_events)
    events_df.to_csv(DATA_DIR / "anomaly_events.csv", index=False)

    logs = build_maintenance_logs(all_events, master_rng)
    with open(DATA_DIR / "maintenance_logs.json", "w") as f:
        json.dump(logs, f, indent=2, default=str)

    print(f"telemetry rows: {len(telemetry):,}")
    print(f"events: {len(events_df)}  (anomaly: {int((events_df.event_type != 'curtailment').sum())}, "
          f"curtailment: {int((events_df.event_type == 'curtailment').sum())})")
    print(f"maintenance logs: {len(logs)}")
    print(f"wrote to: {DATA_DIR}")


if __name__ == "__main__":
    main()
