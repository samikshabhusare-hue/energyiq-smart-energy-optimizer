"""Data preparation and explainable energy-consumption analytics for EnergyIQ.

Input schema: timestamp (parseable date/time), energy_kwh (numeric interval energy),
and optional zone (text). Each row is one measurement interval; the interval duration
is determined by the source data and is not inferred or converted by this module.

Public functions:
    load_and_validate_data(file) -> pandas.DataFrame
    calculate_metrics(df) -> dict
    detect_anomalies(df, window=6, min_history=2, threshold_kwh=2.0) -> DataFrame

Validation warnings are returned in ``df.attrs['data_quality_warnings']`` so that
the expected DataFrame interface stays easy to use in Streamlit and recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "energy_kwh")


def load_and_validate_data(file: Any) -> pd.DataFrame:
    """Read CSV data (or a DataFrame) and return cleaned, chronologically sorted rows.

    Invalid timestamps, non-numeric/missing energy values, and negative readings are
    excluded and counted in the warning list. Duplicate timestamps within the same
    zone retain the first row and are reported. If zone is absent, ``All`` is used.
    """
    if isinstance(file, pd.DataFrame):
        raw = file.copy()
    else:
        raw = pd.read_csv(file)

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_columns)
            + ". Required columns are timestamp and energy_kwh."
        )

    warnings: list[str] = []
    original_count = len(raw)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    bad_time = int(raw["timestamp"].isna().sum())
    if bad_time:
        warnings.append(f"{bad_time} row(s) had invalid or missing timestamps and were excluded.")
    raw = raw.loc[raw["timestamp"].notna()].copy()

    raw["energy_kwh"] = pd.to_numeric(raw["energy_kwh"], errors="coerce")
    raw.loc[np.isinf(raw["energy_kwh"]), "energy_kwh"] = np.nan
    bad_energy = int(raw["energy_kwh"].isna().sum())
    if bad_energy:
        warnings.append(f"{bad_energy} row(s) had missing or non-numeric energy_kwh and were excluded.")
    raw = raw.loc[raw["energy_kwh"].notna()].copy()

    negative = int((raw["energy_kwh"] < 0).sum())
    if negative:
        warnings.append(f"{negative} row(s) had negative energy readings and were excluded.")
        raw = raw.loc[raw["energy_kwh"] >= 0].copy()

    if "zone" not in raw.columns:
        raw["zone"] = "All"
    else:
        raw["zone"] = raw["zone"].fillna("").astype(str).str.strip()
        raw.loc[raw["zone"] == "", "zone"] = "Unknown"

    duplicates = int(raw.duplicated(subset=["timestamp", "zone"], keep="first").sum())
    if duplicates:
        warnings.append(
            f"{duplicates} duplicate timestamp/zone row(s) were found; the first reading was retained."
        )
        raw = raw.drop_duplicates(subset=["timestamp", "zone"], keep="first")

    raw = raw.sort_values(["timestamp", "zone"]).reset_index(drop=True)
    if original_count and raw.empty:
        warnings.append("No valid readings remain after data-quality checks.")
    raw.attrs["data_quality_warnings"] = warnings
    raw.attrs["interval_note"] = (
        "Each row is one source measurement interval. Energy remains in kWh per interval; "
        "the interval duration is not inferred or converted."
    )
    return raw


def calculate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Return overall metrics plus hourly, daily, and optional zone summaries.

    The returned mapping contains ``summary`` (plain values), ``hourly`` and ``daily``
    DataFrames, ``zone_comparison`` DataFrame, and ``data_quality_warnings`` list.
    """
    _require_clean_columns(df)
    has_zone = "zone" in df.columns
    valid = df.copy()
    valid["timestamp"] = pd.to_datetime(valid["timestamp"], errors="coerce")
    valid["energy_kwh"] = pd.to_numeric(valid["energy_kwh"], errors="coerce")
    valid.loc[np.isinf(valid["energy_kwh"]), "energy_kwh"] = np.nan
    valid = valid.loc[valid["timestamp"].notna() & valid["energy_kwh"].notna() & (valid["energy_kwh"] >= 0)].copy()

    if valid.empty:
        return {
            "summary": {"reading_count": 0, "total_energy_kwh": 0.0,
                        "average_interval_kwh": None, "maximum_interval_kwh": None,
                        "peak_timestamp": None, "peak_zone": None},
            "hourly": pd.DataFrame(columns=["hour", "average_energy_kwh", "total_energy_kwh"]),
            "daily": pd.DataFrame(columns=["date", "average_energy_kwh", "total_energy_kwh"]),
            "zone_comparison": pd.DataFrame(columns=["zone", "total_energy_kwh", "average_interval_kwh", "reading_count"]),
            "data_quality_warnings": list(df.attrs.get("data_quality_warnings", [])),
        }

    if has_zone:
        valid["zone"] = valid["zone"].fillna("Unknown").astype(str)

    peak_index = valid["energy_kwh"].idxmax()
    peak = valid.loc[peak_index]
    hourly = valid.assign(hour=valid["timestamp"].dt.hour).groupby("hour", as_index=False).agg(
        average_energy_kwh=("energy_kwh", "mean"), total_energy_kwh=("energy_kwh", "sum")
    )
    daily = valid.assign(date=valid["timestamp"].dt.date).groupby("date", as_index=False).agg(
        average_energy_kwh=("energy_kwh", "mean"), total_energy_kwh=("energy_kwh", "sum")
    )

    if has_zone:
        zone_comparison = valid.groupby("zone", as_index=False).agg(
            total_energy_kwh=("energy_kwh", "sum"),
            average_interval_kwh=("energy_kwh", "mean"),
            reading_count=("energy_kwh", "size"),
        )
        peak_zone = str(peak["zone"])
    else:
        zone_comparison = pd.DataFrame(columns=["zone", "total_energy_kwh", "average_interval_kwh", "reading_count"])
        peak_zone = None

    return {
        "summary": {
            "reading_count": int(len(valid)),
            "total_energy_kwh": float(valid["energy_kwh"].sum()),
            "average_interval_kwh": float(valid["energy_kwh"].mean()),
            "maximum_interval_kwh": float(peak["energy_kwh"]),
            "peak_timestamp": peak["timestamp"],
            "peak_zone": peak_zone,
        },
        "hourly": hourly,
        "daily": daily,
        "zone_comparison": zone_comparison,
        "data_quality_warnings": list(df.attrs.get("data_quality_warnings", [])),
    }


def detect_anomalies(
    df: pd.DataFrame,
    window: int = 6,
    min_history: int = 2,
    threshold_kwh: float = 2.0,
) -> pd.DataFrame:
    """Compare each reading with prior readings for the same time-of-day slot and zone.

    This controls for normal daily schedules. Baselines use only preceding days at
    the matching time of day (including minutes), never the current reading. A reading is
    flagged when its absolute deviation exceeds both the robust spread threshold
    (3 × 1.4826 × rolling MAD) and ``threshold_kwh``. Rows without enough history
    receive no baseline and are not flagged. A flag means unusual versus this local
    history, not proof of waste or equipment failure.
    """
    _require_clean_columns(df)
    if window < 1 or min_history < 1 or threshold_kwh < 0:
        raise ValueError("window and min_history must be positive; threshold_kwh cannot be negative.")
    work = df.copy()
    if "zone" not in work:
        work["zone"] = "All"
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["energy_kwh"] = pd.to_numeric(work["energy_kwh"], errors="coerce")
    work.loc[np.isinf(work["energy_kwh"]), "energy_kwh"] = np.nan
    work = work.loc[work["timestamp"].notna() & work["energy_kwh"].notna() & (work["energy_kwh"] >= 0)].copy()
    work["zone"] = work["zone"].fillna("Unknown").astype(str)
    work = work.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    work["expected_baseline_kwh"] = np.nan
    work["deviation_kwh"] = np.nan
    work["anomaly_flag"] = False

    work["_time"] = work["timestamp"].dt.time
    for _, indices in work.groupby(["zone", "_time"], sort=False).groups.items():
        values = work.loc[indices, "energy_kwh"].astype(float)
        prior = values.shift(1)
        rolling = prior.rolling(window=window, min_periods=min_history)
        baseline = rolling.median()
        mad = prior.rolling(window=window, min_periods=min_history).apply(
            lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
        )
        spread_limit = 3.0 * 1.4826 * mad
        deviation = values - baseline
        flags = baseline.notna() & (deviation.abs() > np.maximum(spread_limit.fillna(0), threshold_kwh))
        work.loc[indices, "expected_baseline_kwh"] = baseline.to_numpy()
        work.loc[indices, "deviation_kwh"] = deviation.to_numpy()
        work.loc[indices, "anomaly_flag"] = flags.to_numpy()

    return work[["timestamp", "zone", "energy_kwh", "expected_baseline_kwh", "deviation_kwh", "anomaly_flag"]].sort_values(["timestamp", "zone"]).reset_index(drop=True)


def _require_clean_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("DataFrame must contain: timestamp and energy_kwh.")

