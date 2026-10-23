"""Evidence-based recommendation engine for EnergyIQ.

Consumes the outputs of analytics.calculate_metrics() and analytics.detect_anomalies()
and returns a prioritised list of human-readable, data-backed recommendations.

Public API
----------
generate_recommendations(metrics, anomalies) -> list[dict]

Each recommendation dict contains:
    category    : str  – e.g. "Peak Demand", "Anomaly", "Off-Peak Baseline",
                          "Zone Imbalance", "High Overall Consumption"
    priority    : str  – one of "High", "Medium", "Low"
    title       : str  – short headline
    description : str  – human-readable explanation using actual numbers
    evidence    : dict – the key metrics that justify this recommendation

Design notes
------------
* All thresholds are explicit constants documented below.  They are engineering
  judgement values, not values learned from the data.
* The function never invents values, causes, timestamps or zone names.
* It returns [] rather than raising when inputs are empty or unusable.
* It is a pure function: no I/O, no side effects, no external APIs.
"""

from __future__ import annotations

from typing import Any

import math

import pandas as pd


# ---------------------------------------------------------------------------
# Explicit, documented thresholds
# ---------------------------------------------------------------------------

# Rule 1 – Peak Demand
# How many of the top hours to inspect.
PEAK_TOP_N: int = 3
# A peak hour must exceed the overall average by strictly more than 20%.
PEAK_EXCESS_RATIO: float = 0.20

# Rule 2 – Anomaly
# Positive deviation above which priority escalates to "High".
ANOMALY_HIGH_DEVIATION_KWH: float = 5.0

# Rule 3 – Off-Peak Baseline
# Off-peak window (inclusive) in hours-of-day.
OFF_PEAK_HOURS: set[int] = {0, 1, 2, 3, 4, 5}
# Daytime window (inclusive) in hours-of-day.
DAYTIME_HOURS: set[int] = {8, 9, 10, 11, 12, 13, 14, 15, 16}
# Off-peak average must exceed this fraction of the daytime average (20 %).
OFF_PEAK_DAYTIME_RATIO: float = 0.20

# Rule 4 – Zone Imbalance
# Highest/lowest zone average must be strictly greater than this ratio.
ZONE_IMBALANCE_RATIO: float = 1.5

# Rule 5 – High Overall Consumption
# Peak reading must exceed this multiple of the overall average.
HIGH_CONSUMPTION_MULTIPLE: float = 2.0

# Priority sort order
_PRIORITY_ORDER: dict[str, int] = {"High": 0, "Medium": 1, "Low": 2}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recommendations(
    metrics: dict[str, Any],
    anomalies: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return a prioritised list of evidence-based energy recommendations.

    Parameters
    ----------
    metrics:
        Dict returned by ``analytics.calculate_metrics(df)``.  Expected keys:
        ``summary``, ``hourly``, ``daily``, ``zone_comparison``,
        ``data_quality_warnings``.
    anomalies:
        DataFrame returned by ``analytics.detect_anomalies(df)``.  Expected
        columns: ``timestamp``, ``zone``, ``energy_kwh``,
        ``expected_baseline_kwh``, ``deviation_kwh``, ``anomaly_flag``.

    Returns
    -------
    list[dict]
        Recommendations sorted High → Medium → Low.  Empty list when there is
        insufficient usable evidence.
    """
    if not isinstance(metrics, dict):
        return []

    results: list[dict[str, Any]] = []

    # Run each rule; each appends 0-or-more dicts to *results*.
    results.extend(_rule_peak_demand(metrics))
    results.extend(_rule_anomaly(anomalies))
    results.extend(_rule_off_peak_baseline(metrics))
    results.extend(_rule_zone_imbalance(metrics))
    results.extend(_rule_high_overall_consumption(metrics))

    # Sort High → Medium → Low, preserving insertion order within each band.
    results.sort(key=lambda r: _PRIORITY_ORDER.get(r.get("priority", "Low"), 2))
    return results


# ---------------------------------------------------------------------------
# Private rule implementations
# ---------------------------------------------------------------------------

def _rule_peak_demand(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule 1 – Peak Demand.

    Identify the top-N hours by average consumption.  For each that exceeds
    the overall average by more than PEAK_EXCESS_RATIO, emit a recommendation.
    Only one recommendation is returned (covering all qualifying peak hours)
    to avoid duplication.
    """
    hourly = _safe_dataframe(metrics, "hourly")
    overall_avg = _safe_float(metrics.get("summary", {}), "average_interval_kwh")

    if hourly.empty or overall_avg is None or overall_avg <= 0:
        return []

    required = {"hour", "average_energy_kwh"}
    if not required.issubset(hourly.columns):
        return []

    valid_hourly = _valid_hourly(hourly)
    if valid_hourly.empty:
        return []

    top = (
        valid_hourly
        .nlargest(PEAK_TOP_N, "average_energy_kwh")
        .reset_index(drop=True)
    )

    # Keep only rows whose average exceeds the threshold.
    threshold = overall_avg * (1 + PEAK_EXCESS_RATIO)
    qualifying = top[top["average_energy_kwh"] > threshold]

    if qualifying.empty:
        return []

    # Build evidence: list of (hour, avg_kwh) for qualifying hours.
    peak_hours_evidence = [
        {
            "hour": int(row["hour"]),
            "average_energy_kwh": round(float(row["average_energy_kwh"]), 3),
        }
        for _, row in qualifying.iterrows()
    ]
    highest_avg = float(qualifying["average_energy_kwh"].max())
    excess_pct = round((highest_avg / overall_avg - 1) * 100, 1)

    hour_labels = ", ".join(
        f"{int(r['hour']):02d}:00" for r in peak_hours_evidence
    )

    description = (
        f"The hour(s) {hour_labels} have elevated interval consumption, reaching "
        f"{highest_avg:.2f} kWh per interval — "
        f"{excess_pct}% above the dataset-wide average of "
        f"{overall_avg:.2f} kWh. "
        "Consider reviewing scheduled loads, HVAC set-points, or operational "
        "processes during these windows to understand what drives the elevated "
        "demand. Shifting flexible loads away from peak hours may reduce "
        "demand charges where applicable."
    )

    return [
        {
            "category": "Peak Demand",
            "priority": "High",
            "title": f"Elevated consumption detected during {len(peak_hours_evidence)} peak hour(s)",
            "description": description,
            "evidence": {
                "overall_average_interval_kwh": round(overall_avg, 3),
                "threshold_kwh": round(threshold, 3),
                "qualifying_peak_hours": peak_hours_evidence,
                "threshold_note": (
                    f"Flagged when peak average exceeds overall average "
                    f"by >{PEAK_EXCESS_RATIO * 100:.0f}%."
                ),
            },
        }
    ]


def _rule_anomaly(anomalies: pd.DataFrame) -> list[dict[str, Any]]:
    """Rule 2 – Anomaly.

    Report flagged readings grouped by zone.  Priority is "High" when the
    largest positive deviation exceeds ANOMALY_HIGH_DEVIATION_KWH, else
    "Medium".  One recommendation per zone that has flagged rows.
    """
    if not isinstance(anomalies, pd.DataFrame) or anomalies.empty:
        return []

    required = {"anomaly_flag", "zone", "energy_kwh", "deviation_kwh",
                "timestamp", "expected_baseline_kwh"}
    if not required.issubset(anomalies.columns):
        return []

    flagged = anomalies[anomalies["anomaly_flag"] == True].copy().reset_index(drop=True)  # noqa: E712
    for column in ("energy_kwh", "deviation_kwh", "expected_baseline_kwh"):
        flagged[column] = flagged[column].map(_safe_numeric)
    flagged = flagged.dropna(subset=["energy_kwh", "deviation_kwh", "zone"])
    flagged = flagged[flagged["energy_kwh"] >= 0]
    if flagged.empty:
        return []

    results: list[dict[str, Any]] = []

    for zone, grp in flagged.groupby("zone", sort=True):
        count = len(grp)

        # Largest *positive* deviation in this zone.
        positive_devs = grp["deviation_kwh"].dropna()
        positive_devs = positive_devs[positive_devs > 0]

        if positive_devs.empty:
            max_pos_dev = None
            worst_row = grp.loc[grp["deviation_kwh"].abs().idxmax()]
        else:
            max_pos_dev = float(positive_devs.max())
            worst_idx = positive_devs.idxmax()
            worst_row = grp.loc[worst_idx]

        priority = (
            "High"
            if (max_pos_dev is not None and max_pos_dev > ANOMALY_HIGH_DEVIATION_KWH)
            else "Medium"
        )

        # Safely extract worst-row values.
        worst_ts = _format_timestamp(worst_row.get("timestamp"))
        worst_actual = _safe_numeric(worst_row.get("energy_kwh"))
        worst_baseline = _safe_numeric(worst_row.get("expected_baseline_kwh"))
        worst_dev = _safe_numeric(worst_row.get("deviation_kwh"))

        # Build description without claiming a proven cause.
        desc_parts = [
            f"Zone '{zone}' has {count} reading(s) flagged as statistically "
            "unusual relative to prior readings at the same time of day for that zone."
        ]
        if worst_ts:
            desc_parts.append(
                f"The {'largest positive' if max_pos_dev is not None else 'largest absolute'} deviation occurred at {worst_ts}: "
                f"{worst_actual:.2f} kWh actual"
                + (
                    f" vs {worst_baseline:.2f} kWh baseline"
                    if worst_baseline is not None
                    else ""
                )
                + (
                    f" (deviation: {worst_dev:+.2f} kWh)"
                    if worst_dev is not None
                    else ""
                )
                + "."
            )
        desc_parts.append(
            "A statistical flag indicates an unusual reading versus local history. "
            "It does not confirm energy waste or equipment failure. "
            "Manual investigation is recommended to determine the cause."
        )
        description = " ".join(desc_parts)

        evidence: dict[str, Any] = {
            "zone": str(zone),
            "flagged_reading_count": count,
            "threshold_note": (
                f"Priority is 'High' when largest positive deviation "
                f"> {ANOMALY_HIGH_DEVIATION_KWH} kWh."
            ),
        }
        if worst_ts:
            evidence["worst_timestamp"] = worst_ts
        if worst_actual is not None:
            evidence["worst_actual_kwh"] = round(worst_actual, 3)
        if worst_baseline is not None:
            evidence["worst_baseline_kwh"] = round(worst_baseline, 3)
        if worst_dev is not None:
            evidence["worst_deviation_kwh"] = round(worst_dev, 3)
        if max_pos_dev is not None:
            evidence["largest_positive_deviation_kwh"] = round(max_pos_dev, 3)

        results.append(
            {
                "category": "Anomaly",
                "priority": priority,
                "title": (
                    f"{'Significant' if priority == 'High' else 'Unusual'} reading(s) "
                    f"detected in zone '{zone}'"
                ),
                "description": description,
                "evidence": evidence,
            }
        )

    return results


def _rule_off_peak_baseline(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule 3 – Off-Peak Baseline.

    Compare average consumption during OFF_PEAK_HOURS against DAYTIME_HOURS.
    Flag when off-peak average exceeds OFF_PEAK_DAYTIME_RATIO of the daytime
    average.  Returns [] if either window has no valid hourly data.
    """
    hourly = _safe_dataframe(metrics, "hourly")
    if hourly.empty:
        return []

    required = {"hour", "average_energy_kwh"}
    if not required.issubset(hourly.columns):
        return []

    valid_hourly = _valid_hourly(hourly)
    if valid_hourly.empty:
        return []

    off_peak_rows = valid_hourly[valid_hourly["hour"].isin(OFF_PEAK_HOURS)]
    daytime_rows = valid_hourly[valid_hourly["hour"].isin(DAYTIME_HOURS)]

    if off_peak_rows.empty or daytime_rows.empty:
        return []

    off_peak_avg = _window_average(off_peak_rows)
    daytime_avg = _window_average(daytime_rows)

    if daytime_avg <= 0:
        return []

    ratio = off_peak_avg / daytime_avg
    if ratio <= OFF_PEAK_DAYTIME_RATIO:
        return []

    off_peak_pct = round(ratio * 100, 1)
    threshold_pct = round(OFF_PEAK_DAYTIME_RATIO * 100, 1)

    hour_range_off = "00:00–05:00"
    hour_range_day = "08:00–16:00"

    description = (
        f"The average consumption during off-peak hours ({hour_range_off}) is "
        f"{off_peak_avg:.2f} kWh per interval, which is {off_peak_pct}% of the "
        f"daytime average ({daytime_avg:.2f} kWh per interval, {hour_range_day}). "
        f"This exceeds the reference threshold of {threshold_pct}%. "
        "Persistent off-peak consumption may be worth investigating to understand "
        "whether it reflects expected base loads (servers, security, refrigeration) "
        "or unintended consumption. The data alone does not confirm a specific cause."
        " These are reference clock-hour windows, not verified local tariff periods; "
        "municipal zones may legitimately serve continuous overnight demand."
    )

    return [
        {
            "category": "Off-Peak Baseline",
            "priority": "Medium",
            "title": "Persistent off-peak consumption warrants investigation",
            "description": description,
            "evidence": {
                "off_peak_hours": sorted(OFF_PEAK_HOURS),
                "off_peak_average_kwh": round(off_peak_avg, 3),
                "daytime_hours": sorted(DAYTIME_HOURS),
                "daytime_average_kwh": round(daytime_avg, 3),
                "off_peak_to_daytime_ratio": round(ratio, 4),
                "threshold_note": (
                    f"Flagged when off-peak average exceeds "
                    f"{threshold_pct}% of the daytime average."
                ),
            },
        }
    ]


def _rule_zone_imbalance(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule 4 – Zone Imbalance.

    Require at least two zones with positive average_interval_kwh.
    Flag when max-to-min ratio exceeds ZONE_IMBALANCE_RATIO.
    """
    zone_df = _safe_dataframe(metrics, "zone_comparison")
    if zone_df.empty:
        return []

    required = {"zone", "average_interval_kwh"}
    if not required.issubset(zone_df.columns):
        return []

    valid_zones = zone_df.copy().reset_index(drop=True)
    valid_zones["average_interval_kwh"] = valid_zones["average_interval_kwh"].map(_safe_numeric)
    valid_zones = valid_zones.dropna(subset=["average_interval_kwh", "zone"])
    valid_zones = valid_zones[valid_zones["average_interval_kwh"] > 0]

    if len(valid_zones) < 2:
        return []

    max_row = valid_zones.loc[valid_zones["average_interval_kwh"].idxmax()]
    min_row = valid_zones.loc[valid_zones["average_interval_kwh"].idxmin()]

    max_avg = float(max_row["average_interval_kwh"])
    min_avg = float(min_row["average_interval_kwh"])
    ratio = max_avg / min_avg

    if ratio <= ZONE_IMBALANCE_RATIO:
        return []

    max_zone = str(max_row["zone"])
    min_zone = str(min_row["zone"])

    description = (
        f"Zone '{max_zone}' averages {max_avg:.2f} kWh per interval, while "
        f"zone '{min_zone}' averages {min_avg:.2f} kWh per interval — a ratio "
        f"of {ratio:.2f}x (threshold: {ZONE_IMBALANCE_RATIO}x). "
        "This imbalance may reflect differences in occupancy, equipment, floor "
        "area, or usage patterns. A review of zone-level schedules and equipment "
        "may help identify whether the gap is expected or actionable."
    )

    return [
        {
            "category": "Zone Imbalance",
            "priority": "Medium",
            "title": (
                f"Consumption imbalance detected between zones "
                f"'{max_zone}' and '{min_zone}'"
            ),
            "description": description,
            "evidence": {
                "highest_zone": max_zone,
                "highest_zone_average_kwh": round(max_avg, 3),
                "lowest_zone": min_zone,
                "lowest_zone_average_kwh": round(min_avg, 3),
                "ratio": round(ratio, 4),
                "zones_compared": int(len(valid_zones)),
                "threshold_note": (
                    f"Flagged when max/min zone average ratio "
                    f"> {ZONE_IMBALANCE_RATIO}."
                ),
            },
        }
    ]


def _rule_high_overall_consumption(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule 5 – High Overall Consumption.

    Flag a Low-priority advisory when the maximum reading exceeds
    HIGH_CONSUMPTION_MULTIPLE times the overall average.  Explicitly states
    that a high reading alone does not prove waste.
    """
    summary = metrics.get("summary", {})
    if not isinstance(summary, dict):
        return []

    avg = _safe_float(summary, "average_interval_kwh")
    max_val = _safe_float(summary, "maximum_interval_kwh")
    peak_ts = summary.get("peak_timestamp")
    peak_zone = summary.get("peak_zone")

    if avg is None or avg <= 0 or max_val is None:
        return []

    if max_val <= HIGH_CONSUMPTION_MULTIPLE * avg:
        return []

    multiple = round(max_val / avg, 2)
    ts_str = _format_timestamp(peak_ts)

    desc_parts = [
        f"The single highest interval reading is {max_val:.2f} kWh — "
        f"{multiple}x the dataset-wide average of {avg:.2f} kWh."
    ]
    if ts_str:
        desc_parts.append(f"This peak occurred at {ts_str}")
        if peak_zone:
            desc_parts[-1] += f" in zone '{peak_zone}'"
        desc_parts[-1] += "."
    desc_parts.append(
        "A single high reading does not by itself prove waste or equipment "
        "malfunction. It may reflect legitimate high-demand activity, a "
        "measurement spike, or a one-off event. It is included here as a "
        "prompt for awareness, not as a confirmed finding."
    )
    description = " ".join(desc_parts)

    evidence: dict[str, Any] = {
        "maximum_interval_kwh": round(max_val, 3),
        "average_interval_kwh": round(avg, 3),
        "ratio": multiple,
        "threshold_note": (
            f"Flagged when maximum reading exceeds "
            f"{HIGH_CONSUMPTION_MULTIPLE}x the overall average."
        ),
    }
    if ts_str:
        evidence["peak_timestamp"] = ts_str
    if peak_zone:
        evidence["peak_zone"] = str(peak_zone)

    return [
        {
            "category": "High Overall Consumption",
            "priority": "Low",
            "title": (
                f"Single interval peak of {max_val:.2f} kWh is "
                f"{multiple}x the overall average"
            ),
            "description": description,
            "evidence": evidence,
        }
    ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _valid_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Keep finite, nonnegative energy and valid integer clock hours."""
    valid = hourly.copy()
    for column in ("hour", "average_energy_kwh"):
        valid[column] = valid[column].map(_safe_numeric)
    valid = valid.dropna(subset=["hour", "average_energy_kwh"])
    return valid.loc[
        valid["hour"].between(0, 23)
        & (valid["hour"] % 1 == 0)
        & (valid["average_energy_kwh"] >= 0)
    ]


def _window_average(rows: pd.DataFrame) -> float:
    """Weight hour means by observation counts inferred from analytics totals.

    This avoids treating a partially observed hour like a complete hour. Older
    callers supplying only means retain the equal-hour comparison.
    """
    if "total_energy_kwh" in rows:
        totals = rows["total_energy_kwh"].map(_safe_numeric)
        means = rows["average_energy_kwh"]
        if totals.notna().all() and (totals > 0).all() and (means > 0).all():
            counts = totals / means
            return float(totals.sum() / counts.sum())
    return float(rows["average_energy_kwh"].mean())


def _safe_dataframe(mapping: dict[str, Any], key: str) -> pd.DataFrame:
    """Return the value at *key* if it is a non-None DataFrame, else empty DF."""
    val = mapping.get(key)
    if isinstance(val, pd.DataFrame):
        return val
    return pd.DataFrame()


def _safe_float(mapping: dict[str, Any], key: str) -> float | None:
    """Return a finite float from *mapping[key]*, or None if absent/NaN/None."""
    if not isinstance(mapping, dict):
        return None
    val = mapping.get(key)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _safe_numeric(value: Any) -> float | None:
    """Convert *value* to a finite float, or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _format_timestamp(ts: Any) -> str | None:
    """Return a human-readable timestamp string, or None if not parseable."""
    if ts is None:
        return None
    try:
        parsed = pd.Timestamp(ts)
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None
