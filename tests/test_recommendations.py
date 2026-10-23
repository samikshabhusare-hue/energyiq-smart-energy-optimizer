"""Tests for the EnergyIQ recommendation engine (recommendations.py).

Each test is numbered and annotated with the specification requirement it covers.
Rule-specific tests use minimal synthetic fixtures so that each test exercises
exactly one rule in isolation.  One integration test uses the actual CSV file
through the analytics module.

Running:
    python -m pytest tests/test_recommendations.py -v
"""

import math
import pathlib
import sys

import pandas as pd
import pytest

# Ensure the project root is on sys.path regardless of where pytest is invoked.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import recommendations
from recommendations import (
    ANOMALY_HIGH_DEVIATION_KWH,
    HIGH_CONSUMPTION_MULTIPLE,
    OFF_PEAK_DAYTIME_RATIO,
    PEAK_EXCESS_RATIO,
    ZONE_IMBALANCE_RATIO,
)

# ---------------------------------------------------------------------------
# Constants used across multiple tests
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"category", "priority", "title", "description", "evidence"}
VALID_PRIORITIES = {"High", "Medium", "Low"}


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict:
    """Metrics dict that produces no recommendations (all empty/None)."""
    return {
        "summary": {
            "reading_count": 0,
            "total_energy_kwh": 0.0,
            "average_interval_kwh": None,
            "maximum_interval_kwh": None,
            "peak_timestamp": None,
            "peak_zone": None,
        },
        "hourly": pd.DataFrame(
            columns=["hour", "average_energy_kwh", "total_energy_kwh"]
        ),
        "daily": pd.DataFrame(
            columns=["date", "average_energy_kwh", "total_energy_kwh"]
        ),
        "zone_comparison": pd.DataFrame(
            columns=["zone", "total_energy_kwh", "average_interval_kwh", "reading_count"]
        ),
        "data_quality_warnings": [],
    }


def _anomaly_row(
    *,
    zone: str = "Block A",
    ts: str = "2026-10-22 12:00:00",
    actual_kwh: float = 18.0,
    baseline_kwh: float = 8.1,
    deviation_kwh: float = 9.9,
    flagged: bool = True,
) -> pd.DataFrame:
    """Single-row anomaly DataFrame matching detect_anomalies() column schema."""
    return pd.DataFrame({
        "timestamp":             pd.to_datetime([ts]),
        "zone":                  [zone],
        "energy_kwh":            [actual_kwh],
        "expected_baseline_kwh": [baseline_kwh],
        "deviation_kwh":         [deviation_kwh],
        "anomaly_flag":          [flagged],
    })


def _peak_metrics(
    *,
    overall_avg: float = 4.0,
    peak_avg: float = 8.0,
    peak_hour: int = 12,
) -> dict:
    """Metrics with one clear peak hour; all other hours equal overall_avg."""
    avgs = [overall_avg] * 24
    avgs[peak_hour] = peak_avg
    hourly = pd.DataFrame({
        "hour":               list(range(24)),
        "average_energy_kwh": avgs,
        "total_energy_kwh":   [a * 3 for a in avgs],
    })
    return {
        "summary": {
            "reading_count":        72,
            "total_energy_kwh":     float(sum(avgs) * 3),
            "average_interval_kwh": overall_avg,
            "maximum_interval_kwh": peak_avg,
            "peak_timestamp":       pd.Timestamp("2026-10-22 12:00:00"),
            "peak_zone":            "Block A",
        },
        "hourly": hourly,
        "daily": pd.DataFrame(columns=["date", "average_energy_kwh", "total_energy_kwh"]),
        "zone_comparison": pd.DataFrame({
            "zone":                 ["Block A"],
            "total_energy_kwh":     [float(sum(avgs) * 3)],
            "average_interval_kwh": [overall_avg],
            "reading_count":        [72],
        }),
        "data_quality_warnings": [],
    }


def _two_zone_metrics(*, high_avg: float = 10.0, low_avg: float = 2.0) -> dict:
    """Metrics with two zones; ratio = high_avg / low_avg."""
    return {
        "summary": {
            "reading_count":        48,
            "total_energy_kwh":     (high_avg + low_avg) * 24.0,
            "average_interval_kwh": (high_avg + low_avg) / 2.0,
            "maximum_interval_kwh": high_avg,
            "peak_timestamp":       None,
            "peak_zone":            "High Zone",
        },
        "hourly": pd.DataFrame(
            columns=["hour", "average_energy_kwh", "total_energy_kwh"]
        ),
        "daily": pd.DataFrame(
            columns=["date", "average_energy_kwh", "total_energy_kwh"]
        ),
        "zone_comparison": pd.DataFrame({
            "zone":                 ["High Zone", "Low Zone"],
            "total_energy_kwh":     [high_avg * 24.0, low_avg * 24.0],
            "average_interval_kwh": [high_avg, low_avg],
            "reading_count":        [24, 24],
        }),
        "data_quality_warnings": [],
    }


def _off_peak_metrics(*, off_peak_avg: float, daytime_avg: float) -> dict:
    """Metrics with controlled off-peak (hours 0-5) and daytime (hours 8-16) averages."""
    avgs = []
    for h in range(24):
        if h in {0, 1, 2, 3, 4, 5}:
            avgs.append(off_peak_avg)
        elif h in {8, 9, 10, 11, 12, 13, 14, 15, 16}:
            avgs.append(daytime_avg)
        else:
            avgs.append(daytime_avg * 0.5)

    hourly = pd.DataFrame({
        "hour":               list(range(24)),
        "average_energy_kwh": avgs,
        "total_energy_kwh":   [a * 3 for a in avgs],
    })
    overall = sum(avgs) / len(avgs)
    return {
        "summary": {
            "reading_count":        72,
            "total_energy_kwh":     float(sum(avgs) * 3),
            "average_interval_kwh": overall,
            "maximum_interval_kwh": max(avgs),
            "peak_timestamp":       None,
            "peak_zone":            None,
        },
        "hourly": hourly,
        "daily": pd.DataFrame(columns=["date", "average_energy_kwh", "total_energy_kwh"]),
        "zone_comparison": pd.DataFrame(
            columns=["zone", "total_energy_kwh", "average_interval_kwh", "reading_count"]
        ),
        "data_quality_warnings": [],
    }


def _high_consumption_metrics(*, avg: float = 4.0, max_val: float = 18.0) -> dict:
    """Metrics where maximum_interval_kwh >> average_interval_kwh."""
    return {
        "summary": {
            "reading_count":        72,
            "total_energy_kwh":     avg * 72,
            "average_interval_kwh": avg,
            "maximum_interval_kwh": max_val,
            "peak_timestamp":       pd.Timestamp("2026-10-22 12:00:00"),
            "peak_zone":            "Block A",
        },
        "hourly": pd.DataFrame(
            columns=["hour", "average_energy_kwh", "total_energy_kwh"]
        ),
        "daily": pd.DataFrame(
            columns=["date", "average_energy_kwh", "total_energy_kwh"]
        ),
        "zone_comparison": pd.DataFrame(
            columns=["zone", "total_energy_kwh", "average_interval_kwh", "reading_count"]
        ),
        "data_quality_warnings": [],
    }


# ===========================================================================
# Req 1 – Empty inputs return a list without crashing
# ===========================================================================

class TestEmptyInputs:

    def test_empty_metrics_and_anomaly_df_return_list(self):
        """Empty metrics dict and empty anomaly DataFrame → returns list (no crash)."""
        result = recommendations.generate_recommendations(
            _empty_metrics(), pd.DataFrame()
        )
        assert isinstance(result, list)

    def test_none_metrics_returns_empty_list(self):
        """None passed as metrics → returns [] without raising."""
        result = recommendations.generate_recommendations(None, pd.DataFrame())
        assert result == []

    def test_entirely_empty_dict_returns_list(self):
        """Completely empty dict (no keys at all) → returns list without raising."""
        result = recommendations.generate_recommendations({}, pd.DataFrame())
        assert isinstance(result, list)


# ===========================================================================
# Req 2 – Flagged anomaly produces Anomaly rec with correct evidence values
# ===========================================================================

class TestAnomalyEvidenceContent:

    def test_anomaly_recommendation_produced(self):
        """A single flagged row → exactly one Anomaly recommendation."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(zone="Zone X", deviation_kwh=9.9, flagged=True),
        )
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        assert len(anomaly_recs) == 1

    def test_anomaly_evidence_zone(self):
        """evidence['zone'] matches the zone name in the flagged row."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(zone="Test Zone", deviation_kwh=9.9, flagged=True),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Anomaly")
        assert ev["zone"] == "Test Zone"

    def test_anomaly_evidence_timestamp(self):
        """evidence['worst_timestamp'] is formatted as YYYY-MM-DD HH:MM."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(ts="2026-10-22 12:00:00", deviation_kwh=9.9, flagged=True),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Anomaly")
        assert ev.get("worst_timestamp") == "2026-10-22 12:00"

    def test_anomaly_evidence_actual_kwh(self):
        """evidence['worst_actual_kwh'] matches the observed consumption."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(actual_kwh=18.0, deviation_kwh=9.9, flagged=True),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Anomaly")
        assert ev.get("worst_actual_kwh") == pytest.approx(18.0, rel=1e-3)

    def test_anomaly_evidence_baseline_kwh(self):
        """evidence['worst_baseline_kwh'] matches expected_baseline_kwh."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(actual_kwh=18.0, baseline_kwh=8.1, deviation_kwh=9.9, flagged=True),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Anomaly")
        assert ev.get("worst_baseline_kwh") == pytest.approx(8.1, rel=1e-3)

    def test_anomaly_evidence_deviation_kwh(self):
        """evidence['largest_positive_deviation_kwh'] matches deviation_kwh."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(actual_kwh=18.0, baseline_kwh=8.1, deviation_kwh=9.9, flagged=True),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Anomaly")
        assert ev.get("largest_positive_deviation_kwh") == pytest.approx(9.9, rel=1e-3)

    def test_anomaly_evidence_flagged_reading_count(self):
        """evidence['flagged_reading_count'] equals the number of flagged rows."""
        two_rows = pd.DataFrame({
            "timestamp":             pd.to_datetime(["2026-10-22 12:00", "2026-10-22 13:00"]),
            "zone":                  ["Block A", "Block A"],
            "energy_kwh":            [18.0, 16.0],
            "expected_baseline_kwh": [8.1, 8.0],
            "deviation_kwh":         [9.9, 8.0],
            "anomaly_flag":          [True, True],
        })
        recs = recommendations.generate_recommendations(_empty_metrics(), two_rows)
        ev = next(r["evidence"] for r in recs if r["category"] == "Anomaly")
        assert ev["flagged_reading_count"] == 2


# ===========================================================================
# Req 3 – Positive deviation > 5 kWh → High priority
# ===========================================================================

class TestAnomalyHighPriority:

    def test_deviation_just_above_threshold_is_high(self):
        """Deviation of ANOMALY_HIGH_DEVIATION_KWH + 0.01 → High priority."""
        dev = ANOMALY_HIGH_DEVIATION_KWH + 0.01
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(
                actual_kwh=ANOMALY_HIGH_DEVIATION_KWH + 8.0,
                baseline_kwh=8.0,
                deviation_kwh=dev,
                flagged=True,
            ),
        )
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        assert len(anomaly_recs) == 1
        assert anomaly_recs[0]["priority"] == "High"

    def test_large_deviation_is_high(self):
        """Clearly large deviation (2× threshold) → High priority."""
        dev = ANOMALY_HIGH_DEVIATION_KWH * 2
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(deviation_kwh=dev, flagged=True),
        )
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        assert anomaly_recs[0]["priority"] == "High"


# ===========================================================================
# Req 4 – Positive deviation ≤ 5 kWh → Medium priority
# ===========================================================================

class TestAnomalyMediumPriority:

    def test_deviation_exactly_at_threshold_is_medium(self):
        """Deviation exactly equal to ANOMALY_HIGH_DEVIATION_KWH → Medium (not High)."""
        dev = ANOMALY_HIGH_DEVIATION_KWH   # strictly not greater than → Medium
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(
                actual_kwh=dev + 5.0,
                baseline_kwh=5.0,
                deviation_kwh=dev,
                flagged=True,
            ),
        )
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        assert len(anomaly_recs) == 1
        assert anomaly_recs[0]["priority"] == "Medium"

    def test_deviation_below_threshold_is_medium(self):
        """Deviation below ANOMALY_HIGH_DEVIATION_KWH → Medium priority."""
        dev = ANOMALY_HIGH_DEVIATION_KWH - 1.0
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(
                actual_kwh=dev + 5.0,
                baseline_kwh=5.0,
                deviation_kwh=dev,
                flagged=True,
            ),
        )
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        assert len(anomaly_recs) == 1
        assert anomaly_recs[0]["priority"] == "Medium"


# ===========================================================================
# Req 5 – Clear peak produces Peak Demand with correct numerical evidence
# ===========================================================================

class TestPeakDemand:

    def test_peak_demand_recommendation_fires(self):
        """Peak avg clearly above threshold → Peak Demand recommendation produced."""
        overall_avg = 4.0
        peak_avg = overall_avg * (1 + PEAK_EXCESS_RATIO + 0.5)   # 70% above
        recs = recommendations.generate_recommendations(
            _peak_metrics(overall_avg=overall_avg, peak_avg=peak_avg, peak_hour=14),
            pd.DataFrame(),
        )
        assert any(r["category"] == "Peak Demand" for r in recs)

    def test_peak_demand_evidence_overall_average(self):
        """evidence['overall_average_interval_kwh'] matches the summary average."""
        overall_avg = 4.0
        peak_avg = 9.0
        recs = recommendations.generate_recommendations(
            _peak_metrics(overall_avg=overall_avg, peak_avg=peak_avg, peak_hour=12),
            pd.DataFrame(),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Peak Demand")
        assert ev["overall_average_interval_kwh"] == pytest.approx(overall_avg, rel=1e-3)

    def test_peak_demand_evidence_threshold(self):
        """evidence['threshold_kwh'] == overall_avg × (1 + PEAK_EXCESS_RATIO)."""
        overall_avg = 4.0
        peak_avg = 9.0
        recs = recommendations.generate_recommendations(
            _peak_metrics(overall_avg=overall_avg, peak_avg=peak_avg, peak_hour=12),
            pd.DataFrame(),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Peak Demand")
        expected_threshold = overall_avg * (1 + PEAK_EXCESS_RATIO)
        assert ev["threshold_kwh"] == pytest.approx(expected_threshold, rel=1e-3)

    def test_peak_demand_evidence_qualifying_hours_contains_peak_hour(self):
        """evidence['qualifying_peak_hours'] list includes the injected peak hour."""
        overall_avg = 4.0
        peak_avg = 9.0
        peak_hour = 14
        recs = recommendations.generate_recommendations(
            _peak_metrics(overall_avg=overall_avg, peak_avg=peak_avg, peak_hour=peak_hour),
            pd.DataFrame(),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Peak Demand")
        qualifying = ev["qualifying_peak_hours"]
        assert isinstance(qualifying, list) and len(qualifying) >= 1
        hours_listed = [h["hour"] for h in qualifying]
        assert peak_hour in hours_listed
        match = next(h for h in qualifying if h["hour"] == peak_hour)
        assert match["average_energy_kwh"] == pytest.approx(peak_avg, rel=1e-3)

    def test_peak_demand_does_not_fire_below_threshold(self):
        """Peak only 10% above overall (below 20% threshold) → no Peak Demand."""
        overall_avg = 4.0
        peak_avg = overall_avg * (1 + PEAK_EXCESS_RATIO - 0.10)  # 10% above
        recs = recommendations.generate_recommendations(
            _peak_metrics(overall_avg=overall_avg, peak_avg=peak_avg, peak_hour=12),
            pd.DataFrame(),
        )
        assert not any(r["category"] == "Peak Demand" for r in recs)


# ===========================================================================
# Req 6 – Single zone does not produce a Zone Imbalance recommendation
# ===========================================================================

class TestSingleZoneNoImbalance:

    def test_single_zone_produces_no_imbalance(self):
        """One zone in zone_comparison → no Zone Imbalance recommendation."""
        recs = recommendations.generate_recommendations(
            _peak_metrics(),   # fixture has only "Block A"
            pd.DataFrame(),
        )
        assert not any(r["category"] == "Zone Imbalance" for r in recs)


# ===========================================================================
# Req 7 – Two zones with ratio > 1.5 → Zone Imbalance
# ===========================================================================

class TestZoneImbalance:

    def test_imbalance_fires_when_ratio_exceeds_threshold(self):
        """Two zones with ratio > ZONE_IMBALANCE_RATIO → Zone Imbalance produced."""
        high_avg, low_avg = 10.0, 2.0     # ratio = 5.0 >> 1.5
        assert high_avg / low_avg > ZONE_IMBALANCE_RATIO
        recs = recommendations.generate_recommendations(
            _two_zone_metrics(high_avg=high_avg, low_avg=low_avg),
            pd.DataFrame(),
        )
        assert any(r["category"] == "Zone Imbalance" for r in recs)

    def test_imbalance_evidence_zone_names_and_values(self):
        """Zone Imbalance evidence contains correct zone names, averages and ratio."""
        high_avg, low_avg = 10.0, 2.0
        recs = recommendations.generate_recommendations(
            _two_zone_metrics(high_avg=high_avg, low_avg=low_avg),
            pd.DataFrame(),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Zone Imbalance")
        assert ev["highest_zone"] == "High Zone"
        assert ev["lowest_zone"] == "Low Zone"
        assert ev["highest_zone_average_kwh"] == pytest.approx(high_avg, rel=1e-3)
        assert ev["lowest_zone_average_kwh"] == pytest.approx(low_avg, rel=1e-3)
        assert ev["ratio"] == pytest.approx(high_avg / low_avg, rel=1e-3)
        assert ev["zones_compared"] == 2

    def test_imbalance_does_not_fire_below_threshold(self):
        """Two zones with ratio ≤ ZONE_IMBALANCE_RATIO → no Zone Imbalance."""
        high_avg, low_avg = 3.0, 2.5    # ratio = 1.2, below 1.5
        assert high_avg / low_avg <= ZONE_IMBALANCE_RATIO
        recs = recommendations.generate_recommendations(
            _two_zone_metrics(high_avg=high_avg, low_avg=low_avg),
            pd.DataFrame(),
        )
        assert not any(r["category"] == "Zone Imbalance" for r in recs)


# ===========================================================================
# Req 8 – High off-peak consumption → Off-Peak Baseline recommendation
# ===========================================================================

class TestOffPeakBaseline:

    def test_off_peak_fires_when_ratio_above_threshold(self):
        """off_peak_avg > OFF_PEAK_DAYTIME_RATIO × daytime_avg → Off-Peak Baseline."""
        daytime = 6.0
        off_peak = daytime * (OFF_PEAK_DAYTIME_RATIO + 0.30)   # 50% of daytime
        recs = recommendations.generate_recommendations(
            _off_peak_metrics(off_peak_avg=off_peak, daytime_avg=daytime),
            pd.DataFrame(),
        )
        assert any(r["category"] == "Off-Peak Baseline" for r in recs)

    def test_off_peak_evidence_averages_and_ratio(self):
        """Off-Peak Baseline evidence contains correct averages and ratio."""
        daytime = 6.0
        off_peak = daytime * (OFF_PEAK_DAYTIME_RATIO + 0.30)
        recs = recommendations.generate_recommendations(
            _off_peak_metrics(off_peak_avg=off_peak, daytime_avg=daytime),
            pd.DataFrame(),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "Off-Peak Baseline")
        assert ev["off_peak_average_kwh"] == pytest.approx(off_peak, rel=1e-3)
        assert ev["daytime_average_kwh"] == pytest.approx(daytime, rel=1e-3)
        assert ev["off_peak_to_daytime_ratio"] == pytest.approx(
            off_peak / daytime, rel=1e-3
        )
        assert isinstance(ev["off_peak_hours"], list)
        assert isinstance(ev["daytime_hours"], list)


# ===========================================================================
# Req 9 – Normal off-peak consumption does not produce the recommendation
# ===========================================================================

class TestOffPeakDoesNotFire:

    def test_off_peak_does_not_fire_when_ratio_below_threshold(self):
        """off_peak_avg ≤ OFF_PEAK_DAYTIME_RATIO × daytime_avg → no Off-Peak Baseline."""
        daytime = 6.0
        off_peak = daytime * (OFF_PEAK_DAYTIME_RATIO - 0.10)   # only 10% of daytime
        recs = recommendations.generate_recommendations(
            _off_peak_metrics(off_peak_avg=off_peak, daytime_avg=daytime),
            pd.DataFrame(),
        )
        assert not any(r["category"] == "Off-Peak Baseline" for r in recs)


# ===========================================================================
# Req 10 – maximum > 2× average → Low-priority High Consumption advisory
# ===========================================================================

class TestHighOverallConsumption:

    def test_high_consumption_fires_above_threshold(self):
        """max > HIGH_CONSUMPTION_MULTIPLE × avg → High Consumption recommendation."""
        avg = 4.0
        max_val = avg * (HIGH_CONSUMPTION_MULTIPLE + 1.0)   # 3× average
        recs = recommendations.generate_recommendations(
            _high_consumption_metrics(avg=avg, max_val=max_val),
            pd.DataFrame(),
        )
        assert any(r["category"] == "High Overall Consumption" for r in recs)

    def test_high_consumption_is_low_priority(self):
        """High Overall Consumption advisory must have priority 'Low'."""
        avg = 4.0
        max_val = avg * (HIGH_CONSUMPTION_MULTIPLE + 1.0)
        recs = recommendations.generate_recommendations(
            _high_consumption_metrics(avg=avg, max_val=max_val),
            pd.DataFrame(),
        )
        hc = next(r for r in recs if r["category"] == "High Overall Consumption")
        assert hc["priority"] == "Low"

    def test_high_consumption_evidence_values(self):
        """evidence contains correct maximum, average and ratio."""
        avg = 4.0
        max_val = 18.0
        recs = recommendations.generate_recommendations(
            _high_consumption_metrics(avg=avg, max_val=max_val),
            pd.DataFrame(),
        )
        ev = next(r["evidence"] for r in recs if r["category"] == "High Overall Consumption")
        assert ev["maximum_interval_kwh"] == pytest.approx(max_val, rel=1e-3)
        assert ev["average_interval_kwh"] == pytest.approx(avg, rel=1e-3)
        assert ev["ratio"] == pytest.approx(max_val / avg, rel=1e-2)

    def test_high_consumption_does_not_fire_at_threshold(self):
        """max exactly equal to HIGH_CONSUMPTION_MULTIPLE × avg → no advisory."""
        avg = 4.0
        max_val = avg * HIGH_CONSUMPTION_MULTIPLE   # not strictly greater → no fire
        recs = recommendations.generate_recommendations(
            _high_consumption_metrics(avg=avg, max_val=max_val),
            pd.DataFrame(),
        )
        assert not any(r["category"] == "High Overall Consumption" for r in recs)


# ===========================================================================
# Req 11 – All anomaly_flag values False → no Anomaly recommendation
# ===========================================================================

class TestNoAnomalyWhenAllFlagsFalse:

    def test_all_false_flags_produce_no_anomaly(self):
        """DataFrame with anomaly_flag all False → no Anomaly recommendation."""
        df = pd.DataFrame({
            "timestamp":             pd.to_datetime(
                ["2026-10-20 10:00:00", "2026-10-20 11:00:00"]
            ),
            "zone":                  ["Block A", "Block A"],
            "energy_kwh":            [7.0, 7.2],
            "expected_baseline_kwh": [7.1, 7.0],
            "deviation_kwh":         [-0.1, 0.2],
            "anomaly_flag":          [False, False],
        })
        recs = recommendations.generate_recommendations(_empty_metrics(), df)
        assert not any(r["category"] == "Anomaly" for r in recs)


# ===========================================================================
# Req 12 – Every recommendation contains all five required keys
# ===========================================================================

class TestRequiredKeys:

    def test_all_recommendations_have_required_keys(self):
        """category, priority, title, description, evidence must be present in every rec."""
        metrics = _two_zone_metrics()
        anomaly_df = _anomaly_row(deviation_kwh=9.9, flagged=True)
        recs = recommendations.generate_recommendations(metrics, anomaly_df)
        assert len(recs) > 0, "Need at least one recommendation to test keys"
        for rec in recs:
            missing = REQUIRED_KEYS - rec.keys()
            assert not missing, (
                f"Recommendation missing keys {missing!r}: {rec.get('title', '<no title>')}"
            )

    def test_evidence_is_dict_in_every_recommendation(self):
        """evidence must always be a dict, never None or a string."""
        metrics = _two_zone_metrics()
        anomaly_df = _anomaly_row(deviation_kwh=9.9, flagged=True)
        recs = recommendations.generate_recommendations(metrics, anomaly_df)
        for rec in recs:
            assert isinstance(rec["evidence"], dict), (
                f"evidence is not a dict in: {rec.get('title')}"
            )


# ===========================================================================
# Req 13 – All priorities valid and output sorted High → Medium → Low
# ===========================================================================

class TestPriorityOrder:

    def test_all_priorities_are_valid_strings(self):
        """Every priority value must be exactly 'High', 'Medium', or 'Low'."""
        metrics = _two_zone_metrics()
        anomaly_df = _anomaly_row(deviation_kwh=9.9, flagged=True)
        recs = recommendations.generate_recommendations(metrics, anomaly_df)
        for rec in recs:
            assert rec["priority"] in VALID_PRIORITIES, (
                f"Invalid priority '{rec['priority']}' in: {rec.get('title')}"
            )

    def test_output_sorted_high_then_medium_then_low(self):
        """Output list must be ordered: all High, then all Medium, then all Low."""
        # Build a scenario that triggers all three priority bands:
        #   High   → Anomaly (dev=9.9) and/or Peak Demand
        #   Medium → Zone Imbalance
        #   Low    → High Consumption (max=18 vs avg=4)
        metrics = _two_zone_metrics(high_avg=10.0, low_avg=2.0)
        metrics["summary"]["average_interval_kwh"] = 4.0
        metrics["summary"]["maximum_interval_kwh"] = 18.0
        anomaly_df = _anomaly_row(deviation_kwh=9.9, flagged=True)

        recs = recommendations.generate_recommendations(metrics, anomaly_df)
        assert len(recs) >= 2, "Need at least two recommendations to test sort order"

        order = {"High": 0, "Medium": 1, "Low": 2}
        order_vals = [order[r["priority"]] for r in recs]
        assert order_vals == sorted(order_vals), (
            f"Recommendations not sorted High→Medium→Low: "
            f"{[r['priority'] for r in recs]}"
        )


# ===========================================================================
# Req 14 – Missing keys, empty DataFrames and NaN values handled safely
# ===========================================================================

class TestSafeHandlingOfBadInputs:

    def test_completely_empty_dict_does_not_raise(self):
        """Empty dict → no exception, returns list."""
        result = recommendations.generate_recommendations({}, pd.DataFrame())
        assert isinstance(result, list)

    def test_summary_with_none_values_does_not_raise(self):
        """None and NaN in summary dict → no crash."""
        metrics = {
            "summary": {
                "reading_count":        0,
                "total_energy_kwh":     None,
                "average_interval_kwh": None,
                "maximum_interval_kwh": float("nan"),
                "peak_timestamp":       None,
                "peak_zone":            None,
            },
            "hourly": pd.DataFrame(
                columns=["hour", "average_energy_kwh", "total_energy_kwh"]
            ),
            "daily": pd.DataFrame(
                columns=["date", "average_energy_kwh", "total_energy_kwh"]
            ),
            "zone_comparison": pd.DataFrame(
                columns=["zone", "total_energy_kwh", "average_interval_kwh", "reading_count"]
            ),
            "data_quality_warnings": [],
        }
        result = recommendations.generate_recommendations(metrics, pd.DataFrame())
        assert isinstance(result, list)

    def test_nan_baseline_in_anomaly_does_not_raise(self):
        """NaN expected_baseline_kwh in a flagged row → no crash."""
        df = pd.DataFrame({
            "timestamp":             pd.to_datetime(["2026-10-22 12:00:00"]),
            "zone":                  ["Block A"],
            "energy_kwh":            [18.0],
            "expected_baseline_kwh": [float("nan")],
            "deviation_kwh":         [9.9],
            "anomaly_flag":          [True],
        })
        result = recommendations.generate_recommendations(_empty_metrics(), df)
        assert isinstance(result, list)

    def test_nan_baseline_omitted_from_evidence(self):
        """When baseline is NaN, 'worst_baseline_kwh' must not appear in evidence."""
        df = pd.DataFrame({
            "timestamp":             pd.to_datetime(["2026-10-22 12:00:00"]),
            "zone":                  ["Block A"],
            "energy_kwh":            [18.0],
            "expected_baseline_kwh": [float("nan")],
            "deviation_kwh":         [9.9],
            "anomaly_flag":          [True],
        })
        recs = recommendations.generate_recommendations(_empty_metrics(), df)
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        if anomaly_recs:
            assert "worst_baseline_kwh" not in anomaly_recs[0]["evidence"]

    def test_anomaly_df_with_wrong_columns_does_not_raise(self):
        """Anomaly DataFrame missing required columns → no crash, no Anomaly rec."""
        bad_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        result = recommendations.generate_recommendations(_empty_metrics(), bad_df)
        assert isinstance(result, list)
        assert not any(r["category"] == "Anomaly" for r in result)

    def test_hourly_df_with_nan_averages_does_not_raise(self):
        """Hourly DataFrame with all NaN average_energy_kwh → no crash."""
        metrics = _empty_metrics()
        metrics["hourly"] = pd.DataFrame({
            "hour": list(range(24)),
            "average_energy_kwh": [float("nan")] * 24,
            "total_energy_kwh": [0.0] * 24,
        })
        result = recommendations.generate_recommendations(metrics, pd.DataFrame())
        assert isinstance(result, list)


# ===========================================================================
# Req 15 – Anomaly descriptions do not claim energy waste or invent causes
# ===========================================================================

class TestAnomalyDescriptionLanguage:

    # Phrases that would constitute an unwarranted causal claim.
    FORBIDDEN_CLAIM_PHRASES = [
        "confirms energy waste",
        "confirmed energy waste",
        "proves energy waste",
        "proven energy waste",
        "is energy waste",
        "confirmed equipment failure",
        "proves equipment failure",
        "caused by",
    ]

    def test_anomaly_description_does_not_assert_waste(self):
        """Anomaly description must not contain affirmative waste/failure claims."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(deviation_kwh=9.9, flagged=True),
        )
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        assert len(anomaly_recs) == 1

        desc = anomaly_recs[0]["description"].lower()
        for phrase in self.FORBIDDEN_CLAIM_PHRASES:
            assert phrase not in desc, (
                f"Anomaly description contains forbidden claim '{phrase}'"
            )

    def test_anomaly_description_contains_disclaimer(self):
        """Anomaly description must state that the flag does not confirm waste."""
        recs = recommendations.generate_recommendations(
            _empty_metrics(),
            _anomaly_row(deviation_kwh=9.9, flagged=True),
        )
        anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
        desc = anomaly_recs[0]["description"].lower()
        assert "does not confirm" in desc or "does not prove" in desc, (
            "Anomaly description must include a disclaimer that the flag "
            "does not confirm waste or equipment failure"
        )


# ===========================================================================
# Integration test – real CSV through analytics pipeline
# ===========================================================================

class TestIntegrationWithSyntheticCSV:

    @pytest.fixture(scope="class")
    @staticmethod
    def recs_from_csv():
        """Keep the intentional 18 kWh spike test on the synthetic fixture."""
        import analytics  # noqa: PLC0415

        csv_path = PROJECT_ROOT / "data" / "sample_energy_synthetic.csv"

        df = analytics.load_and_validate_data(str(csv_path))
        metrics = analytics.calculate_metrics(df)
        anomalies = analytics.detect_anomalies(df)
        return recommendations.generate_recommendations(metrics, anomalies)

    def test_integration_returns_list(self, recs_from_csv):
        """generate_recommendations returns a list when given real analytics output."""
        assert isinstance(recs_from_csv, list)

    def test_integration_anomaly_category_present(self, recs_from_csv):
        """The 2026-10-22 12:00 spike (18 kWh vs ~8.1 baseline) must trigger Anomaly."""
        cats = {r["category"] for r in recs_from_csv}
        assert "Anomaly" in cats, (
            "Expected an Anomaly recommendation from the 18 kWh spike in the CSV"
        )

    def test_integration_high_consumption_present(self, recs_from_csv):
        """18 kWh is ~4× the 4.43 kWh average → High Consumption advisory expected."""
        cats = {r["category"] for r in recs_from_csv}
        assert "High Overall Consumption" in cats

    def test_integration_all_keys_present(self, recs_from_csv):
        """Every recommendation from the real pipeline must have all five required keys."""
        for rec in recs_from_csv:
            missing = REQUIRED_KEYS - rec.keys()
            assert not missing, (
                f"Integration rec missing keys {missing!r}: {rec.get('title')}"
            )

    def test_integration_output_is_sorted(self, recs_from_csv):
        """Integration output must be sorted High → Medium → Low."""
        order = {"High": 0, "Medium": 1, "Low": 2}
        order_vals = [order[r["priority"]] for r in recs_from_csv]
        assert order_vals == sorted(order_vals), (
            f"Integration output not sorted: {[r['priority'] for r in recs_from_csv]}"
        )


@pytest.fixture(scope="module")
def tetouan_pipeline():
    """Exercise all 157,248 actual ten-minute zone readings, without sampling."""
    import analytics

    df = analytics.load_and_validate_data(PROJECT_ROOT / "data" / "sample_energy.csv")
    metrics = analytics.calculate_metrics(df)
    anomalies = analytics.detect_anomalies(df)
    recs = recommendations.generate_recommendations(metrics, anomalies)
    return df, metrics, anomalies, recs


def test_tetouan_dataset_contract(tetouan_pipeline):
    df, _, _, _ = tetouan_pipeline
    assert len(df) == 157248
    assert set(df["zone"]) == {"Zone 1", "Zone 2", "Zone 3"}
    assert df["timestamp"].min() == pd.Timestamp("2017-01-01")
    assert df["timestamp"].max() == pd.Timestamp("2017-12-30 23:50")
    for _, zone in df.groupby("zone"):
        assert (zone["timestamp"].diff().dropna() == pd.Timedelta(minutes=10)).all()


def test_tetouan_recommendations_match_analytics_evidence(tetouan_pipeline):
    import json

    _, metrics, anomalies, recs = tetouan_pipeline
    assert recs
    json.dumps(recs, allow_nan=False)
    flagged = anomalies[anomalies["anomaly_flag"]]
    anomaly_recs = [r for r in recs if r["category"] == "Anomaly"]
    assert {r["evidence"]["zone"] for r in anomaly_recs} == set(flagged["zone"])
    for rec in anomaly_recs:
        evidence = rec["evidence"]
        zone = flagged[flagged["zone"] == evidence["zone"]]
        assert evidence["flagged_reading_count"] == len(zone)
        selected = zone[zone["timestamp"].dt.strftime("%Y-%m-%d %H:%M") == evidence["worst_timestamp"]]
        assert len(selected) == 1
        assert evidence["worst_actual_kwh"] == pytest.approx(selected.iloc[0]["energy_kwh"], abs=0.001)
    categories = {r["category"] for r in recs}
    summary = metrics["summary"]
    hourly = metrics["hourly"]
    top = hourly.nlargest(recommendations.PEAK_TOP_N, "average_energy_kwh")
    expected_peaks = top[top["average_energy_kwh"] > summary["average_interval_kwh"] * (1 + PEAK_EXCESS_RATIO)]
    assert ("Peak Demand" in categories) == (not expected_peaks.empty)
    if not expected_peaks.empty:
        peak = next(r["evidence"] for r in recs if r["category"] == "Peak Demand")
        assert [r["hour"] for r in peak["qualifying_peak_hours"]] == expected_peaks["hour"].tolist()
    zones = metrics["zone_comparison"]
    expected_ratio = zones["average_interval_kwh"].max() / zones["average_interval_kwh"].min()
    assert ("Zone Imbalance" in categories) == (expected_ratio > ZONE_IMBALANCE_RATIO)
    if expected_ratio > ZONE_IMBALANCE_RATIO:
        imbalance = next(r["evidence"] for r in recs if r["category"] == "Zone Imbalance")
        assert imbalance["ratio"] == pytest.approx(expected_ratio, abs=0.0001)
    assert ("High Overall Consumption" in categories) == (
        summary["maximum_interval_kwh"] > HIGH_CONSUMPTION_MULTIPLE * summary["average_interval_kwh"]
    )
    off = next(r["evidence"] for r in recs if r["category"] == "Off-Peak Baseline")
    df = tetouan_pipeline[0]
    assert off["off_peak_average_kwh"] == pytest.approx(
        df.loc[df["timestamp"].dt.hour.isin(recommendations.OFF_PEAK_HOURS), "energy_kwh"].mean(), abs=0.001
    )
    for rec in recs:
        assert REQUIRED_KEYS <= rec.keys()
        assert "2026-10-22" not in rec["description"]
    priorities = [{"High": 0, "Medium": 1, "Low": 2}[r["priority"]] for r in recs]
    assert priorities == sorted(priorities)


def test_negative_only_anomaly_reports_largest_drop():
    df = pd.concat([
        _anomaly_row(actual_kwh=7, baseline_kwh=8, deviation_kwh=-1),
        _anomaly_row(ts="2026-10-22 13:00", actual_kwh=2, baseline_kwh=8, deviation_kwh=-6),
    ])  # Deliberately repeated index labels.
    rec = recommendations.generate_recommendations({}, df)[0]
    assert rec["priority"] == "Medium"
    assert rec["evidence"]["worst_deviation_kwh"] == -6
    assert rec["evidence"]["worst_timestamp"] == "2026-10-22 13:00"
    assert "largest positive deviation" not in rec["description"]
    assert "-6.00 kWh" in rec["description"]


def test_positive_anomalies_with_duplicate_indices():
    df = pd.concat([_anomaly_row(deviation_kwh=2), _anomaly_row(deviation_kwh=9)])
    rec = recommendations.generate_recommendations({}, df)[0]
    assert rec["evidence"]["largest_positive_deviation_kwh"] == 9


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "invalid"])
def test_unusable_anomaly_actual_is_ignored(value):
    df = _anomaly_row(actual_kwh=value)
    assert recommendations.generate_recommendations({}, df) == []


def test_off_peak_weights_unequal_observation_counts():
    metrics = _empty_metrics()
    metrics["hourly"] = pd.DataFrame({
        "hour": [0, 1, 8],
        "average_energy_kwh": [1., 9., 10.],
        "total_energy_kwh": [100., 9., 100.],
    })
    # Overnight mean is 109/101, not the equally weighted hour mean of 5.
    assert recommendations.generate_recommendations(metrics, pd.DataFrame()) == []


@pytest.mark.parametrize("summary", [None, [], "invalid"])
def test_malformed_summary_does_not_crash(summary):
    metrics = _empty_metrics()
    metrics["summary"] = summary
    assert recommendations.generate_recommendations(metrics, pd.DataFrame()) == []


@pytest.mark.parametrize("hour,average", [(None, 20), (24, 20), (1.5, 20), (12, float("inf")), (12, "invalid"), (12, -20)])
def test_invalid_hourly_evidence_is_ignored(hour, average):
    metrics = _peak_metrics()
    metrics["hourly"] = pd.DataFrame({"hour": [hour], "average_energy_kwh": [average]})
    assert recommendations.generate_recommendations(metrics, pd.DataFrame()) == []


@pytest.mark.parametrize("deviation,priority", [(5., "Medium"), (5.01, "High"), (-6., "Medium")])
def test_anomaly_priority_boundary(deviation, priority):
    rec = recommendations.generate_recommendations({}, _anomaly_row(deviation_kwh=deviation))[0]
    assert rec["priority"] == priority


def test_rules_do_not_mutate_inputs():
    import copy

    metrics = _two_zone_metrics()
    anomalies = _anomaly_row()
    original = copy.deepcopy(metrics)
    original_anomalies = anomalies.copy(deep=True)
    recommendations.generate_recommendations(metrics, anomalies)
    for key in ("hourly", "daily", "zone_comparison"):
        pd.testing.assert_frame_equal(metrics[key], original[key])
    assert metrics["summary"] == original["summary"]
    pd.testing.assert_frame_equal(anomalies, original_anomalies)


def test_peak_threshold_equality():
    metrics = _peak_metrics(overall_avg=10, peak_avg=10 * (1 + PEAK_EXCESS_RATIO))
    recs = recommendations.generate_recommendations(metrics, pd.DataFrame())
    assert not any(r['category'] == 'Peak Demand' for r in recs)


def test_zone_threshold_equality():
    metrics = _two_zone_metrics(high_avg=3, low_avg=2)
    recs = recommendations.generate_recommendations(metrics, pd.DataFrame())
    assert not any(r['category'] == 'Zone Imbalance' for r in recs)


def test_off_peak_threshold_equality():
    metrics = _off_peak_metrics(off_peak_avg=2, daytime_avg=10)
    recs = recommendations.generate_recommendations(metrics, pd.DataFrame())
    assert not any(r['category'] == 'Off-Peak Baseline' for r in recs)


def test_peak_hours_limited_to_top_three():
    metrics = _peak_metrics(overall_avg=1)
    metrics['hourly'] = pd.DataFrame({'hour': [1, 2, 3, 4], 'average_energy_kwh': [2, 3, 4, 5]})
    recs = recommendations.generate_recommendations(metrics, pd.DataFrame())
    evidence = next(r['evidence'] for r in recs if r['category'] == 'Peak Demand')
    assert [row['hour'] for row in evidence['qualifying_peak_hours']] == [4, 3, 2]


def test_equal_priority_order_is_stable():
    recs = recommendations.generate_recommendations(_peak_metrics(), _anomaly_row())
    high_categories = [r['category'] for r in recs if r['priority'] == 'High']
    assert high_categories == ['Peak Demand', 'Anomaly']


def test_anomaly_zones_are_independent():
    anomalies = pd.concat([
        _anomaly_row(zone='Zone 1', deviation_kwh=9),
        _anomaly_row(zone='Zone 2', deviation_kwh=2),
    ], ignore_index=True)
    recs = recommendations.generate_recommendations({}, anomalies)
    assert [(r['evidence']['zone'], r['priority']) for r in recs] == [('Zone 1', 'High'), ('Zone 2', 'Medium')]
    assert all(r['evidence']['flagged_reading_count'] == 1 for r in recs)


def test_invalid_anomaly_timestamp_is_omitted():
    anomalies = _anomaly_row()
    anomalies['timestamp'] = 'invalid'
    rec = recommendations.generate_recommendations({}, anomalies)[0]
    assert 'worst_timestamp' not in rec['evidence']
    assert 'invalid' not in rec['description']


@pytest.mark.parametrize('anomalies', [None, [], {}])
def test_metric_rules_without_anomaly_dataframe(anomalies):
    recs = recommendations.generate_recommendations(_peak_metrics(), anomalies)
    assert any(r['category'] == 'Peak Demand' for r in recs)
    assert not any(r['category'] == 'Anomaly' for r in recs)


@pytest.mark.parametrize('value', [float('inf'), float('-inf'), 'invalid'])
def test_nonfinite_summary_values_are_ignored(value):
    metrics = _high_consumption_metrics()
    metrics['summary']['average_interval_kwh'] = value
    assert recommendations.generate_recommendations(metrics, pd.DataFrame()) == []


def test_off_peak_requires_daytime_data():
    metrics = _empty_metrics()
    metrics['hourly'] = pd.DataFrame({'hour': [0, 1], 'average_energy_kwh': [4, 5]})
    assert recommendations.generate_recommendations(metrics, pd.DataFrame()) == []
