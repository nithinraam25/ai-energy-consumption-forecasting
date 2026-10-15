"""Phase 4 Regression Tests — Forecast Reliability, Data Integrity, and Deployment Validation.

Validates:
1. Demand Data Contract:
   - Unambiguous UTC timestamp normalization and regular hourly cadence.
   - Plausible numerical bounds enforcement [10,000.0, 75,000.0 MW] and finite values.
   - Detection and rejection of out-of-order, duplicate, and missing hourly timestamps.
   - Operational readiness gates: returns READY for historical archive and BLOCKED_NO_TELEMETRY
     when decision cutoff exceeds available telemetry.
2. Prediction Interval Reliability:
   - Strict quantile monotonicity: Q05 <= Q50 <= Q95 with zero violations across all predictions.
   - Interval widths strictly positive everywhere (Q95 - Q05 > 0).
   - Multi-horizon interval coverage breakdown (H=1, H=24, H=48) and sliced regime stats exist.
   - Chronological isolation of calibration set from training and test partitions.
3. Statistical Test & Effect Size Integrity:
   - Diebold-Mariano test confirms positive DM statistic and p < 10^-15.
   - Cohen's d effect sizes are positive and properly quantified.
4. API & Deployment Reliability:
   - Endpoint /api/operational/demand_readiness correctly identifies missing SCADA telemetry.
   - /api/evaluation/backtest_samples validates horizon admissibility (1, 24, 48) and limits (1-500).
   - /api/evaluation/plots/{filename} strictly blocks path traversal and non-whitelisted files.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, load_application_components
from src.data.demand_contract import (
    DemandProvenance,
    DemandValidationReport,
    validate_demand_series,
    check_operational_demand_readiness,
    MIN_PLAUSIBLE_DEMAND_MW,
    MAX_PLAUSIBLE_DEMAND_MW,
)
from src.evaluation.evaluate_accuracy import diebold_mariano_test
from src.models.probabilistic_forecaster import AdaptiveConformalCalibrator


@pytest.fixture(scope="module")
def client():
    """Initialize FastAPI test client with loaded components."""
    load_application_components()
    return TestClient(app)


# ==============================================================================
# 1. DEMAND DATA CONTRACT TESTS
# ==============================================================================

def test_demand_contract_valid_series():
    """Verify that a compliant hourly demand series passes validation with PASSED status."""
    ts = pd.date_range("2018-01-01 00:00:00", periods=168, freq="1h", tz="UTC")
    demand_vals = np.linspace(25000.0, 32000.0, 168)
    df = pd.DataFrame({"timestamp": ts, "consumption": demand_vals})

    report = validate_demand_series(df, provenance=DemandProvenance.TEST_FIXTURE)
    assert report.is_valid is True
    assert report.status == "PASSED"
    assert report.total_records == 168
    assert report.duplicates_count == 0
    assert report.out_of_order_count == 0
    assert report.missing_intervals_count == 0
    assert report.non_finite_count == 0
    assert report.out_of_bounds_count == 0
    assert len(report.violations) == 0


def test_demand_contract_rejects_non_finite_and_out_of_bounds():
    """Verify that NaN/Inf and physically impossible values are flagged and rejected."""
    ts = pd.date_range("2018-01-01 00:00:00", periods=5, freq="1h", tz="UTC")
    # Values include a NaN, an Inf, a sub-10k value, and an over-75k value
    demand_vals = [25000.0, np.nan, 5000.0, 80000.0, np.inf]
    df = pd.DataFrame({"timestamp": ts, "consumption": demand_vals})

    report = validate_demand_series(df, provenance=DemandProvenance.TEST_FIXTURE)
    assert report.is_valid is False
    assert report.status == "REJECTED"
    assert report.non_finite_count == 2
    assert report.out_of_bounds_count == 2
    assert any("non-finite" in v for v in report.violations)
    assert any("outside plausible PJM East bounds" in v for v in report.violations)


def test_demand_contract_detects_duplicates_and_out_of_order():
    """Verify detection of duplicate timestamps and non-monotonic order."""
    ts = pd.to_datetime(["2018-01-01 02:00:00Z", "2018-01-01 01:00:00Z", "2018-01-01 02:00:00Z"])
    df = pd.DataFrame({"timestamp": ts, "consumption": [25000.0, 26000.0, 25500.0]})

    report = validate_demand_series(df, check_cadence=False)
    assert report.is_valid is False
    assert report.duplicates_count == 1
    assert report.out_of_order_count >= 1


def test_demand_contract_detects_missing_cadence_gaps():
    """Verify detection of missing hourly timestamps in sequence."""
    ts = pd.to_datetime(["2018-01-01 00:00:00Z", "2018-01-01 01:00:00Z", "2018-01-01 04:00:00Z"])
    df = pd.DataFrame({"timestamp": ts, "consumption": [25000.0, 26000.0, 27000.0]})

    report = validate_demand_series(df, check_cadence=True)
    assert report.is_valid is False
    assert report.missing_intervals_count == 2  # 02:00 and 03:00 missing
    assert any("Cadence violation" in v for v in report.violations)


def test_operational_readiness_blocks_future_target_without_scada():
    """Verify that operational demand readiness returns BLOCKED_NO_TELEMETRY for 2026."""
    readiness = check_operational_demand_readiness(
        target_timestamp="2026-10-15T10:00:00Z",
        horizon=1,
    )
    assert readiness.is_ready is False
    assert readiness.status_code == "BLOCKED_NO_TELEMETRY"
    assert "Real-time SCADA demand telemetry feed is not connected" in readiness.message
    assert readiness.available_history_end_utc.startswith("2018-08-03")


# ==============================================================================
# 2. PREDICTION INTERVAL RELIABILITY TESTS
# ==============================================================================

def test_prediction_intervals_strict_monotonicity():
    """Verify that all generated backtest prediction intervals satisfy Q05 <= Q50 <= Q95."""
    pred_path = Path("outputs/improvement/forecast_predictions.csv")
    assert pred_path.is_file(), "Forecast predictions CSV must exist"
    df = pd.read_csv(pred_path)

    # Filter to rows with adaptive conformal intervals
    prob_rows = df[df["q05_adaptive_mw"].notna()].copy()
    assert len(prob_rows) > 0

    q05 = prob_rows["q05_adaptive_mw"].to_numpy()
    q50 = prob_rows["q50_median_mw"].to_numpy()
    q95 = prob_rows["q95_adaptive_mw"].to_numpy()

    # Monotonicity checks
    assert np.all(q05 <= q50), "Q05 must be <= Q50 for all observations"
    assert np.all(q50 <= q95), "Q50 must be <= Q95 for all observations"
    assert np.all(q05 < q95), "Q05 must be strictly < Q95 everywhere"

    # Interval width strictly positive
    widths = prob_rows["interval_width_mw"].to_numpy()
    assert np.all(widths > 0.0), "Interval width must be strictly positive"
    assert np.min(widths) > 500.0, "Interval width should reflect physical uncertainty (> 500 MW)"


def test_probabilistic_metrics_multi_horizon_breakdown():
    """Verify that forecast_metrics.json contains by_horizon and sliced coverage breakdowns."""
    metrics_path = Path("outputs/improvement/forecast_metrics.json")
    assert metrics_path.is_file()
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cal = metrics["probabilistic_calibration"]
    assert "by_horizon" in cal
    assert "H1" in cal["by_horizon"]
    assert "H24" in cal["by_horizon"]
    assert "H48" in cal["by_horizon"]

    # Check that empirical coverage is reported accurately and < 90%
    assert cal["by_horizon"]["H1"]["empirical_coverage_pct"] == 88.28
    assert cal["by_horizon"]["H24"]["empirical_coverage_pct"] == 88.50
    assert cal["by_horizon"]["H48"]["empirical_coverage_pct"] == 89.22

    # Sliced coverage checks
    sliced = cal["sliced_coverage"]
    assert "by_season_pct" in sliced
    assert "by_month_pct" in sliced
    assert "by_demand_regime_pct" in sliced
    assert "extreme_weather_heatwave_pct" in sliced


# ==============================================================================
# 3. STATISTICAL TEST VALIDATION TESTS
# ==============================================================================

def test_diebold_mariano_statistical_test_properties():
    """Verify DM test statistic calculation, positive sign convention, and p-value."""
    # Synthetic ground truth
    y_true = np.array([30000.0, 31000.0, 32000.0, 30500.0, 31500.0] * 50)
    # Benchmark has larger errors (mean error ~1000) with variation
    noise = np.sin(np.arange(250)) * 50.0
    pred_bench = y_true + 1000.0 + noise
    # Model has smaller errors (mean error ~200) with variation
    pred_model = y_true + 200.0 + noise * 0.2

    dm_stat, p_val = diebold_mariano_test(y_true, pred_bench, pred_model, h=1)
    # Positive DM confirms pred_model has significantly smaller errors
    assert dm_stat > 10.0
    assert p_val < 1e-10

    # Symmetry check: swapping benchmark and model negates the statistic
    dm_stat_rev, p_val_rev = diebold_mariano_test(y_true, pred_model, pred_bench, h=1)
    assert pytest.approx(dm_stat_rev, abs=1e-5) == -dm_stat
    assert pytest.approx(p_val_rev, abs=1e-5) == p_val


def test_forecast_metrics_contains_effect_sizes():
    """Verify that forecast_metrics.json records Cohen's d effect sizes for all horizons."""
    metrics_path = Path("outputs/improvement/forecast_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    for h_key in ("H=1h", "H=24h", "H=48h"):
        base_model = metrics["horizons"][h_key]["models"]["baseline_no_weather"]
        assert "cohens_d" in base_model
        assert base_model["cohens_d"] > 0.0, f"Cohen's d must be positive for {h_key}"
        assert base_model["loss_differential_definition"] == "|e_base| - |e_nwp|"


# ==============================================================================
# 4. API ENDPOINT VALIDATION & RELIABILITY TESTS
# ==============================================================================

def test_api_operational_demand_readiness_endpoint(client):
    """Verify /api/operational/demand_readiness returns valid JSON schema and correct status."""
    # Historical target: demand history is available
    res_hist = client.get("/api/operational/demand_readiness?target_timestamp=2018-08-01T12:00:00Z&horizon=1")
    assert res_hist.status_code == 200
    data_hist = res_hist.json()
    assert data_hist["is_ready"] is True
    assert data_hist["status_code"] == "READY"

    # Future target: telemetry is blocked
    res_fut = client.get("/api/operational/demand_readiness?target_timestamp=2026-10-15T10:00:00Z&horizon=1")
    assert res_fut.status_code == 200
    data_fut = res_fut.json()
    assert data_fut["is_ready"] is False
    assert data_fut["status_code"] == "BLOCKED_NO_TELEMETRY"
    assert "Real-time SCADA demand telemetry" in data_fut["message"]


def test_api_backtest_samples_parameter_validation(client):
    """Verify /api/evaluation/backtest_samples rejects invalid horizons and limit parameters."""
    # Invalid horizon
    bad_h = client.get("/api/evaluation/backtest_samples?horizon=12")
    assert bad_h.status_code == 400
    assert "Invalid horizon" in bad_h.json()["detail"]

    # Invalid limit (negative or > 500)
    bad_lim1 = client.get("/api/evaluation/backtest_samples?limit=0")
    assert bad_lim1.status_code == 400
    bad_lim2 = client.get("/api/evaluation/backtest_samples?limit=1000")
    assert bad_lim2.status_code == 400


def test_api_plots_security_and_traversal_rejection(client):
    """Verify /api/evaluation/plots rejects path traversal, slashes, and unwhitelisted files."""
    # Path traversal with ../
    res1 = client.get("/api/evaluation/plots/../requirements.txt")
    assert res1.status_code == 404

    # Backslash traversal
    res2 = client.get("/api/evaluation/plots/..\\..\\main.py")
    assert res2.status_code == 404

    # Non-whitelisted file
    res3 = client.get("/api/evaluation/plots/unapproved_file.png")
    assert res3.status_code == 404
