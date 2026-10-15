"""Deterministic regression tests for Phase 2 forecast accuracy, robustness, and interval evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluate_accuracy import (
    compute_winkler_score,
    diebold_mariano_test,
)
from src.evaluation.metrics import calculate_metrics
from src.features.horizon_features import (
    SUPPORTED_HORIZONS,
    get_horizon_feature_names,
)


class TestMetricsWAPE:
    """Validate Weighted Absolute Percentage Error (WAPE) metric."""

    def test_wape_perfect_prediction(self):
        y_true = np.array([1000.0, 2000.0, 3000.0])
        y_pred = np.array([1000.0, 2000.0, 3000.0])
        res = calculate_metrics(y_true, y_pred)
        assert res.wape == 0.0
        assert res.mae == 0.0

    def test_wape_known_values(self):
        y_true = np.array([100.0, 200.0, 300.0])  # sum = 600
        y_pred = np.array([110.0, 180.0, 330.0])  # errors = 10, 20, 30 -> sum = 60
        # WAPE = 60 / 600 * 100% = 10.0%
        res = calculate_metrics(y_true, y_pred)
        assert pytest.approx(res.wape, rel=1e-5) == 10.0
        assert pytest.approx(res.mae, rel=1e-5) == 20.0

    def test_wape_scale_independence_over_mape(self):
        """WAPE avoids division by near-zero points that corrupt MAPE."""
        y_true = np.array([1.0, 10000.0])
        y_pred = np.array([2.0, 10000.0])
        res = calculate_metrics(y_true, y_pred)
        # MAPE is (100% + 0%) / 2 = 50.0%
        assert pytest.approx(res.mape, rel=1e-3) == 50.0
        # WAPE is 1.0 / 10001.0 * 100% = 0.009999%
        assert res.wape < 0.02


class TestWinklerScore:
    """Validate Winkler Interval Sharpness Score."""

    def test_winkler_inside_interval(self):
        """When y is inside [lower, upper], Winkler score equals interval width."""
        y = np.array([150.0, 200.0])
        lower = np.array([100.0, 100.0])
        upper = np.array([300.0, 300.0])
        # Width = 200 for both
        score = compute_winkler_score(y, lower, upper, alpha=0.10)
        assert pytest.approx(score, rel=1e-5) == 200.0

    def test_winkler_below_lower_penalty(self):
        """When y < lower, penalty is (2 / alpha) * (lower - y)."""
        y = np.array([80.0])  # 20 units below lower
        lower = np.array([100.0])
        upper = np.array([300.0])
        # width = 200, penalty = (2 / 0.10) * 20 = 20 * 20 = 400
        # total = 200 + 400 = 600
        score = compute_winkler_score(y, lower, upper, alpha=0.10)
        assert pytest.approx(score, rel=1e-5) == 600.0

    def test_winkler_above_upper_penalty(self):
        """When y > upper, penalty is (2 / alpha) * (y - upper)."""
        y = np.array([350.0])  # 50 units above upper
        lower = np.array([100.0])
        upper = np.array([300.0])
        # width = 200, penalty = (2 / 0.10) * 50 = 20 * 50 = 1000
        # total = 200 + 1000 = 1200
        score = compute_winkler_score(y, lower, upper, alpha=0.10)
        assert pytest.approx(score, rel=1e-5) == 1200.0

    def test_winkler_empty_or_nan_guard(self):
        y = np.array([np.nan, 200.0])
        lower = np.array([100.0, 100.0])
        upper = np.array([300.0, 300.0])
        score = compute_winkler_score(y, lower, upper)
        # Should filter out NaN row and evaluate on valid row
        assert pytest.approx(score, rel=1e-5) == 200.0


class TestDieboldMariano:
    """Validate Diebold-Mariano test with Newey-West lag adjustment."""

    def test_dm_identical_forecasts(self):
        y = np.sin(np.linspace(0, 10, 100)) + 10.0
        p1 = y + np.random.RandomState(42).normal(0, 0.1, 100)
        p2 = p1.copy()
        stat, p_val = diebold_mariano_test(y, p1, p2, h=1)
        assert stat == 0.0
        assert p_val == 1.0

    def test_dm_superior_forecast_detection(self):
        """Better model has negative DM statistic when placed as model 2."""
        rng = np.random.RandomState(42)
        y = np.sin(np.linspace(0, 20, 500)) + 10.0
        bad_pred = y + rng.normal(0, 5.0, 500)
        good_pred = y + rng.normal(0, 0.5, 500)
        # d_t = e_bad^2 - e_good^2 > 0 -> DM statistic > 0 (or vice versa)
        stat, p_val = diebold_mariano_test(y, bad_pred, good_pred, h=1)
        assert stat > 0  # bad_pred has significantly larger error than good_pred
        assert p_val < 0.001


class TestTemporalCausalityGuards:
    """Verify that horizon feature matrices never leak future demand lags."""

    def test_supported_horizons(self):
        assert tuple(SUPPORTED_HORIZONS) == (1, 24, 48)

    def test_h1_feature_set(self):
        feats_w = get_horizon_feature_names(1, include_weather=True)
        # lag_1 is origin t_origin = T - 1, which is admissible
        assert "lag_1" in feats_w
        assert "lag_24" in feats_w
        assert "lag_168" in feats_w
        assert "temp_f_fcst" in feats_w

    def test_h24_feature_set_forbids_future_lags(self):
        feats_w = get_horizon_feature_names(24, include_weather=True)
        # For H=24, origin is T - 24.
        # lag_1 (T - 1) is 23 hours in the future and strictly FORBIDDEN
        assert "lag_1" not in feats_w
        assert "lag_24" in feats_w
        assert "lag_48" in feats_w
        assert "lag_168" in feats_w

    def test_h48_feature_set_forbids_future_lags(self):
        feats_w = get_horizon_feature_names(48, include_weather=True)
        # For H=48, origin is T - 48.
        # lag_1 (T - 1) and lag_24 (T - 24) are strictly in the future and FORBIDDEN
        assert "lag_1" not in feats_w
        assert "lag_24" not in feats_w
        assert "lag_48" in feats_w
        assert "lag_72" in feats_w
        assert "lag_168" in feats_w


class TestPhase2Deliverables:
    """Verify that Phase 2 evaluation artifacts exist, are valid, and satisfy scientific invariants."""

    def test_metrics_json_structure(self):
        metrics_file = Path("outputs/improvement/forecast_metrics.json")
        assert metrics_file.is_file(), "forecast_metrics.json must exist"

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "horizons" in data
        assert "probabilistic_calibration" in data

        for h_key in ["H=1h", "H=24h", "H=48h"]:
            assert h_key in data["horizons"]
            h_data = data["horizons"][h_key]
            assert h_data["test_sample_count"] > 20000
            models = h_data["models"]

            for m_key in ["production_nwp", "baseline_no_weather", "persistence", "weekly_seasonal_naive"]:
                assert m_key in models
                m_metrics = models[m_key]
                assert "mae_mw" in m_metrics
                assert "rmse_mw" in m_metrics
                assert "r2" in m_metrics
                assert "wape_pct" in m_metrics

            # NWP skill gain must be positive (skill improvement over no-weather)
            assert models["baseline_no_weather"]["gain_nwp_pct"] > 0
            assert models["baseline_no_weather"]["statistically_significant"] is True

        # Adaptive Conformal Calibration
        prob = data["probabilistic_calibration"]
        assert prob["nominal_target_coverage"] == 0.90
        assert prob["raw_quantiles"]["ordering_violations"] == 0
        assert prob["champion_adaptive_diurnal_conformal"]["ordering_violations"] == 0
        assert prob["champion_adaptive_diurnal_conformal"]["coverage_pct"] >= 88.0

    def test_predictions_csv_monotonicity(self):
        csv_file = Path("outputs/improvement/forecast_predictions.csv")
        assert csv_file.is_file(), "forecast_predictions.csv must exist"

        import pandas as pd
        df = pd.read_csv(csv_file)
        assert len(df) > 60000

        # For H=1 rows with conformal intervals, assert strict monotonicity
        h1_prob = df[df["horizon"] == 1].dropna(subset=["q05_adaptive_mw", "q50_median_mw", "q95_adaptive_mw"])
        assert len(h1_prob) > 20000
        # No crossing violations
        violations = ((h1_prob["q05_adaptive_mw"] > h1_prob["q50_median_mw"]) |
                      (h1_prob["q50_median_mw"] > h1_prob["q95_adaptive_mw"])).sum()
        assert violations == 0

