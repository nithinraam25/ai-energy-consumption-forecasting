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


class TestTemporalInformationLeakageGuards:
    """Automated assertions against future-information leakage, origin alignment, and boundary isolation."""

    @pytest.mark.parametrize("h", [1, 24, 48])
    def test_origin_target_mathematical_consistency(self, h: int):
        """Verify that target_timestamp is strictly origin + horizon for every single row."""
        import pandas as pd
        from src.features.horizon_features import create_horizon_feature_matrix

        dates = pd.date_range("2018-01-01", periods=300, freq="1h")
        dummy_df = pd.DataFrame({
            "timestamp": dates,
            "consumption": np.sin(np.linspace(0, 20, 300)) * 5000.0 + 30000.0,
        })
        feat_df = create_horizon_feature_matrix(dummy_df, horizon=h)
        assert not feat_df.empty

        diffs = feat_df["target_timestamp"] - feat_df["timestamp"]
        expected_delta = pd.Timedelta(hours=h)
        assert (diffs == expected_delta).all(), f"All rows in H={h} must have target - origin == {h} hours"

    @pytest.mark.parametrize("h", [1, 24, 48])
    def test_features_never_use_future_observations(self, h: int):
        """Verify that all lag features correspond strictly to timestamps <= t_origin = T - H."""
        import pandas as pd
        from src.features.horizon_features import create_horizon_feature_matrix, get_horizon_feature_names

        dates = pd.date_range("2018-01-01", periods=300, freq="1h")
        raw_consumption = np.arange(300, dtype=float) + 10000.0
        dummy_df = pd.DataFrame({
            "timestamp": dates,
            "consumption": raw_consumption,
        })
        feat_df = create_horizon_feature_matrix(dummy_df, horizon=h)
        feats = get_horizon_feature_names(h, include_weather=False)

        # For every row i, the value of consumption at t_origin is raw_consumption[idx]
        # Test that no lag feature has a value corresponding to t > t_origin
        for _, row in feat_df.iterrows():
            t_orig = row["timestamp"]
            t_target = row["target_timestamp"]
            assert t_orig == t_target - pd.Timedelta(hours=h)

            # In this linear sequence, consumption = index + 10000.0
            val_at_origin = dummy_df.loc[dummy_df["timestamp"] == t_orig, "consumption"].values[0]

            if f"lag_{h}" in feats:
                assert row[f"lag_{h}"] == val_at_origin, f"lag_{h} must strictly equal consumption at origin t_origin"

            # Check that all lag features are <= val_at_origin (since consumption is monotonically increasing)
            for f in feats:
                if f.startswith("lag_"):
                    lag_num = int(f.split("_")[1])
                    assert lag_num >= h, f"Feature {f} violates horizon {h}: lag number must be >= {h}"
                    assert row[f] <= val_at_origin, f"Feature {f} leaks future demand (> val_at_origin)"

    def test_chronological_boundaries_strict_isolation(self):
        """Verify chronological split guarantees non-overlapping, strictly isolated partitions."""
        import pandas as pd
        from src.evaluation.metrics import chronological_split

        dates = pd.date_range("2016-01-01", periods=1000, freq="1h")
        dummy_df = pd.DataFrame({
            "timestamp": dates,
            "val": np.arange(1000),
        })
        train_df, val_df, test_df = chronological_split(dummy_df, 0.70, 0.15, 0.15)

        assert train_df["timestamp"].max() < val_df["timestamp"].min()
        assert val_df["timestamp"].max() < test_df["timestamp"].min()
        assert len(train_df) + len(val_df) + len(test_df) == 1000


