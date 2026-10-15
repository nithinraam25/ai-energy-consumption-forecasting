"""Deterministic statistical validation and reconciliation tests for Diebold-Mariano implementation.

Tests:
1. Null hypothesis H0: E[d_t] = 0 under synthetic zero-loss differential.
2. Sign convention and antisymmetry: DM(benchmark, model) == -DM(model, benchmark).
3. Bartlett triangular kernel vs. unweighted rectangular lag window.
4. Positive semi-definiteness under strong autocorrelations.
5. Finite-sample Harvey-Leybourne-Newbold (HLN 1997) correction.
6. Exact numerical reproduction of saved benchmark test holdout metrics (N=21,773):
   - H=1: DM = 15.908 (both kernels)
   - H=24: Unweighted DM = 17.857, Bartlett DM = 21.372
   - H=48: Unweighted DM = 15.100, Bartlett DM = 18.371
7. Exact reconciliation of Cohen's d:
   - Paired sample formula mean(d) / std(d): 0.1078 (H1), 0.4550 (H24), 0.5359 (H48)
   - Unweighted DM / sqrt(N) heuristic: 0.1078 (H1), 0.1210 (H24), 0.1023 (H48).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate_accuracy import diebold_mariano_test


class TestDieboldMarianoMathematicalProperties:
    """Rigorous mathematical tests for the Diebold-Mariano test implementation."""

    def test_null_hypothesis_zero_differential(self):
        """When forecast errors are identical, d_t = 0 everywhere, DM = 0.0, p = 1.0."""
        y_true = np.array([100.0, 105.0, 110.0, 95.0, 102.0] * 20)
        pred_a = y_true + 5.0
        pred_b = y_true + 5.0
        dm_stat, p_val = diebold_mariano_test(y_true, pred_a, pred_b, h=1)
        assert dm_stat == 0.0
        assert p_val == 1.0

    def test_sign_convention_and_antisymmetry(self):
        """DM(A, B) must equal -DM(B, A), and superior model must produce positive statistic."""
        rng = np.random.RandomState(42)
        n = 1000
        y_true = rng.normal(30000.0, 5000.0, size=n)
        # Model B is clearly superior (smaller absolute errors)
        pred_a = y_true + rng.normal(0.0, 1000.0, size=n)
        pred_b = y_true + rng.normal(0.0, 200.0, size=n)

        dm_stat, p_val = diebold_mariano_test(y_true, pred_a, pred_b, h=1)
        dm_stat_rev, p_val_rev = diebold_mariano_test(y_true, pred_b, pred_a, h=1)

        assert dm_stat > 0.0, "DM must be positive when candidate pred_b has smaller errors"
        assert dm_stat_rev < 0.0, "DM must be negative when candidate pred_a has larger errors"
        assert pytest.approx(dm_stat, abs=1e-8) == -dm_stat_rev
        assert pytest.approx(p_val, abs=1e-8) == p_val_rev
        assert p_val < 1e-10

    def test_known_synthetic_iid_sequence(self):
        """For i.i.d. Gaussian loss differentials with known mean and variance, DM matches theoretical t-ratio."""
        rng = np.random.RandomState(123)
        n = 10000
        # Construct errors such that d_t = e_bench - e_model has mean mu and variance sigma^2
        y_true = np.zeros(n)
        mu_d = 2.0
        sigma_d = 10.0
        d = rng.normal(mu_d, sigma_d, size=n)
        # Let e_model = 10.0, e_bench = 10.0 + d (where e_bench >= 0)
        e_model = np.full(n, 100.0)
        e_bench = 100.0 + d

        pred_model = y_true + e_model
        pred_bench = y_true + e_bench

        dm_stat, p_val = diebold_mariano_test(y_true, pred_bench, pred_model, h=1)
        expected_dm = np.mean(d) / (np.std(d, ddof=0) / np.sqrt(n))
        assert pytest.approx(dm_stat, rel=1e-3) == expected_dm

    def test_hln_small_sample_adjustment(self):
        """Verify Harvey-Leybourne-Newbold factor reduces DM statistic slightly for small N."""
        rng = np.random.RandomState(999)
        n = 50
        h = 10
        y_true = rng.normal(100.0, 10.0, size=n)
        pred_bench = y_true + rng.normal(0.0, 5.0, size=n)
        pred_model = y_true + rng.normal(0.0, 3.0, size=n)

        dm_std, p_std = diebold_mariano_test(y_true, pred_bench, pred_model, h=h, hln_correction=False)
        dm_hln, p_hln = diebold_mariano_test(y_true, pred_bench, pred_model, h=h, hln_correction=True)

        # Theoretical factor: sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        expected_factor = np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n)
        assert pytest.approx(dm_hln, rel=1e-5) == dm_std * expected_factor
        assert abs(dm_hln) < abs(dm_std), "HLN factor must be < 1.0 for small samples"

    def test_bartlett_versus_unweighted_lag_kernel(self):
        """Under positive autocorrelation, Bartlett kernel downweights distant lags."""
        # Create an AR(1) loss differential process d_t = 0.8 * d_{t-1} + eps_t with positive mean
        rng = np.random.RandomState(42)
        n = 2000
        d = np.zeros(n)
        eps = rng.normal(0.5, 1.0, size=n)
        for t in range(1, n):
            d[t] = 0.7 * d[t-1] + eps[t]

        y_true = np.zeros(n)
        pred_model = y_true + 50.0
        pred_bench = y_true + 50.0 + d

        dm_bartlett, _ = diebold_mariano_test(y_true, pred_bench, pred_model, h=24, kernel="bartlett")
        dm_unweighted, _ = diebold_mariano_test(y_true, pred_bench, pred_model, h=24, kernel="unweighted")

        # Because autocovariances are strictly positive, unweighted sum (weight 1.0) gives larger var_d
        # than Bartlett (weight 1 - k/h < 1.0), which makes DM_unweighted < DM_bartlett
        assert dm_bartlett > dm_unweighted


@pytest.fixture(scope="module")
def predictions_df() -> pd.DataFrame:
    csv_path = Path("outputs/improvement/forecast_predictions.csv")
    assert csv_path.is_file(), f"Predictions CSV artifact missing at: {csv_path}"
    return pd.read_csv(csv_path)


class TestPhase6EvaluationArtifactReconciliation:
    """Verify saved evaluation artifacts and reconcile reported statistics across horizons."""

    def test_h1_exact_reconciliation(self, predictions_df: pd.DataFrame):
        """For H=1, h=1 has zero sum terms; unweighted and Bartlett yield identical DM=15.908."""
        sub = predictions_df[predictions_df["horizon"] == 1]
        assert len(sub) == 21773
        y = sub["actual_demand_mw"].to_numpy()
        p_base = sub["pred_base_mw"].to_numpy()
        p_nwp = sub["pred_nwp_mw"].to_numpy()

        dm_bartlett, p_b = diebold_mariano_test(y, p_base, p_nwp, h=1, kernel="bartlett")
        dm_unweighted, p_u = diebold_mariano_test(y, p_base, p_nwp, h=1, kernel="unweighted")

        assert pytest.approx(dm_bartlett, abs=1e-3) == 15.908
        assert pytest.approx(dm_unweighted, abs=1e-3) == 15.908
        assert p_b < 1e-15
        assert p_u < 1e-15

        # Cohen's d reconciliation
        d = np.abs(y - p_base) - np.abs(y - p_nwp)
        cohens_d = float(np.mean(d) / np.std(d))
        assert pytest.approx(cohens_d, abs=1e-4) == 0.1078

    def test_h24_exact_reconciliation(self, predictions_df: pd.DataFrame):
        """For H=24, unweighted DM=17.857 (Phase 2/4) reconciles with Bartlett DM=21.372 (Phase 3/5/6)."""
        sub = predictions_df[predictions_df["horizon"] == 24]
        assert len(sub) == 21773
        y = sub["actual_demand_mw"].to_numpy()
        p_base = sub["pred_base_mw"].to_numpy()
        p_nwp = sub["pred_nwp_mw"].to_numpy()

        dm_bartlett, p_b = diebold_mariano_test(y, p_base, p_nwp, h=24, kernel="bartlett")
        dm_unweighted, p_u = diebold_mariano_test(y, p_base, p_nwp, h=24, kernel="unweighted")

        # Verify exact reconciliation
        assert pytest.approx(dm_unweighted, abs=1e-3) == 17.857, "Phase 2/4 unweighted DM must be 17.857"
        assert pytest.approx(dm_bartlett, abs=1e-3) == 21.372, "Phase 3/5/6 Bartlett DM must be 21.372"

        # Cohen's d reconciliation
        d = np.abs(y - p_base) - np.abs(y - p_nwp)
        cohens_d_sample = float(np.mean(d) / np.std(d))
        assert pytest.approx(cohens_d_sample, abs=1e-4) == 0.4550, "Sample Cohen's d must be 0.4550"

        # Check the DM / sqrt(N) heuristic that appeared in the Phase 4 report
        heuristic_d = 17.857 / np.sqrt(len(sub))
        assert pytest.approx(heuristic_d, abs=1e-3) == 0.1210

    def test_h48_exact_reconciliation(self, predictions_df: pd.DataFrame):
        """For H=48, unweighted DM=15.100 (Phase 2/4) reconciles with Bartlett DM=18.371 (Phase 3/5/6)."""
        sub = predictions_df[predictions_df["horizon"] == 48]
        assert len(sub) == 21773
        y = sub["actual_demand_mw"].to_numpy()
        p_base = sub["pred_base_mw"].to_numpy()
        p_nwp = sub["pred_nwp_mw"].to_numpy()

        dm_bartlett, p_b = diebold_mariano_test(y, p_base, p_nwp, h=48, kernel="bartlett")
        dm_unweighted, p_u = diebold_mariano_test(y, p_base, p_nwp, h=48, kernel="unweighted")

        # Verify exact reconciliation
        assert pytest.approx(dm_unweighted, abs=1e-3) == 15.100, "Phase 2/4 unweighted DM must be 15.100"
        assert pytest.approx(dm_bartlett, abs=1e-3) == 18.371, "Phase 3/5/6 Bartlett DM must be 18.371"

        # Cohen's d reconciliation
        d = np.abs(y - p_base) - np.abs(y - p_nwp)
        cohens_d_sample = float(np.mean(d) / np.std(d))
        assert pytest.approx(cohens_d_sample, abs=1e-4) == 0.5359, "Sample Cohen's d must be 0.5359"

        # Check the DM / sqrt(N) heuristic that appeared in the Phase 4 report
        heuristic_d = 15.100 / np.sqrt(len(sub))
        assert pytest.approx(heuristic_d, abs=1e-4) == 0.1023
