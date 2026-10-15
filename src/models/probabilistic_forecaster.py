"""Probabilistic and Conformal Forecasting Engine for Energy Consumption.

Provides:
- Multi-quantile gradient boosting regression (Q05, Q50, Q95) via HistGradientBoostingRegressor.
- Point forecasting via mean squared error regression.
- Conformalized Quantile Regression (CQR) interval calibration on isolated calibration partitions.
- Quantile monotonicity enforcement (lower <= median <= upper) to eliminate quantile crossing.
- Rigorous diagnostic evaluation: empirical coverage, mean/median interval width,
  and stratified coverage by season, hour of day, demand quantile, and forecast horizon.
- Persistence and loading of probabilistic models and conformal calibration artifacts.
"""

from pathlib import Path
from typing import Any, Sequence
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import RANDOM_STATE, MODELS_DIR
from src.models.base_model import BaseForecaster
from src.models.tree_models import FORBIDDEN_FEATURE_KEYWORDS

DEFAULT_PROBABILISTIC_MODEL_PATH: Path = MODELS_DIR / "pjm_probabilistic_forecaster_phase9.joblib"
DEFAULT_CALIBRATION_ARTIFACT_PATH: Path = MODELS_DIR / "conformal_calibration_params.json"


class MultiQuantileForecaster:
    """Multi-quantile tree forecaster producing point and quantile forecasts (Q05, Q50, Q95)."""

    def __init__(
        self,
        feature_names: Sequence[str],
        quantiles: Sequence[float] = (0.05, 0.50, 0.95),
        random_state: int = RANDOM_STATE,
        max_iter: int = 150,
        learning_rate: float = 0.05,
        max_depth: int | None = 8,
        max_leaf_nodes: int | None = 45,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Initialize multi-quantile forecaster.

        Args:
            feature_names: Predictor feature names.
            quantiles: Quantile levels to estimate. Defaults to (0.05, 0.50, 0.95).
            random_state: Random state for reproducibility.
            max_iter: Number of boosting iterations.
            learning_rate: Shrinkage step size.
            max_depth: Maximum tree depth.
            max_leaf_nodes: Maximum leaves per tree.
            min_samples_leaf: Minimum samples per leaf.
            l2_regularization: L2 regularization strength.
            **kwargs: Extra arguments forwarded to HistGradientBoostingRegressor.
        """
        if not feature_names:
            raise ValueError("feature_names cannot be empty.")

        for feat in feature_names:
            clean_feat = str(feat).strip().lower()
            if clean_feat in FORBIDDEN_FEATURE_KEYWORDS or clean_feat.startswith("target"):
                raise ValueError(f"Target leakage violation: '{feat}' is forbidden in features.")

        self.feature_names = list(feature_names)
        self.quantiles = sorted(list(quantiles))
        self.random_state = random_state
        self.params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            **kwargs,
        }

        # Estimators dict: quantile -> HistGradientBoostingRegressor
        self.quantile_models_: dict[float, HistGradientBoostingRegressor] = {}
        self.point_model_: HistGradientBoostingRegressor | None = None
        self.is_fitted: bool = False

    def _prepare_features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Validate and extract feature matrix."""
        if isinstance(X, pd.DataFrame):
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Feature matrix missing required features: {sorted(missing)}")
            X_df = X[self.feature_names]
            X_mat = X_df.to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"Feature matrix has {X.shape[1]} columns, expected {len(self.feature_names)}")
            X_mat = X.astype(float)
        else:
            raise TypeError(f"Unsupported feature type: {type(X)}")

        if np.isnan(X_mat).any() or np.isinf(X_mat).any():
            nan_cnt = int(np.isnan(X_mat).sum())
            inf_cnt = int(np.isinf(X_mat).sum())
            raise ValueError(f"Non-finite values detected in feature matrix (NaN: {nan_cnt}, Inf: {inf_cnt})")

        return X_mat

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "MultiQuantileForecaster":
        """Fit point and quantile regression estimators.

        Args:
            X: Input features.
            y: Energy demand target values.

        Returns:
            self: Fitted instance.
        """
        X_mat = self._prepare_features(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        if len(X_mat) != len(y_arr):
            raise ValueError(f"X rows ({len(X_mat)}) != y rows ({len(y_arr)})")

        if np.isnan(y_arr).any() or np.isinf(y_arr).any():
            raise ValueError("Target array contains non-finite values (NaN / Inf).")

        # 1. Fit Point Model (mean squared error)
        self.point_model_ = HistGradientBoostingRegressor(
            loss="squared_error",
            random_state=self.random_state,
            **self.params,
        )
        self.point_model_.fit(X_mat, y_arr)

        # 2. Fit Quantile Models
        self.quantile_models_ = {}
        for q in self.quantiles:
            qm = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                random_state=self.random_state + int(q * 100),
                **self.params,
            )
            qm.fit(X_mat, y_arr)
            self.quantile_models_[q] = qm

        self.is_fitted = True
        return self

    def predict_raw_quantiles(self, X: pd.DataFrame | np.ndarray) -> dict[float, np.ndarray]:
        """Generate unconstrained quantile predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_mat = self._prepare_features(X)
        return {q: self.quantile_models_[q].predict(X_mat) for q in self.quantiles}

    def predict(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        """Generate point and monotonicity-enforced quantile forecasts.

        Enforces: Q05 <= Q50 <= Q95 across all samples to prevent quantile crossing.

        Returns:
            dict containing:
                'point': Mean prediction (MW)
                'q05': 5th percentile (MW)
                'q50': 50th percentile / median (MW)
                'q95': 95th percentile (MW)
                'interval_width': q95 - q05 (MW)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_mat = self._prepare_features(X)

        point_preds = self.point_model_.predict(X_mat)
        raw_q = {q: self.quantile_models_[q].predict(X_mat) for q in self.quantiles}

        q05 = raw_q.get(0.05, point_preds)
        q50 = raw_q.get(0.50, point_preds)
        q95 = raw_q.get(0.95, point_preds)

        # Monotonic sorting to enforce Q05 <= Q50 <= Q95
        q_matrix = np.column_stack([q05, q50, q95])
        q_sorted = np.sort(q_matrix, axis=1)

        q05_clean = q_sorted[:, 0]
        q50_clean = q_sorted[:, 1]
        q95_clean = q_sorted[:, 2]

        return {
            "point": point_preds,
            "q05": q05_clean,
            "q50": q50_clean,
            "q95": q95_clean,
            "interval_width": q95_clean - q05_clean,
        }

    def save(self, file_path: Path | str = DEFAULT_PROBABILISTIC_MODEL_PATH) -> Path:
        """Serialize model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, file_path: Path | str = DEFAULT_PROBABILISTIC_MODEL_PATH) -> "MultiQuantileForecaster":
        """Load serialized model from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}: {type(model)}")
        return model


class ConformalIntervalCalibrator:
    """Conformalized Quantile Regression (CQR) calibrator.

    Calculates nonconformity scores on an isolated calibration partition:
        E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))
    and computes the finite-sample empirical correction factor s to guarantee
    target nominal coverage under exchangeability.
    """

    def __init__(self, nominal_coverage: float = 0.90) -> None:
        """Initialize calibrator.

        Args:
            nominal_coverage: Desired target coverage (e.g. 0.90 for 90% intervals).
        """
        if not (0.50 < nominal_coverage < 1.0):
            raise ValueError(f"Nominal coverage must be in (0.50, 1.0), got {nominal_coverage}")
        self.nominal_coverage = float(nominal_coverage)
        self.alpha = 1.0 - self.nominal_coverage
        self.adjustment_factor: float = 0.0
        self.calibration_sample_count: int = 0
        self.raw_calibration_coverage: float = 0.0
        self.calibrated_calibration_coverage: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(
        self,
        y_true: pd.Series | np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
    ) -> "ConformalIntervalCalibrator":
        """Compute conformal calibration adjustment factor on validation/calibration data.

        Args:
            y_true: Ground truth target demand values.
            q_low: Lower quantile predictions (e.g. Q05).
            q_high: Upper quantile predictions (e.g. Q95).

        Returns:
            self: Calibrated instance.
        """
        y_arr = np.asarray(y_true, dtype=float).ravel()
        q_l = np.asarray(q_low, dtype=float).ravel()
        q_h = np.asarray(q_high, dtype=float).ravel()

        n = len(y_arr)
        if not (len(q_l) == len(q_h) == n):
            raise ValueError(f"Dimension mismatch: y ({n}), q_low ({len(q_l)}), q_high ({len(q_h)})")

        if np.isnan(y_arr).any() or np.isnan(q_l).any() or np.isnan(q_h).any():
            raise ValueError("Calibration inputs contain non-finite values.")

        # Nonconformity score: signed distance outside quantile interval
        # E_i <= 0 if y_i in [q_l, q_h]; E_i > 0 if y_i falls outside interval
        nonconformity = np.maximum(q_l - y_arr, y_arr - q_h)

        # Conformal quantile level with finite-sample correction: ceil((n+1)*(1-alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / float(n))
        adjustment = float(np.quantile(nonconformity, level))

        self.adjustment_factor = adjustment
        self.calibration_sample_count = n
        self.raw_calibration_coverage = float(np.mean((y_arr >= q_l) & (y_arr <= q_h)))

        adj_low = q_l - adjustment
        adj_high = q_h + adjustment
        self.calibrated_calibration_coverage = float(np.mean((y_arr >= adj_low) & (y_arr <= adj_high)))

        self.is_calibrated = True
        return self

    def adjust(
        self,
        q_low: np.ndarray,
        q_median: np.ndarray,
        q_high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply conformal correction factor and enforce quantile monotonicity.

        Args:
            q_low: Raw lower quantile predictions.
            q_median: Raw median predictions.
            q_high: Raw upper quantile predictions.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                (calibrated_low, median, calibrated_high) guaranteed lower <= median <= upper.
        """
        if not self.is_calibrated:
            raise RuntimeError("Calibrator is not calibrated. Call calibrate() first.")

        adj_low = q_low - self.adjustment_factor
        adj_high = q_high + self.adjustment_factor

        # Monotonicity guard: ensure adj_low <= q_median <= adj_high
        m_matrix = np.column_stack([adj_low, q_median, adj_high])
        m_sorted = np.sort(m_matrix, axis=1)

        return m_sorted[:, 0], m_sorted[:, 1], m_sorted[:, 2]

    def save(self, file_path: Path | str = DEFAULT_CALIBRATION_ARTIFACT_PATH) -> Path:
        """Save calibration metadata to JSON."""
        if not self.is_calibrated:
            raise RuntimeError("Cannot save uncalibrated parameters.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nominal_coverage": self.nominal_coverage,
            "alpha": self.alpha,
            "adjustment_factor_mw": round(self.adjustment_factor, 4),
            "calibration_sample_count": self.calibration_sample_count,
            "raw_calibration_coverage": round(self.raw_calibration_coverage, 4),
            "calibrated_calibration_coverage": round(self.calibrated_calibration_coverage, 4),
            "epistemic_limitation": (
                "Conformal coverage guarantees strictly assume exchangeability. Serial autocorrelation "
                "and seasonal distribution shifts in energy time-series yield approximate rather than "
                "exact marginal coverage on non-stationary test partitions."
            ),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def load(cls, file_path: Path | str = DEFAULT_CALIBRATION_ARTIFACT_PATH) -> "ConformalIntervalCalibrator":
        """Load calibration parameters from JSON."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(nominal_coverage=data["nominal_coverage"])
        instance.adjustment_factor = float(data["adjustment_factor_mw"])
        instance.calibration_sample_count = int(data["calibration_sample_count"])
        instance.raw_calibration_coverage = float(data["raw_calibration_coverage"])
        instance.calibrated_calibration_coverage = float(data["calibrated_calibration_coverage"])
        instance.is_calibrated = True
        return instance


DEFAULT_ADAPTIVE_CALIBRATION_PATH: Path = MODELS_DIR / "adaptive_calibration_params.json"


class AdaptiveConformalCalibrator:
    """Adaptive Conformalized Quantile Regression (CQR) Calibrator.

    Extends scalar conformal prediction by conditioning the nonconformity
    adjustment factor on operational covariates (time-of-day, demand regime, season,
    temperature regime, or rolling chronological windows):
        adj_low = q_low - eta(g(x))
        adj_high = q_high + eta(g(x))

    Provides narrower prediction intervals during low-uncertainty conditions (e.g.
    overnight base load) while maintaining target coverage during volatile conditions
    (e.g. morning ramps, summer heatwaves).
    """

    SUPPORTED_METHODS = [
        "GLOBAL_SCALAR",
        "DIURNAL",
        "DEMAND_REGIME",
        "SEASONAL",
        "WEATHER_REGIME",
        "ROLLING_WINDOW",
    ]

    def __init__(
        self,
        nominal_coverage: float = 0.90,
        method: str = "DIURNAL",
        rolling_window_size: int = 1000,
        min_group_samples: int = 50,
    ) -> None:
        """Initialize adaptive calibrator.

        Args:
            nominal_coverage: Target interval coverage. Defaults to 0.90 (90%).
            method: Conditioning strategy ('GLOBAL_SCALAR', 'DIURNAL', 'DEMAND_REGIME',
                'SEASONAL', 'WEATHER_REGIME', 'ROLLING_WINDOW'). Defaults to 'DIURNAL'.
            rolling_window_size: Window size when method='ROLLING_WINDOW'.
            min_group_samples: Minimum group sample size before falling back to global scalar.
        """
        if not (0.50 < nominal_coverage < 1.0):
            raise ValueError(f"Nominal coverage must be in (0.50, 1.0), got {nominal_coverage}")
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{method}'. Choose from {self.SUPPORTED_METHODS}")

        self.nominal_coverage = float(nominal_coverage)
        self.alpha = 1.0 - self.nominal_coverage
        self.method = method
        self.rolling_window_size = int(rolling_window_size)
        self.min_group_samples = int(min_group_samples)

        self.global_adjustment: float = 0.0
        self.group_adjustments: dict[str, float] = {}
        self.group_sample_counts: dict[str, int] = {}
        self.group_raw_coverages: dict[str, float] = {}
        self.group_calibrated_coverages: dict[str, float] = {}

        self.overall_raw_coverage: float = 0.0
        self.overall_calibrated_coverage: float = 0.0
        self.total_calibration_samples: int = 0
        self.is_calibrated: bool = False

    def _assign_groups(
        self,
        n: int,
        timestamps: Sequence[Any] | None = None,
        demand_preds: np.ndarray | None = None,
        temperatures: np.ndarray | None = None,
    ) -> list[str]:
        """Map each observation to its conditioning group label."""
        if self.method == "GLOBAL_SCALAR":
            return ["GLOBAL"] * n

        if self.method == "DIURNAL":
            if timestamps is None or len(timestamps) != n:
                return ["GLOBAL"] * n
            ts = pd.DatetimeIndex(timestamps)
            return [f"H{h:02d}" for h in ts.hour]

        if self.method == "SEASONAL":
            if timestamps is None or len(timestamps) != n:
                return ["GLOBAL"] * n
            ts = pd.DatetimeIndex(timestamps)
            season_map = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Fall", 10: "Fall", 11: "Fall",
            }
            return [season_map.get(m, "Other") for m in ts.month]

        if self.method == "DEMAND_REGIME":
            if demand_preds is None or len(demand_preds) != n:
                return ["GLOBAL"] * n
            p = np.asarray(demand_preds)
            # Define 3 operational demand regimes
            groups = []
            for val in p:
                if val < 28000.0:
                    groups.append("BASE_LOAD")
                elif val <= 35000.0:
                    groups.append("INTERMEDIATE_LOAD")
                else:
                    groups.append("PEAK_LOAD")
            return groups

        if self.method == "WEATHER_REGIME":
            if temperatures is None or len(temperatures) != n:
                return ["GLOBAL"] * n
            t = np.asarray(temperatures)
            groups = []
            for val in t:
                if val < 50.0:
                    groups.append("COLD_HEATING")
                elif val <= 75.0:
                    groups.append("MILD_NEUTRAL")
                else:
                    groups.append("HOT_COOLING")
            return groups

        if self.method == "ROLLING_WINDOW":
            # For rolling window, group by block chunks
            chunk_size = max(100, self.rolling_window_size)
            return [f"BLOCK_{i // chunk_size}" for i in range(n)]

        return ["GLOBAL"] * n

    def calibrate(
        self,
        y_true: pd.Series | np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
        timestamps: Sequence[Any] | None = None,
        demand_preds: np.ndarray | None = None,
        temperatures: np.ndarray | None = None,
    ) -> "AdaptiveConformalCalibrator":
        """Fit adaptive conformal adjustment factors across conditioning groups on calibration data."""
        y_arr = np.asarray(y_true, dtype=float).ravel()
        q_l = np.asarray(q_low, dtype=float).ravel()
        q_h = np.asarray(q_high, dtype=float).ravel()
        n = len(y_arr)

        if not (len(q_l) == len(q_h) == n):
            raise ValueError(f"Dimension mismatch: y ({n}), q_low ({len(q_l)}), q_high ({len(q_h)})")

        if np.isnan(y_arr).any() or np.isnan(q_l).any() or np.isnan(q_h).any():
            raise ValueError("Calibration inputs contain non-finite values.")

        # Compute signed nonconformity scores
        nonconformity = np.maximum(q_l - y_arr, y_arr - q_h)

        # Global baseline adjustment
        level_global = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / float(n))
        self.global_adjustment = float(np.quantile(nonconformity, level_global))

        # Assign groups
        groups = self._assign_groups(
            n=n, timestamps=timestamps, demand_preds=demand_preds, temperatures=temperatures
        )
        groups_arr = np.array(groups)

        self.group_adjustments = {}
        self.group_sample_counts = {}
        self.group_raw_coverages = {}
        self.group_calibrated_coverages = {}

        unique_groups = sorted(list(set(groups)))
        for g in unique_groups:
            mask = (groups_arr == g)
            g_count = int(np.sum(mask))
            self.group_sample_counts[g] = g_count

            g_nonconf = nonconformity[mask]
            g_raw_cov = float(np.mean((y_arr[mask] >= q_l[mask]) & (y_arr[mask] <= q_h[mask])))
            self.group_raw_coverages[g] = round(g_raw_cov, 4)

            if g_count >= self.min_group_samples:
                lvl = min(1.0, np.ceil((g_count + 1) * (1.0 - self.alpha)) / float(g_count))
                g_adj = float(np.quantile(g_nonconf, lvl))
            else:
                g_adj = self.global_adjustment

            self.group_adjustments[g] = round(g_adj, 4)
            g_cal_cov = float(np.mean(
                (y_arr[mask] >= (q_l[mask] - g_adj)) & (y_arr[mask] <= (q_h[mask] + g_adj))
            ))
            self.group_calibrated_coverages[g] = round(g_cal_cov, 4)

        self.total_calibration_samples = n
        self.overall_raw_coverage = float(np.mean((y_arr >= q_l) & (y_arr <= q_h)))

        # Evaluate overall calibrated coverage on calibration set
        adj_per_sample = np.array([self.group_adjustments.get(g, self.global_adjustment) for g in groups])
        cal_l = q_l - adj_per_sample
        cal_h = q_h + adj_per_sample
        self.overall_calibrated_coverage = float(np.mean((y_arr >= cal_l) & (y_arr <= cal_h)))

        self.is_calibrated = True
        return self

    def adjust(
        self,
        q_low: np.ndarray,
        q_median: np.ndarray,
        q_high: np.ndarray,
        timestamps: Sequence[Any] | None = None,
        demand_preds: np.ndarray | None = None,
        temperatures: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply group-specific conformal adjustments and enforce non-crossing monotonicity."""
        if not self.is_calibrated:
            raise RuntimeError("Calibrator is not calibrated. Call calibrate() first.")

        q_l = np.asarray(q_low, dtype=float).ravel()
        q_m = np.asarray(q_median, dtype=float).ravel()
        q_h = np.asarray(q_high, dtype=float).ravel()
        n = len(q_l)

        groups = self._assign_groups(
            n=n, timestamps=timestamps, demand_preds=demand_preds, temperatures=temperatures
        )
        adj_factors = np.array([self.group_adjustments.get(g, self.global_adjustment) for g in groups])

        adj_low = q_l - adj_factors
        adj_high = q_h + adj_factors

        # Monotonicity guard: ensure adj_low <= q_median <= adj_high
        m_matrix = np.column_stack([adj_low, q_m, adj_high])
        m_sorted = np.sort(m_matrix, axis=1)

        return m_sorted[:, 0], m_sorted[:, 1], m_sorted[:, 2]

    def save(self, file_path: Path | str = DEFAULT_ADAPTIVE_CALIBRATION_PATH) -> Path:
        """Save adaptive calibration parameters to JSON."""
        if not self.is_calibrated:
            raise RuntimeError("Cannot save uncalibrated parameters.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nominal_coverage": self.nominal_coverage,
            "method": self.method,
            "global_adjustment_mw": round(self.global_adjustment, 4),
            "group_adjustments_mw": self.group_adjustments,
            "group_sample_counts": self.group_sample_counts,
            "group_raw_coverages": self.group_raw_coverages,
            "group_calibrated_coverages": self.group_calibrated_coverages,
            "total_calibration_samples": self.total_calibration_samples,
            "overall_raw_coverage": round(self.overall_raw_coverage, 4),
            "overall_calibrated_coverage": round(self.overall_calibrated_coverage, 4),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def load(cls, file_path: Path | str = DEFAULT_ADAPTIVE_CALIBRATION_PATH) -> "AdaptiveConformalCalibrator":
        """Load adaptive calibration parameters from JSON."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Adaptive calibration file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(nominal_coverage=data["nominal_coverage"], method=data["method"])
        instance.global_adjustment = float(data["global_adjustment_mw"])
        instance.group_adjustments = {k: float(v) for k, v in data["group_adjustments_mw"].items()}
        instance.group_sample_counts = {k: int(v) for k, v in data.get("group_sample_counts", {}).items()}
        instance.group_raw_coverages = {k: float(v) for k, v in data.get("group_raw_coverages", {}).items()}
        instance.group_calibrated_coverages = {
            k: float(v) for k, v in data.get("group_calibrated_coverages", {}).items()
        }
        instance.total_calibration_samples = int(data.get("total_calibration_samples", 0))
        instance.overall_raw_coverage = float(data.get("overall_raw_coverage", 0.0))
        instance.overall_calibrated_coverage = float(data.get("overall_calibrated_coverage", 0.0))
        instance.is_calibrated = True
        return instance


def evaluate_probabilistic_forecasts(
    y_true: pd.Series | np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    q_median: np.ndarray | None = None,
    point_forecast: np.ndarray | None = None,
    timestamps: Sequence[Any] | None = None,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics for probabilistic forecasts.

    Computes:
    - Marginal empirical coverage: fraction of actuals inside [q_low, q_high]
    - Mean and median prediction interval width (MW)
    - Point and median accuracy (MAE, RMSE, MAPE)
    - Stratified coverage by season, hour of day, demand quantile, and horizon.

    Args:
        y_true: Actual demand values (MW).
        q_low: Lower prediction bound (MW).
        q_high: Upper prediction bound (MW).
        q_median: Optional median forecast (MW).
        point_forecast: Optional conditional mean point forecast (MW).
        timestamps: Optional timestamp sequence for temporal stratification.
        horizons: Optional horizon sequence for horizon stratification.

    Returns:
        dict: Evaluation metric summary.
    """
    y_arr = np.asarray(y_true, dtype=float).ravel()
    l_arr = np.asarray(q_low, dtype=float).ravel()
    h_arr = np.asarray(q_high, dtype=float).ravel()
    n = len(y_arr)

    if not (len(l_arr) == len(h_arr) == n):
        raise ValueError("Array length mismatch in probabilistic evaluation.")

    # Coverage indicator
    inside = (y_arr >= l_arr) & (y_arr <= h_arr)
    coverage = float(np.mean(inside))

    # Interval width
    widths = h_arr - l_arr
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))

    # Point accuracy metrics
    eval_pred = q_median if q_median is not None else point_forecast
    if eval_pred is not None:
        p_arr = np.asarray(eval_pred, dtype=float).ravel()
        mae = float(np.mean(np.abs(y_arr - p_arr)))
        rmse = float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))
        mape = float(np.mean(np.abs((y_arr - p_arr) / np.maximum(1e-5, y_arr))) * 100.0)
    else:
        mae, rmse, mape = 0.0, 0.0, 0.0

    result: dict[str, Any] = {
        "sample_count": n,
        "empirical_coverage": round(coverage, 4),
        "empirical_coverage_pct": round(coverage * 100.0, 2),
        "mean_interval_width_mw": round(mean_width, 2),
        "median_interval_width_mw": round(median_width, 2),
        "mae_mw": round(mae, 2),
        "rmse_mw": round(rmse, 2),
        "mape_pct": round(mape, 2),
    }

    # Demand quantile stratification (Quartiles)
    demand_quantiles = pd.qcut(y_arr, q=4, labels=["Q1_Low", "Q2_MedLow", "Q3_MedHigh", "Q4_Peak"])
    cov_by_demand = {}
    for cat in ["Q1_Low", "Q2_MedLow", "Q3_MedHigh", "Q4_Peak"]:
        mask = (demand_quantiles == cat)
        if mask.any():
            cov_by_demand[cat] = round(float(np.mean(inside[mask])) * 100.0, 2)
    result["coverage_by_demand_tier_pct"] = cov_by_demand

    # Temporal stratification (if timestamps supplied)
    if timestamps is not None and len(timestamps) == n:
        ts_index = pd.DatetimeIndex(timestamps)
        hours = ts_index.hour.to_numpy()
        months = ts_index.month.to_numpy()

        # By Hour
        cov_by_hour = {}
        for h in range(24):
            mask = (hours == h)
            if mask.any():
                cov_by_hour[f"H{h:02d}"] = round(float(np.mean(inside[mask])) * 100.0, 2)
        result["coverage_by_hour_pct"] = cov_by_hour

        # By Season
        # Winter: 12, 1, 2; Spring: 3, 4, 5; Summer: 6, 7, 8; Fall: 9, 10, 11
        season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                      3: "Spring", 4: "Spring", 5: "Spring",
                      6: "Summer", 7: "Summer", 8: "Summer",
                      9: "Fall", 10: "Fall", 11: "Fall"}
        seasons = np.array([season_map.get(m, "Other") for m in months])
        cov_by_season = {}
        for s in ["Winter", "Spring", "Summer", "Fall"]:
            mask = (seasons == s)
            if mask.any():
                cov_by_season[s] = round(float(np.mean(inside[mask])) * 100.0, 2)
        result["coverage_by_season_pct"] = cov_by_season

    # By Horizon (if horizons supplied)
    if horizons is not None and len(horizons) == n:
        h_arr = np.asarray(horizons)
        cov_by_horizon = {}
        for h_val in sorted(list(set(h_arr))):
            mask = (h_arr == h_val)
            if mask.any():
                cov_by_horizon[f"H={h_val}h"] = round(float(np.mean(inside[mask])) * 100.0, 2)
        result["coverage_by_horizon_pct"] = cov_by_horizon

    return result
