"""Tree-based regression forecaster for AI-Powered Energy Consumption Forecasting.

Wraps scikit-learn's HistGradientBoostingRegressor with strict feature validation,
anti-leakage checks, and reproducibility guarantees.
"""

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import RANDOM_STATE
from src.models.base_model import BaseForecaster

FORBIDDEN_FEATURE_KEYWORDS = {"target", "target_next_hour", "target_timestamp"}


class EnergyTreeForecaster(BaseForecaster):
    """Gradient boosted tree forecaster for energy demand time-series.

    Enforces strict feature boundary verification to ensure future target
    values never contaminate the predictor matrix.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        random_state: int = RANDOM_STATE,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = 6,
        max_leaf_nodes: int | None = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize tree-based forecaster with explicit hyperparameters.

        Args:
            feature_names: Explicit ordered list of predictor column names.
            random_state: Reproducibility seed. Defaults to config RANDOM_STATE.
            max_iter: Maximum number of boosting trees. Defaults to 100.
            learning_rate: Shrinkage step size. Defaults to 0.1.
            max_depth: Maximum tree depth. Defaults to 6.
            max_leaf_nodes: Maximum number of leaves per tree. Defaults to 31.
            min_samples_leaf: Minimum samples required per leaf. Defaults to 20.
            l2_regularization: L2 regularization strength. Defaults to 0.0.
            **kwargs: Extra arguments forwarded to HistGradientBoostingRegressor.

        Raises:
            ValueError: If feature_names is empty or contains forbidden target columns.
        """
        if not feature_names:
            raise ValueError("feature_names cannot be empty. Please specify predictor columns.")

        # Anti-leakage guard: Prohibit target-derived names from feature lists
        for feat in feature_names:
            clean_feat = str(feat).strip().lower()
            if clean_feat in FORBIDDEN_FEATURE_KEYWORDS or clean_feat.startswith("target"):
                raise ValueError(
                    f"Anti-leakage violation: Column '{feat}' cannot be included in feature_names."
                )

        super().__init__(model_name="HistGradientBoosting", feature_names=feature_names)
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.extra_kwargs = kwargs

        self.estimator_: HistGradientBoostingRegressor | None = None
        self.metadata_: dict[str, Any] = {}

    def _prepare_features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract and validate feature matrix from input."""
        if isinstance(X, pd.DataFrame):
            assert self.feature_names is not None
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")
            X_arr = X[list(self.feature_names)].to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            X_arr = np.asarray(X, dtype=float)
            if self.feature_names is not None and X_arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Feature array shape mismatch: Expected {len(self.feature_names)} features, "
                    f"got {X_arr.shape[1]}."
                )
        else:
            raise TypeError(f"X must be a pandas DataFrame or numpy array, got {type(X)}.")

        # Reject non-finite values (NaN, Inf)
        if not np.isfinite(X_arr).all():
            nan_count = int(np.isnan(X_arr).sum())
            inf_count = int(np.isinf(X_arr).sum())
            raise ValueError(
                f"Feature matrix contains non-finite values (NaN: {nan_count}, Inf: {inf_count}). "
                f"Please drop or impute missing feature rows before feeding into the estimator."
            )

        return X_arr

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "EnergyTreeForecaster":
        """Fit HistGradientBoostingRegressor on training partition.

        Args:
            X: Training features (DataFrame or 2D array).
            y: Training targets (Series or 1D array).

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If lengths do not match or target contains non-finite values.
        """
        X_mat = self._prepare_features(X)

        if isinstance(y, pd.Series):
            y_arr = y.to_numpy(dtype=float)
        else:
            y_arr = np.asarray(y, dtype=float).ravel()

        if len(X_mat) != len(y_arr):
            raise ValueError(f"Length mismatch between features ({len(X_mat)}) and target ({len(y_arr)}).")

        if len(y_arr) == 0:
            raise ValueError("Training target array cannot be empty.")

        if not np.isfinite(y_arr).all():
            raise ValueError("Training target array contains non-finite values (NaN or Inf).")

        self.estimator_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            **self.extra_kwargs,
        )

        self.estimator_.fit(X_mat, y_arr)
        self.is_fitted = True
        self.metadata_ = {
            "model_name": self.model_name,
            "feature_names": list(self.feature_names) if self.feature_names else [],
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "l2_regularization": self.l2_regularization,
            "n_samples_fit": len(y_arr),
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate forecasts from feature inputs.

        Args:
            X: Input feature matrix.

        Returns:
            np.ndarray: 1D array of predicted energy consumption values.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted or self.estimator_ is None:
            raise RuntimeError("EnergyTreeForecaster is not fitted yet. Call fit() before predict().")

        X_mat = self._prepare_features(X)
        predictions = self.estimator_.predict(X_mat)
        return np.asarray(predictions, dtype=float)
