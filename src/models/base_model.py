"""Base model abstraction for AI-Powered Energy Consumption Forecasting.

Defines the contract for trainable forecasters, including standardized
fit, predict, and joblib-based serialization interfaces.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence
import joblib
import numpy as np
import pandas as pd

from src.config import DEFAULT_MODEL_PATH


class BaseForecaster(ABC):
    """Abstract base class for all energy consumption forecasters."""

    def __init__(self, model_name: str, feature_names: Sequence[str] | None = None) -> None:
        self.model_name = model_name
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseForecaster":
        """Fit the forecasting model on training data.

        Args:
            X: Input feature matrix.
            y: Target values.

        Returns:
            self: Fitted estimator instance.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate forecasts from feature inputs.

        Args:
            X: Input feature matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    def save(self, file_path: Path | str | None = None) -> Path:
        """Serialize the fitted model and metadata to disk using joblib.

        Args:
            file_path: Destination path. Defaults to DEFAULT_MODEL_PATH from src.config.

        Returns:
            Path: Resolved absolute path where model was persisted.

        Raises:
            RuntimeError: If attempting to save an unfitted model.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Cannot save unfitted model '{self.model_name}'. Call fit() first.")

        target_path = Path(file_path) if file_path is not None else DEFAULT_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "estimator": getattr(self, "estimator_", None),
            "metadata": getattr(self, "metadata_", {}),
        }
        joblib.dump(payload, target_path)
        return target_path.resolve()

    @classmethod
    def load(cls, file_path: Path | str | None = None) -> "BaseForecaster":
        """Load a persisted model and its metadata from disk.

        Args:
            file_path: Path to serialized model file. Defaults to DEFAULT_MODEL_PATH.

        Returns:
            BaseForecaster: Reconstructed model instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        target_path = Path(file_path) if file_path is not None else DEFAULT_MODEL_PATH
        if not target_path.is_file():
            raise FileNotFoundError(f"Serialized model file not found at: {target_path}")

        payload = joblib.load(target_path)
        instance = cls(
            model_name=payload.get("model_name", cls.__name__),
            feature_names=payload.get("feature_names"),
        )
        instance.is_fitted = payload.get("is_fitted", False)
        if "estimator" in payload and payload["estimator"] is not None:
            setattr(instance, "estimator_", payload["estimator"])
        if "metadata" in payload:
            setattr(instance, "metadata_", payload["metadata"])
        return instance
