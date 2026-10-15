"""Sliding-window sequence generator for AI-Powered Energy Consumption Forecasting.

Structures chronological multivariate time-series data into 3D sequential arrays
(num_samples, sequence_length, num_features) and aligned targets for sequence models.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    """Immutable container for sliding-window sequence tensors and temporal alignments.

    Attributes:
        X: 3D NumPy array of shape (num_samples, sequence_length, num_features).
        y: 1D or 2D NumPy array of target values. Shape (num_samples,) when
           prediction_horizon == 1, else (num_samples, prediction_horizon).
        target_timestamps: Array of timestamps corresponding to each target observation.
        window_end_timestamps: Array of timestamps corresponding to the last input observation
           in each sequence (forecast origin timestamp).
        feature_names: Tuple of feature column names included in X.
    """

    X: np.ndarray
    y: np.ndarray
    target_timestamps: np.ndarray
    window_end_timestamps: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def num_samples(self) -> int:
        """Total number of generated sequence windows."""
        return len(self.X)

    @property
    def sequence_length(self) -> int:
        """Number of historical time-steps in each input window."""
        return self.X.shape[1] if self.X.ndim == 3 else 0

    @property
    def num_features(self) -> int:
        """Number of features per time-step."""
        return self.X.shape[2] if self.X.ndim == 3 else 0


def build_sliding_windows(
    df: pd.DataFrame,
    sequence_length: int = 168,
    prediction_horizon: int = 1,
    feature_columns: Sequence[str] | None = None,
    target_column: str = "consumption",
    expected_frequency: str | None = None,
) -> SequenceDataset:
    """Generate sliding-window sequence tensors and aligned targets without lookahead leakage.

    ALIGNMENT & ANTI-LEAKAGE SPECIFICATION:
        For window index i:
        - Input sequence X[i] contains observations at indices:
            [i, i + 1, ..., i + sequence_length - 1]
        - Forecast origin timestamp: timestamp at index i + sequence_length - 1
        - Target y[i] contains observation(s) strictly in the future:
            indices [i + sequence_length, ..., i + sequence_length + prediction_horizon - 1]

        Example with sequence_length=3 and prediction_horizon=1:
            Input sequence:  observations at 10:00, 11:00, 12:00
            Target:          observation at 13:00 (target timestamp = 13:00)
            Zero leakage: The target observation at 13:00 is strictly excluded from input X.

    Args:
        df: Chronologically ordered DataFrame containing 'timestamp', target_column,
            and any requested feature columns.
        sequence_length: Positive integer number of historical steps per window. Default 168.
        prediction_horizon: Positive integer number of steps ahead to predict. Default 1.
        feature_columns: Sequence of column names to include in input tensor X.
            If None, defaults to [target_column].
        target_column: Name of the target variable column. Defaults to 'consumption'.
        expected_frequency: Optional pandas frequency string (e.g. 'h'). When supplied,
            strictly asserts that timestamps contain no gaps or irregular steps.

    Returns:
        SequenceDataset: Structured container with X, y, and timestamp metadata.

    Raises:
        ValueError: If sequence_length or horizon is <= 0, timestamps are unsorted or
            duplicated, history is insufficient, or gaps are detected under expected_frequency.
    """
    if not isinstance(sequence_length, (int, np.integer)) or sequence_length < 1:
        raise ValueError(f"sequence_length must be a positive integer (>= 1). Received: {sequence_length}")

    if not isinstance(prediction_horizon, (int, np.integer)) or prediction_horizon < 1:
        raise ValueError(f"prediction_horizon must be a positive integer (>= 1). Received: {prediction_horizon}")

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # 1. Monotonic ordering and duplicate verification
    if df["timestamp"].duplicated().any():
        dup_count = int(df["timestamp"].duplicated().sum())
        raise ValueError(f"Duplicate timestamps detected ({dup_count} duplicates). Sequences require unique timestamps.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted in strictly increasing chronological order.")

    # 2. Strict frequency gap validation when expected_frequency is supplied
    if expected_frequency is not None:
        if len(df) >= 2:
            expected_delta = pd.Timedelta(pd.tseries.frequencies.to_offset(expected_frequency))
            actual_deltas = df["timestamp"].diff().iloc[1:]
            if not (actual_deltas == expected_delta).all():
                irregular_count = int((actual_deltas != expected_delta).sum())
                raise ValueError(
                    f"Irregular timestamps / gaps detected for expected_frequency='{expected_frequency}'. "
                    f"Found {irregular_count} step(s) not matching delta of {expected_delta}. "
                    f"Sliding window generation refuses to silently bridge missing time intervals."
                )

    # 3. History sufficiency check
    required_rows = sequence_length + prediction_horizon
    total_rows = len(df)
    if total_rows < required_rows:
        raise ValueError(
            f"Insufficient history: DataFrame has {total_rows} row(s), but sequence_length={sequence_length} "
            f"and prediction_horizon={prediction_horizon} require at least {required_rows} continuous rows."
        )

    # 4. Resolve feature columns
    if feature_columns is None:
        selected_features = [target_column]
    else:
        selected_features = list(feature_columns)
        for col in selected_features:
            if col not in df.columns:
                raise ValueError(f"Specified feature column '{col}' not found in DataFrame.")

    feature_values = df[selected_features].to_numpy(dtype=float)
    target_values = df[target_column].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy()

    # 5. Sliding window generation
    num_samples = total_rows - sequence_length - prediction_horizon + 1
    num_features = len(selected_features)

    X = np.empty((num_samples, sequence_length, num_features), dtype=float)
    if prediction_horizon == 1:
        y = np.empty((num_samples,), dtype=float)
        target_ts = np.empty((num_samples,), dtype="datetime64[ns]")
    else:
        y = np.empty((num_samples, prediction_horizon), dtype=float)
        target_ts = np.empty((num_samples, prediction_horizon), dtype="datetime64[ns]")

    window_end_ts = np.empty((num_samples,), dtype="datetime64[ns]")

    for i in range(num_samples):
        # Input features: [i : i + sequence_length]
        X[i] = feature_values[i : i + sequence_length]
        window_end_ts[i] = timestamps[i + sequence_length - 1]

        # Target slice: [i + sequence_length : i + sequence_length + prediction_horizon]
        if prediction_horizon == 1:
            y[i] = target_values[i + sequence_length]
            target_ts[i] = timestamps[i + sequence_length]
        else:
            y[i] = target_values[i + sequence_length : i + sequence_length + prediction_horizon]
            target_ts[i] = timestamps[i + sequence_length : i + sequence_length + prediction_horizon]

    return SequenceDataset(
        X=X,
        y=y,
        target_timestamps=target_ts,
        window_end_timestamps=window_end_ts,
        feature_names=tuple(selected_features),
    )
