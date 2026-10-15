"""Baseline forecasting models for AI-Powered Energy Consumption Forecasting.

Implements standard benchmark forecasters:
1. Persistence (naive next-step persistence: y_{t+1} = y_t)
2. Daily Seasonal-Naive (seasonal persistence: y_{t+1} = y_{t+1 - 24h})
"""

from typing import Literal, Sequence
import numpy as np
import pandas as pd


class PersistenceForecaster:
    """Naive persistence baseline forecaster.

    Forecast specification:
        At forecast origin timestamp t, the prediction for target timestamp t+h is
        the observed consumption at forecast origin t:
            y_hat(T_target) = y(T_target - step_hours)

    Gap & Duplicate Safety:
        - Rejects duplicate timestamps with an explicit ValueError.
        - Verifies that the observation at (T_target - step_hours) exists in history.
        - Does not silently bridge irregular gaps when missing_policy='error'.
    """

    def __init__(
        self,
        step_hours: int = 1,
        missing_policy: Literal["exclude", "error"] = "exclude",
    ) -> None:
        """Initialize persistence forecaster.

        Args:
            step_hours: Horizon step size in hours (default 1 for 1-hour-ahead).
            missing_policy: How to handle missing forecast origin observations:
                'exclude': Sets predicted_consumption to NaN.
                'error': Raises ValueError indicating missing historical anchor.
        """
        if step_hours < 1:
            raise ValueError(f"step_hours must be a positive integer (>= 1). Got: {step_hours}")
        if missing_policy not in ("exclude", "error"):
            raise ValueError(f"missing_policy must be 'exclude' or 'error'. Got: {missing_policy}")

        self.model_name = "Persistence"
        self.step_hours = step_hours
        self.missing_policy = missing_policy

    def predict(
        self,
        df: pd.DataFrame,
        target_timestamps: Sequence[pd.Timestamp] | None = None,
        consumption_column: str = "consumption",
        timestamp_column: str = "timestamp",
    ) -> pd.DataFrame:
        """Generate one-step persistence forecasts with explicit timestamp alignment.

        Args:
            df: Historical and evaluation DataFrame.
            target_timestamps: Explicit list of target timestamps to forecast for.
                If None, targets correspond to rows starting after step_hours.
            consumption_column: Column name for consumption.
            timestamp_column: Column name for timestamp.

        Returns:
            pd.DataFrame: Contains columns:
                - target_timestamp: Evaluated moment T_target
                - origin_timestamp: Observation moment (T_target - step_hours)
                - predicted_consumption: Observed value at origin_timestamp
                - actual_consumption: True observation at target_timestamp (if in df)

        Raises:
            ValueError: If required columns are missing, duplicate timestamps exist,
                or missing_policy='error' and an origin observation is missing.
        """
        if timestamp_column not in df.columns or consumption_column not in df.columns:
            raise ValueError(f"DataFrame must contain '{timestamp_column}' and '{consumption_column}'.")

        # Explicit duplicate timestamp detection
        if df[timestamp_column].duplicated().any():
            dup_count = int(df[timestamp_column].duplicated().sum())
            raise ValueError(f"Duplicate timestamps detected in PersistenceForecaster ({dup_count} duplicates).")

        lookup_map = dict(zip(df[timestamp_column], df[consumption_column]))
        delta = pd.Timedelta(hours=self.step_hours)

        if target_timestamps is None:
            min_time = df[timestamp_column].min()
            eval_df = df[df[timestamp_column] >= min_time + delta].sort_values(by=timestamp_column)
            target_ts_list = list(eval_df[timestamp_column])
        else:
            target_ts_list = [pd.Timestamp(t) for t in target_timestamps]

        out_targets: list[pd.Timestamp] = []
        out_origins: list[pd.Timestamp] = []
        out_preds: list[float] = []
        out_actuals: list[float] = []

        for t_target in target_ts_list:
            t_origin = t_target - delta
            if t_origin in lookup_map:
                pred_val = float(lookup_map[t_origin])
            else:
                if self.missing_policy == "error":
                    raise ValueError(
                        f"Missing historical observation for persistence lookup at {t_origin} "
                        f"(exactly {self.step_hours}h prior to target {t_target})."
                    )
                pred_val = float("nan")

            actual_val = float(lookup_map.get(t_target, float("nan")))

            out_targets.append(t_target)
            out_origins.append(t_origin)
            out_preds.append(pred_val)
            out_actuals.append(actual_val)

        return pd.DataFrame({
            "target_timestamp": out_targets,
            "origin_timestamp": out_origins,
            "predicted_consumption": out_preds,
            "actual_consumption": out_actuals,
        })


class SeasonalNaiveForecaster:
    """Daily seasonal-naive baseline forecaster.

    Forecast specification:
        For a target at timestamp T_target, the prediction is the observation recorded
        exactly 24 hours prior:
            y_hat(T_target) = y(T_target - season_hours)

    Timestamp-Aware Lookup:
        Does not assume row positions correspond to elapsed time. Instead, performs
        an exact timestamp lookup for (T_target - season_hours).
    """

    def __init__(
        self,
        season_hours: int = 24,
        missing_policy: Literal["exclude", "error"] = "exclude",
    ) -> None:
        """Initialize seasonal-naive forecaster.

        Args:
            season_hours: Number of hours in seasonal cycle (default 24 for daily cycle).
            missing_policy: Policy when (T_target - season_hours) is not present in history:
                'exclude': Sets predicted_consumption to NaN.
                'error': Raises ValueError indicating missing historical anchor.
        """
        if season_hours < 1:
            raise ValueError(f"season_hours must be a positive integer (>= 1). Got: {season_hours}")
        if missing_policy not in ("exclude", "error"):
            raise ValueError(f"missing_policy must be 'exclude' or 'error'. Got: {missing_policy}")

        self.model_name = "SeasonalNaive_24h"
        self.season_hours = season_hours
        self.missing_policy = missing_policy

    def predict(
        self,
        df: pd.DataFrame,
        target_timestamps: Sequence[pd.Timestamp] | None = None,
        consumption_column: str = "consumption",
        timestamp_column: str = "timestamp",
    ) -> pd.DataFrame:
        """Generate seasonal-naive forecasts using exact timestamp matching.

        Args:
            df: Historical and evaluation DataFrame.
            target_timestamps: Explicit list of target timestamps to forecast for.
                If None, targets correspond to all rows starting after the first season_hours window.
            consumption_column: Column name for consumption.
            timestamp_column: Column name for timestamp.

        Returns:
            pd.DataFrame: Contains columns:
                - target_timestamp: Evaluated moment T_target
                - lookup_timestamp: Historical source timestamp (T_target - season_hours)
                - predicted_consumption: Observed value at lookup_timestamp
                - actual_consumption: True observation at target_timestamp (if in df)

        Raises:
            ValueError: If duplicate timestamps exist, or missing_policy='error' and an anchor is missing.
        """
        if timestamp_column not in df.columns or consumption_column not in df.columns:
            raise ValueError(f"DataFrame must contain '{timestamp_column}' and '{consumption_column}'.")

        # Explicit duplicate timestamp detection
        if df[timestamp_column].duplicated().any():
            dup_count = int(df[timestamp_column].duplicated().sum())
            raise ValueError(f"Duplicate timestamps detected in SeasonalNaiveForecaster ({dup_count} duplicates).")

        # Map timestamp to observed consumption
        lookup_map = dict(zip(df[timestamp_column], df[consumption_column]))

        delta = pd.Timedelta(hours=self.season_hours)

        if target_timestamps is None:
            min_time = df[timestamp_column].min()
            eval_df = df[df[timestamp_column] >= min_time + delta].sort_values(by=timestamp_column)
            target_ts_list = list(eval_df[timestamp_column])
        else:
            target_ts_list = [pd.Timestamp(t) for t in target_timestamps]

        out_targets: list[pd.Timestamp] = []
        out_lookups: list[pd.Timestamp] = []
        out_preds: list[float] = []
        out_actuals: list[float] = []

        for t_target in target_ts_list:
            t_lookup = t_target - delta
            if t_lookup in lookup_map:
                pred_val = float(lookup_map[t_lookup])
            else:
                if self.missing_policy == "error":
                    raise ValueError(
                        f"Missing historical observation for seasonal-naive lookup at {t_lookup} "
                        f"(exactly {self.season_hours}h prior to target {t_target})."
                    )
                pred_val = float("nan")

            actual_val = float(lookup_map.get(t_target, float("nan")))

            out_targets.append(t_target)
            out_lookups.append(t_lookup)
            out_preds.append(pred_val)
            out_actuals.append(actual_val)

        return pd.DataFrame({
            "target_timestamp": out_targets,
            "lookup_timestamp": out_lookups,
            "predicted_consumption": out_preds,
            "actual_consumption": out_actuals,
        })
