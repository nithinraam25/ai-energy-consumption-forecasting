"""Leakage-safe temporal feature engineering for AI-Powered Energy Consumption Forecasting.

Generates calendar, cyclical, autoregressive lag, and rolling statistics
features with strict temporal alignment to prevent target leakage.
"""

from typing import Sequence
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar indicator features from the timestamp column.

    Features generated:
        - hour: Hour of the day (0-23)
        - day_of_week: Day of the week (0=Monday, 6=Sunday)
        - day_of_month: Day of the month (1-31)
        - month: Month of the year (1-12)
        - is_weekend: Binary indicator (1 if Saturday or Sunday, else 0)

    Args:
        df: DataFrame containing a datetime 'timestamp' column.

    Returns:
        pd.DataFrame: New DataFrame with calendar features added.
    """
    df_out = df.copy()
    ts = df_out["timestamp"].dt
    df_out["hour"] = ts.hour
    df_out["day_of_week"] = ts.dayofweek
    df_out["day_of_month"] = ts.day
    df_out["month"] = ts.month
    df_out["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    return df_out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform periodic calendar components using sine and cosine encodings.

    Mathematical formulation:
        hour_sin = sin(2 * pi * hour / 24)
        hour_cos = cos(2 * pi * hour / 24)
        day_of_week_sin = sin(2 * pi * day_of_week / 7)
        day_of_week_cos = cos(2 * pi * day_of_week / 7)

    Ensures smooth continuous boundaries across midnight and week transitions.
    Requires 'hour' and 'day_of_week' columns, creating them if not already present.

    Args:
        df: DataFrame containing 'timestamp' (and optionally 'hour', 'day_of_week').

    Returns:
        pd.DataFrame: New DataFrame with cyclical trigonometric encodings.
    """
    df_out = df.copy()
    if "hour" not in df_out.columns or "day_of_week" not in df_out.columns:
        df_out = add_calendar_features(df_out)

    hour = df_out["hour"].to_numpy(dtype=float)
    dow = df_out["day_of_week"].to_numpy(dtype=float)

    df_out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df_out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df_out["day_of_week_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df_out["day_of_week_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return df_out


def add_lag_features(
    df: pd.DataFrame,
    lags: Sequence[int] = (1, 24, 168),
) -> pd.DataFrame:
    """Create autoregressive lag features for energy consumption.

    For a row at timestamp t, lag_k represents consumption at timestamp t - k hours.
    When a sorted, unique timestamp column is present, exact timestamp delta lookup
    is performed to guarantee lag_24 corresponds to exactly 24 elapsed hours prior.
    If an anchor is missing (e.g. across a daylight saving time gap), NaN is assigned.

    Args:
        df: DataFrame containing a 'consumption' column (and optionally 'timestamp').
        lags: Sequence of positive integer lag offsets (e.g., 1, 24, 168).

    Returns:
        pd.DataFrame: New DataFrame containing lag columns.

    Raises:
        ValueError: If any lag value is not a positive integer (>= 1).
    """
    df_out = df.copy()
    for lag in lags:
        if not isinstance(lag, (int, np.integer)) or lag < 1:
            raise ValueError(f"Lag values must be positive integers (>= 1). Received: {lag}")

    has_valid_ts = (
        "timestamp" in df_out.columns
        and pd.api.types.is_datetime64_any_dtype(df_out["timestamp"])
        and not df_out["timestamp"].duplicated().any()
        and df_out["timestamp"].is_monotonic_increasing
    )

    if has_valid_ts:
        ts_map = dict(zip(df_out["timestamp"], df_out["consumption"]))
        for lag in lags:
            delta = pd.Timedelta(hours=int(lag))
            df_out[f"lag_{lag}"] = [ts_map.get(ts - delta, np.nan) for ts in df_out["timestamp"]]
    else:
        for lag in lags:
            df_out[f"lag_{lag}"] = df_out["consumption"].shift(lag)

    return df_out


def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (24,),
) -> pd.DataFrame:
    """Compute rolling historical statistics over past consumption observations.

    LEAKAGE-PREVENTION CONTRACT:
        For a feature row associated with timestamp t (forecast origin t), the
        rolling window covers observations in the closed interval [t - w + 1, t]:
            rolling_mean_w[t] = mean(consumption[t - w + 1 : t])
            rolling_std_w[t] = std(consumption[t - w + 1 : t])

        The future target at timestamp t+1 is strictly excluded. Modifying any future
        consumption observation (at t+1, t+2, ...) has zero effect on rolling features at t.

    Args:
        df: DataFrame containing a 'consumption' column.
        windows: Sequence of positive integer window sizes (e.g., 24).

    Returns:
        pd.DataFrame: New DataFrame with rolling mean and standard deviation columns.

    Raises:
        ValueError: If any window size is not a positive integer (>= 1).
    """
    df_out = df.copy()
    for w in windows:
        if not isinstance(w, (int, np.integer)) or w < 1:
            raise ValueError(f"Rolling window must be a positive integer (>= 1). Received: {w}")
        df_out[f"rolling_mean_{w}"] = df_out["consumption"].rolling(window=w).mean()
        # For window=1, std is undefined/NaN
        df_out[f"rolling_std_{w}"] = df_out["consumption"].rolling(window=w).std()
    return df_out


def add_target(
    df: pd.DataFrame,
    target_column_name: str = "target_next_hour",
    horizon: int = 1,
) -> pd.DataFrame:
    """Align the future target consumption observation to the current forecast origin.

    ALIGNMENT SPECIFICATION:
        At row index t with forecast origin timestamp t, the target is the consumption
        at timestamp t + horizon hours (e.g., t + 1 hour for horizon=1).
            df[target_column_name] = df['consumption'].shift(-horizon)
            df['target_timestamp'] = df['timestamp'].shift(-horizon)

        If timestamps are present and regular, exact hourly delta alignment is verified.
        If an observation is missing or non-consecutive (e.g. spring forward gap),
        the target is assigned NaN and target_timestamp is assigned NaT rather than
        silently compressing non-consecutive hours into the target label.

    Args:
        df: DataFrame containing 'timestamp' and 'consumption'.
        target_column_name: Name of target column. Defaults to 'target_next_hour'.
        horizon: Positive integer prediction step offset. Defaults to 1.

    Returns:
        pd.DataFrame: New DataFrame with target column and target_timestamp.

    Raises:
        ValueError: If horizon is not a positive integer (>= 1).
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"Target horizon must be a positive integer (>= 1). Received: {horizon}")

    df_out = df.copy()
    has_valid_ts = (
        "timestamp" in df_out.columns
        and pd.api.types.is_datetime64_any_dtype(df_out["timestamp"])
        and not df_out["timestamp"].duplicated().any()
        and df_out["timestamp"].is_monotonic_increasing
    )

    if has_valid_ts:
        expected_delta = pd.Timedelta(hours=int(horizon))
        target_ts_series = df_out["timestamp"].shift(-horizon)
        actual_delta = target_ts_series - df_out["timestamp"]
        is_exact_horizon = actual_delta == expected_delta

        target_vals = df_out["consumption"].shift(-horizon)
        df_out[target_column_name] = target_vals.where(is_exact_horizon, np.nan)
        df_out["target_timestamp"] = target_ts_series.where(is_exact_horizon, pd.NaT)
    else:
        df_out[target_column_name] = df_out["consumption"].shift(-horizon)
        if "timestamp" in df_out.columns:
            df_out["target_timestamp"] = df_out["timestamp"].shift(-horizon)

    return df_out


from pandas.tseries.holiday import USFederalHolidayCalendar


def add_holiday_features(
    df: pd.DataFrame,
    target_timestamp_column: str = "target_timestamp",
    holiday_column_name: str = "is_holiday",
) -> pd.DataFrame:
    """Derive binary public holiday indicator for the forecast target timestamp.

    LEAKAGE-PREVENTION & FORECAST AVAILABILITY CONTRACT:
        - Evaluates holiday status on the nominal calendar date of the forecast target
          timestamp (T_target).
        - For a one-step-ahead forecast origin T_origin, T_target = T_origin + 1 hour.
        - Because public statutory holiday schedules are established by law years in
          advance, the holiday status of T_target is deterministic public calendar
          knowledge fully available at forecast origin T_origin.
        - Future electrical power demand actuals are never accessed.
        - Uses US Federal Statutory Holidays (5 U.S.C. 6103) with standard observed-holiday
          rules (Sunday holidays observed on Monday, Saturday holidays observed on Friday).

    Args:
        df: DataFrame containing the target timestamp column (or 'timestamp').
        target_timestamp_column: Column name containing forecast target datetimes.
            Defaults to 'target_timestamp'. If absent, falls back to 'timestamp'.
        holiday_column_name: Output column name for binary indicator. Defaults to 'is_holiday'.

    Returns:
        pd.DataFrame: New DataFrame with the binary holiday indicator column added ({0, 1}).

    Raises:
        ValueError: If neither target_timestamp_column nor 'timestamp' is present in df.
    """
    df_out = df.copy()
    if target_timestamp_column in df_out.columns:
        ts_series = df_out[target_timestamp_column]
    elif "timestamp" in df_out.columns:
        ts_series = df_out["timestamp"]
    else:
        raise ValueError(
            f"Neither '{target_timestamp_column}' nor 'timestamp' column found in DataFrame. "
            f"Available columns: {list(df_out.columns)}"
        )

    cal = USFederalHolidayCalendar()
    holiday_dates = set(cal.holidays(start="2000-01-01", end="2025-12-31").date)

    target_dates = pd.to_datetime(ts_series).dt.date
    is_holiday = target_dates.isin(holiday_dates).astype(int)
    if ts_series.isna().any():
        is_holiday = is_holiday.where(~ts_series.isna(), 0)

    df_out[holiday_column_name] = is_holiday.astype(int)
    return df_out


def create_temporal_features(
    df: pd.DataFrame,
    expected_frequency: str | None = None,
    lags: Sequence[int] = (1, 24, 168),
    windows: Sequence[int] = (24,),
    target_column_name: str = "target_next_hour",
    include_target: bool = True,
    include_holiday: bool = False,
    drop_na: bool = False,
) -> pd.DataFrame:
    """Execute the complete leakage-safe temporal feature engineering pipeline.

    Validates chronological monotonicity and regularity, then appends calendar,
    cyclical, lag, rolling, and target columns without mutating the input DataFrame.

    Gap-Handling Strategy:
        If expected_frequency is supplied (e.g., 'h'), the function strictly validates
        that timestamps are contiguous at that frequency. If irregular intervals or gaps
        are present, a ValueError is raised to prevent silent compression of non-consecutive
        hours into adjacent rows.
        If expected_frequency is None, no frequency regularity is asserted.

    Args:
        df: Input DataFrame containing 'timestamp' and 'consumption'.
        expected_frequency: Optional pandas frequency string (e.g., 'h', 'D').
        lags: Tuple of integer lag steps. Defaults to (1, 24, 168).
        windows: Tuple of integer rolling window sizes. Defaults to (24,).
        target_column_name: Target column identifier. Defaults to 'target_next_hour'.
        include_target: Whether to append the one-step-ahead target column.
        include_holiday: Whether to derive binary 'is_holiday' feature for forecast target.
        drop_na: If True, drop rows containing any NaN (from initial lag/rolling windows).

    Returns:
        pd.DataFrame: Cleaned feature matrix.

    Raises:
        ValueError: If required columns are missing, duplicates exist, timestamps are
            unsorted, or gaps violate expected_frequency.
    """
    if "timestamp" not in df.columns or "consumption" not in df.columns:
        raise ValueError(
            f"DataFrame must contain 'timestamp' and 'consumption' columns. Found: {list(df.columns)}"
        )

    # Work on a copy to guarantee caller's DataFrame is never mutated
    df_feat = df.copy()

    # 1. Monotonicity and duplicate validation
    if df_feat["timestamp"].duplicated().any():
        dup_count = int(df_feat["timestamp"].duplicated().sum())
        raise ValueError(f"Duplicate timestamps detected ({dup_count} duplicates). Please deduplicate before feature extraction.")

    if not df_feat["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted in strictly increasing chronological order.")

    # 2. Strict frequency gap validation when expected_frequency is supplied
    if expected_frequency is not None:
        if len(df_feat) >= 2:
            expected_delta = pd.Timedelta(pd.tseries.frequencies.to_offset(expected_frequency))
            actual_deltas = df_feat["timestamp"].diff().iloc[1:]
            if not (actual_deltas == expected_delta).all():
                irregular_count = int((actual_deltas != expected_delta).sum())
                raise ValueError(
                    f"Irregular timestamps / gaps detected for expected_frequency='{expected_frequency}'. "
                    f"Found {irregular_count} step(s) not matching delta of {expected_delta}. "
                    f"Feature engineering refuses to silently bridge missing time intervals."
                )

    # 3. Construct features sequentially
    df_feat = add_calendar_features(df_feat)
    df_feat = add_cyclical_features(df_feat)
    df_feat = add_lag_features(df_feat, lags=lags)
    df_feat = add_rolling_features(df_feat, windows=windows)

    if include_target:
        df_feat = add_target(df_feat, target_column_name=target_column_name, horizon=1)

    if include_holiday:
        df_feat = add_holiday_features(df_feat, target_timestamp_column="target_timestamp")

    if drop_na:
        df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat

