"""Unit tests for leakage-safe temporal features and sliding-window sequences."""

import numpy as np
import pandas as pd
import pytest

from src.features.temporal_features import (
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
    add_target,
    create_temporal_features,
)
from src.features.sequence_builder import (
    build_sliding_windows,
    SequenceDataset,
)


@pytest.fixture
def hourly_sample_df() -> pd.DataFrame:
    """Deterministic hourly time series spanning 200 hours."""
    periods = 200
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=periods, freq="h")
    # Linear ramp plus daily wave: 100 + i + 10 * sin(2*pi*hour/24)
    hours = timestamps.hour
    consumption = 100.0 + np.arange(periods, dtype=float) + 10.0 * np.sin(2 * np.pi * hours / 24.0)
    return pd.DataFrame({"timestamp": timestamps, "consumption": consumption})


# ---------------------------------------------------------------------------
# Calendar and Cyclical Tests
# ---------------------------------------------------------------------------


def test_calendar_feature_values() -> None:
    """Test 1: Correct calendar feature values."""
    # 2026-01-03 is Saturday (day_of_week=5, is_weekend=1)
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-03 14:00:00"), pd.Timestamp("2026-01-05 09:00:00")],
        "consumption": [120.0, 130.0],
    })
    res = add_calendar_features(df)
    assert res["hour"].tolist() == [14, 9]
    assert res["day_of_week"].tolist() == [5, 0]  # Saturday=5, Monday=0
    assert res["day_of_month"].tolist() == [3, 5]
    assert res["month"].tolist() == [1, 1]
    assert res["is_weekend"].tolist() == [1, 0]


def test_cyclical_feature_values() -> None:
    """Test 2: Correct cyclical feature values and mathematical properties."""
    df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-01-01 00:00:00"),  # hour 0: sin(0)=0, cos(0)=1
            pd.Timestamp("2026-01-01 06:00:00"),  # hour 6: sin(pi/2)=1, cos(pi/2)=0
            pd.Timestamp("2026-01-01 12:00:00"),  # hour 12: sin(pi)=0, cos(pi)=-1
            pd.Timestamp("2026-01-01 18:00:00"),  # hour 18: sin(3pi/2)=-1, cos(3pi/2)=0
        ],
        "consumption": [10.0, 20.0, 30.0, 40.0],
    })
    res = add_cyclical_features(df)

    np.testing.assert_allclose(res["hour_sin"].iloc[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(res["hour_cos"].iloc[0], 1.0, atol=1e-7)
    np.testing.assert_allclose(res["hour_sin"].iloc[1], 1.0, atol=1e-7)
    np.testing.assert_allclose(res["hour_cos"].iloc[1], 0.0, atol=1e-7)
    np.testing.assert_allclose(res["hour_sin"].iloc[2], 0.0, atol=1e-7)
    np.testing.assert_allclose(res["hour_cos"].iloc[2], -1.0, atol=1e-7)

    # Pythagorean identity: sin^2 + cos^2 == 1
    hour_sq = res["hour_sin"] ** 2 + res["hour_cos"] ** 2
    np.testing.assert_allclose(hour_sq, 1.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Lag Tests
# ---------------------------------------------------------------------------


def test_lag_1_values(hourly_sample_df: pd.DataFrame) -> None:
    """Test 3: Correct lag_1 values."""
    res = add_lag_features(hourly_sample_df, lags=(1,))
    assert pd.isna(res["lag_1"].iloc[0])
    assert res["lag_1"].iloc[1] == hourly_sample_df["consumption"].iloc[0]
    assert res["lag_1"].iloc[50] == hourly_sample_df["consumption"].iloc[49]


def test_lag_24_values(hourly_sample_df: pd.DataFrame) -> None:
    """Test 4: Correct lag_24 values."""
    res = add_lag_features(hourly_sample_df, lags=(24,))
    assert res["lag_24"].iloc[:24].isna().all()
    assert res["lag_24"].iloc[24] == hourly_sample_df["consumption"].iloc[0]
    assert res["lag_24"].iloc[100] == hourly_sample_df["consumption"].iloc[76]


def test_lag_168_values(hourly_sample_df: pd.DataFrame) -> None:
    """Test 5: Correct lag_168 values."""
    res = add_lag_features(hourly_sample_df, lags=(168,))
    assert res["lag_168"].iloc[:168].isna().all()
    assert res["lag_168"].iloc[168] == hourly_sample_df["consumption"].iloc[0]
    assert res["lag_168"].iloc[170] == hourly_sample_df["consumption"].iloc[2]


# ---------------------------------------------------------------------------
# Rolling Statistics Tests
# ---------------------------------------------------------------------------


def test_rolling_mean_values() -> None:
    """Test 6: Correct rolling mean values."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
        "consumption": [10.0, 20.0, 30.0, 40.0],
    })
    res = add_rolling_features(df, windows=(3,))
    # Window=3:
    # row 0: [10] -> NaN (incomplete window in standard pandas rolling without min_periods=1)
    # row 1: [10, 20] -> NaN
    # row 2: [10, 20, 30] -> mean = 20.0
    # row 3: [20, 30, 40] -> mean = 30.0
    assert pd.isna(res["rolling_mean_3"].iloc[0])
    assert pd.isna(res["rolling_mean_3"].iloc[1])
    assert res["rolling_mean_3"].iloc[2] == 20.0
    assert res["rolling_mean_3"].iloc[3] == 30.0


def test_rolling_std_behavior() -> None:
    """Test 7: Correct rolling standard deviation behavior."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
        "consumption": [10.0, 10.0, 10.0],  # Constant series -> std=0
    })
    res = add_rolling_features(df, windows=(2,))
    assert pd.isna(res["rolling_std_2"].iloc[0])
    assert res["rolling_std_2"].iloc[1] == 0.0
    assert res["rolling_std_2"].iloc[2] == 0.0


# ---------------------------------------------------------------------------
# Target Alignment Tests
# ---------------------------------------------------------------------------


def test_next_step_target_alignment() -> None:
    """Test 8: Correct next-step target alignment."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
        "consumption": [100.0, 105.0, 110.0],
    })
    res = add_target(df, target_column_name="target_next_hour", horizon=1)
    # Row 0 target is consumption at Row 1 (105.0)
    assert res["target_next_hour"].iloc[0] == 105.0
    # Row 1 target is consumption at Row 2 (110.0)
    assert res["target_next_hour"].iloc[1] == 110.0
    # Row 2 target is in future -> NaN
    assert pd.isna(res["target_next_hour"].iloc[2])


def test_target_timestamps() -> None:
    """Test 9: Correct target timestamps."""
    ts = pd.date_range("2026-01-01 10:00:00", periods=3, freq="h")
    df = pd.DataFrame({"timestamp": ts, "consumption": [1.0, 2.0, 3.0]})
    res = add_target(df, target_column_name="target_next_hour", horizon=1)
    assert res["target_timestamp"].iloc[0] == pd.Timestamp("2026-01-01 11:00:00")
    assert res["target_timestamp"].iloc[1] == pd.Timestamp("2026-01-01 12:00:00")
    assert pd.isna(res["target_timestamp"].iloc[2])


# ---------------------------------------------------------------------------
# Validation & Error Handling Tests
# ---------------------------------------------------------------------------


def test_rejection_of_nonpositive_lag() -> None:
    """Test 10: Rejection of nonpositive lag values."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")], "consumption": [10.0]})
    with pytest.raises(ValueError, match="positive integers"):
        add_lag_features(df, lags=(0,))
    with pytest.raises(ValueError, match="positive integers"):
        add_lag_features(df, lags=(-5,))


def test_rejection_of_invalid_rolling_window() -> None:
    """Test 11: Rejection of invalid rolling windows."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")], "consumption": [10.0]})
    with pytest.raises(ValueError, match="positive integer"):
        add_rolling_features(df, windows=(0,))
    with pytest.raises(ValueError, match="positive integer"):
        add_rolling_features(df, windows=(-2,))


def test_handling_duplicate_timestamps() -> None:
    """Test 12: Handling and rejection of duplicate timestamps."""
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 00:00")],
        "consumption": [10.0, 20.0],
    })
    with pytest.raises(ValueError, match="Duplicate timestamps detected"):
        create_temporal_features(df)
    with pytest.raises(ValueError, match="Duplicate timestamps detected"):
        build_sliding_windows(df, sequence_length=1, prediction_horizon=1)


def test_handling_unsorted_timestamps() -> None:
    """Test 13: Handling and rejection of unsorted timestamps."""
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 02:00"), pd.Timestamp("2026-01-01 01:00")],
        "consumption": [10.0, 20.0],
    })
    with pytest.raises(ValueError, match="strictly increasing chronological order"):
        create_temporal_features(df)
    with pytest.raises(ValueError, match="strictly increasing chronological order"):
        build_sliding_windows(df, sequence_length=1, prediction_horizon=1)


def test_rejection_of_missing_intervals_when_frequency_supplied() -> None:
    """Test 14: Detection and rejection of missing intervals when expected_frequency is supplied."""
    df = pd.DataFrame({
        # 00:00 and 02:00 (01:00 is missing)
        "timestamp": [pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-01 02:00:00")],
        "consumption": [10.0, 20.0],
    })
    with pytest.raises(ValueError, match="Irregular timestamps / gaps detected"):
        create_temporal_features(df, expected_frequency="h")
    with pytest.raises(ValueError, match="Irregular timestamps / gaps detected"):
        build_sliding_windows(df, sequence_length=1, prediction_horizon=1, expected_frequency="h")


def test_no_gap_assertion_without_frequency() -> None:
    """Test 15: No false assertion of regular hourly data when no frequency is supplied."""
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-01 05:00:00")],
        "consumption": [10.0, 20.0],
    })
    # Should succeed without error because expected_frequency is None
    res = create_temporal_features(df, expected_frequency=None, lags=(1,), windows=(1,))
    assert len(res) == 2
    assert "lag_1" in res.columns


# ---------------------------------------------------------------------------
# Leakage Prevention Tests
# ---------------------------------------------------------------------------


def test_prevention_of_future_leakage_in_rolling_features() -> None:
    """Test 16 & 21: Changing future consumption does not alter earlier features."""
    df_base = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:00", periods=10, freq="h"),
        "consumption": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0],
    })

    # Forecast origin at index t=4 (timestamp 04:00)
    feat_base = create_temporal_features(df_base, windows=(3,), lags=(1, 2), expected_frequency="h")

    # Modify future consumption at index t=5 (timestamp 05:00) by a massive amount
    df_modified = df_base.copy()
    df_modified.loc[5, "consumption"] = 999999.0

    feat_modified = create_temporal_features(df_modified, windows=(3,), lags=(1, 2), expected_frequency="h")

    # Features at forecast origin t=4 must be IDENTICAL between base and modified
    features_cols = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_2", "rolling_mean_3", "rolling_std_3"]
    row_base_features = feat_base.loc[4, features_cols]
    row_mod_features = feat_modified.loc[4, features_cols]
    pd.testing.assert_series_equal(row_base_features, row_mod_features)

    # But the target at t=4 SHOULD reflect the new value at t=5
    assert feat_modified.loc[4, "target_next_hour"] == 999999.0
    assert feat_base.loc[4, "target_next_hour"] == 20.0


def test_prevention_of_future_leakage_in_sequence_inputs() -> None:
    """Test 17: Sequence inputs contain only observations preceding the target."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 10:00:00", periods=5, freq="h"),
        "consumption": [10.0, 11.0, 12.0, 13.0, 14.0],
    })
    # sequence_length=3, horizon=1
    dataset = build_sliding_windows(df, sequence_length=3, prediction_horizon=1)

    # First window:
    # Inputs: observations at 10:00, 11:00, 12:00 -> values [10.0, 11.0, 12.0]
    # Target: observation at 13:00 -> value 13.0
    np.testing.assert_array_equal(dataset.X[0, :, 0], [10.0, 11.0, 12.0])
    assert dataset.y[0] == 13.0
    assert dataset.target_timestamps[0] == np.datetime64("2026-01-01T13:00:00")
    assert dataset.window_end_timestamps[0] == np.datetime64("2026-01-01T12:00:00")

    # Second window:
    # Inputs: observations at 11:00, 12:00, 13:00 -> values [11.0, 12.0, 13.0]
    # Target: observation at 14:00 -> value 14.0
    np.testing.assert_array_equal(dataset.X[1, :, 0], [11.0, 12.0, 13.0])
    assert dataset.y[1] == 14.0


# ---------------------------------------------------------------------------
# Sequence Builder Specifications Tests
# ---------------------------------------------------------------------------


def test_sliding_window_lengths_and_multivariate() -> None:
    """Test 18: Correct sliding-window dimensions and multi-feature support."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        "consumption": np.arange(10, dtype=float),
        "feature_b": np.arange(10, dtype=float) * 2.0,
    })
    dataset = build_sliding_windows(
        df,
        sequence_length=4,
        prediction_horizon=2,
        feature_columns=["consumption", "feature_b"],
        target_column="consumption",
    )
    # Total samples = 10 - 4 - 2 + 1 = 5
    assert dataset.num_samples == 5
    assert dataset.sequence_length == 4
    assert dataset.num_features == 2
    assert dataset.X.shape == (5, 4, 2)
    assert dataset.y.shape == (5, 2)


def test_insufficient_history_handling() -> None:
    """Test 19: Handling insufficient history with informative ValueError."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
        "consumption": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    # sequence_length=5 + horizon=1 = 6 > 5 rows
    with pytest.raises(ValueError, match="Insufficient history"):
        build_sliding_windows(df, sequence_length=5, prediction_horizon=1)


def test_input_dataframe_preservation() -> None:
    """Test 20: Preservation of input DataFrames without unexpected mutation."""
    df_original = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
        "consumption": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    copy_before = df_original.copy(deep=True)

    _ = create_temporal_features(df_original, lags=(1,), windows=(2,))
    _ = build_sliding_windows(df_original, sequence_length=2, prediction_horizon=1)

    # Verify original df was not mutated in place
    pd.testing.assert_frame_equal(df_original, copy_before)


def test_timestamp_aware_lags_and_target_across_gap() -> None:
    """Test 21: Timestamp-aware lags and target handling across a daylight saving time gap."""
    # Simulate a spring forward gap: 00:00, 01:00, 03:00 (02:00 skipped)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-03-08 00:00", "2026-03-08 01:00", "2026-03-08 03:00", "2026-03-08 04:00"]),
        "consumption": [100.0, 105.0, 110.0, 115.0],
    })

    # 1. Target alignment across gap:
    # Row 1 (01:00) next contiguous hour should be 02:00, but next row is 03:00.
    # It must NOT assign 03:00 consumption (110.0) as target_next_hour!
    res_target = add_target(df, target_column_name="target_next_hour", horizon=1)
    assert res_target["target_next_hour"].iloc[0] == 105.0  # 00:00 -> 01:00
    assert pd.isna(res_target["target_next_hour"].iloc[1])  # 01:00 -> 02:00 is missing, so NaN
    assert pd.isna(res_target["target_timestamp"].iloc[1])  # target_timestamp is NaT
    assert res_target["target_next_hour"].iloc[2] == 115.0  # 03:00 -> 04:00
    assert res_target["target_timestamp"].iloc[2] == pd.Timestamp("2026-03-08 04:00")

    # 2. Lag alignment across gap:
    # Row 2 (03:00) 1 hour prior is 02:00, which does not exist in df.
    # lag_1 for 03:00 must be NaN, NOT 01:00 (105.0)!
    res_lag = add_lag_features(df, lags=(1,))
    assert pd.isna(res_lag["lag_1"].iloc[0])  # no history
    assert res_lag["lag_1"].iloc[1] == 100.0  # 01:00 -> lag_1 is 00:00
    assert pd.isna(res_lag["lag_1"].iloc[2])  # 03:00 -> lag_1 is 02:00 (missing)
    assert res_lag["lag_1"].iloc[3] == 110.0  # 04:00 -> lag_1 is 03:00
