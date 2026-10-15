"""Unit tests for public holiday feature engineering in AI-Powered Energy Consumption Forecasting.

Verifies observed holiday rules, midnight/year boundaries, forecast-target alignment,
forecast-time availability, missing/duplicate timestamp resilience, absence of target
leakage, and feature matrix reproducibility.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.features.temporal_features import (
    add_holiday_features,
    create_temporal_features,
)


def test_ordinary_and_observed_holidays() -> None:
    """Verify ordinary statutory holidays and weekend observation shifting rules."""
    dates = pd.date_range("2009-07-01", "2010-07-10", freq="D")
    df = pd.DataFrame({"target_timestamp": dates})
    res = add_holiday_features(df, target_timestamp_column="target_timestamp")

    res_map = dict(zip(res["target_timestamp"].dt.strftime("%Y-%m-%d"), res["is_holiday"]))

    # 1. July 4, 2009 was Saturday -> observed Friday July 3, 2009
    assert res_map["2009-07-03"] == 1, "Friday July 3, 2009 should be observed holiday for Saturday July 4"
    assert res_map["2009-07-04"] == 0, "Saturday July 4, 2009 is shifted to Friday observation"
    assert res_map["2009-07-05"] == 0, "Sunday July 5, 2009 is not a holiday"

    # 2. July 4, 2010 was Sunday -> observed Monday July 5, 2010
    assert res_map["2010-07-04"] == 0, "Sunday July 4, 2010 is shifted to Monday observation"
    assert res_map["2010-07-05"] == 1, "Monday July 5, 2010 should be observed holiday for Sunday July 4"
    assert res_map["2010-07-06"] == 0, "Tuesday July 6, 2010 is a regular working day"

    # 3. Thanksgiving Day (4th Thursday in November)
    assert res_map.get("2009-11-26") == 1, "Thanksgiving 2009 should be tagged as holiday"
    assert res_map.get("2009-11-27") == 0, "Day after Thanksgiving is not a federal statutory holiday"


def test_holiday_midnight_boundaries() -> None:
    """Verify that nominal midnight timestamps align exactly to the calendar date."""
    # Hours around Independence Day 2016 (observed on Monday July 4, 2016)
    hours = pd.date_range("2016-07-03 22:00:00", "2016-07-05 02:00:00", freq="h")
    df = pd.DataFrame({"target_timestamp": hours})
    res = add_holiday_features(df, target_timestamp_column="target_timestamp")

    res_map = dict(zip(res["target_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"), res["is_holiday"]))

    # July 3rd evening hours
    assert res_map["2016-07-03 22:00:00"] == 0
    assert res_map["2016-07-03 23:00:00"] == 0

    # July 4th midnight boundary to final hour
    assert res_map["2016-07-04 00:00:00"] == 1, "00:00:00 on holiday date should have is_holiday=1"
    assert res_map["2016-07-04 12:00:00"] == 1
    assert res_map["2016-07-04 23:00:00"] == 1, "23:00:00 on holiday date should have is_holiday=1"

    # July 5th midnight boundary
    assert res_map["2016-07-05 00:00:00"] == 0, "00:00:00 on day after holiday should have is_holiday=0"
    assert res_map["2016-07-05 01:00:00"] == 0


def test_holiday_year_boundaries() -> None:
    """Verify holiday status across December 31 / January 1 transitions."""
    # Year-end 2015 to New Year 2016 (Jan 1, 2016 was Friday)
    ts = pd.date_range("2015-12-31 20:00:00", "2016-01-01 04:00:00", freq="h")
    df = pd.DataFrame({"target_timestamp": ts})
    res = add_holiday_features(df, target_timestamp_column="target_timestamp")

    res_map = dict(zip(res["target_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"), res["is_holiday"]))

    assert res_map["2015-12-31 23:00:00"] == 0, "New Year's Eve is not a statutory federal holiday"
    assert res_map["2016-01-01 00:00:00"] == 1, "New Year's Day midnight should have is_holiday=1"
    assert res_map["2016-01-01 01:00:00"] == 1

    # In 2011, Jan 1 was Saturday; observed holiday was Friday Dec 31, 2010
    ts_2011 = pd.date_range("2010-12-30", "2011-01-03", freq="D")
    df_2011 = pd.DataFrame({"target_timestamp": ts_2011})
    res_2011 = add_holiday_features(df_2011, target_timestamp_column="target_timestamp")
    res_2011_map = dict(zip(res_2011["target_timestamp"].dt.strftime("%Y-%m-%d"), res_2011["is_holiday"]))

    assert res_2011_map["2010-12-31"] == 1, "Dec 31, 2010 is observed New Year's Day"
    assert res_2011_map["2011-01-01"] == 0, "Jan 1, 2011 is shifted to Friday observation"


def test_forecast_target_timestamp_alignment() -> None:
    """Verify is_holiday describes the forecast target timestamp, not forecast origin."""
    # At origin 2016-07-03 23:00, target is 2016-07-04 00:00 (holiday)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2016-07-03 22:00:00", periods=5, freq="h"),
        "consumption": [30000.0, 31000.0, 32000.0, 28000.0, 27000.0],
    })

    feat = create_temporal_features(df, include_target=True, include_holiday=True, drop_na=False)

    # Check origin 2016-07-03 23:00:00:
    row_eve = feat[feat["timestamp"] == pd.Timestamp("2016-07-03 23:00:00")].iloc[0]
    assert row_eve["target_timestamp"] == pd.Timestamp("2016-07-04 00:00:00")
    assert row_eve["is_holiday"] == 1, "At origin July 3 23:00, target is July 4 00:00, so is_holiday must be 1"

    # Check origin 2016-07-04 23:00:00:
    df2 = pd.DataFrame({
        "timestamp": pd.date_range("2016-07-04 22:00:00", periods=5, freq="h"),
        "consumption": [30000.0, 31000.0, 32000.0, 28000.0, 27000.0],
    })
    feat2 = create_temporal_features(df2, include_target=True, include_holiday=True, drop_na=False)
    row_end = feat2[feat2["timestamp"] == pd.Timestamp("2016-07-04 23:00:00")].iloc[0]
    assert row_end["target_timestamp"] == pd.Timestamp("2016-07-05 00:00:00")
    assert row_end["is_holiday"] == 0, "At origin July 4 23:00, target is July 5 00:00, so is_holiday must be 0"


def test_feature_availability_at_forecast_time() -> None:
    """Verify holiday feature derives strictly from deterministic calendar knowledge."""
    # Calendar status is known decades in advance; no future demand or test statistics used
    future_dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"target_timestamp": future_dates})
    res = add_holiday_features(df)
    assert res["is_holiday"].iloc[0] == 1, "Jan 1, 2024 is known in advance to be New Year's Day"
    assert res["is_holiday"].iloc[1] == 0


def test_missing_and_duplicate_timestamps() -> None:
    """Verify add_holiday_features handles NaT, missing hours, and duplicate timestamps safely."""
    ts = pd.Series([
        pd.Timestamp("2016-07-04 01:00:00"),
        pd.NaT,
        pd.Timestamp("2016-07-04 02:00:00"),
        pd.Timestamp("2016-07-04 02:00:00"),  # Duplicate clock shift
    ])
    df = pd.DataFrame({"target_timestamp": ts})
    res = add_holiday_features(df, target_timestamp_column="target_timestamp")

    assert res["is_holiday"].isna().sum() == 0, "Output must contain zero NaNs"
    assert res["is_holiday"].iloc[0] == 1
    assert res["is_holiday"].iloc[1] == 0, "NaT timestamp must evaluate safely to 0"
    assert res["is_holiday"].iloc[2] == 1
    assert res["is_holiday"].iloc[3] == 1


def test_no_target_leakage() -> None:
    """Verify holiday feature generation is invariant to target consumption perturbations."""
    df1 = pd.DataFrame({
        "timestamp": pd.date_range("2016-07-03", periods=48, freq="h"),
        "consumption": np.linspace(20000, 40000, 48),
    })
    # Scramble consumption in df2
    df2 = df1.copy()
    df2["consumption"] = np.random.default_rng(42).permutation(df1["consumption"].to_numpy())

    feat1 = create_temporal_features(df1, include_holiday=True, drop_na=True)
    feat2 = create_temporal_features(df2, include_holiday=True, drop_na=True)

    np.testing.assert_array_equal(
        feat1["is_holiday"].to_numpy(),
        feat2["is_holiday"].to_numpy(),
        err_msg="is_holiday must not depend on consumption values",
    )


def test_identical_eligible_timestamps_for_baseline_and_candidate() -> None:
    """Verify baseline and candidate feature matrices share identical row counts and target timestamps."""
    dates = pd.date_range("2016-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "consumption": np.linspace(25000, 35000, 200),
    })

    base_feat = create_temporal_features(df, include_holiday=False, drop_na=True)
    cand_feat = create_temporal_features(df, include_holiday=True, drop_na=True)

    assert len(base_feat) == len(cand_feat), "Sample count must be identical"
    pd.testing.assert_series_equal(base_feat["timestamp"], cand_feat["timestamp"])
    pd.testing.assert_series_equal(base_feat["target_timestamp"], cand_feat["target_timestamp"])
    pd.testing.assert_series_equal(base_feat["target_next_hour"], cand_feat["target_next_hour"])


def test_reproducible_holiday_feature_generation() -> None:
    """Verify that multiple consecutive runs yield identical deterministic bitwise results."""
    dates = pd.date_range("2014-01-01", periods=1000, freq="h")
    df = pd.DataFrame({"target_timestamp": dates})

    run1 = add_holiday_features(df)["is_holiday"].to_numpy()
    run2 = add_holiday_features(df)["is_holiday"].to_numpy()

    np.testing.assert_array_equal(run1, run2)
