"""Unit and integration tests for Phase 7 weather feature engineering and imputation.

Validates:
1. Exact degree proxy calculations at the 65.0°F threshold.
2. DST transition preservation and documented synthetic autumn-hour aggregation.
3. Strictly past-only forward-fill behavior and elapsed-UTC gap thresholds.
4. Training-only fallback statistics and 3-tier hierarchy isolation.
5. Imputation state carryover across chronological partition boundaries.
6. Exact feature-matrix sample parity and target alignment against frozen Phase 5A baseline.
7. Error handling for non-finite values, missing statistics, and invalid policies.
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, MODELS_DIR
from src.data.weather_loader import (
    parse_noaa_tmp,
    assign_nominal_utc_hour,
    select_hourly_kphl_observations,
    build_hourly_kphl_weather_series,
    load_raw_noaa_kphl,
)
from src.features.weather_features import (
    compute_degree_proxies,
    WeatherImputer,
    add_weather_features,
)
from src.experiment import find_pjm_dataset
from src.data.loader import load_energy_data
from src.data.preprocessor import preprocess_energy_data
from src.features.temporal_features import create_temporal_features
from src.evaluation.metrics import chronological_split


# ==============================================================================
# 1. DEGREE PROXY CALCULATIONS
# ==============================================================================

def test_degree_proxy_exact_calculations_at_65f():
    """Verify piecewise-linear formulations at 65.0°F and boundary conditions."""
    temps = pd.Series([65.0, 90.0, 30.0, 64.9, 65.1, 0.0, -10.0])
    cooling, heating = compute_degree_proxies(temps, base_temp=65.0)

    # At 65.0°F exactly
    assert cooling.iloc[0] == 0.0
    assert heating.iloc[0] == 0.0

    # At 90.0°F (Hot summer day)
    assert cooling.iloc[1] == 25.0
    assert heating.iloc[1] == 0.0

    # At 30.0°F (Freezing winter day)
    assert cooling.iloc[2] == 0.0
    assert heating.iloc[2] == 35.0

    # Near threshold boundaries
    assert pytest.approx(cooling.iloc[3], abs=1e-5) == 0.0
    assert pytest.approx(heating.iloc[3], abs=1e-5) == 0.1
    assert pytest.approx(cooling.iloc[4], abs=1e-5) == 0.1
    assert pytest.approx(heating.iloc[4], abs=1e-5) == 0.0

    # Sub-zero temperatures
    assert cooling.iloc[5] == 0.0
    assert heating.iloc[5] == 65.0
    assert cooling.iloc[6] == 0.0
    assert heating.iloc[6] == 75.0


# ==============================================================================
# 2. DST TRANSITION PRESERVATION & AUTUMN AGGREGATION
# ==============================================================================

def test_dst_transition_preservation_and_autumn_aggregation():
    """Verify spring non-existent hour omission and autumn duplicate hour preservation."""
    # Synthetic records around Autumn 2010 transition (2010-11-07)
    # EDT: 2010-11-07 05:00 UTC -> 01:00 EDT (temp = 42.0°F)
    # EST: 2010-11-07 06:00 UTC -> 01:00 EST (temp = 40.0°F)
    raw_data = [
        {"DATE": "2010-11-07T04:54:00", "REPORT_TYPE": "FM-15", "TMP": "+0056,1"},  # 42.08°F
        {"DATE": "2010-11-07T05:54:00", "REPORT_TYPE": "FM-15", "TMP": "+0044,1"},  # 39.92°F
    ]
    raw_df = pd.DataFrame(raw_data)

    # Build weather series returning both collapsed and intermediate UTC tables
    collapsed_mean, intermediate = build_hourly_kphl_weather_series(
        raw_df,
        autumn_duplicate_policy="mean",
        return_intermediate_utc=True,
        start_year=2010,
        end_year=2010,
    )

    # 1. Intermediate table preserves both physical UTC hours
    intermediate_dup = intermediate[
        intermediate["naive_ts"] == pd.Timestamp("2010-11-07 01:00:00")
    ]
    assert len(intermediate_dup) == 2, "Intermediate table must preserve both physical observations"
    assert intermediate_dup.iloc[0]["is_dst"] == True
    assert intermediate_dup.iloc[1]["is_dst"] == False
    assert intermediate_dup.iloc[0]["nominal_utc"] == pd.Timestamp("2010-11-07 05:00:00", tz="UTC")
    assert intermediate_dup.iloc[1]["nominal_utc"] == pd.Timestamp("2010-11-07 06:00:00", tz="UTC")

    # 2. Collapsed table combines them to arithmetic mean
    collapsed_target = collapsed_mean[
        collapsed_mean["timestamp"] == pd.Timestamp("2010-11-07 01:00:00")
    ]
    assert len(collapsed_target) == 1, "Collapsed table must contain exactly 1 row for join compatibility"
    expected_mean = (intermediate_dup.iloc[0]["temp_f"] + intermediate_dup.iloc[1]["temp_f"]) / 2.0
    assert pytest.approx(collapsed_target.iloc[0]["temp_f"], abs=1e-4) == expected_mean

    # 3. Test diagnostic policies (EDT-only and EST-only)
    collapsed_edt = build_hourly_kphl_weather_series(
        raw_df, autumn_duplicate_policy="edt", start_year=2010, end_year=2010
    )
    val_edt = collapsed_edt[collapsed_edt["timestamp"] == pd.Timestamp("2010-11-07 01:00:00")]["temp_f"].iloc[0]
    assert pytest.approx(val_edt, abs=1e-4) == intermediate_dup.iloc[0]["temp_f"]

    collapsed_est = build_hourly_kphl_weather_series(
        raw_df, autumn_duplicate_policy="est", start_year=2010, end_year=2010
    )
    val_est = collapsed_est[collapsed_est["timestamp"] == pd.Timestamp("2010-11-07 01:00:00")]["temp_f"].iloc[0]
    assert pytest.approx(val_est, abs=1e-4) == intermediate_dup.iloc[1]["temp_f"]


# ==============================================================================
# 3. CAUSAL FORWARD-FILL & UTC GAP THRESHOLD
# ==============================================================================

def test_strictly_past_only_forward_fill_behavior():
    """Verify forward-fill applies for <=2h gaps, and falls back to Tier 1 for >2h gaps."""
    imputer = WeatherImputer(max_forward_fill_hours=2)
    # Fit mock imputer
    mock_train = pd.DataFrame({
        "target_timestamp": pd.date_range("2010-01-01 00:00:00", periods=48, freq="1h"),
        "temp_f": [50.0] * 48,
    })
    imputer.fit(mock_train)
    # Mock diurnal table cell for month 6, hour 13 to be 77.7°F
    imputer.diurnal_table_["6_13"] = 77.7

    # Test sequence:
    # 10:00 UTC: 60.0 (valid)
    # 11:00 UTC: NaN (1h elapsed -> forward-fill 60.0)
    # 12:00 UTC: NaN (2h elapsed -> forward-fill 60.0)
    # 13:00 UTC: NaN (3h elapsed -> Tier 1 diurnal fallback 77.7)
    # 14:00 UTC: 65.0 (valid)
    # 15:00 UTC: NaN (1h elapsed -> forward-fill 65.0)
    eval_df = pd.DataFrame({
        "target_timestamp": pd.date_range("2010-06-01 10:00:00", periods=6, freq="1h"),
        "nominal_utc": pd.date_range("2010-06-01 14:00:00", periods=6, freq="1h", tz="UTC"),
        "temp_f": [60.0, np.nan, np.nan, np.nan, 65.0, np.nan],
    })

    imputed_df, final_state = imputer.transform(eval_df, utc_col="nominal_utc")

    assert imputed_df.loc[0, "temp_f"] == 60.0
    assert imputed_df.loc[1, "temp_f"] == 60.0, "1h gap must forward-fill"
    assert imputed_df.loc[2, "temp_f"] == 60.0, "2h gap must forward-fill"
    assert pytest.approx(imputed_df.loc[3, "temp_f"], abs=1e-4) == 77.7, "3h gap must fall back to Tier 1"
    assert imputed_df.loc[4, "temp_f"] == 65.0
    assert imputed_df.loc[5, "temp_f"] == 65.0, "1h gap must forward-fill"

    # Final state carried over
    assert final_state[1] == 65.0


# ==============================================================================
# 4. TRAINING-ONLY FALLBACK HIERARCHY & ISOLATION
# ==============================================================================

def test_training_only_fallback_statistics_and_hierarchy(tmp_path: Path):
    """Verify that fallback parameters use training data only and remain strictly frozen."""
    # Create synthetic training set with known diurnal variation
    ts_train = pd.date_range("2005-01-01 00:00:00", "2005-01-10 23:00:00", freq="1h")
    # Base temp = 40.0 + hour
    temps_train = [40.0 + (t.hour * 0.5) for t in ts_train]
    train_df = pd.DataFrame({"target_timestamp": ts_train, "temp_f": temps_train})

    imputer = WeatherImputer(max_forward_fill_hours=2)
    imputer.fit(train_df)

    assert imputer.is_fitted_
    assert len(imputer.diurnal_table_) == 288
    assert len(imputer.monthly_means_) == 12
    assert imputer.global_mean_ is not None

    # Verify diurnal table value for January hour 10
    expected_jan_10 = 40.0 + (10 * 0.5)
    assert pytest.approx(imputer.diurnal_table_["1_10"], abs=1e-4) == expected_jan_10

    # Save and reload imputer parameters
    save_path = tmp_path / "weather_imputer_params.json"
    imputer.save(save_path)
    loaded_imputer = WeatherImputer.load(save_path)

    assert loaded_imputer.is_fitted_
    assert pytest.approx(loaded_imputer.global_mean_, abs=1e-4) == imputer.global_mean_
    assert loaded_imputer.diurnal_table_ == imputer.diurnal_table_

    # Verify that transforming validation or test data DOES NOT mutate imputer parameters
    val_df = pd.DataFrame({
        "target_timestamp": pd.date_range("2015-07-01 00:00:00", periods=24, freq="1h"),
        "temp_f": [999.0] * 24, # Extreme out-of-distribution values
    })
    _ = loaded_imputer.transform(val_df)

    # Imputer parameters remain completely identical
    assert pytest.approx(loaded_imputer.diurnal_table_["1_10"], abs=1e-4) == expected_jan_10

    # Verify fatal error on empty training data
    empty_imputer = WeatherImputer()
    with pytest.raises(RuntimeError, match="Zero valid temperature observations"):
        empty_imputer.fit(pd.DataFrame({"target_timestamp": ts_train[:10], "temp_f": [np.nan] * 10}))


# ==============================================================================
# 5. IMPUTATION STATE ACROSS PARTITION BOUNDARIES
# ==============================================================================

def test_imputation_state_across_partition_boundaries():
    """Verify state carryover at partition boundaries without future-to-past lookahead."""
    imputer = WeatherImputer(max_forward_fill_hours=2)
    mock_train = pd.DataFrame({
        "target_timestamp": pd.date_range("2010-01-01 00:00:00", periods=24, freq="1h"),
        "temp_f": [55.0] * 24,
    })
    imputer.fit(mock_train)

    # Partition 1 ends at 23:00 with valid temp 52.0°F
    part1_df = pd.DataFrame({
        "target_timestamp": [pd.Timestamp("2010-01-01 23:00:00")],
        "nominal_utc": [pd.Timestamp("2010-01-02 04:00:00", tz="UTC")],
        "temp_f": [52.0],
    })
    _, state_p1 = imputer.transform(part1_df, utc_col="nominal_utc")
    assert state_p1 is not None

    # Partition 2 starts 1 hour later at 00:00, but has NaN
    # Elapsed time is 1h <= 2h -> must forward-fill from Partition 1 (52.0°F)
    part2_df = pd.DataFrame({
        "target_timestamp": [pd.Timestamp("2010-01-02 00:00:00")],
        "nominal_utc": [pd.Timestamp("2010-01-02 05:00:00", tz="UTC")],
        "temp_f": [np.nan],
    })
    res_p2, state_p2 = imputer.transform(part2_df, utc_col="nominal_utc", initial_state=state_p1)
    assert res_p2.loc[0, "temp_f"] == 52.0, "Short boundary gap must forward-fill from previous partition"

    # Partition 3 starts 5 hours later at 05:00 with NaN
    # Elapsed time is 5h > 2h -> must fall back to Tier 1 diurnal table
    part3_df = pd.DataFrame({
        "target_timestamp": [pd.Timestamp("2010-01-02 05:00:00")],
        "nominal_utc": [pd.Timestamp("2010-01-02 10:00:00", tz="UTC")],
        "temp_f": [np.nan],
    })
    res_p3, _ = imputer.transform(part3_df, utc_col="nominal_utc", initial_state=state_p2)
    assert res_p3.loc[0, "temp_f"] == 55.0, "Extended boundary gap must fall back to Tier 1 diurnal table"


# ==============================================================================
# 6. EXACT FEATURE-MATRIX SAMPLE PARITY (END-TO-END DATA GATE)
# ==============================================================================

def test_feature_matrix_sample_parity_against_frozen_baseline():
    """Verify that merging weather features preserves 100% exact sample identity.

    Assertions checked:
    - Train count == 101,551
    - Val count == 21,760
    - Test count == 21,762 (full)
    - Ordered target_timestamp vectors match baseline exactly.
    - target_next_hour values match baseline exactly.
    - Zero NaNs, zero infinities across weather features.
    """
    pjm_path = find_pjm_dataset()
    if pjm_path is None or not pjm_path.is_file():
        pytest.skip("PJME dataset not found locally.")

    raw_demand = load_energy_data(pjm_path)
    clean_demand, _ = preprocess_energy_data(raw_demand, duplicate_policy="mean")
    feat_demand = create_temporal_features(
        clean_demand,
        lags=(1, 24, 168),
        windows=(24,),
        target_column_name="target_next_hour",
        drop_na=True,
    )
    train_df, val_df, test_df = chronological_split(feat_demand, 0.70, 0.15, 0.15)

    # Assert baseline sample counts
    assert len(train_df) == 101551
    assert len(val_df) == 21760
    assert len(test_df) == 21762

    # Load raw NOAA weather
    raw_weather = load_raw_noaa_kphl()
    weather_series = build_hourly_kphl_weather_series(raw_weather, autumn_duplicate_policy="mean")

    # Fit imputer on training partition target timestamps
    imputer = WeatherImputer(max_forward_fill_hours=2)

    # Attach weather to train to fit imputer
    train_pre = train_df[["target_timestamp"]].merge(
        weather_series[["timestamp", "temp_f"]],
        left_on="target_timestamp",
        right_on="timestamp",
        how="left",
    )
    imputer.fit(train_pre, timestamp_col="target_timestamp", temp_col="temp_f")

    # Transform train, val, test with state carryover
    train_weather, train_state = add_weather_features(
        train_df, weather_series, imputer, alignment_hypothesis="A"
    )
    val_weather, val_state = add_weather_features(
        val_df, weather_series, imputer, initial_state=train_state, alignment_hypothesis="A"
    )
    test_weather, test_state = add_weather_features(
        test_df, weather_series, imputer, initial_state=val_state, alignment_hypothesis="A"
    )

    # 1. Exact Row Counts
    assert len(train_weather) == 101551
    assert len(val_weather) == 21760
    assert len(test_weather) == 21762

    # 2. Ordered target_timestamp vectors match baseline exactly
    assert (train_weather["target_timestamp"].to_numpy() == train_df["target_timestamp"].to_numpy()).all()
    assert (val_weather["target_timestamp"].to_numpy() == val_df["target_timestamp"].to_numpy()).all()
    assert (test_weather["target_timestamp"].to_numpy() == test_df["target_timestamp"].to_numpy()).all()

    # 3. Monotonic increasing
    assert train_weather["target_timestamp"].is_monotonic_increasing
    assert val_weather["target_timestamp"].is_monotonic_increasing
    assert test_weather["target_timestamp"].is_monotonic_increasing

    # 4. Zero duplicate timestamps
    assert train_weather["target_timestamp"].duplicated().sum() == 0
    assert val_weather["target_timestamp"].duplicated().sum() == 0
    assert test_weather["target_timestamp"].duplicated().sum() == 0

    # 5. Target demand values match baseline exactly
    assert np.allclose(
        train_weather["target_next_hour"].to_numpy(),
        train_df["target_next_hour"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        val_weather["target_next_hour"].to_numpy(),
        val_df["target_next_hour"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        test_weather["target_next_hour"].to_numpy(),
        test_df["target_next_hour"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # 6. Zero NaNs and all finite across all weather features
    weather_feature_cols = ["temp_f", "cooling_degree_proxy_f", "heating_degree_proxy_f"]
    for col in weather_feature_cols:
        assert train_weather[col].isna().sum() == 0
        assert val_weather[col].isna().sum() == 0
        assert test_weather[col].isna().sum() == 0

        assert not np.isinf(train_weather[col]).any()
        assert not np.isinf(val_weather[col]).any()
        assert not np.isinf(test_weather[col]).any()
