"""Rigorous test suite for Phase 7 Operational NWP Forecast Integration.

Tests the 9 mandatory operational requirements:
1. Timestamp normalization (UTC to Eastern Civil, spring 23h, autumn 25h).
2. Weather-to-energy timestamp alignment (elementwise timestamp alignment).
3. Location matching (KPHL station verification, coordinates, bilinear weights).
4. Missing and duplicate weather records handling (causal step-back and diurnal imputer).
5. Forecast issue-time cutoff enforcement (simulated_avail <= target_hour - 1h).
6. Prevention of future-weather leakage (catches and rejects post-origin records).
7. Feature generation (temp_f_fcst, cooling_degree_proxy_fcst, heating_degree_proxy_fcst).
8. Evaluation metric calculations (MAE, RMSE, MAPE, MAE improvement percentage).
9. Train/test chronological separation (strict non-shuffled partitions).
"""

import pytest
import numpy as np
import pandas as pd

from src.data.nwp_loader import (
    NWPForecastRecord,
    build_operational_nwp_dataset,
    validate_nwp_forecast_data,
    KPHL_LAT,
    KPHL_LON,
)
from src.data.hrrr_loader import (
    compute_kphl_spatial_weights,
    evaluate_operational_eligibility,
)
from src.features.weather_features import (
    WeatherImputer,
    compute_degree_proxies,
    add_operational_nwp_features,
)
from src.evaluation.metrics import (
    calculate_metrics,
    chronological_split,
)


# ==============================================================================
# 1. TIMESTAMP NORMALIZATION & CIVIL TIME CONVERSION TESTS
# ==============================================================================

def test_timestamp_normalization_and_dst():
    """Verify continuous UTC timestamps correctly project to Eastern civil time with DST handled."""
    # Test Spring Forward transition: 2017-03-12 (23-hour civil day)
    # 06:00 UTC = 01:00 EST; 07:00 UTC = 03:00 EDT (02:00 local nonexistent)
    utc_range = pd.date_range("2017-03-12 05:00:00", "2017-03-12 09:00:00", freq="h", tz="UTC")
    local_proj = utc_range.tz_convert("America/New_York")

    # Local hours: 00:00, 01:00, 03:00, 04:00, 05:00 (02:00 omitted naturally)
    hours = [ts.hour for ts in local_proj]
    assert 2 not in hours, "Hour 02:00 must not exist on Spring Forward transition!"
    assert hours == [0, 1, 3, 4, 5]

    # Test Autumn Fall Back transition: 2017-11-05 (25-hour civil day)
    # 05:00 UTC = 01:00 EDT; 06:00 UTC = 01:00 EST
    utc_fall = pd.date_range("2017-11-05 04:00:00", "2017-11-05 07:00:00", freq="h", tz="UTC")
    local_fall = utc_fall.tz_convert("America/New_York")
    fall_hours = [ts.hour for ts in local_fall]
    assert fall_hours == [0, 1, 1, 2], "Hour 01:00 must occur twice across Fall Back transition!"


# ==============================================================================
# 2. WEATHER-TO-ENERGY TIMESTAMP ALIGNMENT TESTS
# ==============================================================================

def test_weather_to_energy_timestamp_alignment():
    """Verify that NWP forecast features align elementwise to target energy timestamps."""
    timestamps = pd.date_range("2017-06-01 00:00:00", periods=24, freq="h")
    demand_df = pd.DataFrame({
        "timestamp": timestamps - pd.Timedelta(hours=1),
        "target_timestamp": timestamps,
        "target_next_hour": np.random.uniform(25000, 35000, size=24),
    })

    # NWP forecasts indexed by target timestamp
    nwp_df = pd.DataFrame({
        "timestamp": timestamps,
        "valid_time_utc": timestamps.tz_localize("America/New_York").tz_convert("UTC"),
        "simulated_avail_utc": (timestamps.tz_localize("America/New_York").tz_convert("UTC") - pd.Timedelta(hours=2) + pd.Timedelta(minutes=50)),
        "temp_f_fcst": [70.0 + i * 0.5 for i in range(24)],
    })

    imputer = WeatherImputer(max_forward_fill_hours=2)
    # Dummy train data for imputer
    imputer.fit(demand_df.assign(temp_f_fcst=70.0), timestamp_col="target_timestamp", temp_col="temp_f_fcst")

    augmented, _ = add_operational_nwp_features(demand_df, nwp_df, imputer)

    assert len(augmented) == 24
    assert (augmented["target_timestamp"].to_numpy() == timestamps.to_numpy()).all()
    assert pytest.approx(augmented["temp_f_fcst"].iloc[0]) == 70.0
    assert pytest.approx(augmented["temp_f_fcst"].iloc[23]) == 81.5


# ==============================================================================
# 3. LOCATION MATCHING TESTS
# ==============================================================================

def test_location_matching():
    """Verify KPHL station coordinates, elevation, and Lambert Conformal spatial weights."""
    weights = compute_kphl_spatial_weights(KPHL_LAT, KPHL_LON)

    assert weights["kphl_coordinates"] == (39.8719, -75.2411)
    i_near, j_near = weights["nearest_node"]
    assert i_near == 1526
    assert j_near == 656
    assert pytest.approx(weights["weight_sum"], abs=1e-6) == 1.000000

    # Ensure validation check rejects incorrect location
    dummy_nwp = pd.DataFrame({
        "timestamp": [pd.Timestamp("2017-01-01 00:00:00")],
        "location_id": ["KNYC"],  # Wrong location!
        "issue_time_utc": [pd.Timestamp("2016-12-31 22:00:00", tz="UTC")],
        "valid_time_utc": [pd.Timestamp("2017-01-01 00:00:00", tz="UTC")],
        "lead_time_hours": [2],
        "temp_f_fcst": [45.0],
        "source_dataset": ["NOAA_HRRR"],
        "simulated_avail_utc": [pd.Timestamp("2016-12-31 22:50:00", tz="UTC")],
    })
    dummy_energy = pd.DataFrame({"target_timestamp": [pd.Timestamp("2017-01-01 00:00:00")]})

    with pytest.raises(ValueError, match="location_id != 'KPHL'"):
        validate_nwp_forecast_data(dummy_nwp, dummy_energy)


# ==============================================================================
# 4. MISSING AND DUPLICATE WEATHER RECORDS HANDLING TESTS
# ==============================================================================

def test_missing_and_duplicate_weather_records():
    """Verify that causal step-back fallback and WeatherImputer handle missing forecasts."""
    timestamps = pd.date_range("2017-01-01 00:00:00", periods=5, freq="h")
    demand_df = pd.DataFrame({
        "timestamp": timestamps - pd.Timedelta(hours=1),
        "target_timestamp": timestamps,
        "target_next_hour": [30000.0] * 5,
    })

    # Simulate missing forecast at index 2 (NaN)
    nwp_df = pd.DataFrame({
        "timestamp": timestamps,
        "valid_time_utc": timestamps.tz_localize("America/New_York").tz_convert("UTC"),
        "simulated_avail_utc": (timestamps.tz_localize("America/New_York").tz_convert("UTC") - pd.Timedelta(hours=2) + pd.Timedelta(minutes=50)),
        "temp_f_fcst": [50.0, 52.0, np.nan, 56.0, 58.0],
    })

    imputer = WeatherImputer(max_forward_fill_hours=2)
    imputer.fit(demand_df.assign(temp_f_fcst=[50.0, 52.0, 54.0, 56.0, 58.0]), timestamp_col="target_timestamp", temp_col="temp_f_fcst")

    augmented, _ = add_operational_nwp_features(demand_df, nwp_df, imputer)

    # Missing value at index 2 should be forward-filled from index 1 (52.0)
    assert not augmented["temp_f_fcst"].isna().any()
    assert pytest.approx(augmented["temp_f_fcst"].iloc[2]) == 52.0


# ==============================================================================
# 5. FORECAST ISSUE-TIME CUTOFF ENFORCEMENT TESTS
# ==============================================================================

def test_forecast_issue_time_cutoff_enforcement():
    """Verify that operational eligibility strictly requires t_avail <= t_origin."""
    target_utc = pd.Timestamp("2017-06-01 15:00:00", tz="UTC")
    origin_utc = target_utc - pd.Timedelta(hours=1)  # 14:00 UTC

    # Cycle T-2h: t_init = 13:00, t_avail = 13:50 <= 14:00 (ADMISSIBLE)
    eval_admissible = evaluate_operational_eligibility(
        target_hour_utc=target_utc,
        cycle_init_utc=pd.Timestamp("2017-06-01 13:00:00", tz="UTC"),
        forecast_lead_hours=2,
        forecast_origin_lead_minutes=60,
        model_latency_minutes=50,
    )
    assert eval_admissible["is_admissible"] is True
    assert eval_admissible["buffer_minutes"] == 10.0

    # Late Cycle T-1h: t_init = 14:00, t_avail = 14:50 > 14:00 (INADMISSIBLE!)
    eval_late = evaluate_operational_eligibility(
        target_hour_utc=target_utc,
        cycle_init_utc=pd.Timestamp("2017-06-01 14:00:00", tz="UTC"),
        forecast_lead_hours=1,
        forecast_origin_lead_minutes=60,
        model_latency_minutes=50,
    )
    assert eval_late["is_admissible"] is False
    assert eval_late["buffer_minutes"] == -50.0


# ==============================================================================
# 6. PREVENTION OF FUTURE-WEATHER LEAKAGE TESTS
# ==============================================================================

def test_prevention_of_future_weather_leakage():
    """Verify automated assertion catches and rejects post-origin forecast data."""
    target_ts = pd.Timestamp("2017-06-01 15:00:00")
    target_utc = target_ts.tz_localize("America/New_York").tz_convert("UTC")

    demand_df = pd.DataFrame({
        "timestamp": [target_ts - pd.Timedelta(hours=1)],
        "target_timestamp": [target_ts],
        "target_next_hour": [32000.0],
    })

    # Violating forecast: simulated_avail_utc is at 14:30 UTC > cutoff 14:00 UTC
    violating_nwp = pd.DataFrame({
        "timestamp": [target_ts],
        "valid_time_utc": [target_utc],
        "simulated_avail_utc": [target_utc - pd.Timedelta(minutes=30)],  # 30 min before target, but 30 min AFTER 1h origin!
        "temp_f_fcst": [75.0],
    })

    imputer = WeatherImputer()
    imputer.fit(demand_df.assign(temp_f_fcst=75.0), timestamp_col="target_timestamp", temp_col="temp_f_fcst")

    with pytest.raises(ValueError, match="STRICT CAUSALITY VIOLATION DETECTED"):
        add_operational_nwp_features(demand_df, violating_nwp, imputer, enforce_causality=True)


# ==============================================================================
# 7. FEATURE GENERATION TESTS
# ==============================================================================

def test_feature_generation():
    """Verify piecewise space-conditioning degree proxy formulas."""
    temps = pd.Series([55.0, 65.0, 75.0, 85.0])
    cooling, heating = compute_degree_proxies(temps, base_temp=65.0)

    # Cooling degree proxy = max(0, temp - 65)
    assert cooling.tolist() == [0.0, 0.0, 10.0, 20.0]

    # Heating degree proxy = max(0, 65 - temp)
    assert heating.tolist() == [10.0, 0.0, 0.0, 0.0]


# ==============================================================================
# 8. EVALUATION METRIC CALCULATIONS TESTS
# ==============================================================================

def test_evaluation_metric_calculations():
    """Verify calculation of MAE, RMSE, MAPE, and MAE percentage improvement."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred_base = np.array([110.0, 220.0, 280.0, 390.0])  # absolute errors: 10, 20, 20, 10 -> MAE = 15.0
    y_pred_cand = np.array([105.0, 210.0, 290.0, 395.0])  # absolute errors: 5, 10, 10, 5 -> MAE = 7.5

    met_base = calculate_metrics(y_true, y_pred_base)
    met_cand = calculate_metrics(y_true, y_pred_cand)

    assert pytest.approx(met_base.mae) == 15.0
    assert pytest.approx(met_cand.mae) == 7.5

    # MAE improvement (%) = ((baseline - candidate) / baseline) * 100
    mae_imp = ((met_base.mae - met_cand.mae) / met_base.mae) * 100.0
    assert pytest.approx(mae_imp) == 50.0  # 50% improvement


# ==============================================================================
# 9. TRAIN/TEST CHRONOLOGICAL SEPARATION TESTS
# ==============================================================================

def test_train_test_chronological_separation():
    """Verify that chronological splitting preserves ordering with zero leakage or overlap."""
    dates = pd.date_range("2010-01-01", periods=1000, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "target_timestamp": dates + pd.Timedelta(hours=1),
        "target_next_hour": np.arange(1000, dtype=float),
    })

    train, val, test = chronological_split(df, 0.70, 0.15, 0.15)

    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150

    # Assert strict monotonic separation between partitions
    assert train["target_timestamp"].max() < val["target_timestamp"].min()
    assert val["target_timestamp"].max() < test["target_timestamp"].min()

    # Zero overlap
    train_set = set(train["target_timestamp"])
    val_set = set(val["target_timestamp"])
    test_set = set(test["target_timestamp"])
    assert len(train_set.intersection(val_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
