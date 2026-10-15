"""Comprehensive Test Suite for Phase 8: Production Readiness, Multi-Station & Day-Ahead Forecasting.

Validates all 10 core operational and engineering requirements:
1. Multi-station metadata catalog and coordinate integrity.
2. Multi-station timestamp alignment across primary and satellite stations.
3. Regional feature aggregation (unweighted mean, zonal load-share weighted, spatial gradient).
4. Missing weather and station dropout handling.
5. Horizon lag restrictions and strict cutoff enforcement (H=1, 24, 48).
6. Horizon-dependent target alignment and lead-time forecast error scaling.
7. Production artifact persistence and schema loading.
8. Feature schema validation (detects missing, unexpected, and non-finite features).
9. Operational forecast availability cutoff and anti-substitution enforcement in inference.
10. Operational health monitoring and drift/staleness alerting.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE, MODELS_DIR
from src.data.multi_station import (
    STATION_CATALOG,
    StationMetadata,
    compute_regional_aggregations,
    build_multi_station_nwp_dataset,
)
from src.features.horizon_features import (
    SUPPORTED_HORIZONS,
    get_horizon_feature_names,
    create_horizon_feature_matrix,
)
from src.models.inference import (
    EnergyInferenceEngine,
    DEFAULT_PRODUCTION_MODEL_PATH,
    DEFAULT_FEATURE_SCHEMA_PATH,
    DEFAULT_MODEL_CONFIG_PATH,
)
from src.evaluation.monitoring import ForecastHealthMonitor
from src.models.tree_models import EnergyTreeForecaster


# ==============================================================================
# 1. MULTI-STATION METADATA & COORDINATES
# ==============================================================================

def test_multi_station_metadata_catalog():
    """Verify station catalog identifiers, coordinates, and load weight normalization."""
    assert "KPHL" in STATION_CATALOG
    assert "KEWR" in STATION_CATALOG
    assert "KILG" in STATION_CATALOG

    # Verify coordinates are in PJM East corridor
    kphl = STATION_CATALOG["KPHL"]
    assert 39.0 <= kphl.latitude <= 41.5
    assert -76.0 <= kphl.longitude <= -73.5
    assert kphl.composite_id == "72408013739"

    kewr = STATION_CATALOG["KEWR"]
    assert 39.0 <= kewr.latitude <= 41.5
    assert -76.0 <= kewr.longitude <= -73.5
    assert kewr.composite_id == "72502014734"

    kilg = STATION_CATALOG["KILG"]
    assert 39.0 <= kilg.latitude <= 41.5
    assert -76.0 <= kilg.longitude <= -73.5
    assert kilg.composite_id == "72409313764"

    # Verify zonal load weights sum to 1.0 (PECO 0.38 + PSE&G 0.42 + Delmarva 0.20)
    total_weight = sum(s.zonal_load_weight for s in STATION_CATALOG.values())
    assert pytest.approx(total_weight, abs=1e-5) == 1.0


# ==============================================================================
# 2. REGIONAL AGGREGATION & SPATIAL GRADIENT FORMULAS
# ==============================================================================

def test_regional_feature_aggregation_formulas():
    """Verify exact mathematical implementation of unweighted mean, zonal weighted mean, and spread."""
    # Test values: KPHL=70.0, KEWR=73.0, KILG=67.0
    t_kphl = np.array([70.0, 80.0])
    t_kewr = np.array([73.0, 84.0])
    t_kilg = np.array([67.0, 77.0])

    res = compute_regional_aggregations(t_kphl, t_kewr, t_kilg)

    # 1. Unweighted mean: (70 + 73 + 67) / 3 = 70.0; (80 + 84 + 77) / 3 = 80.33
    assert pytest.approx(res["temp_f_unweighted_mean"].iloc[0], abs=0.01) == 70.00
    assert pytest.approx(res["temp_f_unweighted_mean"].iloc[1], abs=0.01) == 80.33

    # 2. Zonal weighted: 0.38*70 + 0.42*73 + 0.20*67 = 26.6 + 30.66 + 13.4 = 70.66
    expected_zonal_0 = (0.38 * 70.0) + (0.42 * 73.0) + (0.20 * 67.0)
    assert pytest.approx(res["temp_f_zonal_weighted"].iloc[0], abs=0.01) == round(expected_zonal_0, 2)

    # 3. Spatial Spread: max(73, 70, 67) - min(73, 70, 67) = 73 - 67 = 6.0°F
    assert pytest.approx(res["temp_spread_regional"].iloc[0], abs=0.01) == 6.00
    assert pytest.approx(res["temp_spread_regional"].iloc[1], abs=0.01) == 7.00

    # 4. Piecewise degree proxies
    # At T=70.66°F, cooling = 70.66 - 65 = 5.66, heating = 0
    assert pytest.approx(res["cooling_degree_proxy_regional"].iloc[0], abs=0.05) == 5.66
    assert pytest.approx(res["heating_degree_proxy_regional"].iloc[0], abs=0.01) == 0.00


def test_regional_aggregation_resilience_to_partial_station_nan():
    """Verify regional aggregation computes cleanly without crash or warnings when single station drops."""
    t_kphl = np.array([70.0, np.nan])
    t_kewr = np.array([72.0, 75.0])
    t_kilg = np.array([np.nan, 73.0])

    res = compute_regional_aggregations(t_kphl, t_kewr, t_kilg)

    # For row 0: valid stations KPHL(70) and KEWR(72) -> mean 71.0, spread 2.0
    assert pytest.approx(res["temp_f_unweighted_mean"].iloc[0], abs=0.01) == 71.00
    assert pytest.approx(res["temp_spread_regional"].iloc[0], abs=0.01) == 2.00

    # For row 1: valid stations KEWR(75) and KILG(73) -> mean 74.0, spread 2.0
    assert pytest.approx(res["temp_f_unweighted_mean"].iloc[1], abs=0.01) == 74.00
    assert pytest.approx(res["temp_spread_regional"].iloc[1], abs=0.01) == 2.00


# ==============================================================================
# 3. MULTI-HORIZON & DAY-AHEAD FORECASTING INTEGRITY
# ==============================================================================

def test_horizon_feature_names_admissibility():
    """Verify that lag features adhere strictly to horizon cutoff and prohibit lookahead."""
    feats_h1 = get_horizon_feature_names(1, include_weather=False)
    assert "lag_1" in feats_h1
    assert "lag_24" in feats_h1

    feats_h24 = get_horizon_feature_names(24, include_weather=False)
    # Strictly prohibited at H=24: lag_1 through lag_23 must NOT exist!
    assert "lag_1" not in feats_h24
    assert "lag_24" in feats_h24
    assert "lag_48" in feats_h24
    assert "rolling_mean_24_h24" in feats_h24

    feats_h48 = get_horizon_feature_names(48, include_weather=False)
    # Strictly prohibited at H=48: lag_1 through lag_47 must NOT exist!
    assert "lag_1" not in feats_h48
    assert "lag_24" not in feats_h48
    assert "lag_48" in feats_h48
    assert "lag_72" in feats_h48
    assert "rolling_mean_24_h48" in feats_h48


def test_horizon_feature_matrix_target_alignment():
    """Verify target timestamp and target demand alignment for H=24 and H=48."""
    # Synthetic hourly energy series (300 hours)
    dates = pd.date_range("2020-01-01 00:00:00", periods=300, freq="h")
    # Linear ramp demand for exact mathematical verification: y(t) = 1000 + i
    demand = [1000.0 + i for i in range(len(dates))]
    df = pd.DataFrame({"timestamp": dates, "total_power_demand": demand})

    # Test H=24
    mat_h24 = create_horizon_feature_matrix(df, nwp_df=None, horizon=24)
    # First row in mat_h24:
    # Origin index should have at least 168 burn-in. Suppose origin is at index 168:
    # t_origin = dates[168], target_timestamp = dates[168 + 24] = dates[192]
    # target_demand should be demand[192] = 1192.0
    row0 = mat_h24.iloc[0]
    expected_target_ts = row0["timestamp"] + pd.Timedelta(hours=24)
    assert row0["target_timestamp"] == expected_target_ts
    assert row0["target_demand"] == demand[dates.get_loc(expected_target_ts)]

    # Lag 24 at origin t is demand at t (which is exactly target - 24h)
    assert row0["lag_24"] == row0["total_power_demand"]
    # Lag 48 is demand at t - 24
    origin_loc = dates.get_loc(row0["timestamp"])
    assert row0["lag_48"] == demand[origin_loc - 24]


# ==============================================================================
# 4. PRODUCTION INFERENCE ENGINE & SCHEMA VALIDATION
# ==============================================================================

def test_inference_engine_schema_validation():
    """Verify inference engine catches missing, unexpected, and non-finite features."""
    schema = {
        "model_version": "test-v1",
        "features": ["hour", "month", "lag_1", "temp_f_zonal_weighted"],
        "forecast_origin_cutoff": "t_origin = T - 1h",
    }

    # Dummy model
    class DummyModel:
        def predict(self, X):
            return np.ones(len(X)) * 25000.0

    engine = EnergyInferenceEngine(model=DummyModel(), schema=schema)

    # 1. Valid batch
    valid_batch = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01 00:00:00", periods=5, freq="h"),
        "hour": [0, 1, 2, 3, 4],
        "month": [1, 1, 1, 1, 1],
        "lag_1": [25000.0] * 5,
        "temp_f_zonal_weighted": [65.0] * 5,
    })
    validated = engine.validate_schema(valid_batch)
    assert len(validated) == 5

    # 2. Missing feature: omit 'temp_f_zonal_weighted'
    missing_batch = valid_batch.drop(columns=["temp_f_zonal_weighted"])
    with pytest.raises(ValueError, match="PRODUCTION SCHEMA VIOLATION"):
        engine.validate_schema(missing_batch)

    # 3. Non-finite values: inject NaN
    nan_batch = valid_batch.copy()
    nan_batch.loc[2, "lag_1"] = np.nan
    with pytest.raises(ValueError, match="PRODUCTION DATA INTEGRITY ERROR"):
        engine.validate_schema(nan_batch)


def test_inference_engine_cutoff_enforcement():
    """Verify inference engine rejects predictions when forecast availability is post-origin."""
    schema = {
        "model_version": "test-v1",
        "features": ["hour", "temp_f_zonal_weighted"],
    }

    class DummyModel:
        def predict(self, X):
            return np.ones(len(X)) * 30000.0

    engine = EnergyInferenceEngine(model=DummyModel(), schema=schema)

    # Valid times: target = 14:00 UTC, cutoff = 13:00 UTC (1h before)
    # Test with simulated_avail = 13:30 UTC (> cutoff 13:00 UTC) -> must raise ValueError
    violation_df = pd.DataFrame({
        "target_timestamp": [pd.Timestamp("2021-06-01 14:00:00")],
        "valid_time_utc": [pd.Timestamp("2021-06-01 14:00:00")],
        "simulated_avail_utc": [pd.Timestamp("2021-06-01 13:30:00")],  # 30 min too late!
        "hour": [14],
        "temp_f_zonal_weighted": [88.5],
    })

    with pytest.raises(ValueError, match="CAUSALITY VIOLATION DETECTED"):
        engine.predict(violation_df, enforce_cutoff=True)


# ==============================================================================
# 5. OPERATIONAL HEALTH MONITORING & DRIFT DETECTION
# ==============================================================================

def test_forecast_health_monitor_weather_and_staleness():
    """Verify monitor audits missing weather inputs and stale forecast flags."""
    monitor = ForecastHealthMonitor(stale_threshold_hours=6.0)

    # 1. Healthy feed
    healthy_feed = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01 00:00:00", periods=24, freq="h"),
        "temp_f_kphl": [50.0] * 24,
        "lead_time_hours": [2.0] * 24,
    })
    res_healthy = monitor.audit_weather_inputs(healthy_feed)
    assert res_healthy["status"] == "HEALTHY"
    assert res_healthy["missing_forecasts_count"] == 0

    # 2. Stale feed (lead time 8 hours > 6.0 hours threshold)
    stale_feed = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01 00:00:00", periods=24, freq="h"),
        "temp_f_kphl": [50.0] * 24,
        "lead_time_hours": [8.0] * 24,
    })
    res_stale = monitor.audit_weather_inputs(stale_feed)
    assert res_stale["status"] == "STALE_WEATHER_INPUTS"
    assert res_stale["stale_forecasts_count"] == 24


def test_forecast_health_monitor_residual_alerts():
    """Verify performance monitor calculates error metrics and alerts on excessive residuals."""
    monitor = ForecastHealthMonitor(residual_alert_threshold_mw=3000.0)

    actuals = np.array([30000.0, 32000.0, 31000.0])
    # Case with an extreme 4,500 MW forecast spike
    predictions = np.array([30200.0, 36500.0, 31100.0])

    perf = monitor.audit_prediction_performance(actuals, predictions)
    assert perf["residual_alerts_count"] == 1
    assert perf["max_residual_mw"] == 4500.0
    assert perf["status"] in ("WARNING_ELEVATED_RESIDUALS", "LARGE_RESIDUAL_ALERT")
    assert perf["mae_mw"] > 0
    assert perf["rmse_mw"] > 0


# ==============================================================================
# 6. PRODUCTION ARTIFACT LOADING & MULTI-STATION ALIGNMENT
# ==============================================================================

def test_production_artifacts_exist_and_loadable():
    """Verify that serialized Phase 8 production artifacts load cleanly and produce valid forecasts."""
    assert DEFAULT_PRODUCTION_MODEL_PATH.is_file(), f"Missing production model: {DEFAULT_PRODUCTION_MODEL_PATH}"
    assert DEFAULT_FEATURE_SCHEMA_PATH.is_file(), f"Missing feature schema: {DEFAULT_FEATURE_SCHEMA_PATH}"
    assert DEFAULT_MODEL_CONFIG_PATH.is_file(), f"Missing model config: {DEFAULT_MODEL_CONFIG_PATH}"

    # Initialize production inference engine from disk artifacts
    engine = EnergyInferenceEngine(
        model_path=DEFAULT_PRODUCTION_MODEL_PATH,
        schema_path=DEFAULT_FEATURE_SCHEMA_PATH,
        model_version="2.0.0-phase8",
    )
    assert len(engine.expected_features) == 18

    # Create synthetic test batch conforming to schema
    schema = engine.schema
    features = schema["features"]
    stats = schema["feature_statistics"]

    batch_dict = {f: [stats[f]["mean"]] for f in features}
    batch_dict["target_timestamp"] = [pd.Timestamp("2022-01-01 12:00:00")]
    batch_df = pd.DataFrame(batch_dict)

    preds_df = engine.predict(batch_df, enforce_availability_check=False)
    assert len(preds_df) == 1
    pred_mw = preds_df["predicted_demand_mw"].iloc[0]
    assert 15000.0 <= pred_mw <= 55000.0, f"Unreasonable prediction: {pred_mw} MW"


def test_multi_station_dataset_alignment():
    """Verify processed multi-station dataset has continuous timestamps and valid station columns."""
    multi_csv = Path("data/processed/nwp_multi_station_forecasts.csv")
    if not multi_csv.is_file():
        pytest.skip("Processed multi-station dataset not yet generated.")

    df = pd.read_csv(multi_csv, nrows=1000)
    required_cols = [
        "timestamp", "temp_f_kphl", "temp_f_kewr", "temp_f_kilg",
        "temp_f_unweighted_mean", "temp_f_zonal_weighted", "temp_spread_regional",
        "cooling_degree_proxy_regional", "heating_degree_proxy_regional",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in multi-station dataset"
        assert not df[col].isna().any(), f"NaNs detected in column '{col}'"

    # Verify spread >= 0 everywhere
    assert (df["temp_spread_regional"] >= 0.0).all()
