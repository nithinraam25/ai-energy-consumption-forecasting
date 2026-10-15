"""Comprehensive Test Suite for Phase 15: Automated Forecast Orchestration.

Tests:
1. Automatic weather-cache consumption and regional feature calculation.
2. Fallback weather provider activation when cached data is missing or causally ineligible.
3. Automatic demand-lag and rolling statistics construction without future lookahead.
4. Missing, incomplete, and non-finite demand history rejection with clear errors.
5. Strict temporal ineligibility rejection when direct lag_1 extends beyond decision cutoff.
6. Exact production schema alignment and feature ordering (18 features).
7. End-to-end orchestration with controlled fixtures.
8. End-to-end integration test using actual cached HRRR GRIB2 cycle (labeled explicitly).
9. Multi-step recursive roll-forward for H=24.
10. FastAPI integration: POST /predict/orchestrated and POST /predict (features=None).
11. Causality, invalid horizon, and missing history error responses.
12. Dashboard endpoint compatibility and updated UI controls.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, load_application_components
from src.data.demand_provider import (
    DemandHistoryProvider,
    DemandLagResult,
    MissingDemandHistoryError,
    IncompleteDemandHistoryError,
    TemporalIneligibilityError,
)
from src.models.inference import DEFAULT_FEATURE_SCHEMA_PATH
from src.services.orchestrator import (
    ForecastOrchestrator,
    OrchestratedForecastResult,
    InvalidHorizonError,
)
from src.weather.service import (
    WeatherFeatureService,
    WeatherFeatureResult,
    WeatherUnavailableError,
)


@pytest.fixture(scope="module")
def client():
    """Initialize FastAPI test client with loaded components."""
    load_application_components()
    return TestClient(app)


@pytest.fixture
def synthetic_demand_fixture() -> pd.DataFrame:
    """Generate 200 hours of regular hourly demand data for deterministic unit tests.

    Period: 2025-01-01 00:00:00 to 2025-01-09 07:00:00 UTC.
    Pattern: linear ramp demand[i] = 20000.0 + i * 50.0.
    """
    dates = pd.date_range("2025-01-01 00:00:00", periods=300, freq="h", tz="UTC")
    demand = [20000.0 + i * 50.0 for i in range(len(dates))]
    return pd.DataFrame({"timestamp": dates, "consumption": demand})


# ==============================================================================
# 1. WEATHER FEATURE SERVICE TESTS
# ==============================================================================

def test_weather_feature_service_consumes_cached_hrrr():
    """Integration test: Verify consumption of actual cached genuine HRRR GRIB2 cycle.

    Uses `data/weather_cache/hrrr_20261015_t08z_f02.json`.
    Target: 2026-10-15 10:00:00 UTC, Cutoff: 2026-10-15 09:00:00 UTC (H=1).
    """
    svc = WeatherFeatureService()
    res = svc.get_weather_features(
        target_timestamp="2026-10-15T10:00:00Z",
        horizon=1,
        cutoff_time_utc="2026-10-15T09:00:00Z",
        allow_fallback=False,
    )

    assert isinstance(res, WeatherFeatureResult)
    assert res.is_fallback is False
    assert res.provenance == "NOAA_HRRR_OPERATIONAL_ECCODES"
    assert res.temp_f_kphl == 58.06
    assert res.temp_f_kewr == 54.91
    assert res.temp_f_kilg == 59.61
    assert pytest.approx(res.temp_f_zonal_weighted, abs=0.01) == 57.05
    assert pytest.approx(res.temp_spread_regional, abs=0.01) == 4.70
    assert res.cooling_degree_proxy_regional == 0.0
    assert pytest.approx(res.heating_degree_proxy_regional, abs=0.01) == 7.95


def test_weather_feature_service_fallback_on_missing_cache():
    """Unit test: Verify automatic diurnal climatology fallback when cache is missing."""
    svc = WeatherFeatureService()
    # Historical timestamp with no cached HRRR file
    res = svc.get_weather_features(
        target_timestamp="2018-08-01T12:00:00Z",
        horizon=1,
        cutoff_time_utc="2018-08-01T11:00:00Z",
        allow_fallback=True,
    )

    assert isinstance(res, WeatherFeatureResult)
    assert res.is_fallback is True
    assert "FALLBACK" in res.provenance
    assert res.temp_f_zonal_weighted > 0.0
    assert res.cooling_degree_proxy_regional >= 0.0
    assert res.heating_degree_proxy_regional >= 0.0
    assert any("FALLBACK_WEATHER_USED" in w for w in res.warnings)


def test_weather_feature_service_rejects_when_fallback_disallowed():
    """Unit test: Verify explicit error when cache is missing and allow_fallback=False."""
    svc = WeatherFeatureService()
    with pytest.raises(WeatherUnavailableError, match="No causally eligible NOAA HRRR forecast"):
        svc.get_weather_features(
            target_timestamp="2018-08-01T12:00:00Z",
            horizon=1,
            cutoff_time_utc="2018-08-01T11:00:00Z",
            allow_fallback=False,
        )


# ==============================================================================
# 2. DEMAND HISTORY PROVIDER TESTS
# ==============================================================================

def test_demand_history_provider_automatic_lag_construction(synthetic_demand_fixture):
    """Unit test: Verify exact mathematical computation of lag and rolling features."""
    prov = DemandHistoryProvider(demand_df=synthetic_demand_fixture)

    # Pick index 180 as target (T = dates[180])
    dates = synthetic_demand_fixture["timestamp"]
    target_ts = dates.iloc[180]  # Horizon 1 -> cutoff at index 179

    res = prov.get_demand_lags(target_timestamp=target_ts, horizon=1)
    assert isinstance(res, DemandLagResult)

    # Pattern: demand[i] = 20000 + i * 50
    expected_lag_1 = 20000.0 + 179 * 50.0      # Index 179 (T - 1h)
    expected_lag_24 = 20000.0 + (180 - 24) * 50.0  # Index 156 (T - 24h)
    expected_lag_168 = 20000.0 + (180 - 168) * 50.0  # Index 12 (T - 168h)

    assert res.lag_1 == expected_lag_1
    assert res.lag_24 == expected_lag_24
    assert res.lag_168 == expected_lag_168

    # Rolling 24h: indices 156 to 179 inclusive (24 observations)
    window_vals = [20000.0 + i * 50.0 for i in range(156, 180)]
    assert pytest.approx(res.rolling_mean_24, abs=0.01) == float(np.mean(window_vals))
    assert pytest.approx(res.rolling_std_24, abs=0.01) == float(np.std(window_vals, ddof=1))


def test_demand_history_provider_temporal_ineligibility(synthetic_demand_fixture):
    """Unit test: Verify that asking for direct lags at H=24 is rejected as future lookahead."""
    prov = DemandHistoryProvider(demand_df=synthetic_demand_fixture)
    dates = synthetic_demand_fixture["timestamp"]
    target_ts = dates.iloc[180]

    # For H=24, cutoff is T - 24h. Observation at T - 1h occurred 23 hours AFTER cutoff!
    with pytest.raises(TemporalIneligibilityError, match="TEMPORAL INELIGIBILITY"):
        prov.get_demand_lags(target_timestamp=target_ts, horizon=24)


def test_demand_history_provider_missing_history_error(synthetic_demand_fixture):
    """Unit test: Verify clear error when target requires history beyond dataset boundaries."""
    prov = DemandHistoryProvider(demand_df=synthetic_demand_fixture)

    # Date far in the future
    with pytest.raises(MissingDemandHistoryError, match="No demand observations available at or before decision cutoff"):
        prov.get_demand_lags(target_timestamp="2030-01-01T12:00:00Z", horizon=1)

    # Early date with fewer than 168 hours of burn-in history
    early_ts = synthetic_demand_fixture["timestamp"].iloc[50]
    with pytest.raises(MissingDemandHistoryError, match="Missing required demand history timestamps"):
        prov.get_demand_lags(target_timestamp=early_ts, horizon=1)


def test_demand_history_provider_incomplete_nan_history(synthetic_demand_fixture):
    """Unit test: Verify rejection when demand window contains NaNs."""
    corrupted_df = synthetic_demand_fixture.copy()
    corrupted_df.loc[170, "consumption"] = np.nan
    prov = DemandHistoryProvider(demand_df=corrupted_df)

    dates = corrupted_df["timestamp"]
    target_ts = dates.iloc[180]

    with pytest.raises(IncompleteDemandHistoryError, match="NaN values detected in rolling 24h window"):
        prov.get_demand_lags(target_timestamp=target_ts, horizon=1)


# ==============================================================================
# 3. FORECAST ORCHESTRATOR TESTS
# ==============================================================================

def test_orchestration_service_end_to_end_fixture(synthetic_demand_fixture):
    """Unit test: Complete orchestration pipeline using controlled synthetic fixture."""
    prov = DemandHistoryProvider(demand_df=synthetic_demand_fixture)
    orch = ForecastOrchestrator(demand_provider=prov)

    dates = synthetic_demand_fixture["timestamp"]
    target_ts = dates.iloc[180]

    res = orch.orchestrate_forecast(
        target_timestamp=target_ts,
        horizon=1,
        enforce_cutoff=True,
        allow_fallback_weather=True,
    )

    assert isinstance(res, OrchestratedForecastResult)
    assert res.point_forecast_mw > 0.0
    assert res.q05_mw <= res.q50_mw <= res.q95_mw
    assert res.interval_width_mw > 0.0
    assert "FALLBACK" in res.weather_data_provenance

    # Verify exact schema feature keys and order
    with open(DEFAULT_FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    expected_features = schema["features"]

    assert list(res.features_used.keys()) == expected_features
    for col in expected_features:
        val = res.features_used[col]
        assert not np.isnan(val) and not np.isinf(val)


def test_orchestration_service_with_actual_cached_hrrr_integration():
    """Integration test: Complete orchestration using actual genuine cached HRRR cycle.

    Combines:
    - Genuine decoded NOAA HRRR GRIB2 file `hrrr_20261015_t08z_f02.json`
    - Controlled 168-hour demand history fixture ending at 2026-10-15 09:00:00 UTC.
    """
    # Create required 168-hour history ending at 09:00:00 UTC
    dates = pd.date_range("2026-10-08 10:00:00", "2026-10-15 09:00:00", freq="h", tz="UTC")
    demand_vals = [32000.0 + (i % 24) * 150.0 for i in range(len(dates))]
    fixture_df = pd.DataFrame({"timestamp": dates, "consumption": demand_vals})

    prov = DemandHistoryProvider(demand_df=fixture_df)
    orch = ForecastOrchestrator(demand_provider=prov)

    res = orch.orchestrate_forecast(
        target_timestamp="2026-10-15T10:00:00Z",
        horizon=1,
        enforce_cutoff=True,
        allow_fallback_weather=False,
    )

    assert isinstance(res, OrchestratedForecastResult)
    assert res.point_forecast_mw > 0.0
    assert res.q05_mw <= res.q50_mw <= res.q95_mw
    assert res.weather_data_provenance == "NOAA_HRRR_OPERATIONAL_ECCODES"
    assert res.features_used["temp_f_zonal_weighted"] == 57.05
    assert res.features_used["temp_spread_regional"] == 4.70


def test_orchestration_service_recursive_h24(synthetic_demand_fixture):
    """Unit test: Multi-step recursive roll-forward for Day-Ahead horizon H=24."""
    prov = DemandHistoryProvider(demand_df=synthetic_demand_fixture)
    orch = ForecastOrchestrator(demand_provider=prov)

    dates = synthetic_demand_fixture["timestamp"]
    target_ts = dates.iloc[210]

    res = orch.orchestrate_forecast(
        target_timestamp=target_ts,
        horizon=24,
        enforce_cutoff=True,
        allow_fallback_weather=True,
    )

    assert isinstance(res, OrchestratedForecastResult)
    assert res.horizon == 24
    assert res.point_forecast_mw > 0.0
    assert res.q05_mw <= res.q50_mw <= res.q95_mw
    assert any("RECURSIVE_DISPATCH_ROLLFORWARD" in w for w in res.data_quality_warnings)

    # Mathematical consistency checks:
    # At T (index 210), lag_24 is consumption at T - 24h (index 186 = origin_ts)
    expected_lag_24 = float(synthetic_demand_fixture["consumption"].iloc[186])
    expected_lag_168 = float(synthetic_demand_fixture["consumption"].iloc[210 - 168])
    assert pytest.approx(res.features_used["lag_24"], abs=0.01) == expected_lag_24
    assert pytest.approx(res.features_used["lag_168"], abs=0.01) == expected_lag_168


def test_orchestration_service_recursive_h48(synthetic_demand_fixture):
    """Regression test: Multi-step recursive roll-forward for Two-Day horizon H=48."""
    prov = DemandHistoryProvider(demand_df=synthetic_demand_fixture)
    orch = ForecastOrchestrator(demand_provider=prov)

    dates = synthetic_demand_fixture["timestamp"]
    target_ts = dates.iloc[240]  # Cutoff at index 192 (192 > 168)

    res = orch.orchestrate_forecast(
        target_timestamp=target_ts,
        horizon=48,
        enforce_cutoff=True,
        allow_fallback_weather=True,
    )

    assert isinstance(res, OrchestratedForecastResult)
    assert res.horizon == 48
    assert res.point_forecast_mw > 0.0
    assert res.q05_mw <= res.q50_mw <= res.q95_mw
    assert any("48 steps from origin" in w for w in res.data_quality_warnings)
    # At T (index 240), lag_168 is at index 72 (<= origin index 192)
    expected_lag_168 = float(synthetic_demand_fixture["consumption"].iloc[240 - 168])
    assert pytest.approx(res.features_used["lag_168"], abs=0.01) == expected_lag_168


def test_orchestration_service_rejects_invalid_horizon():
    """Unit test: Verify that requesting an unsupported horizon raises InvalidHorizonError."""
    orch = ForecastOrchestrator()
    with pytest.raises(InvalidHorizonError, match="Unsupported forecast horizon: 12"):
        orch.orchestrate_forecast("2018-08-01T12:00:00Z", horizon=12)


# ==============================================================================
# 4. API & DASHBOARD INTEGRATION TESTS
# ==============================================================================

def test_api_predict_orchestrated_endpoint_success(client):
    """API Integration: Test POST /predict/orchestrated on historical holdout archive."""
    payload = {
        "timestamp": "2018-08-01T12:00:00Z",
        "horizon": 1,
        "enforce_cutoff": True,
        "allow_fallback_weather": True,
    }
    resp = client.post("/predict/orchestrated", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["point_forecast_mw"] > 0.0
    assert data["q05_mw"] <= data["q50_mw"] <= data["q95_mw"]
    assert "FALLBACK" in data["weather_data_provenance"]


def test_api_predict_orchestrated_missing_demand_error(client):
    """API Integration: Verify that future targets without SCADA feed return HTTP 422."""
    payload = {
        "timestamp": "2026-10-15T10:00:00Z",
        "horizon": 1,
        "enforce_cutoff": True,
        "allow_fallback_weather": True,
    }
    resp = client.post("/predict/orchestrated", json=payload)
    assert resp.status_code == 422
    err_detail = resp.json()["detail"]
    assert "Missing required demand history timestamps" in err_detail or "No demand observations" in err_detail


def test_api_predict_route_supports_orchestration_mode(client):
    """API Integration: Verify POST /predict with features=None invokes automated orchestration."""
    payload = {
        "timestamp": "2018-08-01T12:00:00Z",
        "horizon": 1,
        "features": None,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["point_forecast_mw"] > 0.0


def test_dashboard_endpoint_serves_updated_orchestrated_dashboard(client):
    """Verify that root and dashboard endpoints serve updated Phase 15 orchestration controls."""
    for path in ("/", "/dashboard"):
        res = client.get(path)
        assert res.status_code == 200
        assert "Automated Orchestration" in res.text
        assert "Workflow Mode" in res.text
        assert "Holdout Archive" in res.text
        assert "setWorkflowMode" in res.text
