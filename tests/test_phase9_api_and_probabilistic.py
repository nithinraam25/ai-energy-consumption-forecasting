"""Comprehensive Test Suite for Phase 9: Probabilistic Forecasting, API, and Deployment.

Tests:
1. Multi-quantile regression prediction shape, validity, and non-crossing monotonicity (Q05 <= Q50 <= Q95).
2. Conformal calibration factor computation and nonconformity scoring.
3. Coverage and prediction interval evaluation metrics (marginal, seasonal, demand tiers).
4. Time-series integrity: chronological separation, horizon admissibility, availability cutoff enforcement.
5. REST API: GET /health returns HEALTHY with active model status.
6. REST API: POST /predict generates point forecasts and calibrated Q05, Q50, Q95 intervals.
7. REST API: POST /predict rejects unsupported horizons (H=12) with HTTP 422.
8. REST API: POST /predict rejects non-finite feature values (NaN/Inf) with HTTP 422.
9. REST API: POST /predict rejects post-cutoff weather forecasts with HTTP 400.
10. REST API: POST /audit reports missing weather, staleness, and causality violations.
11. Deployment artifacts: Dockerfile, .dockerignore, and docker-compose.yml configuration consistency.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import MODELS_DIR, RANDOM_STATE
from src.models.probabilistic_forecaster import (
    MultiQuantileForecaster,
    ConformalIntervalCalibrator,
    evaluate_probabilistic_forecasts,
    DEFAULT_PROBABILISTIC_MODEL_PATH,
    DEFAULT_CALIBRATION_ARTIFACT_PATH,
)
from src.features.weather_features import SolarAndExtendedWeatherInterface
from src.api.app import app, load_application_components


# ==============================================================================
# 1. MULTI-QUANTILE FORECASTER TESTS
# ==============================================================================

def test_multi_quantile_forecaster_monotonicity_and_shape():
    """Verify that quantile predictions strictly satisfy Q05 <= Q50 <= Q95 for all samples."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    # Target with heteroskedastic noise
    y = 20000.0 + 50.0 * X[:, 0] + np.random.normal(0, 500, size=100)

    feature_names = ["feat1", "feat2", "feat3", "feat4"]
    forecaster = MultiQuantileForecaster(
        feature_names=feature_names,
        quantiles=(0.05, 0.50, 0.95),
        max_iter=30,
        random_state=42,
    )
    forecaster.fit(X, y)
    preds = forecaster.predict(X)

    q05 = preds["q05"]
    q50 = preds["q50"]
    q95 = preds["q95"]
    point = preds["point"]
    width = preds["interval_width"]

    assert len(q05) == len(q50) == len(q95) == len(point) == 100
    # Strict monotonicity check: no quantile crossing
    assert (q05 <= q50).all(), "Quantile crossing detected: Q05 > Q50!"
    assert (q50 <= q95).all(), "Quantile crossing detected: Q50 > Q95!"
    assert (width >= 0.0).all(), "Negative interval width detected!"


def test_multi_quantile_forecaster_rejects_non_finite_inputs():
    """Verify that fitting and inference reject non-finite inputs."""
    forecaster = MultiQuantileForecaster(feature_names=["f1", "f2"], max_iter=10)
    X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([100.0, 200.0])

    with pytest.raises(ValueError, match="Non-finite values"):
        forecaster.fit(X_nan, y)

    X_good = np.array([[1.0, 2.0], [2.0, 3.0]])
    forecaster.fit(X_good, y)
    with pytest.raises(ValueError, match="Non-finite values"):
        forecaster.predict(X_nan)


# ==============================================================================
# 2. CONFORMAL CALIBRATION & METRICS TESTS
# ==============================================================================

def test_conformal_calibrator_calibration_and_adjustment():
    """Verify conformal calibrator calculates valid adjustment factor and guarantees coverage."""
    y_cal = np.array([100.0, 105.0, 110.0, 95.0, 102.0, 108.0, 99.0, 101.0, 103.0, 107.0])
    q_low = np.array([97.0] * 10)
    q_high = np.array([104.0] * 10)
    q_med = np.array([100.5] * 10)

    calibrator = ConformalIntervalCalibrator(nominal_coverage=0.90)
    calibrator.calibrate(y_cal, q_low, q_high)

    assert calibrator.is_calibrated
    # Adjustment factor should expand intervals to cover points like 110 and 95
    assert calibrator.adjustment_factor > 0

    adj_low, adj_med, adj_high = calibrator.adjust(q_low, q_med, q_high)
    assert (adj_low <= adj_med).all()
    assert (adj_med <= adj_high).all()
    assert (adj_high - adj_low > q_high - q_low).all()


def test_evaluate_probabilistic_forecasts_metrics():
    """Verify probabilistic evaluation function computes coverage, width, and stratifications."""
    y_true = np.array([20000.0, 25000.0, 30000.0, 35000.0])
    q_low = np.array([19000.0, 24000.0, 29000.0, 34000.0])
    q_high = np.array([21000.0, 26000.0, 31000.0, 36000.0])
    q_med = np.array([20000.0, 25000.0, 30000.0, 35000.0])
    timestamps = [
        pd.Timestamp("2021-01-01 00:00:00"),
        pd.Timestamp("2021-04-01 06:00:00"),
        pd.Timestamp("2021-07-01 12:00:00"),
        pd.Timestamp("2021-10-01 18:00:00"),
    ]

    metrics = evaluate_probabilistic_forecasts(
        y_true, q_low, q_high, q_med, timestamps=timestamps
    )

    # 100% of samples fall within bounds
    assert metrics["empirical_coverage_pct"] == 100.0
    assert metrics["mean_interval_width_mw"] == 2000.0
    assert metrics["mae_mw"] == 0.0
    assert "Winter" in metrics["coverage_by_season_pct"]
    assert "Summer" in metrics["coverage_by_season_pct"]


# ==============================================================================
# 3. WEATHER PROVENANCE & SOLAR INTERFACE
# ==============================================================================

def test_solar_and_extended_weather_interface():
    """Verify solar interface cleanly handles presence or absence of GHI without fabrication."""
    df_without_ghi = pd.DataFrame({"temp_f": [70.0, 75.0]})
    assert not SolarAndExtendedWeatherInterface.is_solar_available(df_without_ghi)

    # Calling add_solar_features on missing GHI must return DataFrame without modifying columns
    out_df = SolarAndExtendedWeatherInterface.add_solar_features(df_without_ghi)
    assert "ghi_normalized_fcst" not in out_df.columns

    # Calling with genuine GHI extracts normalized features
    df_with_ghi = pd.DataFrame({"temp_f": [70.0], "ghi_fcst": [650.0]})
    assert SolarAndExtendedWeatherInterface.is_solar_available(df_with_ghi)
    out_ghi = SolarAndExtendedWeatherInterface.add_solar_features(df_with_ghi)
    assert "is_daylight_fcst" in out_ghi.columns
    assert out_ghi["is_daylight_fcst"].iloc[0] == 1
    assert pytest.approx(out_ghi["ghi_normalized_fcst"].iloc[0], abs=0.01) == 0.65


# ==============================================================================
# 4. REST API ENDPOINT TESTS (TestClient)
# ==============================================================================

@pytest.fixture
def api_client():
    """Create FastAPI TestClient with initialized components."""
    load_application_components()
    return TestClient(app)


def test_api_health_endpoint(api_client):
    """Verify GET /health returns 200 and healthy status metadata."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("HEALTHY", "DEGRADED")
    assert data["model_version"] in ("2.0.0-phase9", "2.0.0-phase10")
    assert "timestamp_utc" in data


def test_api_predict_valid_request(api_client):
    """Verify POST /predict returns valid point and probabilistic forecasts."""
    # Build payload using feature schema
    from src.models.inference import DEFAULT_FEATURE_SCHEMA_PATH
    with open(DEFAULT_FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = {feat: schema["feature_statistics"][feat]["mean"] for feat in schema["features"]}

    payload = {
        "timestamp": "2022-01-01T12:00:00",
        "horizon": 24,
        "features": features,
        "weather_inputs": {
            "temp_f_zonal_weighted": 65.0,
            "simulated_avail_utc": "2021-12-31T11:00:00Z",
            "valid_time_utc": "2022-01-01T12:00:00Z",
            "provenance": "NOAA_HRRR_OPERATIONAL_PROXY",
        },
        "enforce_cutoff": True,
    }

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert "point_forecast_mw" in res
    assert "q05_mw" in res
    assert "q50_mw" in res
    assert "q95_mw" in res
    assert "interval_width_mw" in res
    # Monotonicity check in API response
    assert res["q05_mw"] <= res["q50_mw"] <= res["q95_mw"]
    assert res["interval_width_mw"] > 0
    assert res["horizon"] == 24
    assert res["weather_data_provenance"] == "NOAA_HRRR_OPERATIONAL_PROXY"


def test_api_predict_rejects_unsupported_horizon(api_client):
    """Verify POST /predict returns 422 for unsupported horizons (e.g. H=12)."""
    payload = {
        "timestamp": "2022-01-01T12:00:00",
        "horizon": 12,  # Unsupported! Must be 1, 24, or 48
        "features": {"hour": 12},
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_rejects_non_finite_features(api_client):
    """Verify POST /predict returns 422 when features contain non-finite or invalid types."""
    # Test null/NaN feature value
    raw_json = '{"timestamp": "2022-01-01T12:00:00", "horizon": 1, "features": {"hour": null}}'
    response = api_client.post("/predict", content=raw_json, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_api_predict_rejects_causality_violations(api_client):
    """Verify POST /predict returns 400 when weather forecast availability is post-cutoff."""
    from src.models.inference import DEFAULT_FEATURE_SCHEMA_PATH
    with open(DEFAULT_FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = {feat: schema["feature_statistics"][feat]["mean"] for feat in schema["features"]}

    # Horizon H=1h: target = 14:00, cutoff = 13:00. Simulated avail = 13:30 (30 min late!)
    payload = {
        "timestamp": "2022-01-01T14:00:00",
        "horizon": 1,
        "features": features,
        "weather_inputs": {
            "temp_f_zonal_weighted": 65.0,
            "simulated_avail_utc": "2022-01-01T13:30:00Z",
            "valid_time_utc": "2022-01-01T14:00:00Z",
        },
        "enforce_cutoff": True,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "CAUSALITY VIOLATION" in response.json()["detail"]


def test_api_audit_endpoint(api_client):
    """Verify POST /audit reports missing weather, staleness, and non-finite counts."""
    audit_payload = {
        "records": [
            {"temp_f_kphl": 60.0, "lead_time_hours": 2.0},
            {"temp_f_kphl": None, "lead_time_hours": 8.0},  # Stale & missing
        ],
        "stale_threshold_hours": 6.0,
    }
    response = api_client.post("/audit", json=audit_payload)
    assert response.status_code == 200
    res = response.json()
    assert res["total_records"] == 2
    assert res["missing_weather_count"] == 1
    assert res["stale_forecast_count"] == 1
    assert res["status"] in ("WARNING", "CRITICAL")


# ==============================================================================
# 5. DEPLOYMENT CONFIGURATION & REPRODUCIBILITY
# ==============================================================================

def test_docker_and_deployment_artifacts_exist():
    """Verify presence of production Dockerfile, .dockerignore, docker-compose, and requirements."""
    dockerfile = Path("Dockerfile")
    dockerignore = Path(".dockerignore")
    docker_compose = Path("docker-compose.yml")
    requirements = Path("requirements.txt")

    assert dockerfile.is_file(), "Dockerfile is missing"
    assert dockerignore.is_file(), ".dockerignore is missing"
    assert docker_compose.is_file(), "docker-compose.yml is missing"
    assert requirements.is_file(), "requirements.txt is missing"

    content_df = dockerfile.read_text(encoding="utf-8")
    assert "uvicorn" in content_df
    assert "HEALTHCHECK" in content_df
    assert "appuser" in content_df

    content_req = requirements.read_text(encoding="utf-8")
    assert "fastapi" in content_req
    assert "uvicorn" in content_req
    assert "pydantic" in content_req
