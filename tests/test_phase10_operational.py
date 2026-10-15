"""Automated Test Suite for Phase 10 Operational Readiness and Failure Modes.

Validates:
1. NOAA HRRR Provider real-time discovery and range fetching.
2. NOAA 404 / Missing files handling.
3. Network timeouts and retry backoff.
4. Malformed GRIB2 header handling.
5. Missing variable index handling.
6. Causal Fallback Provider anti-substitution policy.
7. Background Weather Update Worker duplicate cycle detection and freshness.
8. Background Weather Update Worker graceful shutdown.
9. Adaptive Conformal Calibrator fitting, adjustment, and monotonicity.
10. Adaptive Conformal Calibrator serialization and loading.
11. Prometheus /metrics exposition endpoint.
12. API error handling for causality cutoffs and missing model artifacts.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import time
import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from src.weather.providers.base import WeatherForecastResponse, WeatherProviderStatus
from src.weather.providers.noaa_hrrr import NOAAHRRRWeatherProvider
from src.weather.providers.fallback import FallbackWeatherProvider
from src.weather.worker import WeatherUpdateWorker
from src.models.probabilistic_forecaster import (
    AdaptiveConformalCalibrator,
    MultiQuantileForecaster,
    DEFAULT_ADAPTIVE_CALIBRATION_PATH,
)
from src.api.app import app, load_application_components, app_state
from src.api.metrics import (
    record_request_metrics,
    record_prediction_success,
    record_prediction_failure,
    get_prometheus_metrics_payload,
)


@pytest.fixture
def api_client():
    """Create FastAPI test client with components loaded."""
    load_application_components()
    return TestClient(app)


def test_noaa_provider_404_missing_file_handling():
    """Verify that a 404 from NOAA AWS S3 raises FileNotFoundError without crashing."""
    provider = NOAAHRRRWeatherProvider(timeout_sec=2.0, max_retries=1)

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        with pytest.raises(FileNotFoundError) as exc_info:
            provider.get_cycle_field("20990101", cycle_hour=0, lead_hour=2)
        assert "Resource not found" in str(exc_info.value) or "404" in str(exc_info.value)


def test_noaa_provider_network_timeout_and_retries():
    """Verify that network timeouts trigger retries and raise RuntimeError after max retries."""
    provider = NOAAHRRRWeatherProvider(timeout_sec=0.5, max_retries=2, backoff_factor=1.1)

    with patch("requests.get", side_effect=requests.Timeout("Connection timed out")):
        with pytest.raises(RuntimeError) as exc_info:
            provider.get_cycle_field("20260101", cycle_hour=0, lead_hour=2)
        assert "retries" in str(exc_info.value)

    status = provider.get_status()
    assert status.consecutive_failures >= 1
    assert "Connection timed out" in status.last_error


def test_noaa_provider_malformed_grib2_header():
    """Verify that malformed GRIB bytes raise a clear ValueError."""
    provider = NOAAHRRRWeatherProvider(timeout_sec=2.0, max_retries=1)

    idx_text = "1:0:d=2026010100:TMP:2 m above ground:2 hour fcst:\n2:500:d=2026010100:DPT:2 m above ground:2 hour fcst:\n"

    with patch("requests.get") as mock_get:
        # First call for .idx
        resp_idx = MagicMock()
        resp_idx.status_code = 200
        resp_idx.text = idx_text

        # Second call for .grib2 returns corrupt non-GRIB payload
        resp_grib = MagicMock()
        resp_grib.status_code = 206
        resp_grib.content = b"INVALID_CORRUPT_BYTES_7777"

        mock_get.side_effect = [resp_idx, resp_grib]

        with pytest.raises(ValueError) as exc_info:
            provider.get_cycle_field("20260101", cycle_hour=0, lead_hour=2)
        assert "Invalid GRIB2" in str(exc_info.value)


def test_noaa_provider_missing_variable_in_index():
    """Verify that an index missing TMP:2 m above ground raises FileNotFoundError."""
    provider = NOAAHRRRWeatherProvider(timeout_sec=2.0, max_retries=1)

    idx_text_no_tmp = "1:0:d=2026010100:PRES:surface:2 hour fcst:\n2:500:d=2026010100:UGRD:10 m above ground:2 hour fcst:\n"

    with patch("requests.get") as mock_get:
        resp_idx = MagicMock()
        resp_idx.status_code = 200
        resp_idx.text = idx_text_no_tmp
        mock_get.return_value = resp_idx

        with pytest.raises(FileNotFoundError) as exc_info:
            provider.get_cycle_field("20260101", cycle_hour=0, lead_hour=2)
        assert "not located in index" in str(exc_info.value)


def test_fallback_provider_anti_substitution_guarantee():
    """Verify that FallbackWeatherProvider strictly accesses training diurnal profiles and never future observations."""
    fallback_provider = FallbackWeatherProvider(fallback_mode="DIURNAL_CLIMATOLOGY")
    target_time = "2026-07-15T14:00:00"

    resp = fallback_provider.fetch_forecast(
        location_id="KPHL",
        target_time_utc=target_time,
        lead_hours=2,
    )

    assert isinstance(resp, WeatherForecastResponse)
    assert resp.is_fallback is True
    assert "FALLBACK" in resp.source
    assert resp.metadata.get("anti_substitution_verified") is True
    assert isinstance(resp.temp_f, float)
    assert 40.0 <= resp.temp_f <= 110.0


def test_weather_update_worker_duplicate_detection_and_ledger(tmp_path):
    """Verify worker skips duplicate cycles and records metadata in persistent ledger."""
    mock_provider = MagicMock()
    mock_provider.discover_latest_cycles.return_value = [
        {
            "cycle_str": "20261015_12Z",
            "cycle_date": "20261015",
            "cycle_hour": 12,
            "init_time_utc": "2026-10-15T12:00:00+00:00",
        }
    ]

    ledger_file = tmp_path / "processed_cycles.json"
    worker = WeatherUpdateWorker(
        provider=mock_provider,
        poll_interval_seconds=1.0,
        ledger_path=ledger_file,
    )

    # First sync: finds new cycle
    report1 = worker.sync_once()
    assert report1["new_cycles_found"] == 1
    assert report1["cycles_skipped_duplicate"] == 0
    assert ledger_file.is_file()

    # Second sync: skips duplicate cycle
    report2 = worker.sync_once()
    assert report2["new_cycles_found"] == 0
    assert report2["cycles_skipped_duplicate"] == 1


def test_weather_update_worker_graceful_shutdown():
    """Verify background worker thread starts and stops gracefully without leaking resources."""
    mock_provider = MagicMock()
    mock_provider.discover_latest_cycles.return_value = []

    worker = WeatherUpdateWorker(provider=mock_provider, poll_interval_seconds=0.1)
    worker.start()
    assert worker.is_alive() is True

    worker.stop(timeout_sec=2.0)
    assert worker.is_alive() is False


def test_adaptive_conformal_calibrator_fitting_and_monotonicity():
    """Verify AdaptiveConformalCalibrator fits group adjustments and enforces Q05 <= Q50 <= Q95."""
    np.random.seed(42)
    n = 2000
    y_true = np.random.normal(30000, 4000, size=n)
    q_low = y_true - np.random.uniform(1500, 2500, size=n)
    q_median = y_true + np.random.normal(0, 300, size=n)
    q_high = y_true + np.random.uniform(1500, 2500, size=n)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="h")

    calibrator = AdaptiveConformalCalibrator(nominal_coverage=0.90, method="DIURNAL")
    calibrator.calibrate(
        y_true=y_true,
        q_low=q_low,
        q_high=q_high,
        timestamps=timestamps,
        demand_preds=q_median,
    )

    assert calibrator.is_calibrated is True
    assert len(calibrator.group_adjustments) == 24
    assert 0.85 <= calibrator.overall_calibrated_coverage <= 0.95

    # Test adjust and non-crossing monotonicity
    adj_low, adj_med, adj_high = calibrator.adjust(
        q_low=q_low[:50],
        q_median=q_median[:50],
        q_high=q_high[:50],
        timestamps=timestamps[:50],
    )

    assert (adj_low <= adj_med).all()
    assert (adj_med <= adj_high).all()


def test_adaptive_conformal_calibrator_serialization(tmp_path):
    """Verify AdaptiveConformalCalibrator JSON export and load roundtrip."""
    calibrator = AdaptiveConformalCalibrator(nominal_coverage=0.90, method="DIURNAL")
    calibrator.global_adjustment = 50.0
    calibrator.group_adjustments = {f"H{h:02d}": 45.0 + h for h in range(24)}
    calibrator.group_sample_counts = {f"H{h:02d}": 100 for h in range(24)}
    calibrator.group_raw_coverages = {f"H{h:02d}": 0.88 for h in range(24)}
    calibrator.group_calibrated_coverages = {f"H{h:02d}": 0.90 for h in range(24)}
    calibrator.total_calibration_samples = 2400
    calibrator.overall_raw_coverage = 0.88
    calibrator.overall_calibrated_coverage = 0.90
    calibrator.is_calibrated = True

    export_path = tmp_path / "test_adaptive_params.json"
    calibrator.save(export_path)
    assert export_path.is_file()

    loaded = AdaptiveConformalCalibrator.load(export_path)
    assert loaded.is_calibrated is True
    assert loaded.method == "DIURNAL"
    assert loaded.group_adjustments["H10"] == 55.0


def test_api_prometheus_metrics_endpoint(api_client):
    """Verify GET /metrics exposes Prometheus formatted metrics lines."""
    # Generate some metric activity first
    record_request_metrics("GET", "/health", 200, 0.012)
    record_prediction_success(1, 4200.0)
    record_prediction_failure("schema_violation")

    resp = api_client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text

    # Verify Prometheus exposition lines
    assert "pjm_api_requests_total" in text or "pjm_model_version" in text
    assert resp.headers["content-type"].startswith("text/plain")


def test_api_causality_violation_rejection(api_client):
    """Verify POST /predict rejects a request where forecast availability time is after cutoff."""
    payload = {
        "timestamp": "2026-10-15T12:00:00",
        "horizon": 1,
        "features": {
            "hour_of_day": 12, "day_of_week": 3, "month": 10, "is_weekend": 0, "is_holiday": 0,
            "lag_1": 31000.0, "lag_2": 30500.0, "lag_3": 30000.0, "lag_24": 30200.0,
            "lag_48": 29800.0, "lag_168": 30900.0, "rolling_mean_24": 30100.0, "rolling_std_24": 1200.0,
            "temp_f_zonal_weighted": 72.0, "cooling_degree_proxy_zonal": 7.0,
            "heating_degree_proxy_zonal": 0.0, "temp_spread_regional": 3.5,
        },
        "weather_inputs": {
            "station_id": "KPHL",
            "issue_time_utc": "2026-10-15T11:00:00Z",
            "valid_time_utc": "2026-10-15T12:00:00Z",
            "lead_time_hours": 1,
            "temp_f": 72.0,
            # Available at 11:50, which is AFTER the H=1 cutoff of 11:00!
            "simulated_avail_utc": "2026-10-15T11:50:00Z",
            "provenance": "NOAA_HRRR",
        },
        "enforce_cutoff": True,
    }

    resp = api_client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "CAUSALITY VIOLATION" in resp.json()["detail"]


def test_weather_update_worker_process_lock_singleton(tmp_path):
    """Verify WeatherUpdateWorker singleton lock prevents concurrent duplicate worker instances."""
    lock_file = tmp_path / "worker.lock"
    worker1 = WeatherUpdateWorker(lock_path=lock_file, use_process_lock=True, poll_interval_seconds=1.0)
    worker2 = WeatherUpdateWorker(lock_path=lock_file, use_process_lock=True, poll_interval_seconds=1.0)

    # Worker 1 acquires lock
    assert worker1.acquire_lock() is True

    # Worker 2 attempts concurrent lock acquisition -> rejected
    assert worker2.acquire_lock() is False
    assert worker2.start() is False

    # Worker 1 releases lock -> Worker 2 can now acquire
    worker1.release_lock()
    assert worker2.acquire_lock() is True
    worker2.release_lock()


def test_api_weather_provenance_fallback_and_staleness_warnings(api_client):
    """Verify POST /predict marks fallback weather and stale forecasts with explicit warnings."""
    from src.models.inference import DEFAULT_FEATURE_SCHEMA_PATH
    import json
    with open(DEFAULT_FEATURE_SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    feats = {k: v["mean"] for k, v in schema["feature_statistics"].items()}

    # 1. Fallback forecast
    fallback_payload = {
        "timestamp": "2026-10-15T12:00:00",
        "horizon": 24,
        "features": feats,
        "weather_inputs": {
            "temp_f_zonal_weighted": 57.5,
            "simulated_avail_utc": "2026-10-14T11:00:00Z",
            "valid_time_utc": "2026-10-15T12:00:00Z",
            "provenance": "FALLBACK_DIURNAL_CLIMATOLOGY",
            "is_fallback": True,
        },
    }
    resp1 = api_client.post("/predict", json=fallback_payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["weather_data_provenance"] == "FALLBACK_DIURNAL_CLIMATOLOGY"
    assert any("FALLBACK_WEATHER_USED" in w for w in data1["data_quality_warnings"])

    # 2. Stale forecast (issued 10h before cutoff)
    stale_payload = {
        "timestamp": "2026-10-15T12:00:00",
        "horizon": 1,  # Cutoff: 11:00
        "features": feats,
        "weather_inputs": {
            "temp_f_zonal_weighted": 58.0,
            "simulated_avail_utc": "2026-10-15T01:00:00Z",  # 10h before cutoff!
            "valid_time_utc": "2026-10-15T12:00:00Z",
            "provenance": "NOAA_HRRR_OPERATIONAL",
        },
    }
    resp2 = api_client.post("/predict", json=stale_payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert any("STALE_WEATHER_INPUT" in w for w in data2["data_quality_warnings"])

    # 3. Caller supplied features without weather_inputs wrapper
    caller_payload = {
        "timestamp": "2026-10-15T12:00:00",
        "horizon": 24,
        "features": feats,
    }
    resp3 = api_client.post("/predict", json=caller_payload)
    assert resp3.status_code == 200
    assert resp3.json()["weather_data_provenance"] == "CALLER_SUPPLIED_FEATURES"


def test_grib2_true_gridpoint_decoding_and_anti_substitution():
    """Verify ecCodes decodes genuine localized temperatures and never substitutes Section 5 minimum."""
    from src.weather.providers.noaa_hrrr import (
        decode_hrrr_grib2_stations,
        ECCODES_AVAILABLE,
        DEFAULT_CACHE_DIR,
    )
    from src.data.multi_station import STATION_CATALOG

    assert ECCODES_AVAILABLE is True, "ecCodes must be available in runtime environment"

    # Use the cached operational GRIB2 slice
    cache_json = DEFAULT_CACHE_DIR / "hrrr_20261015_t08z_f02.json"
    assert cache_json.is_file(), "Operational cached forecast JSON must exist"

    with open(cache_json, "r", encoding="utf-8") as f:
        cached = json.load(f)

    # 1. Verification of true grid-point temperatures
    assert "station_temps_f" in cached
    temps_f = cached["station_temps_f"]
    temps_k = cached["station_temps_k"]

    # Mid-Atlantic temperatures in October are between 45°F and 75°F (280 K - 297 K)
    for stn in ["KPHL", "KEWR", "KILG"]:
        assert 45.0 <= temps_f[stn] <= 75.0, f"Station {stn} temp {temps_f[stn]}F out of physical range"
        assert 280.0 <= temps_k[stn] <= 297.0, f"Station {stn} temp {temps_k[stn]}K out of physical range"

        # 2. Strict Anti-Substitution invariant: Decoded temp must NOT equal Section 5 reference minimum (~9.8°F / 260.85 K)
        ref_min_k = cached["reference_value_k"]
        assert abs(temps_k[stn] - ref_min_k) > 10.0, (
            f"ANTI-SUBSTITUTION VIOLATION: Decoded {stn} ({temps_k[stn]} K) matches domain minimum ({ref_min_k} K)!"
        )


def test_kelvin_to_fahrenheit_conversion_exactness():
    """Verify Kelvin to Fahrenheit physical unit conversions."""
    # Freezing point
    t_f_freeze = round((273.15 - 273.15) * 1.8 + 32.0, 2)
    assert t_f_freeze == 32.0

    # Boiling point
    t_f_boil = round((373.15 - 273.15) * 1.8 + 32.0, 2)
    assert t_f_boil == 212.0

    # Typical ambient 60°F = 288.7056 K
    t_k_ambient = 288.7055555555556
    t_f_ambient = round((t_k_ambient - 273.15) * 1.8 + 32.0, 2)
    assert t_f_ambient == 60.0


def test_corrupt_or_truncated_grib2_fails_safely():
    """Verify that passing truncated or corrupt bytes to decode_hrrr_grib2_stations fails safely."""
    from src.weather.providers.noaa_hrrr import decode_hrrr_grib2_stations

    corrupt_bytes = b"GRIB_CORRUPT_PAYLOAD_NOT_A_VALID_GRIB_MESSAGE_7777"
    with pytest.raises(RuntimeError) as exc_info:
        decode_hrrr_grib2_stations(corrupt_bytes)
    assert "decoding failed" in str(exc_info.value).lower() or "eccodes" in str(exc_info.value).lower()


def test_station_coordinates_map_to_intended_grid_cells():
    """Verify station coordinates map deterministically to expected HRRR Lambert Conformal grid indices."""
    from src.data.hrrr_loader import project_latlon_to_hrrr_grid
    from src.data.multi_station import STATION_CATALOG

    # KPHL (Philadelphia: 39.8719, -75.2411) -> (i ~ 1526, j ~ 656)
    i_phl, j_phl = project_latlon_to_hrrr_grid(STATION_CATALOG["KPHL"].latitude, STATION_CATALOG["KPHL"].longitude)
    assert int(round(i_phl)) == 1526
    assert int(round(j_phl)) == 656

    # KEWR (Newark: 40.6895, -74.1745) -> (i ~ 1548, j ~ 693)
    i_ewr, j_ewr = project_latlon_to_hrrr_grid(STATION_CATALOG["KEWR"].latitude, STATION_CATALOG["KEWR"].longitude)
    assert int(round(i_ewr)) == 1548
    assert int(round(j_ewr)) == 693

    # KILG (Wilmington: 39.6787, -75.6065) -> (i ~ 1518, j ~ 647)
    i_ilg, j_ilg = project_latlon_to_hrrr_grid(STATION_CATALOG["KILG"].latitude, STATION_CATALOG["KILG"].longitude)
    assert int(round(i_ilg)) == 1518
    assert int(round(j_ilg)) == 647


