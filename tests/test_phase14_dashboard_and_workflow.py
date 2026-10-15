"""Smoke and Integration Test for Complete Supported API Workflow and Dashboard.

Validates:
- GET / and GET /dashboard serve HTML dashboard interface with HTTP 200.
- GET /health reports healthy status and model versions.
- POST /predict with genuine NOAA HRRR operational inputs succeeds and enforces Q05 <= Q50 <= Q95.
- POST /predict with fallback weather triggers explicit warning and provenance.
- POST /predict with post-cutoff weather rejects with HTTP 400.
- POST /audit evaluates batch health and flags issues.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, load_application_components
from src.models.inference import DEFAULT_FEATURE_SCHEMA_PATH


@pytest.fixture(scope="module")
def client():
    """Initialize FastAPI test client with loaded components."""
    load_application_components()
    return TestClient(app)


def test_dashboard_endpoint_serves_html(client):
    """Verify that root and dashboard endpoints serve the production HTML dashboard."""
    for path in ("/", "/dashboard"):
        res = client.get(path)
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "PJM East AI Energy Forecasting Dashboard" in res.text
        assert "Probabilistic Demand Forecast" in res.text


def test_complete_end_to_end_forecasting_workflow(client):
    """Execute end-to-end operational inference cycle with genuine HRRR weather."""
    # 1. Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["status"] == "HEALTHY"
    assert h_data["model_loaded"] is True
    assert h_data["probabilistic_capable"] is True

    # 2. Load schema template
    with open(DEFAULT_FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    feats = {k: v["mean"] for k, v in schema["feature_statistics"].items()}

    # 3. Predict with genuine HRRR f02 operational inputs (H=1)
    hrrr_cache = Path("data/weather_cache/hrrr_20261015_t08z_f02.json")
    assert hrrr_cache.is_file(), "Cached genuine HRRR cycle must exist"
    with open(hrrr_cache, "r", encoding="utf-8") as f:
        hrrr_data = json.load(f)

    pred_payload = {
        "timestamp": "2026-10-15T10:00:00",
        "horizon": 1,
        "features": feats,
        "weather_inputs": {
            "temp_f_kphl": hrrr_data["station_temps_f"]["KPHL"],
            "temp_f_kewr": hrrr_data["station_temps_f"]["KEWR"],
            "temp_f_kilg": hrrr_data["station_temps_f"]["KILG"],
            "temp_f_zonal_weighted": 57.05,
            "simulated_avail_utc": hrrr_data["available_time_utc"],
            "valid_time_utc": hrrr_data["valid_time_utc"],
            "provenance": "NOAA_HRRR_OPERATIONAL_ECCODES",
        },
        "enforce_cutoff": True,
    }
    p_res = client.post("/predict", json=pred_payload)
    assert p_res.status_code == 200
    p_data = p_res.json()
    assert p_data["point_forecast_mw"] > 0
    assert p_data["q05_mw"] <= p_data["q50_mw"] <= p_data["q95_mw"]
    assert p_data["weather_data_provenance"] == "NOAA_HRRR_OPERATIONAL_ECCODES"

    # 4. Fallback weather prediction
    fb_payload = {
        "timestamp": "2026-10-15T12:00:00",
        "horizon": 24,
        "features": feats,
        "weather_inputs": {
            "temp_f_zonal_weighted": 57.05,
            "simulated_avail_utc": "2026-10-14T11:00:00Z",
            "valid_time_utc": "2026-10-15T12:00:00Z",
            "provenance": "FALLBACK_DIURNAL_CLIMATOLOGY",
            "is_fallback": True,
        },
        "enforce_cutoff": True,
    }
    fb_res = client.post("/predict", json=fb_payload)
    assert fb_res.status_code == 200
    fb_data = fb_res.json()
    assert fb_data["weather_data_provenance"] == "FALLBACK_DIURNAL_CLIMATOLOGY"
    assert any("FALLBACK_WEATHER_USED" in w for w in fb_data["data_quality_warnings"])

    # 5. Strict Causality violation rejection
    viol_payload = {
        "timestamp": "2026-10-15T12:00:00",
        "horizon": 1,
        "features": feats,
        "weather_inputs": {
            "temp_f_zonal_weighted": 57.05,
            "simulated_avail_utc": "2026-10-15T11:45:00Z",  # 45m after H=1 cutoff
            "valid_time_utc": "2026-10-15T12:00:00Z",
            "provenance": "NOAA_HRRR_OPERATIONAL",
        },
        "enforce_cutoff": True,
    }
    viol_res = client.post("/predict", json=viol_payload)
    assert viol_res.status_code == 400
    assert "CAUSALITY VIOLATION" in viol_res.json()["detail"]

    # 6. Audit endpoint execution
    audit_res = client.post(
        "/audit",
        json={"records": [{"temp_f_kphl": 58.06, "lead_time_hours": 2}], "stale_threshold_hours": 6},
    )
    assert audit_res.status_code == 200
    assert audit_res.json()["status"] in ("HEALTHY", "WARNING", "CRITICAL")
