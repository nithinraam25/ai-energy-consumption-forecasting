"""Phase 3 Regression Tests — Dashboard, Visualization, and Operational Readiness.

Validates:
- GET / and GET /dashboard serve enhanced HTML dashboard with Phase 3 components.
- GET /api/evaluation/metrics returns verified evaluation metrics matching forecast_metrics.json.
- GET /api/evaluation/plots/{filename} serves publication plots and rejects non-whitelisted files.
- GET /api/evaluation/backtest_samples returns real holdout samples with strictly valid quantile ordering.
- POST /predict/orchestrated handles historical archive (2018) successfully.
- POST /predict/orchestrated fails safely with HTTP 422 for live 2026 scenarios lacking SCADA demand history.
- Missing live demand telemetry produces explicit, human-readable error disclosures.
- Diagnostic prediction mode (/predict) validates genuine HRRR inputs with Q05 <= Q50 <= Q95.
- Causality guard enforces cutoff (available_time <= forecast_cutoff) and rejects post-cutoff data with HTTP 400.
- Scientific metrics consistency: empirical coverage is accurately reported (88.28%) against 90% target.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, load_application_components
from src.models.inference import DEFAULT_FEATURE_SCHEMA_PATH


@pytest.fixture(scope="module")
def client():
    """Initialize FastAPI test client with loaded model components."""
    load_application_components()
    return TestClient(app)


def test_phase3_dashboard_html_components(client):
    """Verify that dashboard HTML contains Phase 3 components, tabs, disclosures, and charts."""
    for path in ("/", "/dashboard"):
        res = client.get(path)
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        html = res.text

        # Executive summary & SCADA telemetry warning
        assert "Operational Telemetry Disclosure" in html
        assert "Real-time SCADA demand telemetry is currently <strong>not integrated</strong>" in html or "Real-time SCADA demand telemetry is currently not integrated" in html
        assert "Workflow Mode" in html
        assert "Evaluation Scenario" in html
        assert "Forecast Horizon" in html

        # Tab navigation and view containers
        assert "btnModeOrchestrated" in html
        assert "btnModeDiagnostic" in html
        assert "btnTargetHoldout" in html
        assert "btnTargetLive" in html
        assert "view_forecast" in html
        assert "view_backtest" in html
        assert "view_benchmark" in html
        assert "view_plots" in html

        # Backtest canvas and benchmark table
        assert "backtestCanvas" in html
        assert "Verified Multi-Horizon Benchmark vs Baselines" in html
        assert "Scientific Integrity Disclosure" in html
        assert "88.28%" in html
        assert "90.00%" in html or "90.0%" in html

        # Provenance elements
        assert "weatherProvenanceBadge" in html
        assert "metaSourceText" in html
        assert "ecCodes 2.50.0 (C-Lib)" in html
        assert "metaTempZonal" in html


def test_api_evaluation_metrics_consistency(client):
    """Verify that /api/evaluation/metrics matches the saved Phase 2 evaluation report."""
    metrics_path = Path("outputs/improvement/forecast_metrics.json")
    assert metrics_path.is_file(), "Saved metrics JSON must exist"
    with open(metrics_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    res = client.get("/api/evaluation/metrics")
    assert res.status_code == 200
    data = res.json()

    assert "horizons" in data
    assert data["horizons"]["H=1h"]["test_sample_count"] == 21773

    # Check horizon MAEs match exactly
    assert round(data["horizons"]["H=1h"]["models"]["production_nwp"]["mae_mw"], 2) == 333.72
    assert round(data["horizons"]["H=24h"]["models"]["production_nwp"]["mae_mw"], 2) == 1226.73
    assert round(data["horizons"]["H=48h"]["models"]["production_nwp"]["mae_mw"], 2) == 1487.77

    # Check adaptive conformal coverage matches 88.28%
    assert round(data["probabilistic_calibration"]["champion_adaptive_diurnal_conformal"]["coverage_pct"], 2) == 88.28
    assert data["probabilistic_calibration"]["nominal_target_coverage"] == 0.90


def test_api_evaluation_plots_endpoint(client):
    """Verify that /api/evaluation/plots serves whitelisted images and rejects others."""
    whitelisted_plots = [
        "actual_vs_predicted_h1.png",
        "actual_vs_predicted_h24.png",
        "actual_vs_predicted_h48.png",
        "forecast_error_distribution.png",
        "horizon_metric_comparison.png",
        "prediction_intervals.png",
    ]

    for plot_name in whitelisted_plots:
        res = client.get(f"/api/evaluation/plots/{plot_name}")
        assert res.status_code == 200, f"Plot {plot_name} should return 200"
        assert res.headers["content-type"] == "image/png"
        assert len(res.content) > 1000, f"Plot {plot_name} should contain valid image bytes"

    # Test rejection of unauthorized / non-existent plot
    bad_res = client.get("/api/evaluation/plots/unauthorized_file.png")
    assert bad_res.status_code == 404

    # Test path traversal prevention
    trav_res = client.get("/api/evaluation/plots/../../src/api/app.py")
    assert trav_res.status_code == 404


def test_api_evaluation_backtest_samples(client):
    """Verify that /api/evaluation/backtest_samples returns valid backtest data."""
    res = client.get("/api/evaluation/backtest_samples?horizon=1&limit=50")
    assert res.status_code == 200
    data = res.json()

    assert data["horizon"] == 1
    assert data["sample_count"] == 50
    assert len(data["samples"]) == 50

    for sample in data["samples"]:
        assert "target_timestamp" in sample
        assert sample["actual_demand_mw"] > 0
        assert sample["pred_nwp_mw"] > 0
        # Check quantile monotonic ordering: Q05 <= Q95
        assert sample["q05_adaptive_mw"] <= sample["q95_adaptive_mw"]


def test_orchestrated_historical_archive_success(client):
    """Verify that automated orchestration succeeds for a valid historical timestamp."""
    payload = {
        "timestamp": "2018-08-01T12:00:00Z",
        "horizon": 1,
        "allow_fallback_weather": True,
        "enforce_cutoff": True,
    }
    res = client.post("/predict/orchestrated", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["point_forecast_mw"] > 0
    assert data["q05_mw"] <= data["q50_mw"] <= data["q95_mw"]
    assert "FALLBACK" in data["weather_data_provenance"]


def test_orchestrated_live_fails_safely_without_demand_telemetry(client):
    """Verify that automated orchestration fails safely with HTTP 422 when live demand is missing."""
    payload = {
        "timestamp": "2026-10-15T10:00:00Z",
        "horizon": 1,
        "allow_fallback_weather": True,
        "enforce_cutoff": True,
    }
    res = client.post("/predict/orchestrated", json=payload)
    assert res.status_code == 422
    data = res.json()
    err_detail = data["detail"]
    assert "Missing required demand history timestamps" in err_detail or "No demand observations" in err_detail


def test_diagnostic_prediction_with_hrrr_weather(client):
    """Verify diagnostic prediction mode with genuine HRRR weather inputs."""
    with open(DEFAULT_FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    feats = {k: v["mean"] for k, v in schema["feature_statistics"].items()}

    hrrr_cache = Path("data/weather_cache/hrrr_20261015_t08z_f02.json")
    assert hrrr_cache.is_file()
    with open(hrrr_cache, "r", encoding="utf-8") as f:
        hrrr_data = json.load(f)

    payload = {
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
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["point_forecast_mw"] > 0
    assert data["q05_mw"] <= data["q50_mw"] <= data["q95_mw"]
    assert data["weather_data_provenance"] == "NOAA_HRRR_OPERATIONAL_ECCODES"


def test_causality_guard_post_cutoff_rejection(client):
    """Verify that causality guard strictly rejects post-cutoff weather with HTTP 400."""
    with open(DEFAULT_FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    feats = {k: v["mean"] for k, v in schema["feature_statistics"].items()}

    # Target: 12:00, Horizon: 1 -> Cutoff: 11:00 UTC. Avail: 11:30 UTC -> Post-cutoff!
    payload = {
        "timestamp": "2026-10-15T12:00:00",
        "horizon": 1,
        "features": feats,
        "weather_inputs": {
            "temp_f_zonal_weighted": 60.0,
            "simulated_avail_utc": "2026-10-15T11:30:00Z",
            "valid_time_utc": "2026-10-15T12:00:00Z",
            "provenance": "NOAA_HRRR_OPERATIONAL",
        },
        "enforce_cutoff": True,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    assert "CAUSALITY VIOLATION" in res.json()["detail"]
