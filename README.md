# AI-Powered Energy Consumption Forecasting

A modular, production-ready machine learning system for temporal multivariate energy consumption forecasting.

---

## 📁 Project Structure

```text
AI-Powered Energy Consumption Forecasting/
├── data/
│   ├── raw/                  # Immutable raw energy datasets
│   ├── processed/            # Cleaned, resampled, and aligned series
│   └── simulated/            # Controlled test datasets
├── docs/                     # Technical documentation
│   ├── README.md             # Master documentation index
│   ├── architecture.md       # System architecture and layer topology
│   ├── data_pipeline_and_flow.md # Data lifecycle and sequencing
│   ├── dataset_specifications.md # Target schema and quality contracts
│   ├── model_learning_and_methodology.md # ML/DL formulation and losses
│   ├── modules_catalog.md    # Codebase modules and interfaces
│   └── execution_and_validation_log.md # Implementation and verification records
├── models/                   # Serialized model weights and scalers (.joblib, .pt)
├── outputs/
│   ├── forecasts/            # Generated prediction tables (CSV/Parquet)
│   ├── metrics/              # Evaluation summaries (MAE, RMSE, MAPE, R2)
│   ├── plots/                # Forecast vs. actual visualization figures
│   └── logs/                 # Execution and pipeline logs
├── src/
│   ├── __init__.py           # Package marker
│   └── config.py             # Centralized path and environment configuration
├── tests/
│   └── test_config.py        # Scaffolding and configuration unit tests
├── .gitignore                # Git exclusions for environments, models, and data
├── main.py                   # Initial execution entry point
└── requirements.txt          # Core dependencies
```

---

## 🚀 Setup & Getting Started

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.13)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize Runtime Directories
```bash
python main.py
```

### 4. Run Tests
```bash
python -m pytest -q
```

### 5. Run Phase 10 Operational & Adaptive Calibration Benchmark
```bash
python -m src.experiment --phase10
```

### 6. Launch Production REST API & Interactive Dashboard
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Available Endpoints & Interfaces:
- `GET  /` or `GET /dashboard`: Interactive HTML5/CSS3 glassmorphic forecasting dashboard with Automated Orchestration and Diagnostic modes.
- `GET  /health`: Model readiness, calibration state, schema status, and orchestrator health.
- `POST /predict`: Dual-mode forecast endpoint — accepts manual `features` dictionary (diagnostic mode) or automatically orchestrates cached weather and demand lags if `features` is omitted.
- `POST /predict/orchestrated`: Dedicated automated forecast orchestration endpoint requiring only `timestamp` and `horizon`.
- `POST /audit`: Operational data quality, causality, staleness, and drift gate.
- `GET  /metrics`: Prometheus exposition format metrics exporter.

#### Example 1: Automated Forecast Orchestration (`POST /predict/orchestrated`)
```bash
curl -X POST "http://localhost:8000/predict/orchestrated" \
     -H "Content-Type: application/json" \
     -d '{
       "timestamp": "2018-08-01T12:00:00Z",
       "horizon": 1,
       "enforce_cutoff": true,
       "allow_fallback_weather": true
     }'
```

#### Example 2: Diagnostic / Template Forecast (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "timestamp": "2026-10-15T10:00:00Z",
       "horizon": 1,
       "features": {
         "hour": 10.0, "day_of_week": 3.0, "day_of_month": 15.0, "month": 10.0, "is_weekend": 0.0,
         "hour_sin": 0.5, "hour_cos": -0.866, "day_of_week_sin": 0.434, "day_of_week_cos": -0.901,
         "lag_1": 31200.0, "lag_24": 30500.0, "lag_168": 30900.0, "rolling_mean_24": 30800.0, "rolling_std_24": 1250.0,
         "temp_f_zonal_weighted": 57.5, "cooling_degree_proxy_regional": 0.0, "heating_degree_proxy_regional": 7.5, "temp_spread_regional": 4.7
       }
     }'
```

### 7. Run Weather Update Worker Daemon
```bash
python -m src.weather.worker
```

### 8. Run Full Multi-Service Stack via Docker Compose (API + Worker + Prometheus)
```bash
docker compose build --no-cache
docker compose up -d
```
*(Note: Docker runtime validation is currently BLOCKED on local Windows hosts where Docker is not installed on PATH, but fully configured for Docker-enabled Linux/Cloud environments.)*

---

## 🔬 Benchmark Performance & Provenance Reality

Evaluated across the 21,773-hour held-out test partition (2016–2018):
- **Real-Time ($H=1\text{h}$)**: **$332.90\text{ MW}$ MAE** ($R^2 = 0.9952$, $\text{MAPE} = 1.07\%$) vs Persistence ($1,054.91\text{ MW}$).
- **Day-Ahead ($H=24\text{h}$)**: **$1,227.22\text{ MW}$ MAE** ($R^2 = 0.9364$, $\text{MAPE} = 3.95\%$) — **33.3% improvement** over no-weather tree ($1,840.80\text{ MW}$).
- **Two-Day ($H=48\text{h}$)**: **$1,485.31\text{ MW}$ MAE** ($R^2 = 0.9082$, $\text{MAPE} = 4.80\%$) — **40.3% improvement** over no-weather tree ($2,485.69\text{ MW}$).
- **Probabilistic Coverage**: **$88.21\%$ empirical coverage** against 90% target using Adaptive Diurnal CQR ($4,180.61\text{ MW}$ mean width); reaches **$92.16\%$ coverage** during peak heatwaves ($T > 75^\circ\text{F}$). Zero quantile-crossing violations across all test hours.
- **Meteorological Provenance**: Genuine operational GRIB2 ingestion is verified via ECMWF `ecCodes` on live 2026 cycles; historical 2002–2018 benchmark metrics represent a retrospective simulation of operational forecast quality using AR(1) error innovations on surface observations.

---

## ⚠️ Operational Boundary & Verification Disclosures

1. **Demand Telemetry Requirement**:
   - The historical archive (`data/processed/pjm_cleaned.csv`) spans from 2002-01-01 through 2018-08-03 00:00:00 UTC.
   - Automated operational forecasting for future dates (e.g., 2026-10-15) requires live SCADA demand observations at or before the decision cutoff ($T - H\text{ hours}$). If caller invokes automated orchestration without registered operational telemetry, the system returns `HTTP 422 Unprocessable Content` (`MissingDemandHistoryError`) rather than silently hallucinating demand history or substituting schema means.
2. **Containerization & Docker Status**:
   - Multi-container Docker deployment (`docker compose up`) is currently **BLOCKED / UNVERIFIED** on the local Windows host where the Docker daemon is not installed on PATH. Container configurations (`Dockerfile`, `docker-compose.yml`, `deploy/prometheus.yml`) are fully syntax-validated and prepared for Docker-enabled Linux environments.
3. **Visual Browser Rendering Status**:
   - Visual UI rendering in a live browser is recorded as **NOT TESTED** because headless browser drivers (Playwright) could not be downloaded in this sandboxed offline environment. Backend HTTP response contracts, HTML delivery, and JSON schemas are fully verified via automated test suites.
4. **Historical Weather Provenance**:
   - Historical evaluation metrics (2002–2018) represent a retrospective simulation of operational forecast quality using AR(1) error innovations on surface observations (NOAA ISD). Genuine operational GRIB2 ingestion is verified on live 2026 cycles via ECMWF `ecCodes`.

---

## ⚖️ License & Data Redistribution Notice

- **Software License**: Pending explicit human operator selection prior to public repository release.
- **Data Distribution**:
  - PJM East hourly electricity consumption data is published by PJM Interconnection.
  - Meteorological observations (NOAA ISD) and numerical weather predictions (NOAA HRRR) are provided by the US National Oceanic and Atmospheric Administration (NOAA).
  - Users are responsible for complying with all upstream data use policies and redistribution agreements.

---

## 📚 Technical Documentation Directory

For complete architectural specifications, mathematical formulations, and validation reports, refer to [`docs/README.md`](docs/README.md):
- [Phase 8: Production Readiness & Multi-Station NWP](docs/phase8_production_readiness.md)
- [Phase 9: Probabilistic Energy Forecasting & API](docs/phase9_probabilistic_api.md)
- [Phase 10: Real-Time NOAA Integration & Adaptive Calibration](docs/phase10_operational_readiness.md)
- [Phase 11.2: NOAA HRRR Grid-Point Temperature Validation](docs/phase11_2_hrrr_gridpoint_validation.md)
- [Phase 12: Deployment and Security Validation](docs/phase12_deployment_and_security_validation.md)
- [Phase 13: End-to-End Forecasting Validation & Scientific Audit](docs/phase13_end_to_end_forecast_validation.md)
- [Phase 14: Final System Demonstration, Dashboard, & Handoff](docs/phase14_final_demo_and_handoff.md)
- [Comprehensive Final Project Readiness & Integration Audit](docs/final_project_readiness_audit.md)
- [Execution & Validation Log](docs/execution_and_validation_log.md)

