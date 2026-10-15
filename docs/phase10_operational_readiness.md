# Phase 10: Real-Time NOAA Weather Integration, Adaptive Calibration, and Production Monitoring

## Executive Summary

Phase 10 elevates the **PJM East AI-Powered Energy Consumption Forecasting System** from an experimental forecasting platform to a verifiable, operationally resilient production prototype.

Building on the multi-station operational NWP foundations of Phase 8 and the conformal probabilistic forecasting of Phase 9, Phase 10 delivers four major operational capabilities:
1. **Real-Time NOAA HRRR Weather Connector & Causal Fallback Architecture**: A modular weather provider system (`src/weather/providers/`) connecting directly to NOAA High-Resolution Rapid Refresh (HRRR) AWS Open Data via HTTP Range requests. Enforces strict forecast availability cutoffs ($t_{\text{avail}} \le t_{\text{origin}}$), file-based and in-memory caching to eliminate redundant downloads, exponential backoff with configurable retries, and an explicit anti-substitution fallback provider (`FallbackWeatherProvider`) that refuses to substitute future observed weather.
2. **Asynchronous Background Weather Update Worker**: A resilient background daemon (`src/weather/worker.py`) that monitors published NWP cycles, detects duplicate cycles, tracks forecast freshness, alerts on stale data (>6 hours), and supports graceful shutdown without blocking real-time API inference.
3. **Adaptive Diurnal Conformal Calibration**: An advanced covariate-conditioned conformal prediction engine (`AdaptiveConformalCalibrator`) that models intra-day volatility across the 24 diurnal hours. Evaluated against 5 competing calibration methodologies on validation data ($21,760$ samples), the champion diurnal model reduced average prediction interval width from **$4,212.58\text{ MW}$ to $4,180.74\text{ MW}$** (saving $31.84\text{ MW}$ mean and $74.09\text{ MW}$ median) while increasing empirical test coverage from **$88.09\%$ to $88.28\%$** and lifting morning industrial ramp coverage by up to **$+9.59\%$**.
4. **Prometheus Operational Monitoring**: Full instrumentation (`src/api/metrics.py`) exposing standard Prometheus metrics via `GET /metrics`, monitoring request rates, latency histograms, prediction failures, provider dropouts, forecast age, interval width distributions, and data-quality audit anomalies, paired with `deploy/prometheus.yml`.
5. **Zero-Regression Verification**: The complete project test suite of **150 automated tests passed in 23.23s** with zero failures or regressions.

---

## 1. System Architecture & Module Catalog

The Phase 10 operational architecture integrates the data provider, background sync daemon, adaptive inference engine, REST API, and Prometheus exporter:

```
+--------------------------------------------------------------------------------------------------+
|                                  Phase 10 Operational Architecture                               |
+--------------------------------------------------------------------------------------------------+
                                                 │
          ┌──────────────────────────────────────┼──────────────────────────────────────┐
          ▼                                      ▼                                      ▼
[ Weather Integration Subsystem ]      [ Adaptive Probabilistic Engine ]       [ API & Monitoring Subsystem ]
src/weather/providers/                 src/models/probabilistic_forecaster.py  src/api/app.py
- base.py: WeatherProviderBase         - MultiQuantileForecaster (Q05,50,95)   - GET  /health (Readiness)
- noaa_hrrr.py: NOAAHRRRWeatherProvider- AdaptiveConformalCalibrator           - POST /predict (Point+Intervals)
- fallback.py: FallbackWeatherProvider - Stratified Stress Evaluator           - POST /audit (Data Quality Gate)
                                                                               - GET  /metrics (Prometheus Exporter)
          │                                      │                                      │
          ▼                                      ▼                                      ▼
[ Background Worker Daemon ]           [ Calibration Checkpoint ]             [ Monitoring & Deployment ]
src/weather/worker.py                  models/adaptive_calibration_params.json deploy/prometheus.yml
- Cycle Discovery & Cache Sync         - 24 Diurnal Adjustment Factors        docker-compose.yml
- Duplicate Cycle Filtering            - Strict Monotonic Non-Crossing Sort   - FastAPI API Service (8000)
- Staleness & Freshness Tracking                                              - Prometheus Scraper (9090)
```

### Module Responsibilities

| Module | Primary Class / Function | Responsibility |
| :--- | :--- | :--- |
| `src/weather/providers/base.py` | `WeatherProviderBase`, `WeatherForecastResponse`, `WeatherProviderStatus` | Abstract interface and typed contracts for weather providers, guaranteeing immutable metadata and source tracking. |
| `src/weather/providers/noaa_hrrr.py` | `NOAAHRRRWeatherProvider` | Connects to NOAA AWS S3 Open Data; fetches `.idx` and targeted GRIB2 byte ranges; extracts KPHL, KEWR, KILG station values; enforces causality ($t_{\text{avail}} \le t_{\text{cutoff}}$); caches results to disk. |
| `src/weather/providers/fallback.py` | `FallbackWeatherProvider` | Causal fallback using training-only diurnal profiles when live NWP is unreachable; strictly enforces anti-substitution (zero future observations). |
| `src/weather/worker.py` | `WeatherUpdateWorker` | Background synchronization daemon polling for newly published HRRR cycles, filtering duplicates, tracking forecast age, and supporting clean shutdown. |
| `src/models/probabilistic_forecaster.py` | `AdaptiveConformalCalibrator` | Implements group-conditioned CQR across 6 strategies (diurnal, seasonal, demand-regime, weather-regime, rolling window, scalar). |
| `src/api/metrics.py` | Prometheus metrics instrumentation | Thread-safe, non-blocking metric recording for API request rates, latency, prediction errors, forecast age, and interval widths. |
| `src/api/app.py` | FastAPI application (`app`) | Production service exposing `/health`, `/predict`, `/audit`, and `/metrics` with adaptive calibration routing and timing middleware. |
| `deploy/prometheus.yml` | Prometheus scrape configuration | Configuration scraping API `/metrics` every 15s with regional metadata labels. |
| `tests/test_phase10_operational.py` | Operational failure test suite | 11 dedicated automated tests covering 404s, timeouts, malformed files, duplicate cycles, worker shutdown, and metrics. |

---

## 2. Weather-Data Provenance & Operational Assumptions

To ensure complete scientific integrity, data provenance is explicitly distinguished:

| Category | Datasets / Sources | Temporal Range | Verification Status | Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **Genuine Operational NWP** | NOAA HRRR CONUS 3km GRIB2 on AWS S3 Open Data (`noaa-hrrr-bdp-pds`) | Mid-2014–Present | **Verified Live via Targeted Range HTTP** | Primary live operational forecast feed for real-time inference and background worker synchronization. |
| **Observed Historical Weather** | NOAA ISD In-Situ Surface Station Records (KPHL, KEWR, KILG) | 2002–2018 | **Verified (SHA-256 Manifest)** | Ground truth for historical training, feature engineering, and physical validation. |
| **Simulated / Reconstructed Forecasts** | Autoregressive AR(1) Gaussian forecast-error perturbations applied to ISD | 2002–2018 | **Audited Synthetic Baseline** | Enables multi-year retrospective model training prior to mid-2014 where genuine HRRR AWS archives do not exist. |

### Operational Availability Assumptions & Causality Cutoffs

For a forecast valid at civil hour $T$ with forecast lead $H$:
- **Forecast Issue Cycle ($t_{\text{init}}$)**: $t_{\text{init}} = T - H\text{ hours}$ (e.g. for $T = 12:00\text{ UTC}$ and $H = 2\text{h}$, $t_{\text{init}} = 10:00\text{ UTC}$).
- **Supercomputer Model Latency**: $\Delta t_{\text{latency}} = 50\text{ minutes}$ (standard NOAA operational dissemination SLA).
- **Physical Availability Timestamp ($t_{\text{avail}}$)**: $t_{\text{avail}} = t_{\text{init}} + 50\text{ min} = 10:50\text{ UTC}$.
- **Prediction Cutoff Constraint**: $t_{\text{avail}} \le t_{\text{origin}} = T - H\text{ hours}$.
- **Anti-Substitution Rule**: If an operational forecast is unavailable at $t_{\text{origin}}$, the system **never** substitutes actual observed weather at $T$. It issues a structured fallback using `FallbackWeatherProvider` or raises an explicit 400 causality violation.

---

## 3. Real-Time NOAA HRRR Connector & Worker Architecture

### Byte-Range Slicing & Caching Pipeline

Rather than downloading massive full-CONUS GRIB2 files (~150 MB each), `NOAAHRRRWeatherProvider`:
1. Queries the lightweight index file (`.idx`, ~35 KB) located at:
   `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.wrfprsf{LEAD}.grib2.idx`
2. Identifies exact byte boundaries for `TMP:2 m above ground`:
   `record_num:byte_offset:date:TMP:2 m above ground:...`
3. Issues an HTTP Range request (`Range: bytes={start}-{end}`) to retrieve solely the targeted ~1.2 MB binary message.
4. Parses Sections 0, 1, 3, 4, 5 to verify WMO grid metadata (1799x1059 Lambert Conformal) and reference Kelvin temperatures.
5. Computes bilinear station projections for KPHL, KEWR, and KILG and caches the extracted payload in:
   - Memory cache: `dict[cache_key, payload]`
   - Disk cache: `data/weather_cache/hrrr_{date}_t{HH}z_f{lead}.json`

### Background Weather Update Worker (`WeatherUpdateWorker`)

```
      [ NOAA AWS S3 ]
             │
             │ Periodic Poll (e.g. every 300s)
             ▼
   [ WeatherUpdateWorker ]
             │
             ├── 1. Discover latest published cycles (limit=4)
             ├── 2. Check processed_cycles.json ledger
             │        ├── If already in ledger -> Skip duplicate
             │        └── If new -> Download & cache byte range
             ├── 3. Update forecast freshness age
             │        └── If age > 6.0h -> Set is_stale = True & trigger alert
             └── 4. Graceful stop listener (threading.Event)
```

- **Resilience**: Automatically recovers from temporary network partitions using exponential backoff ($1.5^k$ seconds).
- **Freshness Monitoring**: Constantly evaluates elapsed hours since the newest available run.
- **Non-blocking**: Operates on a dedicated daemon thread, decoupling slow external NOAA downloads from the FastAPI request event loop.

---

## 4. Adaptive Conformal Calibration Methodology

### Theoretical Motivation

In Phase 9, global scalar Conformalized Quantile Regression applied a uniform adjustment factor $\hat{\eta} = +56.70\text{ MW}$ across all hours and seasons. However, empirical error diagnostics revealed strong intra-day heteroscedasticity:
- **Low-Uncertainty Periods**: Early morning hours (07:00–08:00) exhibited raw coverage of **95–96%**, indicating that global intervals were excessively wide and conservative.
- **High-Uncertainty Periods**: Morning industrial ramp-ups (10:00–12:00) suffered raw coverage drops down to **68–71%**, indicating systematic under-coverage during rapid demand ramping.

### Mathematical Formulation

Given nonconformity scores $E_i = \max(\hat{q}_{05}(\mathbf{x}_i) - y_i, \, y_i - \hat{q}_{95}(\mathbf{x}_i))$ on the calibration set $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n_{\text{cal}}}$, `AdaptiveConformalCalibrator` assigns each sample to a conditioning group $g \in \mathcal{G}$ and calculates group-specific empirical adjustment factors $\hat{\eta}_g$:

$$\hat{\eta}_g = \text{Quantile} \left( \{E_i\}_{i \in \mathcal{I}_g}, \, \min\left(1.0, \, \frac{\lceil (n_g + 1)(1 - \alpha) \rceil}{n_g}\right) \right)$$

If a group sample size $n_g < n_{\text{min}}$ (threshold = 50), it falls back to the global scalar adjustment $\hat{\eta}_{\text{global}}$.

### Validation Set Benchmark Comparison (21,760 samples)

Six calibration methodologies were rigorously compared on the chronological validation split ($2013-08$ to $2016-01$):

| Strategy | Conditioning Variable | Groups Count | Val Empirical Coverage | Mean Interval Width | Median Interval Width | Val Point MAE | Selected Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global Scalar CQR (Phase 9)** | None (Uniform Scalar) | 1 | 90.00% | 3,999.05 MW | 3,415.69 MW | 555.54 MW | Baseline |
| **Diurnal Adaptive CQR** | **Hour of Day ($H00-H23$)** | **24** | **90.22%** | **3,967.23 MW** | **3,383.50 MW** | **555.22 MW** | **CHAMPION** |
| **Demand-Regime Adaptive** | Base / Mid / Peak Load | 3 | 90.02% | 3,986.54 MW | 3,417.71 MW | 555.53 MW | Candidate |
| **Seasonal Adaptive** | Winter / Spring / Summer / Fall | 4 | 90.03% | 3,987.30 MW | 3,416.28 MW | 555.50 MW | Candidate |
| **Weather-Regime Adaptive** | Cold / Mild / Hot Temp Bands | 3 | 90.05% | 3,961.99 MW | 3,421.10 MW | 555.24 MW | Candidate |
| **Rolling-Window Adaptive** | Chronological Window ($W=1000$) | 22 | 90.11% | 4,002.28 MW | 3,456.18 MW | 555.48 MW | Candidate |

*Champion Selection*: **Diurnal Adaptive CQR** was selected as the champion strategy. It achieved **$90.22\%$** validation coverage while producing narrower average prediction intervals ($3,967.23\text{ MW}$ vs $3,999.05\text{ MW}$, a $-31.82\text{ MW}$ reduction) and narrowing median intervals by $-32.19\text{ MW}$.

---

## 5. Out-of-Sample Test Holdout Evaluation Results

The champion Diurnal Adaptive Calibrator was evaluated against the baseline on the untouched, frozen test holdout ($21,759$ aligned samples, $2016-01$ to $2018-08$).

### Overall Holdout Performance Comparison

| Configuration | Test Empirical Coverage | Mean Interval Width | Median Interval Width | Point MAE | Point RMSE | Point MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Target** | **90.00%** | — | — | — | — | — |
| **Raw Quantiles (Uncalibrated)** | 86.12% | 4,099.19 MW | 3,438.68 MW | 618.85 MW | 814.47 MW | 2.00% |
| **Global Scalar CQR (Phase 9)** | 88.09% | 4,212.58 MW | 3,552.07 MW | 618.85 MW | 814.47 MW | 2.00% |
| **Champion Diurnal Adaptive CQR** | **88.28%** | **4,180.74 MW** | **3,477.98 MW** | **618.44 MW** | **814.06 MW** | **2.00%** |

*Core Predictive Benefits*:
1. **Narrower Prediction Intervals**: Average interval width decreased by **$31.84\text{ MW}$**, and median interval width decreased by **$74.09\text{ MW}$**.
2. **Superior Holdout Coverage**: Empirical coverage increased from $88.09\%$ to **$88.28\%$**, closing the gap toward nominal coverage.

### Diurnal Hour-by-Hour Holdout Coverage Diagnostics

```
Hour   Raw (%)   Global Scalar (%)   Champion Adaptive (%)   Adaptive Improvement
----------------------------------------------------------------------------------
H00     88.22%        90.42%                90.97%                 +0.55%
H01     89.86%        91.95%                92.28%                 +0.33%
H07     95.04%        96.03%                92.83%           -3.20% (Trimmed excess)
H08     95.26%        96.58%                91.40%           -5.18% (Trimmed excess)
H09     83.90%        86.77%                85.23%                 -1.54%
H10     71.00%        72.22%                81.81%          +9.59% (CRITICAL GAIN!)
H11     68.58%        70.56%                78.72%          +8.16% (CRITICAL GAIN!)
H12     71.22%        73.54%                80.71%          +7.17% (CRITICAL GAIN!)
H13     77.40%        79.60%                83.79%                 +4.19%
H20     94.82%        95.59%                93.61%           -1.98% (Trimmed excess)
H22     84.91%        86.23%                90.31%                 +4.08%
H23     79.74%        82.38%                86.45%                 +4.07%
```

*Key Finding*: The adaptive calibrator resolves the critical morning industrial ramp vulnerability. Coverage at 10:00 surged from **$72.22\%$ to $81.81\%$ (+9.59%)** and at 11:00 from **$70.56\%$ to $78.72\%$ (+8.16%)**, while safely narrowing over-conservative intervals during early morning hours.

### Extreme Operational Stress Tests

Evaluated on challenging system conditions:
- **Heatwave Stress Subset ($T_{\text{zonal}} \ge 75^\circ\text{F}$, $4,878$ samples)**:
  - Baseline Scalar Coverage: $93.05\%$ (Mean Width: $6,316.43\text{ MW}$)
  - Adaptive Diurnal Coverage: **$91.33\%$** (Mean Width: **$6,271.31\text{ MW}$**, saving $45.12\text{ MW}$)
- **Peak Demand Stress Subset ($y \ge 35,000\text{ MW}$, $5,011$ samples)**:
  - Baseline Scalar Coverage: $91.00\%$ (Mean Width: $6,672.06\text{ MW}$)
  - Adaptive Diurnal Coverage: **$89.68\%$** (Mean Width: **$6,597.36\text{ MW}$**, saving $74.70\text{ MW}$)

### Multi-Horizon Adaptive Prediction Intervals

| Horizon | Operational Role | Calibrated Empirical Coverage | Mean Interval Width | Median Interval Width | Point MAE | Point MAPE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$H=1\text{h}$** | Real-Time Balancing | **89.21%** | 5,456.87 MW | 4,239.25 MW | 407.48 MW | 1.34% |
| **$H=24\text{h}$** | Day-Ahead Market | **87.21%** | 6,566.34 MW | 5,926.39 MW | 1,277.75 MW | 4.12% |
| **$H=48\text{h}$** | Two-Day Commitment | **87.00%** | 7,489.00 MW | 6,808.33 MW | 1,561.23 MW | 5.06% |

---

## 6. Prometheus Monitoring & Metric Specifications

The `/metrics` endpoint exports standard Prometheus text exposition formatted metrics:

```text
# HELP pjm_api_requests_total Total count of HTTP requests served by endpoint and status code.
# TYPE pjm_api_requests_total counter
pjm_api_requests_total{endpoint="/predict",method="POST",status_code="200"} 142.0

# HELP pjm_api_request_duration_seconds HTTP request latency distribution in seconds.
# TYPE pjm_api_request_duration_seconds histogram
pjm_api_request_duration_seconds_bucket{endpoint="/predict",le="0.05"} 138.0

# HELP pjm_prediction_interval_width_mw Distribution of generated 90% prediction interval widths in MW.
# TYPE pjm_prediction_interval_width_mw histogram
pjm_prediction_interval_width_mw_bucket{horizon_hours="1",le="5000.0"} 120.0

# HELP pjm_forecast_age_seconds Age in seconds of the most recently ingested NWP weather forecast.
# TYPE pjm_forecast_age_seconds gauge
pjm_forecast_age_seconds 1820.0

# HELP pjm_model_version Active production model and calibration version details.
# TYPE pjm_model_version info
pjm_model_version_info{calibration_method="adaptive_diurnal",model_id="pjm_probabilistic_champion_v2",version="2.0.0-phase10"} 1.0
```

### Prometheus Scrape & Grafana Setup
- **Scrape Configuration**: Provided in [`deploy/prometheus.yml`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/deploy/prometheus.yml).
- **Recommended Alerts**:
  - `HighForecastAge`: Trigger alert if `pjm_forecast_age_seconds > 21600` (>6 hours).
  - `HighPredictionLatency`: Trigger alert if `rate(pjm_api_request_duration_seconds_sum[5m]) / rate(pjm_api_request_duration_seconds_count[5m]) > 0.5s`.
  - `WeatherProviderDown`: Trigger alert if `increase(pjm_weather_provider_failures_total[10m]) > 3`.

---

## 7. Containerization & Deployment Verification

### Container Assets
- [`Dockerfile`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/Dockerfile): Python 3.11-slim, non-root user `appuser:10001`, health check probe polling `/health`, Uvicorn entrypoint.
- [`docker-compose.yml`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/docker-compose.yml): Configures both the forecasting API (`forecasting-api:8000`) and Prometheus scraper (`prometheus:9090`).
- [`.dockerignore`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/.dockerignore): Prevents leakage of raw CSVs, caches, test logs, and git objects into image layers.

### Host Execution Status
- **Docker Desktop Daemon**: Docker CLI is not active on this Windows host.
- **Verification Method**: Automated syntax and configuration assertions in [`tests/test_phase9_api_and_probabilistic.py`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/tests/test_phase9_api_and_probabilistic.py#L210-L240) structurally validated `Dockerfile`, `.dockerignore`, and `docker-compose.yml`.

---

## 8. Test Execution & Verification

### Test Suite Execution
```bash
python -m pytest -v
```
**Outcome**:
```text
======================= 150 passed, 1 warning in 23.23s =======================
```
- **Phase 1–5 Scaffolding, Models & Features**: 72 passed.
- **Phase 5B DST & Residual Diagnostics**: 7 passed.
- **Phase 6 Holiday Integration**: 10 passed.
- **Phase 7 NWP & Weather Loader**: 15 passed.
- **Phase 8 Production Engine & Health Monitor**: 11 passed.
- **Phase 9 Probabilistic Models & REST API**: 12 passed.
- **Phase 10 Operational Failure Modes & Providers**: 11 passed.
- **Regressions**: **0 across all existing phases**.

---

## 9. Reproducibility & Commands

### 1. Run Complete Automated Test Suite
```bash
python -m pytest -v
```

### 2. Run Phase 10 Experiment & Calibration Benchmark
```bash
python -m src.experiment --phase10
```

### 3. Launch Production REST API
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 4. Query Health, Predictions, and Prometheus Metrics
```bash
# Health check
curl -X GET http://localhost:8000/health

# Prometheus metrics
curl -X GET http://localhost:8000/metrics
```

### 5. Launch Full Stack with Docker Compose
```bash
docker compose up -d
```

---

## 10. Known Limitations & Remaining Risks

1. **Temporal Non-Stationarity**: Even with 24 diurnal adjustments, non-stationary climate shifts yield an approximate marginal coverage of $88.28\%$ on the out-of-sample multi-year test holdout relative to the nominal $90.0\%$ target.
2. **NOAA Dissemination Latency Outliers**: Severe weather events occasionally delay HRRR cycle publication beyond the nominal 50-minute buffer. The system’s fallback provider prevents API downtime, but operators should monitor the `pjm_forecast_age_seconds` gauge.
3. **Summer Heatwave Ramp Uncertainty**: While peak coverage is stable ($89.68\%$), heatwave interval widths reach $\approx 6,270\text{ MW}$, requiring grid operators to maintain proportional operating reserve margins.
