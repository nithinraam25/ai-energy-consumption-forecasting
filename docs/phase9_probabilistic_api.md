# Phase 9: Probabilistic Energy Forecasting, Production API, and Deployment Readiness

## Executive Summary

Phase 9 operationalizes the **PJM East AI-Powered Energy Consumption Forecasting System** by transitioning from deterministic point forecasting to risk-aware, calibrated **probabilistic forecasting**, exposing the inference and diagnostic pipeline through a production **FastAPI REST API**, and packaging the service for reproducible **Docker containerization**.

Prior phases established high-accuracy deterministic models with multi-station meteorological integration (Phase 8 Test MAE: **528.60 MW**, $R^2 = 0.9883$). However, power system grid operators and dispatch planners require calibrated uncertainty quantification—specifically prediction intervals—to evaluate tail-risk exposure, unit commitment reserves, and extreme peak ramp volatility.

Phase 9 delivers:
1. **Multi-Quantile Regression Engine** producing simultaneous $Q_{05}$ (lower bound), $Q_{50}$ (median), $Q_{95}$ (upper bound), and mean point predictions with strictly guaranteed non-crossing monotonicity ($Q_{05} \le Q_{50} \le Q_{95}$).
2. **Conformalized Quantile Regression (CQR)** interval calibration operating on an isolated chronological calibration split ($21,760$ samples), lifting nominal $90\%$ validation coverage from $87.97\%$ to an exact $90.00\%$, and achieving **$88.09\%$ empirical coverage** on the isolated multi-year test holdout ($21,759$ samples).
3. **Rigorous Solar / Weather Provenance Audit** documenting that historical NOAA ISD airport surface records contain zero solar radiation/GHI measurements, while NOAA HRRR archives GHI only from mid-2014 onward. A modular, leakage-free interface (`SolarAndExtendedWeatherInterface`) was established without manufacturing synthetic GHI.
4. **Production FastAPI Service** exposing `/health`, `/predict`, and `/audit` with strict Pydantic schemas, sub-second latency, causality cutoff verification, in-memory model caching, and structured error responses.
5. **Containerization & Deployment Deliverables** comprising `Dockerfile`, `.dockerignore`, `docker-compose.yml`, health check probes, and configurable runtime parameters.
6. **100% Passing Automated Test Suite** across 139 unit, integration, and security tests with zero regression.

---

## 1. System Architecture & Module Responsibilities

The Phase 9 architecture extends the modular pipeline with four primary subsystems:

```
+--------------------------------------------------------------------------------------------------+
|                                    Phase 9 Production Pipeline                                   |
+--------------------------------------------------------------------------------------------------+
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
     [ Data & Feature Pipeline ]                                  [ Probabilistic Engine ]
  src/features/weather_features.py                             src/models/probabilistic_forecaster.py
  - SolarAndExtendedWeatherInterface                           - MultiQuantileForecaster (Q05, Q50, Q95)
  - Multi-station weather aggregation                          - ConformalIntervalCalibrator (CQR)
  - Strict causality cutoffs (t_avail <= t_origin)             - Stratified Diagnostic Evaluator
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                     [ REST API & Service Layer ]
                                           src/api/app.py
                                           src/api/schemas.py
                                     - GET  /health (Readiness / Liveness)
                                     - POST /predict (Point + Calibrated Intervals)
                                     - POST /audit (Data Quality & Drift Gate)
                                                 │
                                                 ▼
                                     [ Deployment Container ]
                                     Dockerfile & docker-compose.yml
                                     - Python 3.11-slim
                                     - Non-root user (appuser:10001)
                                     - Integrated ASGI Uvicorn server
                                     - Automated Docker HEALTHCHECK
```

### Module Catalog

| Module | Primary Class / Function | Responsibility |
| :--- | :--- | :--- |
| `src/models/probabilistic_forecaster.py` | `MultiQuantileForecaster` | Fits quantile regressors for $\alpha \in \{0.05, 0.50, 0.95\}$ and a point regressor; enforces monotonic non-crossing sorting. |
| `src/models/probabilistic_forecaster.py` | `ConformalIntervalCalibrator` | Implements Conformalized Quantile Regression (CQR); computes nonconformity score adjustment $\hat{\eta}$ on calibration partition. |
| `src/models/probabilistic_forecaster.py` | `evaluate_probabilistic_forecasts` | Computes empirical coverage, mean/median width, point metrics (MAE, RMSE, MAPE), and disaggregates by season, hour, demand tier, and horizon. |
| `src/features/weather_features.py` | `SolarAndExtendedWeatherInterface` | Ingests genuine operational Global Horizontal Irradiance (GHI) and extended variables without temporal leakage; refuses to manufacture synthetic GHI. |
| `src/api/schemas.py` | `HealthResponse`, `PredictionRequest`, `PredictionResponse`, `AuditRequest`, `AuditResponse` | Pydantic data contracts enforcing input validation, finite-value assertions, horizon bounds, and output formatting. |
| `src/api/app.py` | FastAPI application (`app`) | Production REST API exposing `/health`, `/predict`, and `/audit` with lifespan model loading and exception handling. |
| `Dockerfile` & `docker-compose.yml` | Container runtime specifications | Hardened container environment with non-root privileges, configurable environment variables, and health probes. |

---

## 2. Weather-Data Provenance & Solar (GHI) Audit

An essential requirement of Phase 9 was conducting an uncompromised audit of weather data provenance and investigating whether solar irradiance features could be integrated.

### Historical Provenance Audit

| Dataset | Provider / Source | Period | Temporal Resolution | Variables | Provenance Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NOAA ISD Surface Records** | NOAA NCEI (KPHL, KEWR, KILG) | 2002–2018 | Hourly / Sub-hourly | `TMP`, `DEW`, `WND`, `SLP` | **Genuine Historical In-Situ Observations** |
| **Simulated Forecast Errors** | AR(1) Gaussian perturbation applied to ISD | 2002–2018 | Hourly | `temp_f_fcst` | **Simulated / Reconstructed Forecasts** (Retrospective synthetic baseline) |
| **NOAA HRRR Live / Archive** | NOAA / AWS S3 Open Data | Mid-2014–Present | Hourly (15-min update) | `TMP:2 m`, `DSWRF` (GHI), `UGRD`, `VGRD` | **Genuine Operational NWP Forecasts** |

### Solar Radiation (GHI) Audit Findings

1. **NOAA ISD Surface Stations**: Stations KPHL (Philadelphia), KEWR (Newark), and KILG (Wilmington) report meteorological parameters via METAR/ASOS instruments. None of these airport stations archive physical pyranometer solar irradiance measurements (Global Horizontal Irradiance - GHI, Direct Normal Irradiance - DNI, or Diffuse Horizontal Irradiance - DHI).
2. **Operational HRRR Model on AWS S3**: The NOAA High-Resolution Rapid Refresh (HRRR) system models Downward Short-Wave Radiation Flux at surface (`DSWRF`), which is equivalent to GHI. However, public archive records on AWS S3 are only available from **mid-2014 onward**. Operational HRRR forecasts do not exist for the historical period of 2002 to 2013.
3. **Engineering Decision**: In strict accordance with the project's data integrity rules, **we explicitly refused to manufacture synthetic solar irradiance data** for 2002–2013. Instead, we designed and implemented `SolarAndExtendedWeatherInterface`. This interface:
   - Validates physical limits of GHI ($0 \le \text{GHI} \le 1,400\text{ W/m}^2$) and enforces night-time zero bounds ($0\text{ W/m}^2$ when solar elevation $\le 0$).
   - Flags missing solar data gracefully with provenance metadata (`source: "nwp_operational"` or `"unavailable"`).
   - Allows the inference engine to accept genuine GHI whenever operational data feeds provide it, while falling back safely to the baseline meteorological feature set when GHI is absent.

---

## 3. Probabilistic Modeling Methodology

### Quantile Loss Formulation

For a given quantile level $q \in (0, 1)$ and true target $y$, the pinball (quantile) loss function is:

$$\mathcal{L}_q(y, \hat{y}_q) = \max \left( q (y - \hat{y}_q), (1 - q) (\hat{y}_q - y) \right)$$

We train three distinct gradient-boosted quantile regressors using scikit-learn's `HistGradientBoostingRegressor(loss="quantile", quantile=q)`:
- **$Q_{05}$ Estimator**: $q = 0.05$ (5th percentile lower bound).
- **$Q_{50}$ Estimator**: $q = 0.50$ (Median demand forecast).
- **$Q_{95}$ Estimator**: $q = 0.95$ (95th percentile upper bound).
- **Point Estimator**: `HistGradientBoostingRegressor(loss="squared_error")` (Conditional mean demand).

### Hyperparameter Specifications

The quantile models share the validated Phase 8 hyperparameter configuration:
- `learning_rate`: $0.05$
- `max_iter`: $200$
- `max_depth`: $8$
- `max_leaf_nodes`: $45$
- `l2_regularization`: $0.50$
- `random_state`: $42$

### Monotonic Non-Crossing Post-Processing

Standard independent quantile regression models can occasionally suffer from "quantile crossing" ($\hat{q}_{05} > \hat{q}_{50}$ or $\hat{q}_{50} > \hat{q}_{95}$) in regions of sparse data or sudden domain shifts. `MultiQuantileForecaster` mathematically eliminates quantile crossing by applying monotonic non-crossing sorting across predicted percentiles:

$$[\hat{y}_{05}^{\text{sorted}}, \hat{y}_{50}^{\text{sorted}}, \hat{y}_{95}^{\text{sorted}}] = \text{sort}([\hat{y}_{05}, \hat{y}_{50}, \hat{y}_{95}])$$

This guarantees:

$$\hat{y}_{05} \le \hat{y}_{50} \le \hat{y}_{95} \quad \forall \mathbf{x} \in \mathcal{X}$$

---

## 4. Conformal Calibration Methodology

### Conformalized Quantile Regression (CQR)

While raw quantile regression attempts to output percentiles matching nominal levels, empirical coverage often under-covers due to finite sample variance, function approximation bias, and regularization shrinkage.

To address this, we implement a separate **Conformal Interval Calibrator** based on Conformalized Quantile Regression (Romano et al., 2019).

#### 1. Nonconformity Score
For each observation $i$ in the calibration set $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n_{\text{cal}}}$, the signed nonconformity error score $E_i$ is computed:

$$E_i = \max \left( \hat{q}_{05}(\mathbf{x}_i) - y_i, \, y_i - \hat{q}_{95}(\mathbf{x}_i) \right)$$

- $E_i \le 0$ indicates that the true demand $y_i$ fell inside the raw prediction interval $[\hat{q}_{05}(\mathbf{x}_i), \hat{q}_{95}(\mathbf{x}_i)]$.
- $E_i > 0$ measures the distance by which $y_i$ breached the raw interval.

#### 2. Calibration Adjustment Factor
Given a nominal miscoverage level $\alpha = 0.10$ (for $90\%$ prediction intervals), the adjustment factor $\hat{\eta}$ is calculated as the $(1 - \alpha)(1 + 1/n_{\text{cal}})$-th empirical quantile of $\{E_i\}_{i=1}^{n_{\text{cal}}}$:

$$\hat{\eta} = \text{Quantile} \left( \{E_i\}_{i=1}^{n_{\text{cal}}}, \, \min\left(1, \, \frac{\lceil (1 - \alpha)(n_{\text{cal}} + 1) \rceil}{n_{\text{cal}}}\right) \right)$$

On our validation calibration split ($n_{\text{cal}} = 21,760$ samples), the raw quantile model achieved $87.97\%$ coverage. The computed calibration factor was:

$$\hat{\eta} = \mathbf{+56.70\text{ MW}}$$

Expanding the raw intervals by $+56.70\text{ MW}$ adjusted the calibration partition coverage to **exactly $90.00\%$**.

#### 3. Calibrated Interval Construction
For any new feature vector $\mathbf{x}_{\text{new}}$, the calibrated prediction interval is constructed as:

$$\hat{y}_{05}^{\text{cal}} = \hat{q}_{05}(\mathbf{x}_{\text{new}}) - \hat{\eta}$$
$$\hat{y}_{95}^{\text{cal}} = \hat{q}_{95}(\mathbf{x}_{\text{new}}) + \hat{\eta}$$

A post-adjustment safety pass ensures monotonic consistency:

$$\hat{y}_{05}^{\text{final}} = \min(\hat{y}_{05}^{\text{cal}}, \hat{q}_{50})$$
$$\hat{y}_{95}^{\text{final}} = \max(\hat{y}_{95}^{\text{cal}}, \hat{q}_{50})$$

### Crucial Epistemic Limitation: Temporal Dependence & Non-Exchangeability

Standard conformal prediction theory guarantees exact marginal coverage under the assumption of **exchangeability** (i.i.d. observations). In real-world electric power systems, demand and temperature data exhibit:
1. Strong **serial autocorrelation** (e.g., autoregressive memory across hours and days).
2. **Diurnal non-stationarity** and intra-day variance clustering.
3. Macro **climate shifts** and multi-year seasonal distribution changes.

Because exchangeability is violated in temporal domains, the nominal $90\%$ coverage guarantee becomes an **approximate marginal coverage** property rather than a strict finite-sample guarantee. Consequently, when transferred to an out-of-sample multi-year test holdout, coverage may diverge slightly from nominal levels. Documenting this transparently is mandatory for trustworthy grid operation.

---

## 5. Chronological Partitioning & Experimental Protocol

To eliminate temporal leakage, the 17-year dataset (2002–2018) is strictly partitioned along chronological boundaries:

```
2002-01-01                                     2013-08-07              2016-01-30             2018-08-02
├───────────────────────────────────────────────┼───────────────────────┼──────────────────────┤
│               TRAINING SET                    │   CALIBRATION SET     │     TEST SET         │
│             101,551 samples                   │   21,760 samples      │   21,762 samples     │
│       Model Fitting (Loss Minimization)       │  CQR Interval Tuning  │ Isolated Evaluation  │
└───────────────────────────────────────────────┴───────────────────────┴──────────────────────┘
```

1. **Training Split** ($101,551$ rows, 2002-01 to 2013-08): Used solely for estimating tree split thresholds and leaf weights for $Q_{05}, Q_{50}, Q_{95}$ and Point models.
2. **Calibration Split** ($21,760$ rows, 2013-08 to 2016-01): Completely isolated from model fitting. Used solely to compute nonconformity scores and obtain $\hat{\eta}$.
3. **Test Holdout** ($21,762$ total rows / $21,759$ aligned rows, 2016-01 to 2018-08): Untouched throughout training and calibration. Used strictly to report final generalization metrics.

---

## 6. Empirical Evaluation Results

### Overall Prediction Interval Performance (Test Holdout: 21,762 samples)

| Configuration | Empirical Coverage | Mean Interval Width | Median Interval Width | Point MAE | Point RMSE | Point MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Target** | **90.00%** | — | — | — | — | — |
| **Raw Quantile Regression** | 86.12% | 4,099.19 MW | 3,438.68 MW | 618.85 MW | 814.47 MW | 2.00% |
| **Conformal Calibrated (CQR)** | **88.09%** | **4,212.58 MW** | **3,552.07 MW** | **618.85 MW** | **814.47 MW** | **2.00%** |

*Analysis*: The conformal calibrator boosted empirical coverage on the unseen test set from $86.12\%$ to **$88.09\%$** (a $+1.97\%$ absolute coverage gain) at the modest cost of expanding average interval width by only $+113.39\text{ MW}$ ($+2.77\%$). The remaining gap to $90.00\%$ ($1.91\%$) reflects real-world non-stationary distribution shift across multi-year evaluation periods.

### Stratified Coverage Diagnostics (Calibrated Test Holdout)

#### Coverage by Season

| Season | Calibrated Coverage | Mean Width | Status |
| :--- | :---: | :---: | :--- |
| **Winter** | 87.11% | 4,310 MW | Highly Stable |
| **Spring** | 86.19% | 3,920 MW | Stable |
| **Summer** | **92.61%** | 4,680 MW | Conservative (Heatwave buffering) |
| **Fall** | 85.92% | 3,940 MW | Moderate Under-coverage |

#### Coverage by Demand Quartile

| Demand Tier | Demand Range (MW) | Calibrated Coverage |
| :--- | :--- | :---: |
| **Q1 (Base Load)** | $\le 27,000$ | 86.31% |
| **Q2 (Medium-Low)** | $27,000 - 31,500$ | 86.42% |
| **Q3 (Medium-High)** | $31,500 - 36,000$ | 88.46% |
| **Q4 (Peak Demand)** | $> 36,000$ | **91.18%** |

*Key Finding*: The model achieves its highest empirical coverage (**$91.18\%$**) during peak demand conditions (Q4). This provides critical reliability where grid stress and generation reserves are most sensitive.

#### Multi-Horizon Prediction Intervals

| Horizon | Description | Features Count | Calibrated Coverage | Mean Interval Width | Median Interval Width | Point MAE | Point MAPE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=1\text{h}$** | Real-Time Dispatch | 17 | **89.11%** | 5,517.33 MW | 4,270.98 MW | 407.60 MW | 1.34% |
| **$H=24\text{h}$** | Day-Ahead Market | 17 | **87.13%** | 6,565.12 MW | 5,930.01 MW | 1,277.75 MW | 4.12% |
| **$H=48\text{h}$** | Two-Day Unit Commitment | 17 | **86.87%** | 7,504.26 MW | 6,861.73 MW | 1,561.23 MW | 5.06% |

*Physical Interpretation*: As the forecast horizon extends from $1\text{h}$ to $48\text{h}$, the interval width widens monotonically from $5,517\text{ MW}$ to $7,504\text{ MW}$ ($+36.0\%$). This widening accurately reflects atmospheric NWP dispersion and the loss of short-term autoregressive load observations.

---

## 7. Production REST API Reference

The forecasting and diagnostic services are exposed via a production **FastAPI** application (`src/api/app.py`).

### Endpoints Overview

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status, model artifact readiness, and monitoring state. |
| `POST` | `/predict` | Validated point and calibrated probabilistic quantile forecast generation. |
| `POST` | `/audit` | Comprehensive input data quality, causality, and distribution drift audit. |

---

### GET `/health`

Verifies that model artifacts, conformal parameters, and health monitors are initialized and ready for production serving.

#### Example Response (HTTP 200)
```json
{
  "status": "healthy",
  "version": "2.0.0-phase9",
  "model_artifact_loaded": true,
  "model_id": "pjm_probabilistic_champion_v2",
  "conformal_calibration_active": true,
  "required_feature_count": 17,
  "monitoring_status": "active",
  "timestamp": "2026-10-15T15:25:00Z"
}
```

---

### POST `/predict`

Generates point demand predictions and calibrated $90\%$ prediction intervals ($Q_{05}, Q_{50}, Q_{95}$).

#### Example Request Payload
```json
{
  "timestamp": "2026-10-15T12:00:00",
  "horizon_hours": 1,
  "historical_lags": {
    "lag_1": 31250.0,
    "lag_2": 30800.0,
    "lag_3": 30100.0,
    "lag_24": 30500.0,
    "lag_48": 29800.0,
    "lag_168": 31100.0,
    "rolling_mean_24": 30200.0,
    "rolling_std_24": 1450.0
  },
  "weather_forecasts": [
    {
      "station_id": "KPHL",
      "issue_time": "2026-10-15T10:00:00",
      "valid_time": "2026-10-15T12:00:00",
      "lead_time_hours": 2,
      "temp_f": 74.5,
      "dew_point_f": 58.0,
      "wind_speed_mph": 8.5,
      "source": "nwp_operational",
      "available_time": "2026-10-15T10:50:00"
    }
  ]
}
```

#### Example Response (HTTP 200)
```json
{
  "timestamp": "2026-10-15T12:00:00",
  "forecast_horizon_hours": 1,
  "point_forecast_mw": 31450.25,
  "q05_mw": 29320.10,
  "q50_mw": 31445.80,
  "q95_mw": 33532.68,
  "prediction_interval_width_mw": 4212.58,
  "calibrated": true,
  "model_id": "pjm_probabilistic_champion_v2",
  "weather_provenance": "operational_nwp_kphl",
  "data_quality_warnings": []
}
```

---

### POST `/audit`

Performs static and dynamic validation on proposed forecast inputs without invoking model inference.

#### Example Request Payload
```json
{
  "records": [
    {
      "timestamp": "2026-10-15T12:00:00",
      "temp_f": 72.0,
      "cooling_degree_proxy": 7.0,
      "heating_degree_proxy": 0.0,
      "forecast_available_time": "2026-10-15T10:50:00",
      "forecast_origin_time": "2026-10-15T11:00:00",
      "source_dataset": "nwp_operational"
    }
  ],
  "check_staleness": true,
  "check_distribution_drift": true
}
```

#### Example Response (HTTP 200)
```json
{
  "total_records": 1,
  "valid_records": 1,
  "invalid_records": 0,
  "temporal_leakage_detected": false,
  "stale_forecasts_detected": false,
  "distribution_drift_detected": false,
  "issues_found": [],
  "audit_passed": true
}
```

---

## 8. Error Codes & Validation Rules

The API applies defensive validation at the schema and route layers:

| HTTP Status | Condition | Example Reason |
| :---: | :--- | :--- |
| **`400 Bad Request`** | Unsupported horizon or schema mismatch | `"Unsupported forecast horizon: 12. Supported horizons: [1, 24, 48]"` |
| **`422 Unprocessable Entity`** | Pydantic validation failure | Non-finite values (`NaN`, `Infinity`), missing required keys, negative demand lags. |
| **`422 Unprocessable Entity`** | Temporal causality violation | NWP forecast availability time occurs after the prediction cutoff ($t_{\text{avail}} > t_{\text{origin}}$). |
| **`503 Service Unavailable`** | Model artifacts missing | `pjm_probabilistic_forecaster_phase9.joblib` or schema files not present on disk. |

---

## 9. Containerization & Deployment Guide

The service is fully dockerized for container platforms (Kubernetes, AWS ECS, Docker Compose).

### Deliverables
- `Dockerfile`: Multi-stage build based on `python:3.11-slim`, non-root user `appuser` (UID 10001), health probe configured.
- `.dockerignore`: Excludes raw training datasets, `.git`, cache files, and test temporary logs.
- `docker-compose.yml`: Local orchestrator mapping port 8000:8000 with environment configuration.

### Build and Run Instructions

#### 1. Build the Docker Image
```bash
docker build -t pjm-energy-forecaster:v2.0.0 .
```

#### 2. Run Container with Docker
```bash
docker run -d \
  --name pjm-forecaster-api \
  -p 8000:8000 \
  -e PORT=8000 \
  -e MODEL_PATH=models/pjm_probabilistic_forecaster_phase9.joblib \
  pjm-energy-forecaster:v2.0.0
```

#### 3. Run with Docker Compose
```bash
docker compose up -d
```

#### 4. Verify Health Inside Container
```bash
docker inspect --format='{{json .State.Health}}' pjm-forecaster-api
```

#### 5. View Logs and Teardown
```bash
# View real-time logs
docker compose logs -f

# Gracefully stop the service
docker compose down
```

---

## 10. Reproducibility & Test Execution

### 1. Execute Probabilistic Experiments
```bash
python -m src.experiment --phase9
```
*Outputs generated*:
- `models/pjm_probabilistic_forecaster_phase9.joblib`
- `models/conformal_calibration_params.json`
- `outputs/metrics/pjm_phase9_probabilistic_metrics.json`
- `outputs/metrics/pjm_phase9_probabilistic_metrics.csv`
- `outputs/metrics/pjm_phase9_multi_horizon_probabilistic.csv`
- `outputs/forecasts/pjm_phase9_probabilistic_predictions.csv`

### 2. Start the API Locally
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run the Automated Test Suite
```bash
# Run Phase 9 specific tests
pytest tests/test_phase9_api_and_probabilistic.py -v

# Run the complete test suite (139 tests)
pytest -v
```

---

## 11. Known Limitations & Operational Considerations

1. **Temporal Non-Exchangeability**: Conformal prediction mathematically assumes exchangeable observations. Autoregressive load series violate this assumption. As demonstrated, the calibrator achieved exact $90.00\%$ coverage on the calibration split, but $88.09\%$ on the test split due to temporal shift.
2. **Absence of Historical Pyranometer Measurements**: Historical NOAA ISD surface observations for Philadelphia, Newark, and Wilmington do not record solar radiation. While the interface is designed to ingest operational GHI from live NWP GRIB2 streams, historical backtests before 2014 cannot evaluate empirical solar irradiance gains without fabricating data.
3. **Severe Heatwave Tail Uncertainty**: In summer peak hours ($>40,000\text{ MW}$), extreme temperature nonlinearities can widen prediction intervals beyond $6,000\text{ MW}$. Operators should pair point forecasts with interval widths to trigger reserve margins.

---

## 12. Recommended Next Phase: Phase 10

For Phase 10, the recommended engineering milestones are:
1. **Adaptive Locally-Weighted Conformal Inference**: Replace global scalar conformal calibration ($\hat{\eta}$) with localized conformal prediction conditioned on current temperature or forecast horizon to narrow intervals during mild shoulder seasons.
2. **Direct NOAA HRRR GRIB2 Real-Time Ingestion Daemon**: Deploy a background worker service polling NOAA AWS S3 Open Data hourly to unpack real-time GRIB2 slices directly into memory.
3. **OpenTelemetry & Prometheus Instrumentation**: Add exportable metric scrapers for latency, prediction throughput, and drift alarms to connect directly with Grafana dashboards.
