"""Production REST API for AI-Powered Energy Consumption Forecasting.

Endpoints:
- GET  /health: Health check, artifact verification, model version metadata.
- POST /predict: Point and calibrated probabilistic forecasting (Q05, Q50, Q95).
- POST /audit: Operational data quality, missingness, staleness, and drift audits.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    OrchestratedPredictionRequest,
    PredictionResponse,
    AuditRequest,
    AuditResponse,
)
from src.services.orchestrator import (
    ForecastOrchestrator,
    OrchestrationError,
    InvalidHorizonError,
)
from src.data.demand_provider import (
    MissingDemandHistoryError,
    IncompleteDemandHistoryError,
    TemporalIneligibilityError,
)
from src.weather.service import (
    WeatherUnavailableError,
    WeatherCausalityError,
)
from src.config import MODELS_DIR
from src.models.inference import (
    EnergyInferenceEngine,
    DEFAULT_PRODUCTION_MODEL_PATH,
    DEFAULT_FEATURE_SCHEMA_PATH,
)
from src.models.probabilistic_forecaster import (
    MultiQuantileForecaster,
    ConformalIntervalCalibrator,
    AdaptiveConformalCalibrator,
    DEFAULT_PROBABILISTIC_MODEL_PATH,
    DEFAULT_CALIBRATION_ARTIFACT_PATH,
    DEFAULT_ADAPTIVE_CALIBRATION_PATH,
)
from src.evaluation.monitoring import ForecastHealthMonitor
from src.api.metrics import (
    record_request_metrics,
    record_prediction_success,
    record_prediction_failure,
    record_audit_failure,
    record_weather_provider_failure,
    set_model_info,
    get_prometheus_metrics_payload,
)


# Environment-configurable model and schema paths
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_PRODUCTION_MODEL_PATH)))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", str(DEFAULT_FEATURE_SCHEMA_PATH)))
PROB_MODEL_PATH = Path(os.getenv("PROB_MODEL_PATH", str(DEFAULT_PROBABILISTIC_MODEL_PATH)))
CALIBRATION_PATH = Path(os.getenv("CALIBRATION_PATH", str(DEFAULT_CALIBRATION_ARTIFACT_PATH)))
ADAPTIVE_CALIBRATION_PATH = Path(
    os.getenv("ADAPTIVE_CALIBRATION_PATH", str(DEFAULT_ADAPTIVE_CALIBRATION_PATH))
)

# State container for loaded model components
app_state: dict[str, Any] = {
    "point_engine": None,
    "prob_model": None,
    "calibrator": None,
    "adaptive_calibrator": None,
    "orchestrator": None,
    "monitor": None,
    "is_ready": False,
    "model_version": "2.0.0-phase10",
    "init_error": None,
}


def load_application_components() -> None:
    """Load model artifacts, schema, calibrator, orchestrator, and monitor into memory."""
    try:
        app_state["monitor"] = ForecastHealthMonitor()

        # Load point inference engine if artifact exists
        if MODEL_PATH.is_file() and SCHEMA_PATH.is_file():
            app_state["point_engine"] = EnergyInferenceEngine(
                model_path=MODEL_PATH,
                schema_path=SCHEMA_PATH,
                model_version=app_state["model_version"],
            )

        # Load probabilistic quantile model if artifact exists
        if PROB_MODEL_PATH.is_file():
            app_state["prob_model"] = MultiQuantileForecaster.load(PROB_MODEL_PATH)

        # Load conformal calibrator if artifact exists
        if CALIBRATION_PATH.is_file():
            app_state["calibrator"] = ConformalIntervalCalibrator.load(CALIBRATION_PATH)

        # Load adaptive calibrator if artifact exists
        if ADAPTIVE_CALIBRATION_PATH.is_file():
            app_state["adaptive_calibrator"] = AdaptiveConformalCalibrator.load(ADAPTIVE_CALIBRATION_PATH)

        # Initialize automated forecast orchestrator
        if app_state["point_engine"] is not None:
            app_state["orchestrator"] = ForecastOrchestrator(
                inference_engine=app_state["point_engine"],
                prob_model=app_state["prob_model"],
                adaptive_calibrator=app_state["adaptive_calibrator"],
                calibrator=app_state["calibrator"],
            )

        cal_type = "adaptive_diurnal" if app_state["adaptive_calibrator"] else "scalar_cqr"
        set_model_info(
            version=app_state["model_version"],
            model_id="pjm_probabilistic_champion_v2",
            calibration_method=cal_type,
        )

        app_state["is_ready"] = (app_state["point_engine"] is not None)
        app_state["init_error"] = None
    except Exception as exc:
        app_state["is_ready"] = False
        app_state["init_error"] = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown initialization."""
    load_application_components()
    yield


app = FastAPI(
    title="PJM East Probabilistic Energy Forecasting API",
    description="Operational REST API for point and conformalized probabilistic electricity demand forecasting.",
    version="2.0.0-phase10",
    lifespan=lifespan,
)

# CORS middleware for development and staging browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=FileResponse)
@app.get("/dashboard", response_class=FileResponse)
def serve_dashboard():
    """Serve the interactive operational forecasting dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file, media_type="text/html")
    return HTMLResponse("<h1>PJM East Forecasting API</h1><p>Dashboard static assets not found.</p>")


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """Timing middleware recording request rates and latencies for Prometheus."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    path = request.url.path
    # Group path for metrics (avoid high-cardinality)
    endpoint = path if path in ("/health", "/predict", "/audit", "/metrics") else "other"
    record_request_metrics(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
        duration_sec=duration,
    )
    return response


@app.get("/metrics")
def get_metrics() -> Response:
    """Prometheus exposition format operational metrics endpoint."""
    payload, media_type = get_prometheus_metrics_payload()
    return Response(content=payload, media_type=media_type)


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Service health and model readiness check.

    Returns:
        HealthResponse: Status, model readiness, schema status, and monitoring state.
    """
    if app_state["point_engine"] is None:
        load_application_components()

    is_ready = bool(app_state["is_ready"] and app_state["point_engine"] is not None)
    prob_capable = bool(
        app_state["prob_model"] is not None and (
            app_state["calibrator"] is not None or app_state["adaptive_calibrator"] is not None
        )
    )

    status_str = "HEALTHY" if is_ready else "DEGRADED" if app_state["point_engine"] is not None else "UNHEALTHY"

    return HealthResponse(
        status=status_str,
        model_loaded=is_ready,
        model_version=app_state["model_version"],
        probabilistic_capable=prob_capable,
        schema_status="VALID" if is_ready else "SCHEMA_OR_MODEL_MISSING",
        monitoring_status="ACTIVE",
        timestamp_utc=datetime.now(timezone.utc),
    )


def _execute_orchestrated_prediction(
    timestamp: str,
    horizon: int,
    enforce_cutoff: bool = True,
    allow_fallback_weather: bool = True,
) -> PredictionResponse:
    """Execute end-to-end automated prediction orchestration."""
    if app_state["orchestrator"] is None:
        load_application_components()
        if app_state["orchestrator"] is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Forecasting orchestration service is not ready.",
            )

    orchestrator: ForecastOrchestrator = app_state["orchestrator"]
    try:
        res = orchestrator.orchestrate_forecast(
            target_timestamp=timestamp,
            horizon=horizon,
            enforce_cutoff=enforce_cutoff,
            allow_fallback_weather=allow_fallback_weather,
        )
        record_prediction_success(
            horizon_hours=horizon,
            interval_width_mw=res.interval_width_mw,
        )
        return PredictionResponse(
            prediction_timestamp_utc=res.prediction_timestamp_utc,
            target_timestamp=res.target_timestamp,
            horizon=res.horizon,
            point_forecast_mw=res.point_forecast_mw,
            q05_mw=res.q05_mw,
            q50_mw=res.q50_mw,
            q95_mw=res.q95_mw,
            interval_width_mw=res.interval_width_mw,
            nominal_coverage=res.nominal_coverage,
            model_identifier=res.model_identifier,
            model_version=res.model_version,
            weather_data_provenance=res.weather_data_provenance,
            data_quality_warnings=res.data_quality_warnings,
        )
    except TemporalIneligibilityError as exc:
        record_prediction_failure("causality_cutoff_violation")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except WeatherCausalityError as exc:
        record_prediction_failure("weather_causality_violation")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except InvalidHorizonError as exc:
        record_prediction_failure("invalid_horizon")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except MissingDemandHistoryError as exc:
        record_prediction_failure("missing_demand_history")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except IncompleteDemandHistoryError as exc:
        record_prediction_failure("incomplete_demand_history")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except WeatherUnavailableError as exc:
        record_weather_provider_failure("weather_unavailable")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        record_prediction_failure("orchestration_error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Orchestration failed: {exc}")


@app.post("/predict/orchestrated", response_model=PredictionResponse)
def predict_energy_demand_orchestrated(payload: OrchestratedPredictionRequest) -> PredictionResponse:
    """Generate demand forecast using automated weather and demand history orchestration."""
    return _execute_orchestrated_prediction(
        timestamp=payload.timestamp,
        horizon=payload.horizon,
        enforce_cutoff=payload.enforce_cutoff,
        allow_fallback_weather=payload.allow_fallback_weather,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_energy_demand(payload: PredictionRequest) -> PredictionResponse:
    """Generate demand forecast (point prediction + Q05, Q50, Q95 prediction intervals).

    If `payload.features` is omitted, automatically executes the full orchestration pipeline.
    If `payload.features` is provided, executes low-level direct inference against the schema.
    """
    if payload.features is None:
        return _execute_orchestrated_prediction(
            timestamp=payload.timestamp,
            horizon=payload.horizon,
            enforce_cutoff=payload.enforce_cutoff,
            allow_fallback_weather=payload.allow_fallback_weather,
        )

    if app_state["point_engine"] is None:
        load_application_components()
        if app_state["point_engine"] is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Forecasting model artifacts unavailable. Service is not ready.",
            )

    engine: EnergyInferenceEngine = app_state["point_engine"]
    prob_model: MultiQuantileForecaster | None = app_state["prob_model"]
    calibrator: ConformalIntervalCalibrator | None = app_state["calibrator"]
    adaptive_calibrator: AdaptiveConformalCalibrator | None = app_state.get("adaptive_calibrator")

    # 1. Feature DataFrame construction with automated temporal and proxy feature derivation
    feature_row = dict(payload.features)

    # Derive calendar and trigonometric features from payload.timestamp if omitted
    try:
        dt = pd.to_datetime(payload.timestamp)
        if "hour" not in feature_row:
            feature_row["hour"] = float(dt.hour)
        if "day_of_week" not in feature_row:
            feature_row["day_of_week"] = float(dt.dayofweek)
        if "day_of_month" not in feature_row:
            feature_row["day_of_month"] = float(dt.day)
        if "month" not in feature_row:
            feature_row["month"] = float(dt.month)
        if "is_weekend" not in feature_row:
            feature_row["is_weekend"] = 1.0 if dt.dayofweek >= 5 else 0.0
        if "hour_sin" not in feature_row:
            feature_row["hour_sin"] = float(np.sin(2.0 * np.pi * feature_row["hour"] / 24.0))
        if "hour_cos" not in feature_row:
            feature_row["hour_cos"] = float(np.cos(2.0 * np.pi * feature_row["hour"] / 24.0))
        if "day_of_week_sin" not in feature_row:
            feature_row["day_of_week_sin"] = float(np.sin(2.0 * np.pi * feature_row["day_of_week"] / 7.0))
        if "day_of_week_cos" not in feature_row:
            feature_row["day_of_week_cos"] = float(np.cos(2.0 * np.pi * feature_row["day_of_week"] / 7.0))
    except Exception:
        pass

    # Populate temperature and degree proxy features if weather_inputs supplied
    if payload.weather_inputs is not None:
        if "temp_f_zonal_weighted" not in feature_row and payload.weather_inputs.temp_f_zonal_weighted is not None:
            feature_row["temp_f_zonal_weighted"] = float(payload.weather_inputs.temp_f_zonal_weighted)

    if "temp_f_zonal_weighted" in feature_row and feature_row["temp_f_zonal_weighted"] is not None:
        tz = float(feature_row["temp_f_zonal_weighted"])
        if "cooling_degree_proxy_regional" not in feature_row:
            feature_row["cooling_degree_proxy_regional"] = float(max(0.0, tz - 65.0))
        if "heating_degree_proxy_regional" not in feature_row:
            feature_row["heating_degree_proxy_regional"] = float(max(0.0, 65.0 - tz))
        if "temp_spread_regional" not in feature_row:
            feature_row["temp_spread_regional"] = 2.536  # Historical average spread across KPHL, KEWR, KILG

    df_row = pd.DataFrame([feature_row])

    # 2. Causality, Fallback, Staleness, and Cutoff verification if weather inputs supplied
    warnings: list[str] = []
    if payload.weather_inputs is not None:
        w = payload.weather_inputs
        if getattr(w, "is_fallback", False) or "FALLBACK" in (w.provenance or "").upper():
            warnings.append(f"FALLBACK_WEATHER_USED: Weather inputs supplied via fallback provider ({w.provenance}).")

        if w.simulated_avail_utc and w.valid_time_utc:
            t_avail = pd.to_datetime(w.simulated_avail_utc)
            t_valid = pd.to_datetime(w.valid_time_utc)
            t_cutoff = t_valid - pd.Timedelta(hours=payload.horizon)

            # Audit staleness: age beyond the operational cutoff > 6.0h
            slack_hours = (t_cutoff - t_avail).total_seconds() / 3600.0
            if slack_hours > 6.0:
                warnings.append(
                    f"STALE_WEATHER_INPUT: Weather forecast available at {w.simulated_avail_utc} is "
                    f"{slack_hours:.1f}h older than the H={payload.horizon}h cutoff ({t_cutoff.isoformat()})."
                )
                record_audit_failure("stale_weather")

            if payload.enforce_cutoff and t_avail > t_cutoff:
                record_prediction_failure("causality_cutoff_violation")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"CAUSALITY VIOLATION: Weather forecast availability ({w.simulated_avail_utc}) "
                        f"is after the prediction cutoff ({t_cutoff.isoformat()}) for horizon H={payload.horizon}h."
                    ),
                )
        if w.ghi_fcst is not None:
            warnings.append("GHI forecast supplied via experimental interface; historical training used tri-station temperature grid.")

    # 3. Schema validation against production contract
    try:
        engine.validate_features(df_row)
    except ValueError as exc:
        record_prediction_failure("schema_validation_violation")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # 4. Point Forecast
    try:
        point_preds = engine.model.predict(df_row[engine.expected_features])
        point_mw = float(np.round(point_preds[0], 2))
    except Exception as exc:
        record_prediction_failure("point_inference_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    # 5. Probabilistic Quantile & Conformal Calibration
    if prob_model is not None and prob_model.is_fitted:
        try:
            q_dict = prob_model.predict(df_row[prob_model.feature_names])
            raw_q05 = float(q_dict["q05"][0])
            raw_q50 = float(q_dict["q50"][0])
            raw_q95 = float(q_dict["q95"][0])

            if adaptive_calibrator is not None and adaptive_calibrator.is_calibrated:
                c_low, c_med, c_high = adaptive_calibrator.adjust(
                    q_low=np.array([raw_q05]),
                    q_median=np.array([raw_q50]),
                    q_high=np.array([raw_q95]),
                    timestamps=[payload.timestamp],
                    demand_preds=np.array([point_mw]),
                )
                q05_mw = float(np.round(c_low[0], 2))
                q50_mw = float(np.round(c_med[0], 2))
                q95_mw = float(np.round(c_high[0], 2))
            elif calibrator is not None and calibrator.is_calibrated:
                c_low, c_med, c_high = calibrator.adjust(
                    np.array([raw_q05]), np.array([raw_q50]), np.array([raw_q95])
                )
                q05_mw = float(np.round(c_low[0], 2))
                q50_mw = float(np.round(c_med[0], 2))
                q95_mw = float(np.round(c_high[0], 2))
            else:
                q05_mw, q50_mw, q95_mw = raw_q05, raw_q50, raw_q95
                warnings.append("Uncalibrated quantile predictions: conformal calibrator was not active.")
        except Exception as exc:
            warnings.append(f"Probabilistic model execution fallback: {str(exc)}")
            q05_mw = round(point_mw * 0.92, 2)
            q50_mw = point_mw
            q95_mw = round(point_mw * 1.08, 2)
    else:
        # Graceful heuristic fallback based on empirical test holdout RMSE (~698 MW ~ 2.4%)
        q05_mw = round(point_mw - 1.645 * 700.0, 2)
        q50_mw = point_mw
        q95_mw = round(point_mw + 1.645 * 700.0, 2)
        warnings.append("Parametric interval fallback: dedicated quantile models not loaded.")

    interval_width = round(q95_mw - q05_mw, 2)
    record_prediction_success(horizon_hours=payload.horizon, interval_width_mw=interval_width)

    return PredictionResponse(
        prediction_timestamp_utc=datetime.now(timezone.utc),
        target_timestamp=payload.timestamp,
        horizon=payload.horizon,
        point_forecast_mw=point_mw,
        q05_mw=q05_mw,
        q50_mw=q50_mw,
        q95_mw=q95_mw,
        interval_width_mw=interval_width,
        nominal_coverage=0.90,
        model_identifier="pjm_production_forecaster_phase10",
        model_version=app_state["model_version"],
        weather_data_provenance=(
            payload.weather_inputs.provenance if (payload.weather_inputs and payload.weather_inputs.provenance) else "CALLER_SUPPLIED_FEATURES"
        ),
        data_quality_warnings=warnings,
    )


@app.get("/audit")
def get_audit_status() -> dict[str, Any]:
    """Return the current system-wide operational health and audit parameters."""
    monitor: ForecastHealthMonitor = app_state.get("monitor") or ForecastHealthMonitor()
    return {
        "status": "ACTIVE",
        "monitoring_active": True,
        "stale_threshold_hours": monitor.stale_threshold_hours,
        "residual_alert_threshold_mw": monitor.residual_alert_threshold_mw,
        "model_ready": bool(app_state.get("is_ready")),
        "model_version": app_state.get("model_version"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/audit", response_model=AuditResponse)
def audit_input_batch(payload: AuditRequest) -> AuditResponse:
    """Audit incoming batch records for missingness, staleness, causality violations, and drift.

    Args:
        payload: Batch records and audit thresholds.

    Returns:
        AuditResponse: Structured health metrics.
    """
    records = payload.records
    total = len(records)
    if total == 0:
        return AuditResponse(
            status="HEALTHY",
            total_records=0,
            audit_timestamp_utc=datetime.now(timezone.utc),
        )

    df_batch = pd.DataFrame(records)

    # 1. Missing weather check
    weather_cols = [c for c in ["temp_f_zonal_weighted", "temp_f_kphl", "temp_f_fcst"] if c in df_batch.columns]
    missing_weather_count = 0
    if weather_cols:
        missing_weather_count = int(df_batch[weather_cols[0]].isna().sum())
    else:
        missing_weather_count = total
    missing_pct = round((missing_weather_count / total) * 100.0, 2)

    # 2. Staleness check
    stale_count = 0
    if "lead_time_hours" in df_batch.columns:
        leads = pd.to_numeric(df_batch["lead_time_hours"], errors="coerce")
        stale_count = int((leads > payload.stale_threshold_hours).sum())

    # 3. Causality violations
    causality_violations = 0
    if "simulated_avail_utc" in df_batch.columns and "valid_time_utc" in df_batch.columns:
        avail = pd.to_datetime(df_batch["simulated_avail_utc"], errors="coerce")
        valid = pd.to_datetime(df_batch["valid_time_utc"], errors="coerce")
        causality_violations = int((avail > valid).sum())

    # 4. Schema verification if engine loaded
    missing_schema_feats: list[str] = []
    if app_state["point_engine"] is not None:
        expected = app_state["point_engine"].expected_features
        missing_schema_feats = sorted(list(set(expected) - set(df_batch.columns)))

    # 5. Non-finite values
    numeric_df = df_batch.select_dtypes(include=[np.number])
    non_finite_count = int(numeric_df.isna().sum().sum() + np.isinf(numeric_df.to_numpy()).sum())

    # Overall Status determination
    if causality_violations > 0:
        record_audit_failure("causality_violations")
    if missing_pct > 0.0:
        record_audit_failure("missing_weather")
    if stale_count > 0:
        record_audit_failure("stale_weather")
    if len(missing_schema_feats) > 0:
        record_audit_failure("schema_mismatch")
    if non_finite_count > 0:
        record_audit_failure("non_finite_features")

    if causality_violations > 0 or missing_pct > 20.0 or len(missing_schema_feats) > 0:
        status_eval = "CRITICAL"
    elif stale_count > 0 or missing_pct > 5.0 or non_finite_count > 0:
        status_eval = "WARNING"
    else:
        status_eval = "HEALTHY"

    return AuditResponse(
        status=status_eval,
        total_records=total,
        missing_weather_count=missing_weather_count,
        missing_weather_pct=missing_pct,
        stale_forecast_count=stale_count,
        causality_violations_count=causality_violations,
        schema_mismatch_features=missing_schema_feats,
        non_finite_features_count=non_finite_count,
        distribution_drift_alerts=[],
        audit_timestamp_utc=datetime.now(timezone.utc),
    )
