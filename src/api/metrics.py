"""Prometheus Operational Monitoring and Metric Instrumentation.

Collects and exports production operational metrics for:
- API request rate and latency histograms.
- Prediction failure and exception counters.
- Weather provider connectivity failures.
- Forecast freshness age (seconds).
- Missing weather input counters.
- Prediction interval width distribution.
- Data-quality audit failures.
- Model version metadata.
- Forecast horizon distribution.
- Feature distribution drift alerts.

Defensive design: Metric instrumentation errors are logged and swallowed
so that a monitoring failure NEVER crashes the forecasting API.
"""

from typing import Any
import logging
import time

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


if PROMETHEUS_AVAILABLE:
    # 1. API Request & Latency Metrics
    API_REQUESTS_TOTAL = Counter(
        "pjm_api_requests_total",
        "Total count of HTTP requests served by endpoint and status code.",
        ["method", "endpoint", "status_code"],
    )

    API_REQUEST_DURATION_SECONDS = Histogram(
        "pjm_api_request_duration_seconds",
        "HTTP request latency distribution in seconds.",
        ["endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    # 2. Prediction Failures
    PREDICTION_FAILURES_TOTAL = Counter(
        "pjm_prediction_failures_total",
        "Total count of failed prediction requests.",
        ["failure_reason"],
    )

    # 3. Weather Provider Metrics
    WEATHER_PROVIDER_FAILURES_TOTAL = Counter(
        "pjm_weather_provider_failures_total",
        "Total count of weather provider connection or retrieval failures.",
        ["provider", "error_type"],
    )

    FORECAST_AGE_SECONDS = Gauge(
        "pjm_forecast_age_seconds",
        "Age in seconds of the most recently ingested NWP weather forecast.",
    )

    MISSING_WEATHER_INPUTS_TOTAL = Counter(
        "pjm_missing_weather_inputs_total",
        "Total count of missing or dropped weather variables.",
        ["station_id"],
    )

    # 4. Forecast Distribution & Uncertainty
    PREDICTION_INTERVAL_WIDTH_MW = Histogram(
        "pjm_prediction_interval_width_mw",
        "Distribution of generated 90% prediction interval widths in MW.",
        ["horizon_hours"],
        buckets=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0],
    )

    FORECAST_HORIZON_REQUESTS_TOTAL = Counter(
        "pjm_forecast_horizon_requests_total",
        "Total count of forecast requests segmented by horizon.",
        ["horizon_hours"],
    )

    # 5. Data Quality & Drift
    DATA_QUALITY_AUDIT_FAILURES_TOTAL = Counter(
        "pjm_data_quality_audit_failures_total",
        "Total count of data quality audit failures by rule.",
        ["audit_rule"],
    )

    DRIFT_WARNINGS_TOTAL = Counter(
        "pjm_drift_warnings_total",
        "Total count of feature distribution drift alerts flagged.",
        ["feature_group"],
    )

    # 6. Model Metadata Info
    MODEL_VERSION_INFO = Info(
        "pjm_model_version",
        "Active production model and calibration version details.",
    )


def record_request_metrics(method: str, endpoint: str, status_code: int, duration_sec: float) -> None:
    """Record HTTP request rate and duration safely."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        API_REQUESTS_TOTAL.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
        API_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(duration_sec)
    except Exception as exc:
        logger.debug(f"Failed to record request metric: {exc}")


def record_prediction_success(horizon_hours: int, interval_width_mw: float) -> None:
    """Record successful prediction distribution metrics."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        FORECAST_HORIZON_REQUESTS_TOTAL.labels(horizon_hours=str(horizon_hours)).inc()
        PREDICTION_INTERVAL_WIDTH_MW.labels(horizon_hours=str(horizon_hours)).observe(interval_width_mw)
    except Exception as exc:
        logger.debug(f"Failed to record prediction metric: {exc}")


def record_prediction_failure(reason: str) -> None:
    """Record prediction failure counter."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        PREDICTION_FAILURES_TOTAL.labels(failure_reason=reason).inc()
    except Exception as exc:
        logger.debug(f"Failed to record prediction failure metric: {exc}")


def record_weather_provider_failure(provider: str, error_type: str) -> None:
    """Record weather provider failure counter."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        WEATHER_PROVIDER_FAILURES_TOTAL.labels(provider=provider, error_type=error_type).inc()
    except Exception as exc:
        logger.debug(f"Failed to record weather metric: {exc}")


def set_forecast_age(age_seconds: float) -> None:
    """Update current NWP forecast freshness gauge."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        FORECAST_AGE_SECONDS.set(max(0.0, float(age_seconds)))
    except Exception as exc:
        logger.debug(f"Failed to set forecast age metric: {exc}")


def record_audit_failure(rule_name: str) -> None:
    """Record data quality audit failure counter."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        DATA_QUALITY_AUDIT_FAILURES_TOTAL.labels(audit_rule=rule_name).inc()
    except Exception as exc:
        logger.debug(f"Failed to record audit metric: {exc}")


def record_drift_warning(feature_group: str) -> None:
    """Record feature drift warning."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        DRIFT_WARNINGS_TOTAL.labels(feature_group=feature_group).inc()
    except Exception as exc:
        logger.debug(f"Failed to record drift metric: {exc}")


def set_model_info(version: str, model_id: str, calibration_method: str) -> None:
    """Set model deployment info labels."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        MODEL_VERSION_INFO.info({
            "version": version,
            "model_id": model_id,
            "calibration_method": calibration_method,
        })
    except Exception as exc:
        logger.debug(f"Failed to set model info metric: {exc}")


def get_prometheus_metrics_payload() -> tuple[bytes, str]:
    """Generate Prometheus exposition text format bytes and content-type.

    Returns:
        tuple[bytes, str]: (metric bytes, media content type).
    """
    if not PROMETHEUS_AVAILABLE:
        fallback_text = "# Prometheus client library not available\n"
        return fallback_text.encode("utf-8"), "text/plain; version=0.0.4"

    try:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    except Exception as exc:
        logger.error(f"Error generating Prometheus metrics: {exc}")
        err_msg = f"# Error generating metrics: {str(exc)}\n"
        return err_msg.encode("utf-8"), "text/plain; version=0.0.4"
