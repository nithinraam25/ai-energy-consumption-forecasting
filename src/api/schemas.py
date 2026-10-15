"""Pydantic Request and Response Schemas for Production Forecasting API.

Strictly validates:
- Horizon admissibility (H in {1, 24, 48}).
- Non-finite value prohibition (NaN, Inf rejected with 422).
- Required meteorological and historical feature schemas.
- Clean structured output contracts for health, predictions, and audits.
"""

from datetime import datetime
from typing import Any, Sequence
from pydantic import BaseModel, Field, field_validator
import numpy as np


class HealthResponse(BaseModel):
    """System health and model readiness response contract."""
    status: str = Field(..., description="Service status: HEALTHY, DEGRADED, or UNHEALTHY")
    model_loaded: bool = Field(..., description="Whether production forecasting models are loaded in memory")
    model_version: str = Field(..., description="Active production model version")
    probabilistic_capable: bool = Field(..., description="Whether quantile regression models are active")
    schema_status: str = Field(..., description="Schema integrity status")
    monitoring_status: str = Field(..., description="Operational health monitoring status")
    timestamp_utc: datetime = Field(..., description="Current server UTC timestamp")


class WeatherInputPayload(BaseModel):
    """Operational weather forecast payload for target hour."""
    temp_f_kphl: float | None = Field(None, description="KPHL temperature forecast (°F)")
    temp_f_kewr: float | None = Field(None, description="KEWR temperature forecast (°F)")
    temp_f_kilg: float | None = Field(None, description="KILG temperature forecast (°F)")
    temp_f_zonal_weighted: float | None = Field(None, description="Zonal-weighted composite forecast (°F)")
    temp_spread_regional: float | None = Field(None, description="Spatial thermal gradient spread (°F)")
    ghi_fcst: float | None = Field(None, description="Optional Global Horizontal Irradiance (W/m2)")
    simulated_avail_utc: str | None = Field(None, description="Forecast publication/availability timestamp")
    valid_time_utc: str | None = Field(None, description="Forecast valid timestamp")
    provenance: str = Field("NOAA_HRRR_OPERATIONAL_PROXY", description="Meteorological data provenance")

    @field_validator("temp_f_kphl", "temp_f_kewr", "temp_f_kilg", "temp_f_zonal_weighted", "temp_spread_regional", "ghi_fcst")
    @classmethod
    def validate_finite(cls, v: float | None) -> float | None:
        if v is not None and (np.isnan(v) or np.isinf(v)):
            raise ValueError("Weather values must be finite numerical values.")
        return v


class OrchestratedPredictionRequest(BaseModel):
    """Automated orchestration inference request payload."""
    timestamp: str = Field(..., description="Target prediction timestamp (ISO-8601 or YYYY-MM-DD HH:MM:SS)")
    horizon: int = Field(1, description="Forecast horizon in hours (1, 24, or 48)")
    enforce_cutoff: bool = Field(True, description="Enforce strict causality availability cutoff (t_avail <= t_cutoff)")
    allow_fallback_weather: bool = Field(True, description="Allow fallback climatology if cached HRRR weather is unavailable")

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in (1, 24, 48):
            raise ValueError(f"Unsupported forecast horizon: {v}. Must be one of (1, 24, 48).")
        return v


class PredictionRequest(BaseModel):
    """Inference request payload."""
    timestamp: str = Field(..., description="Target prediction timestamp (ISO-8601 or YYYY-MM-DD HH:MM:SS)")
    horizon: int = Field(1, description="Forecast horizon in hours (1, 24, or 48)")
    features: dict[str, float] | None = Field(None, description="Feature dictionary conforming to production schema. If omitted, automated orchestration service is invoked.")
    weather_inputs: WeatherInputPayload | None = Field(None, description="Optional weather payload for provenance & causality validation")
    enforce_cutoff: bool = Field(True, description="Enforce strict causality availability cutoff (t_avail <= t_cutoff)")
    allow_fallback_weather: bool = Field(True, description="Allow fallback climatological weather when features are orchestrated")

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in (1, 24, 48):
            raise ValueError(f"Unsupported forecast horizon: {v}. Must be one of (1, 24, 48).")
        return v

    @field_validator("features")
    @classmethod
    def validate_features_finite(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            for feat, val in v.items():
                if np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Feature '{feat}' contains non-finite value ({val}).")
        return v


class PredictionResponse(BaseModel):
    """Inference response payload."""
    prediction_timestamp_utc: datetime = Field(..., description="Timestamp when prediction was executed")
    target_timestamp: str = Field(..., description="Target hour for power consumption forecast")
    horizon: int = Field(..., description="Forecast horizon in hours (1, 24, or 48)")
    point_forecast_mw: float = Field(..., description="Point demand forecast in Megawatts (MW)")
    q05_mw: float = Field(..., description="Calibrated lower prediction bound (5th percentile, MW)")
    q50_mw: float = Field(..., description="Median demand forecast (50th percentile, MW)")
    q95_mw: float = Field(..., description="Calibrated upper prediction bound (95th percentile, MW)")
    interval_width_mw: float = Field(..., description="Prediction interval width: q95 - q05 (MW)")
    nominal_coverage: float = Field(0.90, description="Nominal prediction interval coverage")
    model_identifier: str = Field(..., description="Model artifact name")
    model_version: str = Field(..., description="Model release version")
    weather_data_provenance: str = Field(..., description="Data provenance documentation")
    data_quality_warnings: list[str] = Field(default_factory=list, description="Quality and health warnings")


class AuditRequest(BaseModel):
    """Batch audit request payload."""
    records: list[dict[str, Any]] = Field(..., description="Batch records with timestamps and features")
    stale_threshold_hours: float = Field(6.0, description="Maximum allowable forecast age before stale flag")


class AuditResponse(BaseModel):
    """Structured data audit summary."""
    status: str = Field(..., description="Audit status: HEALTHY, WARNING, or CRITICAL")
    total_records: int = Field(..., description="Total records evaluated")
    missing_weather_count: int = Field(0, description="Records with missing weather inputs")
    missing_weather_pct: float = Field(0.0, description="Percentage of missing weather inputs")
    stale_forecast_count: int = Field(0, description="Records exceeding staleness threshold")
    causality_violations_count: int = Field(0, description="Records with simulated_avail > target_cutoff")
    schema_mismatch_features: list[str] = Field(default_factory=list, description="Missing required features")
    non_finite_features_count: int = Field(0, description="Number of NaN or Inf values detected")
    distribution_drift_alerts: list[str] = Field(default_factory=list, description="Features exhibiting significant drift")
    audit_timestamp_utc: datetime = Field(..., description="Audit execution timestamp")
