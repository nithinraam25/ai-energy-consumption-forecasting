"""Forecast Orchestration Service for AI-Powered Energy Consumption Forecasting.

Connects:
- Weather ingestion and cache discovery (`WeatherFeatureService`)
- Demand history and leakage-safe autoregressive lags (`DemandHistoryProvider`)
- Feature engineering and strict production schema validation
- Point and multi-quantile tree models
- Adaptive diurnal conformal calibration

Provides a single, cohesive, decoupled interface for automated forecasting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.data.demand_provider import (
    DemandHistoryProvider,
    DemandHistoryError,
    MissingDemandHistoryError,
    IncompleteDemandHistoryError,
    TemporalIneligibilityError,
)
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
from src.weather.service import (
    WeatherFeatureService,
    WeatherFeatureResult,
    WeatherServiceError,
    WeatherUnavailableError,
    WeatherCausalityError,
)

logger = logging.getLogger(__name__)


class OrchestrationError(Exception):
    """Base exception for orchestration workflow failures."""
    pass


class InvalidHorizonError(OrchestrationError):
    """Raised when an unsupported forecast horizon is requested."""
    pass


@dataclass(frozen=True)
class OrchestratedForecastResult:
    """Immutable data contract for orchestrated prediction results."""
    prediction_timestamp_utc: datetime
    target_timestamp: str
    horizon: int
    point_forecast_mw: float
    q05_mw: float
    q50_mw: float
    q95_mw: float
    interval_width_mw: float
    nominal_coverage: float
    model_identifier: str
    model_version: str
    weather_data_provenance: str
    data_quality_warnings: list[str] = field(default_factory=list)
    features_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prediction_timestamp_utc": self.prediction_timestamp_utc.isoformat(),
            "target_timestamp": self.target_timestamp,
            "horizon": self.horizon,
            "point_forecast_mw": self.point_forecast_mw,
            "q05_mw": self.q05_mw,
            "q50_mw": self.q50_mw,
            "q95_mw": self.q95_mw,
            "interval_width_mw": self.interval_width_mw,
            "nominal_coverage": self.nominal_coverage,
            "model_identifier": self.model_identifier,
            "model_version": self.model_version,
            "weather_data_provenance": self.weather_data_provenance,
            "data_quality_warnings": list(self.data_quality_warnings),
            "features_used": dict(self.features_used),
        }


class ForecastOrchestrator:
    """Automated operational forecasting orchestrator."""

    SUPPORTED_HORIZONS = (1, 24, 48)

    def __init__(
        self,
        weather_service: WeatherFeatureService | None = None,
        demand_provider: DemandHistoryProvider | None = None,
        inference_engine: EnergyInferenceEngine | None = None,
        prob_model: MultiQuantileForecaster | None = None,
        adaptive_calibrator: AdaptiveConformalCalibrator | None = None,
        calibrator: ConformalIntervalCalibrator | None = None,
    ) -> None:
        """Initialize the orchestration service with required components.

        Args:
            weather_service: Weather discovery and feature service.
            demand_provider: Leakage-free demand history provider.
            inference_engine: Production point inference engine.
            prob_model: Multi-quantile pinball loss forecaster.
            adaptive_calibrator: Adaptive diurnal conformal calibrator.
            calibrator: Scalar conformal calibrator fallback.
        """
        self.weather_service = weather_service or WeatherFeatureService()
        self.demand_provider = demand_provider or DemandHistoryProvider()

        if inference_engine is not None:
            self.engine = inference_engine
        elif DEFAULT_PRODUCTION_MODEL_PATH.is_file() and DEFAULT_FEATURE_SCHEMA_PATH.is_file():
            self.engine = EnergyInferenceEngine(
                model_path=DEFAULT_PRODUCTION_MODEL_PATH,
                schema_path=DEFAULT_FEATURE_SCHEMA_PATH,
            )
        else:
            self.engine = None

        if prob_model is not None:
            self.prob_model = prob_model
        elif DEFAULT_PROBABILISTIC_MODEL_PATH.is_file():
            self.prob_model = MultiQuantileForecaster.load(DEFAULT_PROBABILISTIC_MODEL_PATH)
        else:
            self.prob_model = None

        if adaptive_calibrator is not None:
            self.adaptive_calibrator = adaptive_calibrator
        elif DEFAULT_ADAPTIVE_CALIBRATION_PATH.is_file():
            self.adaptive_calibrator = AdaptiveConformalCalibrator.load(DEFAULT_ADAPTIVE_CALIBRATION_PATH)
        else:
            self.adaptive_calibrator = None

        if calibrator is not None:
            self.calibrator = calibrator
        elif DEFAULT_CALIBRATION_ARTIFACT_PATH.is_file():
            self.calibrator = ConformalIntervalCalibrator.load(DEFAULT_CALIBRATION_ARTIFACT_PATH)
        else:
            self.calibrator = None

    def orchestrate_forecast(
        self,
        target_timestamp: pd.Timestamp | str,
        horizon: int = 1,
        enforce_cutoff: bool = True,
        allow_fallback_weather: bool = True,
    ) -> OrchestratedForecastResult:
        """Execute end-to-end automated forecast generation.

        Workflow:
        1. Validate target horizon and calculate decision-time cutoff:
           t_cutoff = T - H hours.
        2. Obtain eligible weather inputs (cached HRRR or causal fallback).
        3. Obtain eligible demand-lag features (actual history or recursive roll-forward).
        4. Derive temporal and cyclical calendar encodings for target T.
        5. Assemble and validate the 18-feature production matrix.
        6. Execute point forecasting.
        7. Execute multi-quantile probabilistic forecasting with non-crossing projection.
        8. Apply adaptive conformal calibration.
        9. Package into standard PredictionResponse with explicit provenance.

        Args:
            target_timestamp: Target valid hour (T).
            horizon: Forecast horizon (1, 24, or 48 hours).
            enforce_cutoff: Enforce causality cutoff on weather and demand inputs.
            allow_fallback_weather: Allow documented climatology fallback if HRRR missing.

        Returns:
            PredictionResponse: Complete point and interval prediction result.

        Raises:
            InvalidHorizonError: If horizon not in (1, 24, 48).
            MissingDemandHistoryError: If required demand history is missing.
            TemporalIneligibilityError: If demand history is temporally ineligible.
            WeatherUnavailableError: If weather is missing and fallback disallowed.
            OrchestrationError: If models are unavailable or inference fails.
        """
        if horizon not in self.SUPPORTED_HORIZONS:
            raise InvalidHorizonError(
                f"Unsupported forecast horizon: {horizon}. Must be one of {self.SUPPORTED_HORIZONS}."
            )

        if self.engine is None:
            raise OrchestrationError("Production inference engine is not initialized or model artifacts missing.")

        target_ts = pd.to_datetime(target_timestamp)
        if target_ts.tz is None:
            target_ts = target_ts.tz_localize("UTC")
        else:
            target_ts = target_ts.tz_convert("UTC")

        cutoff_ts = target_ts - pd.Timedelta(hours=horizon)

        # 1. Weather Inputs Stage
        weather_res: WeatherFeatureResult = self.weather_service.get_weather_features(
            target_timestamp=target_ts,
            horizon=horizon,
            cutoff_time_utc=cutoff_ts if enforce_cutoff else None,
            allow_fallback=allow_fallback_weather,
        )

        # 2. Demand History & Lags Stage
        # For H=1: direct actualized history ending at T-1h
        # For H > 1: recursive roll-forward starting from origin cutoff t_origin = T - H
        warnings: list[str] = list(weather_res.warnings)
        if horizon == 1:
            demand_lags = self.demand_provider.get_demand_lags(
                target_timestamp=target_ts,
                horizon=1,
            )
            lags_dict = demand_lags.to_dict()
        else:
            # Multi-step recursive roll-forward from decision origin
            lags_dict, recursive_warnings = self._roll_forward_demand_lags(
                origin_ts=cutoff_ts,
                steps=horizon,
                weather_res=weather_res,
            )
            warnings.extend(recursive_warnings)

        # 3. Calendar & Cyclical Feature Stage (evaluated strictly at target T)
        dt_local = target_ts  # Evaluated in civil/UTC representation
        hour_val = float(dt_local.hour)
        dow_val = float(dt_local.dayofweek)
        dom_val = float(dt_local.day)
        month_val = float(dt_local.month)
        is_weekend_val = 1.0 if dow_val >= 5 else 0.0

        hour_sin = float(np.sin(2.0 * np.pi * hour_val / 24.0))
        hour_cos = float(np.cos(2.0 * np.pi * hour_val / 24.0))
        dow_sin = float(np.sin(2.0 * np.pi * dow_val / 7.0))
        dow_cos = float(np.cos(2.0 * np.pi * dow_val / 7.0))

        # 4. Assemble Complete 18-Feature Vector
        feature_dict = {
            "hour": hour_val,
            "day_of_week": dow_val,
            "day_of_month": dom_val,
            "month": month_val,
            "is_weekend": is_weekend_val,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week_sin": dow_sin,
            "day_of_week_cos": dow_cos,
            "lag_1": lags_dict["lag_1"],
            "lag_24": lags_dict["lag_24"],
            "lag_168": lags_dict["lag_168"],
            "rolling_mean_24": lags_dict["rolling_mean_24"],
            "rolling_std_24": lags_dict["rolling_std_24"],
            "temp_f_zonal_weighted": weather_res.temp_f_zonal_weighted,
            "cooling_degree_proxy_regional": weather_res.cooling_degree_proxy_regional,
            "heating_degree_proxy_regional": weather_res.heating_degree_proxy_regional,
            "temp_spread_regional": weather_res.temp_spread_regional,
        }

        df_row = pd.DataFrame([feature_dict])

        # 5. Production Schema Validation
        self.engine.validate_features(df_row)

        # 6. Point Forecasting
        X = df_row[self.engine.expected_features]
        point_preds = self.engine.model.predict(X)
        point_mw = float(np.round(point_preds[0], 2))

        # 7. Probabilistic Quantiles & Adaptive Calibration
        if self.prob_model is not None and self.prob_model.is_fitted:
            q_dict = self.prob_model.predict(df_row[self.prob_model.feature_names])
            raw_q05 = float(q_dict["q05"][0])
            raw_q50 = float(q_dict["q50"][0])
            raw_q95 = float(q_dict["q95"][0])

            if self.adaptive_calibrator is not None and self.adaptive_calibrator.is_calibrated:
                c_low, c_med, c_high = self.adaptive_calibrator.adjust(
                    q_low=np.array([raw_q05]),
                    q_median=np.array([raw_q50]),
                    q_high=np.array([raw_q95]),
                    timestamps=[target_ts.isoformat()],
                    demand_preds=np.array([point_mw]),
                )
                q05_mw = float(np.round(c_low[0], 2))
                q50_mw = float(np.round(c_med[0], 2))
                q95_mw = float(np.round(c_high[0], 2))
            elif self.calibrator is not None and self.calibrator.is_calibrated:
                c_low, c_med, c_high = self.calibrator.adjust(
                    np.array([raw_q05]), np.array([raw_q50]), np.array([raw_q95])
                )
                q05_mw = float(np.round(c_low[0], 2))
                q50_mw = float(np.round(c_med[0], 2))
                q95_mw = float(np.round(c_high[0], 2))
            else:
                q05_mw, q50_mw, q95_mw = raw_q05, raw_q50, raw_q95
        else:
            # Heuristic standard error fallback if quantile models missing
            q05_mw = round(point_mw - 1.645 * 700.0, 2)
            q50_mw = point_mw
            q95_mw = round(point_mw + 1.645 * 700.0, 2)

        # Enforce non-crossing monotonicity: Q05 <= Q50 <= Q95
        q05_mw, q50_mw, q95_mw = tuple(np.maximum.accumulate([q05_mw, q50_mw, q95_mw]))
        interval_width = round(q95_mw - q05_mw, 2)

        return OrchestratedForecastResult(
            prediction_timestamp_utc=datetime.now(timezone.utc),
            target_timestamp=target_ts.isoformat(),
            horizon=horizon,
            point_forecast_mw=point_mw,
            q05_mw=round(q05_mw, 2),
            q50_mw=round(q50_mw, 2),
            q95_mw=round(q95_mw, 2),
            interval_width_mw=interval_width,
            nominal_coverage=0.90,
            model_identifier="pjm_production_forecaster_phase15_orchestrated",
            model_version=self.engine.model_version,
            weather_data_provenance=weather_res.provenance,
            data_quality_warnings=warnings,
            features_used=feature_dict,
        )

    def _roll_forward_demand_lags(
        self,
        origin_ts: pd.Timestamp,
        steps: int,
        weather_res: WeatherFeatureResult,
    ) -> tuple[dict[str, float], list[str]]:
        """Perform recursive roll-forward to generate autoregressive lags for H > 1.

        Starts strictly from available history at origin_ts (T - H), predicting step-by-step
        and feeding point forecasts back into autoregressive lag buffers.
        """
        self.demand_provider._ensure_loaded()
        demand_series = self.demand_provider._demand_series
        assert demand_series is not None

        # Filter admissible history up to origin_ts
        admissible = demand_series[demand_series.index <= origin_ts]

        # Dictionary holding generated predictions for t > origin_ts
        simulated_demand: dict[pd.Timestamp, float] = {}

        def get_demand_at(t: pd.Timestamp) -> float:
            if t in simulated_demand:
                return simulated_demand[t]
            if t in admissible.index:
                return float(admissible.loc[t])
            archive_max = demand_series.index.max().isoformat()
            raise MissingDemandHistoryError(
                f"Missing required historical demand observation at {t.isoformat()} for recursive roll-forward. "
                f"Admissible cutoff is {origin_ts.isoformat()}."
            )

        final_lags: dict[str, float] = {}

        # Roll forward from step 1 up to steps
        for step in range(1, steps + 1):
            target_step_ts = origin_ts + pd.Timedelta(hours=step)

            # Lags for target_step_ts:
            lag_1_t = target_step_ts - pd.Timedelta(hours=1)
            lag_24_t = target_step_ts - pd.Timedelta(hours=24)
            lag_168_t = target_step_ts - pd.Timedelta(hours=168)

            lag_1_val = get_demand_at(lag_1_t)
            lag_24_val = get_demand_at(lag_24_t)
            lag_168_val = get_demand_at(lag_168_t)

            # 24-hour rolling window ending at target_step_ts - 1h
            rolling_window_vals = [
                get_demand_at(target_step_ts - pd.Timedelta(hours=h))
                for h in range(24, 0, -1)
            ]
            rolling_mean_val = float(np.mean(rolling_window_vals))
            rolling_std_val = float(np.std(rolling_window_vals, ddof=1))

            if step == steps:
                # We have reached target T! Return the final lags dict
                final_lags = {
                    "lag_1": round(lag_1_val, 2),
                    "lag_24": round(lag_24_val, 2),
                    "lag_168": round(lag_168_val, 2),
                    "rolling_mean_24": round(rolling_mean_val, 2),
                    "rolling_std_24": round(rolling_std_val, 2),
                }
                break

            # For intermediate steps, evaluate model to predict demand at target_step_ts
            hour_val = float(target_step_ts.hour)
            dow_val = float(target_step_ts.dayofweek)
            row_dict = {
                "hour": hour_val,
                "day_of_week": dow_val,
                "day_of_month": float(target_step_ts.day),
                "month": float(target_step_ts.month),
                "is_weekend": 1.0 if dow_val >= 5 else 0.0,
                "hour_sin": float(np.sin(2.0 * np.pi * hour_val / 24.0)),
                "hour_cos": float(np.cos(2.0 * np.pi * hour_val / 24.0)),
                "day_of_week_sin": float(np.sin(2.0 * np.pi * dow_val / 7.0)),
                "day_of_week_cos": float(np.cos(2.0 * np.pi * dow_val / 7.0)),
                "lag_1": lag_1_val,
                "lag_24": lag_24_val,
                "lag_168": lag_168_val,
                "rolling_mean_24": rolling_mean_val,
                "rolling_std_24": rolling_std_val,
                "temp_f_zonal_weighted": weather_res.temp_f_zonal_weighted,
                "cooling_degree_proxy_regional": weather_res.cooling_degree_proxy_regional,
                "heating_degree_proxy_regional": weather_res.heating_degree_proxy_regional,
                "temp_spread_regional": weather_res.temp_spread_regional,
            }
            df_inter = pd.DataFrame([row_dict])[self.engine.expected_features]
            pred_step_mw = float(self.engine.model.predict(df_inter)[0])
            simulated_demand[target_step_ts] = pred_step_mw

        warnings = [
            f"RECURSIVE_DISPATCH_ROLLFORWARD: Generated multi-step autoregressive lags across {steps} steps from origin {origin_ts.isoformat()}."
        ]
        return final_lags, warnings
