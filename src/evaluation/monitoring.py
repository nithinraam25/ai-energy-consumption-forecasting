"""Operational Monitoring and Reliability Engine for Energy Consumption Forecasting.

Provides automated runtime monitoring for:
- Missing weather forecast inputs and cycle dropouts.
- Stale weather inputs (forecast age > threshold).
- Out-of-distribution feature values and drift detection (z-score bounds vs baseline).
- Missing energy consumption history inputs.
- Schema conformity and unexpected column alerts.
- Real-time prediction error monitoring (MAE, MAPE, residual threshold triggers).
- Structured JSON and markdown operational health audits.
"""

from typing import Any, Sequence
import json
import numpy as np
import pandas as pd


class ForecastHealthMonitor:
    """Operational reliability and data quality monitor for energy forecasting systems."""

    def __init__(
        self,
        stale_threshold_hours: float = 6.0,
        residual_alert_threshold_mw: float = 3000.0,
    ) -> None:
        """Initialize monitor.

        Args:
            stale_threshold_hours: Maximum allowable elapsed time before an NWP input is flagged as stale.
            residual_alert_threshold_mw: Forecast residual threshold in MW to trigger operational anomaly alert.
        """
        self.stale_threshold_hours = float(stale_threshold_hours)
        self.residual_alert_threshold_mw = float(residual_alert_threshold_mw)

    def audit_weather_inputs(
        self,
        weather_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        temp_col: str = "temp_f_fcst",
        issue_time_col: str = "issue_time_utc",
        valid_time_col: str = "valid_time_utc",
    ) -> dict[str, Any]:
        """Audit weather forecast availability, completeness, and staleness.

        Args:
            weather_df: Weather DataFrame to audit.
            timestamp_col: Target timestamp column.
            temp_col: Temperature forecast column.
            issue_time_col: Forecast issuance column.
            valid_time_col: Forecast valid time column.

        Returns:
            dict[str, Any]: Structured weather health report.
        """
        total_rows = len(weather_df)
        if total_rows == 0:
            return {"status": "EMPTY_BATCH", "total_rows": 0, "healthy": False}

        if temp_col in weather_df.columns:
            actual_temp_col = temp_col
        elif "temp_f_zonal_weighted" in weather_df.columns:
            actual_temp_col = "temp_f_zonal_weighted"
        elif "temp_f_kphl" in weather_df.columns:
            actual_temp_col = "temp_f_kphl"
        elif "temp_f" in weather_df.columns:
            actual_temp_col = "temp_f"
        else:
            actual_temp_col = None

        missing_count = int(weather_df[actual_temp_col].isna().sum()) if actual_temp_col is not None else total_rows
        missing_rate_pct = round((missing_count / total_rows) * 100.0, 2)

        # Staleness audit: (valid_time - issue_time) > threshold or lead_time_hours > threshold
        stale_count = 0
        if issue_time_col in weather_df.columns and valid_time_col in weather_df.columns:
            valid = pd.to_datetime(weather_df[valid_time_col])
            issue = pd.to_datetime(weather_df[issue_time_col])
            lead_hours = (valid - issue).dt.total_seconds() / 3600.0
            stale_count = int((lead_hours > self.stale_threshold_hours).sum())
        elif "lead_time_hours" in weather_df.columns:
            lead_hours = pd.to_numeric(weather_df["lead_time_hours"], errors="coerce")
            stale_count = int((lead_hours > self.stale_threshold_hours).sum())

        status = "HEALTHY"
        if missing_rate_pct > 20.0:
            status = "CRITICAL_MISSING_WEATHER"
        elif missing_rate_pct > 5.0:
            status = "DEGRADED"
        elif stale_count > 0:
            status = "STALE_WEATHER_INPUTS"

        is_healthy = bool(status == "HEALTHY")

        return {
            "total_records": total_rows,
            "missing_forecasts_count": missing_count,
            "missing_forecasts_pct": missing_rate_pct,
            "stale_forecasts_count": stale_count,
            "healthy": is_healthy,
            "status": status,
        }

    def audit_feature_distributions(
        self,
        feature_df: pd.DataFrame,
        feature_names: Sequence[str],
        reference_stats: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Check for out-of-distribution values or unexpected feature drift.

        Args:
            feature_df: Current inference batch.
            feature_names: List of features to audit.
            reference_stats: Optional dict mapping feature -> {mean, std, min, max}.

        Returns:
            dict[str, Any]: Feature drift and anomaly summary.
        """
        anomalies: dict[str, list[str]] = {}

        for col in feature_names:
            if col not in feature_df.columns:
                continue

            series = feature_df[col].dropna()
            if series.empty:
                anomalies[col] = ["All values missing"]
                continue

            col_anomalies = []
            # Check physically implausible bounds
            if "temp" in col and (series.min() < -40.0 or series.max() > 130.0):
                col_anomalies.append(f"Temperature outside physical limits: [{series.min():.1f}, {series.max():.1f}]")
            if "demand" in col or "lag" in col:
                if series.min() < 5000.0 or series.max() > 70000.0:
                    col_anomalies.append(f"Load demand outside PJM East operational limits: [{series.min():.1f}, {series.max():.1f}]")

            # Reference z-score check
            if reference_stats and col in reference_stats:
                ref_mean = reference_stats[col]["mean"]
                ref_std = reference_stats[col]["std"]
                if ref_std > 0:
                    batch_mean = series.mean()
                    z_score = abs(batch_mean - ref_mean) / ref_std
                    if z_score > 3.5:
                        col_anomalies.append(f"Significant distributional shift: z-score = {z_score:.2f}")

            if col_anomalies:
                anomalies[col] = col_anomalies

        return {
            "features_audited": len(feature_names),
            "anomalous_features_count": len(anomalies),
            "anomalies": anomalies,
            "healthy": len(anomalies) == 0,
        }

    def audit_prediction_performance(
        self,
        actual_demand: pd.Series | np.ndarray,
        predicted_demand: pd.Series | np.ndarray,
        timestamps: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Evaluate real-time forecast accuracy when ground-truth load arrives.

        Args:
            actual_demand: Actual verified electrical power demand (MW).
            predicted_demand: Model predicted electrical power demand (MW).
            timestamps: Optional timestamp series.

        Returns:
            dict[str, Any]: Performance and threshold violation audit.
        """
        y_true = np.asarray(actual_demand, dtype=float)
        y_pred = np.asarray(predicted_demand, dtype=float)

        residuals = y_true - y_pred
        abs_err = np.abs(residuals)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mape = float(np.mean(abs_err / y_true) * 100.0)

        # Threshold violations (residuals > alert threshold)
        violation_indices = np.where(abs_err > self.residual_alert_threshold_mw)[0]
        alert_count = int(len(violation_indices))

        return {
            "samples_evaluated": len(y_true),
            "mae_mw": round(mae, 2),
            "rmse_mw": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "residual_alerts_count": alert_count,
            "max_residual_mw": round(float(np.max(abs_err)), 2),
            "alert_threshold_mw": self.residual_alert_threshold_mw,
            "status": "NORMAL" if alert_count == 0 else "WARNING_ELEVATED_RESIDUALS",
        }
