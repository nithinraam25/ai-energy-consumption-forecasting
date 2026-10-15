"""Evaluation package for AI-Powered Energy Consumption Forecasting.

Exports metrics calculation, chronological splitting, and model comparison workflows.
"""

from src.evaluation.metrics import (
    calculate_metrics,
    chronological_split,
    compare_models,
    MetricResults,
)
from src.evaluation.error_analysis import (
    compute_error_metrics_by_hour,
    compute_error_metrics_by_weekday,
    compute_error_metrics_by_demand_band,
    identify_largest_errors,
    run_forecast_error_analysis,
)

from src.evaluation.monitoring import ForecastHealthMonitor

__all__ = [
    "calculate_metrics",
    "chronological_split",
    "compare_models",
    "MetricResults",
    "compute_error_metrics_by_hour",
    "compute_error_metrics_by_weekday",
    "compute_error_metrics_by_demand_band",
    "identify_largest_errors",
    "run_forecast_error_analysis",
    "ForecastHealthMonitor",
]
