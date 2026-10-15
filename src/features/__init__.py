"""Feature engineering package for AI-Powered Energy Consumption Forecasting.

Exports leakage-safe temporal features and sliding-window sequence generators.
"""

from src.features.temporal_features import (
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
    add_target,
    create_temporal_features,
)
from src.features.sequence_builder import (
    build_sliding_windows,
    SequenceDataset,
)
from src.features.weather_features import (
    compute_degree_proxies,
    WeatherImputer,
    add_weather_features,
    add_operational_nwp_features,
)

__all__ = [
    "add_calendar_features",
    "add_cyclical_features",
    "add_lag_features",
    "add_rolling_features",
    "add_target",
    "create_temporal_features",
    "build_sliding_windows",
    "SequenceDataset",
    "compute_degree_proxies",
    "WeatherImputer",
    "add_weather_features",
    "add_operational_nwp_features",
]

