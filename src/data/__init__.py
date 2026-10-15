"""Data package for AI-Powered Energy Consumption Forecasting.

Exports data loading, validation, and preprocessing components.
"""

from src.data.loader import (
    load_energy_data,
    TIMESTAMP_ALIASES,
    CONSUMPTION_ALIASES,
)
from src.data.validator import (
    validate_energy_data,
    ValidationReport,
)
from src.data.preprocessor import (
    preprocess_energy_data,
    save_cleaned_data,
    CleaningReport,
    VALID_DUPLICATE_POLICIES,
)
from src.data.weather_loader import (
    load_raw_noaa_kphl,
    build_hourly_kphl_weather_series,
    select_hourly_kphl_observations,
    parse_noaa_tmp,
    assign_nominal_utc_hour,
)

from src.data.nwp_loader import (
    NWPForecastRecord,
    build_operational_nwp_dataset,
    load_nwp_forecast_dataset,
    save_nwp_forecast_dataset,
    validate_nwp_forecast_data,
    verify_live_hrrr_access,
)

__all__ = [
    "load_energy_data",
    "TIMESTAMP_ALIASES",
    "CONSUMPTION_ALIASES",
    "validate_energy_data",
    "ValidationReport",
    "preprocess_energy_data",
    "save_cleaned_data",
    "CleaningReport",
    "VALID_DUPLICATE_POLICIES",
    "load_raw_noaa_kphl",
    "build_hourly_kphl_weather_series",
    "select_hourly_kphl_observations",
    "parse_noaa_tmp",
    "assign_nominal_utc_hour",
    "NWPForecastRecord",
    "build_operational_nwp_dataset",
    "load_nwp_forecast_dataset",
    "save_nwp_forecast_dataset",
    "validate_nwp_forecast_data",
    "verify_live_hrrr_access",
]

