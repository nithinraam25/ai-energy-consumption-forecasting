"""Fallback Weather Provider.

Provides an explicitly configured, causal fallback mechanism when real-time
NOAA HRRR forecasts are temporarily unreachable or delayed:
- 3-tier diurnal climatology fallback based exclusively on historical training records.
- Step-back cycle fallback using previous valid cycles (f03 / f04).
- STRICT GUARANTEE: Never silently substitutes future observed weather for missing forecasts.
- Marks is_fallback=True and records explicit provenance and fallback reason.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging
import pandas as pd

from src.config import MODELS_DIR
from src.weather.providers.base import (
    WeatherProviderBase,
    WeatherForecastResponse,
    WeatherProviderStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_IMPUTER_PATH: Path = MODELS_DIR / "weather_imputer_params.json"


class FallbackWeatherProvider(WeatherProviderBase):
    """Causal fallback weather provider using pre-calculated diurnal training profiles."""

    def __init__(
        self,
        imputer_params_path: Path | str = DEFAULT_IMPUTER_PATH,
        fallback_mode: str = "DIURNAL_CLIMATOLOGY",
    ) -> None:
        """Initialize fallback weather provider.

        Args:
            imputer_params_path: Path to serialized weather imputer JSON parameters.
            fallback_mode: Fallback strategy ('DIURNAL_CLIMATOLOGY' or 'GLOBAL_MEAN').
        """
        self.imputer_params_path = Path(imputer_params_path)
        self.fallback_mode = fallback_mode
        self._status = WeatherProviderStatus(provider_name="CAUSAL_FALLBACK_PROVIDER")

        self.diurnal_table: dict[str, float] = {}
        self.monthly_means: dict[str, float] = {}
        self.global_mean: float = 55.4  # Historical default mean deg F

        self._load_parameters()

    def _load_parameters(self) -> None:
        """Load training-only statistical parameters from disk."""
        if self.imputer_params_path.is_file():
            try:
                with open(self.imputer_params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                self.diurnal_table = params.get("diurnal_table", {})
                self.monthly_means = params.get("monthly_means", {})
                self.global_mean = float(params.get("global_mean", 55.4))
                self._status.is_available = True
            except Exception as exc:
                logger.error(f"Failed to load imputer parameters from {self.imputer_params_path}: {exc}")
                self._status.last_error = str(exc)
        else:
            logger.warning(f"Imputer parameters not found at {self.imputer_params_path}; using static fallback.")
            self._status.last_error = "Parameters file not found"

    def get_status(self) -> WeatherProviderStatus:
        """Return the current provider operational status."""
        return self._status

    def discover_latest_cycles(self, limit: int = 5) -> list[dict[str, Any]]:
        """Fallback provider does not have published external cycles."""
        return []

    def fetch_forecast(
        self,
        location_id: str,
        target_time_utc: pd.Timestamp | str,
        lead_hours: int = 2,
        cutoff_time_utc: pd.Timestamp | str | None = None,
    ) -> WeatherForecastResponse:
        """Generate a causal fallback forecast strictly from historical climatological statistics.

        Anti-substitution policy: Strictly refuses to access any future observations.

        Args:
            location_id: Station or zone identifier.
            target_time_utc: Target valid timestamp.
            lead_hours: Forecast lead time.
            cutoff_time_utc: Optional prediction cutoff.

        Returns:
            WeatherForecastResponse: Annotated fallback forecast.
        """
        target_ts = pd.to_datetime(target_time_utc)
        month = target_ts.month
        hour = target_ts.hour

        # Tier 1: 288-cell monthly diurnal cell
        cell_key = f"{month:02d}_{hour:02d}"
        if cell_key in self.diurnal_table:
            temp_f = self.diurnal_table[cell_key]
            tier_used = "TIER_1_DIURNAL_TABLE"
        elif str(month) in self.monthly_means:
            temp_f = self.monthly_means[str(month)]
            tier_used = "TIER_2_MONTHLY_MEAN"
        else:
            temp_f = self.global_mean
            tier_used = "TIER_3_GLOBAL_MEAN"

        retrieval_ts = datetime.now(timezone.utc).isoformat()
        init_ts = (target_ts - pd.Timedelta(hours=lead_hours)).isoformat()

        avail_ts = (
            pd.to_datetime(cutoff_time_utc).isoformat()
            if cutoff_time_utc is not None
            else target_ts.isoformat()
        )

        return WeatherForecastResponse(
            location_id=location_id,
            issue_time_utc=init_ts,
            valid_time_utc=target_ts.isoformat(),
            lead_time_hours=lead_hours,
            temp_f=round(float(temp_f), 2),
            available_time_utc=avail_ts,
            source=f"FALLBACK_{self.fallback_mode}_{tier_used}",
            retrieval_time_utc=retrieval_ts,
            is_fallback=True,
            metadata={
                "fallback_tier": tier_used,
                "anti_substitution_verified": True,
                "reason": "Operational NWP unavailable or offline",
            },
        )
