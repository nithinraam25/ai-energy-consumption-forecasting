"""Abstract Base Class and Contracts for Weather Providers.

Defines:
- WeatherForecastResponse: Structured forecast output contract.
- WeatherProviderStatus: Provider health and availability state.
- WeatherProviderBase: Abstract provider interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class WeatherForecastResponse:
    """Immutable data contract representing a retrieved weather forecast."""
    location_id: str
    issue_time_utc: str
    valid_time_utc: str
    lead_time_hours: int
    temp_f: float
    available_time_utc: str
    source: str
    retrieval_time_utc: str
    is_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeatherProviderStatus:
    """Mutable health and connectivity status of a weather provider."""
    provider_name: str
    is_available: bool = True
    last_successful_cycle: str | None = None
    consecutive_failures: int = 0
    last_error: str | None = None
    last_check_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class WeatherProviderBase(ABC):
    """Abstract interface for all meteorological data providers."""

    @abstractmethod
    def fetch_forecast(
        self,
        location_id: str,
        target_time_utc: pd.Timestamp | str,
        lead_hours: int = 2,
        cutoff_time_utc: pd.Timestamp | str | None = None,
    ) -> WeatherForecastResponse:
        """Retrieve a weather forecast for a target location and valid time.

        Args:
            location_id: Identifier of the station or regional zone (e.g. 'KPHL').
            target_time_utc: Desired forecast valid time in UTC.
            lead_hours: Forecast projection lead horizon.
            cutoff_time_utc: Optional operational cutoff timestamp. If provided,
                the returned forecast's available_time_utc MUST be <= cutoff_time_utc.

        Returns:
            WeatherForecastResponse: Validated forecast response.

        Raises:
            ValueError: If parameters are invalid or cutoff constraint is violated.
            RuntimeError: If data retrieval fails and no fallback is configured.
        """
        pass

    @abstractmethod
    def discover_latest_cycles(self, limit: int = 5) -> list[dict[str, Any]]:
        """List recently published NWP initialization cycles.

        Args:
            limit: Maximum number of recent cycles to discover.

        Returns:
            list[dict[str, Any]]: List of cycle metadata records.
        """
        pass

    @abstractmethod
    def get_status(self) -> WeatherProviderStatus:
        """Return the current operational status of the provider."""
        pass
