"""Operational Real-Time Demand Ingestion Adapters.

Provides production-ready, testable client interfaces for official electric grid
demand telemetry feeds:
1. PJM Data Miner 2 REST API (RTO-level and Zonal load feeds).
2. U.S. Energy Information Administration (EIA) API v2 (Hourly RTO demand).

Security & Integrity Policies:
- Zero Secret Logging: API keys and tokens are NEVER printed or logged.
- Environment-Configured: Credentials must be supplied via PJM_API_KEY or EIA_API_KEY.
- Anti-Fabrication Rule: In the absence of authenticated credentials, the adapter
  strictly refuses to synthesize or mock observations, reporting BLOCKED status.
- Strict Data Contract: All ingested series must pass `validate_demand_series()`:
  unambiguous UTC timestamps, regular 1h cadence, physical bounds [10k, 75k] MW,
  and no NaN or Inf values.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
import logging
import os
from typing import Any
import requests
import numpy as np
import pandas as pd

from src.data.demand_contract import (
    validate_demand_series,
    DemandValidationReport,
    DemandProvenance,
    DemandContractViolationError,
    MIN_PLAUSIBLE_DEMAND_MW,
    MAX_PLAUSIBLE_DEMAND_MW,
)

logger = logging.getLogger(__name__)


class DemandTelemetryAuthError(RuntimeError):
    """Raised when operational demand feed credentials are absent or rejected."""
    pass


class DemandTelemetryUnavailableError(RuntimeError):
    """Raised when operational demand feed is unreachable or encounters an outage."""
    pass


class DemandTelemetryAdapter(ABC):
    """Abstract interface for operational electrical load telemetry providers."""

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Name of the telemetry feed source."""
        pass

    @property
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Whether valid credentials are provided in the environment."""
        pass

    @abstractmethod
    def fetch_recent_demand(
        self,
        start_utc: datetime | pd.Timestamp,
        end_utc: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch continuous hourly electrical demand between start_utc and end_utc.

        Returns DataFrame with 'timestamp' (UTC DatetimeIndex) and 'consumption' (MW).
        """
        pass


class PJMDataMinerAdapter(DemandTelemetryAdapter):
    """Official PJM Data Miner 2 REST API Client.

    Endpoint: `https://api.pjm.com/api/v1/inst_load` (Instantaneous 5-min Load)
    Header: `Ocp-Apim-Subscription-Key` (from environment variable `PJM_API_KEY`)
    """

    DEFAULT_BASE_URL = "https://api.pjm.com/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = 15,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.getenv("PJM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def adapter_name(self) -> str:
        return "PJM_DATA_MINER_2"

    @property
    def is_authenticated(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 8)

    def fetch_recent_demand(
        self,
        start_utc: datetime | pd.Timestamp,
        end_utc: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch demand observations from PJM Data Miner 2."""
        if not self.is_authenticated:
            raise DemandTelemetryAuthError(
                "PJM Data Miner 2 authentication key (PJM_API_KEY) is not configured. "
                "Operational telemetry ingestion is blocked."
            )

        start_ts = pd.to_datetime(start_utc, utc=True)
        end_ts = pd.to_datetime(end_utc, utc=True)

        url = f"{self.base_url}/inst_load"
        # Masked headers: never log or expose _api_key
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key.strip(),
            "Accept": "application/json",
            "User-Agent": "PJM-East-Forecaster/1.0",
        }
        params = {
            "rowCount": 50000,
            "startRow": 1,
            "datetime_beginning_utc": start_ts.strftime("%Y-%m-%d %H:%M"),
            "datetime_ending_utc": end_ts.strftime("%Y-%m-%d %H:%M"),
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=self.timeout_seconds)
                if resp.status_code == 401 or resp.status_code == 403:
                    raise DemandTelemetryAuthError("PJM Data Miner 2 rejected API credentials (HTTP 401/403).")
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                if not items:
                    return pd.DataFrame(columns=["timestamp", "consumption"])

                raw_df = pd.DataFrame(items)
                # PJM inst_load fields: 'datetime_beginning_utc', 'instantaneous_load'
                time_col = "datetime_beginning_utc" if "datetime_beginning_utc" in raw_df.columns else "timestamp"
                val_col = "instantaneous_load" if "instantaneous_load" in raw_df.columns else "load"

                raw_df["timestamp"] = pd.to_datetime(raw_df[time_col], utc=True)
                raw_df["consumption"] = pd.to_numeric(raw_df[val_col], errors="coerce")

                # Resample 5-minute instantaneous load to regular 1-hour average
                hourly_df = (
                    raw_df.set_index("timestamp")["consumption"]
                    .resample("1h")
                    .mean()
                    .dropna()
                    .reset_index()
                )

                # Validate against contract
                report = validate_demand_series(
                    hourly_df,
                    provenance=DemandProvenance.REAL_TIME_SCADA,
                    check_cadence=True,
                    strict_bounds=True,
                )
                if not report.is_valid:
                    raise DemandContractViolationError(
                        f"PJM Data Miner 2 payload violated demand contract: {report.violations}"
                    )

                return hourly_df
            except (DemandTelemetryAuthError, DemandContractViolationError):
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning("PJM Data Miner request attempt %d failed: %s", attempt + 1, exc)

        raise DemandTelemetryUnavailableError(
            f"Failed to fetch demand from PJM Data Miner after {self.max_retries} attempts: {last_exc}"
        )


class EIADemandAdapter(DemandTelemetryAdapter):
    """U.S. Energy Information Administration (EIA) API v2 Client.

    Endpoint: `https://api.eia.gov/v2/electricity/rto/region-data/data/`
    Header / Query: `X-Api-Key` (from environment variable `EIA_API_KEY`)
    """

    DEFAULT_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = 15,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.getenv("EIA_API_KEY")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def adapter_name(self) -> str:
        return "EIA_API_V2"

    @property
    def is_authenticated(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 8)

    def fetch_recent_demand(
        self,
        start_utc: datetime | pd.Timestamp,
        end_utc: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch demand observations from EIA API v2."""
        if not self.is_authenticated:
            raise DemandTelemetryAuthError(
                "EIA API v2 authentication key (EIA_API_KEY) is not configured. "
                "Operational telemetry ingestion is blocked."
            )

        start_ts = pd.to_datetime(start_utc, utc=True)
        end_ts = pd.to_datetime(end_utc, utc=True)

        headers = {
            "X-Api-Key": self._api_key.strip(),
            "Accept": "application/json",
            "User-Agent": "PJM-East-Forecaster/1.0",
        }
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "PJM",
            "facets[type][]": "D",  # Demand
            "start": start_ts.strftime("%Y-%m-%dT%H"),
            "end": end_ts.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(self.base_url, headers=headers, params=params, timeout=self.timeout_seconds)
                if resp.status_code == 401 or resp.status_code == 403:
                    raise DemandTelemetryAuthError("EIA API v2 rejected API credentials (HTTP 401/403).")
                resp.raise_for_status()
                data = resp.json()
                raw_records = data.get("response", {}).get("data", [])
                if not raw_records:
                    return pd.DataFrame(columns=["timestamp", "consumption"])

                raw_df = pd.DataFrame(raw_records)
                # EIA period format: '2026-10-15T12'
                raw_df["timestamp"] = pd.to_datetime(raw_df["period"], utc=True)
                raw_df["consumption"] = pd.to_numeric(raw_df["value"], errors="coerce")

                clean_df = raw_df[["timestamp", "consumption"]].dropna().sort_values("timestamp").reset_index(drop=True)

                # Validate against contract
                report = validate_demand_series(
                    clean_df,
                    provenance=DemandProvenance.REAL_TIME_SCADA,
                    check_cadence=True,
                    strict_bounds=True,
                )
                if not report.is_valid:
                    raise DemandContractViolationError(
                        f"EIA API v2 payload violated demand contract: {report.violations}"
                    )

                return clean_df
            except (DemandTelemetryAuthError, DemandContractViolationError):
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning("EIA API v2 request attempt %d failed: %s", attempt + 1, exc)

        raise DemandTelemetryUnavailableError(
            f"Failed to fetch demand from EIA API v2 after {self.max_retries} attempts: {last_exc}"
        )


def get_operational_demand_adapter() -> DemandTelemetryAdapter:
    """Factory to retrieve configured operational demand adapter, or unconfigured fallback."""
    if os.getenv("PJM_API_KEY"):
        return PJMDataMinerAdapter()
    elif os.getenv("EIA_API_KEY"):
        return EIADemandAdapter()
    else:
        return PJMDataMinerAdapter(api_key=None)
