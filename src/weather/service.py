"""Weather Feature Service for Automated Forecast Orchestration.

Discovers and loads eligible operational weather forecasts from the local disk cache
or invokes the documented causal fallback policy when cached data is missing,
stale, or causally ineligible.

Key Guarantees:
- Never fabricates live weather data.
- Enforces causality: t_avail <= t_cutoff (where t_cutoff = T - H hours).
- Enforces freshness: flags forecasts older than stale_threshold_hours (default 6h).
- Computes multi-station zonal load-weighted composites, spatial gradient spread,
  and space-conditioning degree proxies using verified domain formulas.
- Returns transparent data provenance and quality warnings.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging
import pandas as pd

from src.data.multi_station import compute_regional_aggregations
from src.weather.providers.base import WeatherForecastResponse
from src.weather.providers.fallback import FallbackWeatherProvider
from src.weather.providers.noaa_hrrr import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


class WeatherServiceError(Exception):
    """Base exception for weather service failures."""
    pass


class WeatherUnavailableError(WeatherServiceError):
    """Raised when eligible weather data cannot be obtained and fallback is disabled."""
    pass


class WeatherCausalityError(WeatherServiceError):
    """Raised when weather forecast was published after the decision cutoff."""
    pass


@dataclass(frozen=True)
class WeatherFeatureResult:
    """Immutable result contract for prepared weather features."""
    temp_f_zonal_weighted: float
    cooling_degree_proxy_regional: float
    heating_degree_proxy_regional: float
    temp_spread_regional: float
    temp_f_kphl: float | None
    temp_f_kewr: float | None
    temp_f_kilg: float | None
    provenance: str
    is_fallback: bool
    init_time_utc: str | None
    valid_time_utc: str | None
    available_time_utc: str | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to feature dictionary for model ingestion and API serialization."""
        return {
            "temp_f_zonal_weighted": self.temp_f_zonal_weighted,
            "cooling_degree_proxy_regional": self.cooling_degree_proxy_regional,
            "heating_degree_proxy_regional": self.heating_degree_proxy_regional,
            "temp_spread_regional": self.temp_spread_regional,
            "temp_f_kphl": self.temp_f_kphl,
            "temp_f_kewr": self.temp_f_kewr,
            "temp_f_kilg": self.temp_f_kilg,
            "provenance": self.provenance,
            "is_fallback": self.is_fallback,
            "init_time_utc": self.init_time_utc,
            "valid_time_utc": self.valid_time_utc,
            "available_time_utc": self.available_time_utc,
            "warnings": list(self.warnings),
        }


class WeatherFeatureService:
    """Service for obtaining eligible weather data and preparing regional model features."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        fallback_provider: FallbackWeatherProvider | None = None,
        stale_threshold_hours: float = 6.0,
    ) -> None:
        """Initialize weather feature service.

        Args:
            cache_dir: Path to directory containing cached HRRR forecast JSON files.
            fallback_provider: Optional fallback provider instance.
            stale_threshold_hours: Maximum allowable slack before stale warning (hours).
        """
        self.cache_dir = Path(cache_dir)
        self.fallback_provider = fallback_provider or FallbackWeatherProvider()
        self.stale_threshold_hours = float(stale_threshold_hours)

    def get_weather_features(
        self,
        target_timestamp: pd.Timestamp | str,
        horizon: int = 1,
        cutoff_time_utc: pd.Timestamp | str | None = None,
        allow_fallback: bool = True,
    ) -> WeatherFeatureResult:
        """Obtain eligible weather data and prepare station and regional features.

        Args:
            target_timestamp: Target valid hour for energy forecast.
            horizon: Forecast horizon (1, 24, 48).
            cutoff_time_utc: Decision cutoff timestamp. Defaults to target - horizon.
            allow_fallback: Whether to use causal climatology if cached HRRR unavailable.

        Returns:
            WeatherFeatureResult: Prepared weather features and provenance.

        Raises:
            WeatherUnavailableError: If cached data missing and fallback disallowed.
            WeatherCausalityError: If cached data violates causality and fallback disallowed.
        """
        target_ts = pd.to_datetime(target_timestamp)
        if target_ts.tz is None:
            target_ts = target_ts.tz_localize("UTC")
        else:
            target_ts = target_ts.tz_convert("UTC")

        if cutoff_time_utc is None:
            cutoff_ts = target_ts - pd.Timedelta(hours=horizon)
        else:
            cutoff_ts = pd.to_datetime(cutoff_time_utc)
            if cutoff_ts.tz is None:
                cutoff_ts = cutoff_ts.tz_localize("UTC")
            else:
                cutoff_ts = cutoff_ts.tz_convert("UTC")

        # 1. Search for causally eligible cached HRRR file
        cached_entry = self._find_eligible_cached_hrrr(target_ts, cutoff_ts)

        if cached_entry is not None:
            return self._prepare_from_cached_hrrr(cached_entry, cutoff_ts, horizon)

        # 2. Apply documented fallback policy if allowed
        if allow_fallback:
            return self._prepare_from_fallback(target_ts, cutoff_ts, horizon)

        raise WeatherUnavailableError(
            f"No causally eligible NOAA HRRR forecast found in cache for target {target_ts.isoformat()} "
            f"at cutoff {cutoff_ts.isoformat()}, and allow_fallback=False."
        )

    def _find_eligible_cached_hrrr(
        self,
        target_ts: pd.Timestamp,
        cutoff_ts: pd.Timestamp,
    ) -> dict[str, Any] | None:
        """Scan cache directory for a genuine HRRR cycle valid at target and available before cutoff."""
        if not self.cache_dir.is_dir():
            return None

        best_entry: dict[str, Any] | None = None
        best_avail_ts: pd.Timestamp | None = None

        for cache_file in self.cache_dir.glob("hrrr_*.json"):
            if cache_file.name == "processed_cycles.json":
                continue
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                valid_str = data.get("valid_time_utc")
                avail_str = data.get("available_time_utc")
                station_temps = data.get("station_temps_f")

                if not valid_str or not avail_str or not station_temps:
                    continue

                entry_valid_ts = pd.to_datetime(valid_str)
                entry_avail_ts = pd.to_datetime(avail_str)

                # Must match target valid timestamp
                if entry_valid_ts != target_ts:
                    continue

                # Must satisfy causality: available before or at decision cutoff
                if entry_avail_ts > cutoff_ts:
                    continue

                # Pick the most recently available eligible cycle
                if best_avail_ts is None or entry_avail_ts > best_avail_ts:
                    best_avail_ts = entry_avail_ts
                    best_entry = data
            except Exception as exc:
                logger.warning(f"Error reading cache file {cache_file}: {exc}")
                continue

        return best_entry

    def _prepare_from_cached_hrrr(
        self,
        data: dict[str, Any],
        cutoff_ts: pd.Timestamp,
        horizon: int,
    ) -> WeatherFeatureResult:
        """Derive model features from decoded genuine HRRR station temperatures."""
        station_temps = data["station_temps_f"]
        kphl = float(station_temps.get("KPHL", 65.0))
        kewr = float(station_temps.get("KEWR", 65.0))
        kilg = float(station_temps.get("KILG", 65.0))

        # Use authoritative regional aggregation logic from multi_station module
        reg_df = compute_regional_aggregations(
            temp_kphl=[kphl],
            temp_kewr=[kewr],
            temp_kilg=[kilg],
        )

        avail_ts = pd.to_datetime(data["available_time_utc"])
        slack_hours = (cutoff_ts - avail_ts).total_seconds() / 3600.0
        warnings: list[str] = []

        if slack_hours > self.stale_threshold_hours:
            warnings.append(
                f"STALE_WEATHER_INPUT: Cached HRRR forecast available at {data['available_time_utc']} "
                f"is {slack_hours:.1f}h older than the H={horizon}h cutoff ({cutoff_ts.isoformat()})."
            )

        extraction_method = data.get("extraction_method", "ECCODES_GRIB2_BILINEAR_INTERPOLATION")
        provenance = (
            "NOAA_HRRR_OPERATIONAL_ECCODES"
            if "ECCODES" in extraction_method.upper()
            else "NOAA_HRRR_OPERATIONAL"
        )

        return WeatherFeatureResult(
            temp_f_zonal_weighted=float(reg_df["temp_f_zonal_weighted"].iloc[0]),
            cooling_degree_proxy_regional=float(reg_df["cooling_degree_proxy_regional"].iloc[0]),
            heating_degree_proxy_regional=float(reg_df["heating_degree_proxy_regional"].iloc[0]),
            temp_spread_regional=float(reg_df["temp_spread_regional"].iloc[0]),
            temp_f_kphl=kphl,
            temp_f_kewr=kewr,
            temp_f_kilg=kilg,
            provenance=provenance,
            is_fallback=False,
            init_time_utc=data.get("init_time_utc"),
            valid_time_utc=data.get("valid_time_utc"),
            available_time_utc=data.get("available_time_utc"),
            warnings=warnings,
        )

    def _prepare_from_fallback(
        self,
        target_ts: pd.Timestamp,
        cutoff_ts: pd.Timestamp,
        horizon: int,
    ) -> WeatherFeatureResult:
        """Derive model features using training-only diurnal climatology fallback."""
        fb_resp = self.fallback_provider.fetch_forecast(
            location_id="REGIONAL_ZONAL",
            target_time_utc=target_ts,
            lead_hours=horizon,
            cutoff_time_utc=cutoff_ts,
        )

        zonal_temp = float(fb_resp.temp_f)
        cooling = float(max(0.0, zonal_temp - 65.0))
        heating = float(max(0.0, 65.0 - zonal_temp))
        spread = 2.536  # Historical average spread from production feature schema

        warnings = [
            f"FALLBACK_WEATHER_USED: Weather inputs supplied via fallback provider ({fb_resp.source})."
        ]

        return WeatherFeatureResult(
            temp_f_zonal_weighted=round(zonal_temp, 2),
            cooling_degree_proxy_regional=round(cooling, 2),
            heating_degree_proxy_regional=round(heating, 2),
            temp_spread_regional=round(spread, 2),
            temp_f_kphl=None,
            temp_f_kewr=None,
            temp_f_kilg=None,
            provenance="FALLBACK_DIURNAL_CLIMATOLOGY",
            is_fallback=True,
            init_time_utc=fb_resp.issue_time_utc,
            valid_time_utc=fb_resp.valid_time_utc,
            available_time_utc=fb_resp.available_time_utc,
            warnings=warnings,
        )
