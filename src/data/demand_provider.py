"""Demand History Provider for Automated Forecast Orchestration.

Calculates leakage-free autoregressive lag and rolling demand features
conforming to the production feature schema:
- lag_1
- lag_24
- lag_168
- rolling_mean_24
- rolling_std_24

Key Guarantees:
- Enforces strict operational decision cutoff: only observations at t <= t_cutoff
  (where t_cutoff = T - H hours) are admissible.
- Never uses schema means or static defaults as silent substitutes for missing demand history.
- Returns explicit, actionable errors when history is missing, incomplete, or temporally ineligible.
- Documents exact minimum continuous history requirements for each horizon.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import logging
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_CLEANED_DEMAND_PATH: Path = PROCESSED_DATA_DIR / "pjm_cleaned.csv"


class DemandHistoryError(Exception):
    """Base exception for demand history retrieval errors."""
    pass


class MissingDemandHistoryError(DemandHistoryError):
    """Raised when required historical demand timestamps are not present in dataset."""
    pass


class IncompleteDemandHistoryError(DemandHistoryError):
    """Raised when demand records exist but contain NaNs or insufficient window size."""
    pass


class TemporalIneligibilityError(DemandHistoryError):
    """Raised when an observation required by a feature occurred after the decision cutoff."""
    pass


@dataclass(frozen=True)
class DemandLagResult:
    """Immutable data contract for prepared demand lag features."""
    lag_1: float
    lag_24: float
    lag_168: float
    rolling_mean_24: float
    rolling_std_24: float
    cutoff_time_utc: str
    latest_observation_time: str
    sample_count_in_window: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary matching production feature schema keys."""
        return {
            "lag_1": self.lag_1,
            "lag_24": self.lag_24,
            "lag_168": self.lag_168,
            "rolling_mean_24": self.rolling_mean_24,
            "rolling_std_24": self.rolling_std_24,
        }


class DemandHistoryProvider:
    """Manages access to energy consumption series and prepares autoregressive features."""

    def __init__(
        self,
        demand_data_path: Path | str = DEFAULT_CLEANED_DEMAND_PATH,
        demand_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialize demand history provider.

        Args:
            demand_data_path: Path to cleaned hourly energy consumption CSV.
            demand_df: Optional pre-loaded DataFrame (for testing or real-time buffers).
        """
        self.demand_data_path = Path(demand_data_path)
        self._df: pd.DataFrame | None = None
        self._demand_series: pd.Series | None = None

        if demand_df is not None:
            self._set_data(demand_df)

    def _set_data(self, df: pd.DataFrame) -> None:
        """Process and index demand DataFrame."""
        clean = df.copy()
        if "timestamp" not in clean.columns:
            raise ValueError("Demand DataFrame must contain a 'timestamp' column.")

        val_col = "consumption" if "consumption" in clean.columns else "total_power_demand"
        if val_col not in clean.columns:
            raise ValueError(f"Demand DataFrame must contain 'consumption' or 'total_power_demand'. Columns: {clean.columns.tolist()}")

        clean["timestamp"] = pd.to_datetime(clean["timestamp"])
        if clean["timestamp"].dt.tz is None:
            clean["timestamp"] = clean["timestamp"].dt.tz_localize("UTC")
        else:
            clean["timestamp"] = clean["timestamp"].dt.tz_convert("UTC")

        clean = clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        clean = clean.reset_index(drop=True)

        self._df = clean
        self._demand_series = pd.Series(
            clean[val_col].to_numpy(dtype=float),
            index=clean["timestamp"],
            name="consumption",
        )

    def _ensure_loaded(self) -> None:
        """Load dataset from disk if not already in memory."""
        if self._demand_series is not None:
            return

        if not self.demand_data_path.is_file():
            raise FileNotFoundError(f"Cleaned demand dataset not found at: {self.demand_data_path}")

        raw_df = pd.read_csv(self.demand_data_path)
        self._set_data(raw_df)

    def register_observations(self, new_records_df: pd.DataFrame) -> None:
        """Append or update real-time demand observations in the provider buffer."""
        self._ensure_loaded()
        assert self._df is not None

        combined = pd.concat([self._df, new_records_df], ignore_index=True)
        self._set_data(combined)

    def get_demand_lags(
        self,
        target_timestamp: pd.Timestamp | str,
        horizon: int = 1,
    ) -> DemandLagResult:
        """Calculate exact autoregressive lag and rolling features for target hour and horizon.

        Minimum Continuous History Requirements:
        - For H=1: Continuous history from T - 168h to T - 1h (168 continuous hours).
        - For H > 1: Observations at t > T - H are in the future relative to the decision
          cutoff and cannot be observed directly without future lookahead.

        Args:
            target_timestamp: Target valid hour (T).
            horizon: Forecast horizon (1, 24, 48).

        Returns:
            DemandLagResult: Exact lag and rolling features.

        Raises:
            TemporalIneligibilityError: If horizon H > 1 and actual future demand is requested.
            MissingDemandHistoryError: If required past timestamps are missing from history.
            IncompleteDemandHistoryError: If observations contain NaNs or incomplete window.
        """
        self._ensure_loaded()
        assert self._demand_series is not None

        target_ts = pd.to_datetime(target_timestamp)
        if target_ts.tz is None:
            target_ts = target_ts.tz_localize("UTC")
        else:
            target_ts = target_ts.tz_convert("UTC")

        cutoff_ts = target_ts - pd.Timedelta(hours=horizon)

        # 1. Temporal Eligibility Check:
        # lag_1 is consumption at T - 1h. If H > 1, T - 1h is strictly AFTER cutoff t_cutoff = T - H.
        lag_1_ts = target_ts - pd.Timedelta(hours=1)
        if lag_1_ts > cutoff_ts:
            raise TemporalIneligibilityError(
                f"TEMPORAL INELIGIBILITY: Demand observation at T-1h ({lag_1_ts.isoformat()}) "
                f"occurred after the decision cutoff ({cutoff_ts.isoformat()}) for horizon H={horizon}h. "
                f"Actual future demand cannot be used as lag_1 without target leakage. "
                f"At H={horizon}h, direct actualized demand history ends at {cutoff_ts.isoformat()}."
            )

        # 2. Causality filtering: filter to admissible history t <= cutoff_ts
        admissible = self._demand_series[self._demand_series.index <= cutoff_ts]

        archive_max = self._demand_series.index.max()
        archive_min = self._demand_series.index.min()
        if admissible.empty or cutoff_ts > archive_max:
            raise MissingDemandHistoryError(
                f"No demand observations available at or before decision cutoff ({cutoff_ts.isoformat()}). "
                f"Local demand archive spans {archive_min.isoformat()} to {archive_max.isoformat()}. "
                f"Live or cached SCADA metering feed required for target {target_ts.isoformat()}."
            )

        # 3. Exact Autoregressive Lag Lookup
        lag_24_ts = target_ts - pd.Timedelta(hours=24)
        lag_168_ts = target_ts - pd.Timedelta(hours=168)

        missing_timestamps: list[str] = []
        if lag_1_ts not in admissible.index:
            missing_timestamps.append(f"lag_1 ({lag_1_ts.isoformat()})")
        if lag_24_ts not in admissible.index:
            missing_timestamps.append(f"lag_24 ({lag_24_ts.isoformat()})")
        if lag_168_ts not in admissible.index:
            missing_timestamps.append(f"lag_168 ({lag_168_ts.isoformat()})")

        if missing_timestamps:
            archive_max = self._demand_series.index.max().isoformat()
            raise MissingDemandHistoryError(
                f"Missing required demand history timestamps: {', '.join(missing_timestamps)}. "
                f"Latest available observation in archive is {archive_max}. "
                f"Minimum history required: 168 contiguous hours ending at T-1h."
            )

        val_lag_1 = float(admissible.loc[lag_1_ts])
        val_lag_24 = float(admissible.loc[lag_24_ts])
        val_lag_168 = float(admissible.loc[lag_168_ts])

        # Check for NaNs
        for name, val in [("lag_1", val_lag_1), ("lag_24", val_lag_24), ("lag_168", val_lag_168)]:
            if np.isnan(val) or np.isinf(val):
                raise IncompleteDemandHistoryError(f"Demand observation for {name} is non-finite: {val}")

        # 4. Rolling 24-Hour Statistics ending at T-1h
        window_start = target_ts - pd.Timedelta(hours=24)
        window_end = target_ts - pd.Timedelta(hours=1)
        window_slice = admissible.loc[window_start:window_end]

        if len(window_slice) < 24:
            raise IncompleteDemandHistoryError(
                f"Incomplete rolling 24h window: expected 24 hourly observations between "
                f"{window_start.isoformat()} and {window_end.isoformat()}, found {len(window_slice)}."
            )

        if window_slice.isna().any():
            raise IncompleteDemandHistoryError("NaN values detected in rolling 24h window slice.")

        rolling_mean = float(window_slice.mean())
        rolling_std = float(window_slice.std(ddof=1))

        return DemandLagResult(
            lag_1=round(val_lag_1, 2),
            lag_24=round(val_lag_24, 2),
            lag_168=round(val_lag_168, 2),
            rolling_mean_24=round(rolling_mean, 2),
            rolling_std_24=round(rolling_std, 2),
            cutoff_time_utc=cutoff_ts.isoformat(),
            latest_observation_time=admissible.index.max().isoformat(),
            sample_count_in_window=len(window_slice),
        )

    def get_origin_lags(
        self,
        origin_timestamp: pd.Timestamp | str,
    ) -> DemandLagResult:
        """Calculate lags available at a specific decision origin time (t_origin).

        Useful for multi-step recursive forecasting roll-forward.
        At t_origin, lags correspond to t_origin, t_origin - 23h, t_origin - 167h.
        """
        # When origin is t_origin, target is t_origin + 1h for a 1-step advance
        t_origin = pd.to_datetime(origin_timestamp)
        target_t1 = t_origin + pd.Timedelta(hours=1)
        return self.get_demand_lags(target_timestamp=target_t1, horizon=1)
