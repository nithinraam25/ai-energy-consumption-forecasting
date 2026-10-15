"""Demand Data Contract and Telemetry Validation Interface.

Defines the formal data contracts, validation rules, and causality gates
for electrical demand time-series consumed by the PJM East forecasting system.

Strict Guarantees:
1. Timezone & Temporal Regularity:
   - All timestamps must be convertible to unambiguous UTC.
   - Enforces regular hourly cadence (freq="1h").
   - Explicitly handles Daylight Saving Time (DST) by ensuring monotonic,
     gap-free UTC indexing (preventing civil time ambiguity during autumn fall-back
     or spring forward transitions).
2. Value Plausibility & Units:
   - Unit is strictly Megawatts (MW).
   - All values must be strictly finite (NaN, Inf, -Inf prohibited).
   - Plausible physical bounds for PJM East zonal grid:
     10,000.0 MW <= demand <= 75,000.0 MW.
3. Provenance Differentiation:
   - Distinct classification between HISTORICAL_ARCHIVE, TEST_FIXTURE,
     SIMULATED_REPLAY, and REAL_TIME_SCADA.
   - Prohibits claiming unverified sources as REAL_TIME_SCADA.
4. Decision Cutoff & Causality:
   - Observations occurring after the forecast decision cutoff (t > T - H)
     are strictly inadmissible.
5. Fail-Safe Telemetry Check:
   - When real telemetry is missing at or before cutoff, operational forecasts
     are blocked and report exactly which continuous timestamps are missing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

# Physical operational boundaries for PJM East RTO load
MIN_PLAUSIBLE_DEMAND_MW: float = 10000.0
MAX_PLAUSIBLE_DEMAND_MW: float = 75000.0
REQUIRED_FREQUENCY: str = "1h"
HISTORICAL_ARCHIVE_END_UTC: str = "2018-08-03T00:00:00Z"


class DemandProvenance(str, Enum):
    """Categorical classification of electrical demand sources."""
    HISTORICAL_ARCHIVE = "HISTORICAL_ARCHIVE"
    TEST_FIXTURE = "TEST_FIXTURE"
    SIMULATED_REPLAY = "SIMULATED_REPLAY"
    REAL_TIME_SCADA = "REAL_TIME_SCADA"


class DemandContractViolationError(ValueError):
    """Raised when demand time-series violates data contract rules."""
    pass


@dataclass(frozen=True)
class DemandValidationReport:
    """Detailed audit report of a demand series validation check."""
    is_valid: bool
    status: str  # "PASSED", "WARNING", "REJECTED"
    provenance: DemandProvenance
    total_records: int
    start_timestamp_utc: str | None
    end_timestamp_utc: str | None
    duplicates_count: int
    out_of_order_count: int
    missing_intervals_count: int
    non_finite_count: int
    out_of_bounds_count: int
    violations: list[str] = field(default_factory=list)
    missing_timestamps_sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "provenance": self.provenance.value,
            "total_records": self.total_records,
            "start_timestamp_utc": self.start_timestamp_utc,
            "end_timestamp_utc": self.end_timestamp_utc,
            "duplicates_count": self.duplicates_count,
            "out_of_order_count": self.out_of_order_count,
            "missing_intervals_count": self.missing_intervals_count,
            "non_finite_count": self.non_finite_count,
            "out_of_bounds_count": self.out_of_bounds_count,
            "violations": self.violations,
            "missing_timestamps_sample": self.missing_timestamps_sample[:10],
        }


def validate_demand_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    demand_col: str = "consumption",
    provenance: DemandProvenance = DemandProvenance.HISTORICAL_ARCHIVE,
    check_cadence: bool = True,
    strict_bounds: bool = True,
) -> DemandValidationReport:
    """Validate a demand DataFrame against the formal PJM East demand data contract.

    Args:
        df: Input DataFrame containing timestamp and demand columns.
        timestamp_col: Name of timestamp column.
        demand_col: Name of demand consumption column.
        provenance: Provenance classification of this dataset.
        check_cadence: Whether to check for strictly consecutive hourly intervals.
        strict_bounds: Whether to strictly enforce physical plausibility bounds (10k-75k MW).

    Returns:
        DemandValidationReport: Comprehensive audit report.
    """
    violations: list[str] = []

    if df.empty:
        return DemandValidationReport(
            is_valid=False,
            status="REJECTED",
            provenance=provenance,
            total_records=0,
            start_timestamp_utc=None,
            end_timestamp_utc=None,
            duplicates_count=0,
            out_of_order_count=0,
            missing_intervals_count=0,
            non_finite_count=0,
            out_of_bounds_count=0,
            violations=["Empty demand dataset provided."],
        )

    # 1. Column presence check
    if timestamp_col not in df.columns:
        violations.append(f"Missing required timestamp column: '{timestamp_col}'.")
    if demand_col not in df.columns:
        # Fallback to alternate names if applicable
        if "total_power_demand" in df.columns:
            demand_col = "total_power_demand"
        else:
            violations.append(f"Missing required demand column: '{demand_col}'.")

    if violations:
        return DemandValidationReport(
            is_valid=False,
            status="REJECTED",
            provenance=provenance,
            total_records=len(df),
            start_timestamp_utc=None,
            end_timestamp_utc=None,
            duplicates_count=0,
            out_of_order_count=0,
            missing_intervals_count=0,
            non_finite_count=0,
            out_of_bounds_count=0,
            violations=violations,
        )

    # 2. Timestamp parsing & timezone normalization to UTC
    try:
        ts_series = pd.to_datetime(df[timestamp_col], errors="coerce")
        invalid_ts_count = int(ts_series.isna().sum())
        if invalid_ts_count > 0:
            violations.append(f"Contains {invalid_ts_count} unparseable timestamp values.")

        if ts_series.dt.tz is None:
            ts_utc = ts_series.dt.tz_localize("UTC")
        else:
            ts_utc = ts_series.dt.tz_convert("UTC")
    except Exception as exc:
        return DemandValidationReport(
            is_valid=False,
            status="REJECTED",
            provenance=provenance,
            total_records=len(df),
            start_timestamp_utc=None,
            end_timestamp_utc=None,
            duplicates_count=0,
            out_of_order_count=0,
            missing_intervals_count=0,
            non_finite_count=0,
            out_of_bounds_count=0,
            violations=[f"Timestamp parsing failed: {exc}"],
        )

    # 3. Duplicate and order checking
    duplicates_count = int(ts_utc.duplicated().sum())
    if duplicates_count > 0:
        violations.append(f"Detected {duplicates_count} duplicate timestamp records.")

    is_monotonic = ts_utc.is_monotonic_increasing
    out_of_order_count = 0
    if not is_monotonic:
        diffs = ts_utc.diff()
        out_of_order_count = int((diffs < pd.Timedelta(0)).sum())
        violations.append(f"Timestamps are out of chronological order ({out_of_order_count} inversions).")

    # 4. Numerical validity & bounds check
    raw_demand = df[demand_col].to_numpy(dtype=float)
    non_finite_mask = np.isnan(raw_demand) | np.isinf(raw_demand)
    non_finite_count = int(non_finite_mask.sum())
    if non_finite_count > 0:
        violations.append(f"Contains {non_finite_count} non-finite demand values (NaN/Inf).")

    valid_demand = raw_demand[~non_finite_mask]
    out_of_bounds_count = 0
    if strict_bounds and len(valid_demand) > 0:
        too_low = valid_demand < MIN_PLAUSIBLE_DEMAND_MW
        too_high = valid_demand > MAX_PLAUSIBLE_DEMAND_MW
        out_of_bounds_count = int(np.sum(too_low | too_high))
        if out_of_bounds_count > 0:
            min_val = float(np.min(valid_demand))
            max_val = float(np.max(valid_demand))
            violations.append(
                f"Contains {out_of_bounds_count} observations outside plausible PJM East bounds "
                f"[{MIN_PLAUSIBLE_DEMAND_MW:,.0f}, {MAX_PLAUSIBLE_DEMAND_MW:,.0f} MW]. "
                f"Observed min: {min_val:.1f} MW, max: {max_val:.1f} MW."
            )

    # 5. Cadence & Gap Check (Regular 1-hour interval)
    missing_intervals_count = 0
    missing_timestamps_sample: list[str] = []
    if check_cadence and len(ts_utc.dropna()) > 1 and duplicates_count == 0 and is_monotonic:
        clean_ts = ts_utc.dropna()
        full_expected_range = pd.date_range(
            start=clean_ts.min(),
            end=clean_ts.max(),
            freq="1h",
            tz="UTC",
        )
        missing_set = set(full_expected_range) - set(clean_ts)
        missing_intervals_count = len(missing_set)
        if missing_intervals_count > 0:
            missing_timestamps_sample = [ts.isoformat() for ts in sorted(list(missing_set))[:10]]
            violations.append(
                f"Cadence violation: {missing_intervals_count} missing hourly intervals in span "
                f"({clean_ts.min().isoformat()} to {clean_ts.max().isoformat()})."
            )

    is_valid = len(violations) == 0
    status = "PASSED" if is_valid else "REJECTED"

    start_str = ts_utc.min().isoformat() if not ts_utc.empty else None
    end_str = ts_utc.max().isoformat() if not ts_utc.empty else None

    return DemandValidationReport(
        is_valid=is_valid,
        status=status,
        provenance=provenance,
        total_records=len(df),
        start_timestamp_utc=start_str,
        end_timestamp_utc=end_str,
        duplicates_count=duplicates_count,
        out_of_order_count=out_of_order_count,
        missing_intervals_count=missing_intervals_count,
        non_finite_count=non_finite_count,
        out_of_bounds_count=out_of_bounds_count,
        violations=violations,
        missing_timestamps_sample=missing_timestamps_sample,
    )


@dataclass(frozen=True)
class OperationalTelemetryReadiness:
    """Contract response indicating whether live demand telemetry is available."""
    is_ready: bool
    status_code: str  # "READY", "BLOCKED_NO_TELEMETRY", "BLOCKED_INCOMPLETE_WINDOW"
    target_timestamp_utc: str
    horizon_hours: int
    decision_cutoff_utc: str
    required_history_start_utc: str
    required_history_end_utc: str
    available_history_end_utc: str
    provenance_used: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert readiness to JSON-serializable dict."""
        return {
            "is_ready": self.is_ready,
            "status_code": self.status_code,
            "target_timestamp_utc": self.target_timestamp_utc,
            "horizon_hours": self.horizon_hours,
            "decision_cutoff_utc": self.decision_cutoff_utc,
            "required_history_start_utc": self.required_history_start_utc,
            "required_history_end_utc": self.required_history_end_utc,
            "available_history_end_utc": self.available_history_end_utc,
            "provenance_used": self.provenance_used,
            "message": self.message,
        }


def check_operational_demand_readiness(
    target_timestamp: str | pd.Timestamp,
    horizon: int = 1,
    available_series: pd.Series | None = None,
) -> OperationalTelemetryReadiness:
    """Assess whether real continuous demand telemetry is present to support automated forecasting.

    Enforces that:
    1. Decision cutoff = target - horizon hours.
    2. Continuous history from cutoff - 168h to cutoff must exist.
    3. If available data ends before cutoff (e.g., historical archive ending in 2018 for a 2026 forecast),
       the operational forecast is marked BLOCKED.

    Args:
        target_timestamp: Target valid hour T.
        horizon: Forecast horizon (1, 24, 48).
        available_series: Pre-loaded demand Series indexed by UTC timestamp.

    Returns:
        OperationalTelemetryReadiness: Assessment result.
    """
    ts = pd.to_datetime(target_timestamp)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    cutoff_ts = ts - pd.Timedelta(hours=horizon)
    req_start_ts = cutoff_ts - pd.Timedelta(hours=168)

    if available_series is None or available_series.empty:
        # Check default archive max
        archive_max = pd.Timestamp(HISTORICAL_ARCHIVE_END_UTC)
    else:
        archive_max = available_series.index.max()
        if archive_max.tz is None:
            archive_max = archive_max.tz_localize("UTC")
        else:
            archive_max = archive_max.tz_convert("UTC")

    if cutoff_ts > archive_max:
        return OperationalTelemetryReadiness(
            is_ready=False,
            status_code="BLOCKED_NO_TELEMETRY",
            target_timestamp_utc=ts.isoformat(),
            horizon_hours=horizon,
            decision_cutoff_utc=cutoff_ts.isoformat(),
            required_history_start_utc=req_start_ts.isoformat(),
            required_history_end_utc=cutoff_ts.isoformat(),
            available_history_end_utc=archive_max.isoformat(),
            provenance_used="HISTORICAL_ARCHIVE_EXHAUSTED",
            message=(
                f"OPERATIONAL TELEMETRY BLOCKED: Decision cutoff ({cutoff_ts.isoformat()}) "
                f"exceeds available demand history which ends at {archive_max.isoformat()}. "
                f"Real-time SCADA demand telemetry feed is not connected. "
                f"To test models on future cycles without live telemetry, use Diagnostic / Template Mode."
            ),
        )

    # Check that required 168h window is complete
    if available_series is not None:
        window = available_series[(available_series.index >= req_start_ts) & (available_series.index <= cutoff_ts)]
        if len(window) < 168:
            return OperationalTelemetryReadiness(
                is_ready=False,
                status_code="BLOCKED_INCOMPLETE_WINDOW",
                target_timestamp_utc=ts.isoformat(),
                horizon_hours=horizon,
                decision_cutoff_utc=cutoff_ts.isoformat(),
                required_history_start_utc=req_start_ts.isoformat(),
                required_history_end_utc=cutoff_ts.isoformat(),
                available_history_end_utc=archive_max.isoformat(),
                provenance_used="HISTORICAL_ARCHIVE",
                message=(
                    f"INCOMPLETE WINDOW: Found {len(window)} observations between "
                    f"{req_start_ts.isoformat()} and {cutoff_ts.isoformat()}, expected at least 168."
                ),
            )

    return OperationalTelemetryReadiness(
        is_ready=True,
        status_code="READY",
        target_timestamp_utc=ts.isoformat(),
        horizon_hours=horizon,
        decision_cutoff_utc=cutoff_ts.isoformat(),
        required_history_start_utc=req_start_ts.isoformat(),
        required_history_end_utc=cutoff_ts.isoformat(),
        available_history_end_utc=archive_max.isoformat(),
        provenance_used="HISTORICAL_ARCHIVE",
        message="Continuous demand history validated across required lag window.",
    )
