"""Data validation module for AI-Powered Energy Consumption Forecasting.

Generates comprehensive data quality reports detailing missing values,
ordering integrity, duplicates, temporal gaps, and physical range checks.
"""

from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Immutable validation summary for an energy consumption dataset."""

    total_rows: int
    missing_timestamps: int
    invalid_timestamps: int
    missing_consumption: int
    non_numeric_consumption: int
    duplicate_timestamps: int
    is_monotonic_increasing: bool
    expected_frequency: str | None
    has_gaps: bool | None
    detected_gaps_count: int
    negative_consumption_count: int
    has_negative_consumption: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a dictionary representation."""
        return asdict(self)


def validate_energy_data(
    df: pd.DataFrame,
    expected_frequency: str | None = None,
    allow_negative_consumption: bool = True,
) -> ValidationReport:
    """Validate a loaded energy DataFrame and return a structured quality report.

    This function does NOT modify, interpolate, or remove any records.

    Note on invalid vs missing values:
        If the DataFrame columns have already been coerced to datetime64 or float64
        (e.g., by load_energy_data), unparseable source values appear as NaT/NaN
        and are counted under missing_timestamps / missing_consumption. If string
        object columns are supplied directly, unparseable strings are detected
        and reported under invalid_timestamps / non_numeric_consumption.

    Args:
        df: DataFrame containing at least 'timestamp' and 'consumption' columns.
        expected_frequency: Optional pandas frequency string (e.g., 'h', 'D', '15min').
            If None, gap detection is omitted.
        allow_negative_consumption: If True, negative consumption values generate
            warnings. If False, negative values generate errors and invalidate the dataset.

    Returns:
        ValidationReport: Immutable dataclass summarizing dataset health.

    Raises:
        ValueError: If 'timestamp' or 'consumption' columns are missing from df.
    """
    if "timestamp" not in df.columns or "consumption" not in df.columns:
        raise ValueError(
            f"DataFrame must contain 'timestamp' and 'consumption' columns. Found: {list(df.columns)}"
        )

    warnings: list[str] = []
    errors: list[str] = []
    total_rows = len(df)

    # 1. Timestamp validation
    ts_series = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts_series):
        missing_ts = int(ts_series.isna().sum())
        invalid_ts = 0  # Pre-coerced to NaT prior to validator
    else:
        # String or object column: check missing and unparseable values
        missing_ts = int(ts_series.isna().sum())
        parsed_ts = pd.to_datetime(ts_series, errors="coerce", format="mixed")
        invalid_ts = int((parsed_ts.isna() & ts_series.notna()).sum())
        ts_series = parsed_ts

    if missing_ts > 0:
        errors.append(f"Found {missing_ts} missing timestamp(s).")
    if invalid_ts > 0:
        errors.append(f"Found {invalid_ts} unparseable timestamp string(s).")

    # 2. Consumption validation
    cons_series = df["consumption"]
    if pd.api.types.is_numeric_dtype(cons_series):
        missing_cons = int(cons_series.isna().sum())
        non_numeric_cons = 0  # Pre-coerced to NaN prior to validator
    else:
        missing_cons = int(cons_series.isna().sum())
        parsed_cons = pd.to_numeric(cons_series, errors="coerce")
        non_numeric_cons = int((parsed_cons.isna() & cons_series.notna()).sum())
        cons_series = parsed_cons

    if missing_cons > 0:
        warnings.append(f"Found {missing_cons} missing consumption value(s).")
    if non_numeric_cons > 0:
        errors.append(f"Found {non_numeric_cons} non-numeric consumption value(s).")

    # 3. Duplicate timestamps check
    dup_ts = int(ts_series.duplicated().sum())
    if dup_ts > 0:
        warnings.append(f"Found {dup_ts} duplicate timestamp(s).")

    # 4. Monotonic ordering check
    valid_ts = ts_series.dropna()
    is_monotonic = bool(valid_ts.is_monotonic_increasing)
    if not is_monotonic:
        warnings.append("Timestamps are not strictly monotonically increasing.")

    # 5. Gap detection under expected frequency
    has_gaps: bool | None = None
    detected_gaps = 0
    if expected_frequency is not None:
        if len(valid_ts) >= 2:
            min_ts = valid_ts.min()
            max_ts = valid_ts.max()
            expected_range = pd.date_range(start=min_ts, end=max_ts, freq=expected_frequency)
            actual_unique_ts = set(valid_ts)
            missing_steps = len(set(expected_range) - actual_unique_ts)
            has_gaps = missing_steps > 0
            detected_gaps = missing_steps
            if has_gaps:
                warnings.append(
                    f"Detected {detected_gaps} missing interval step(s) under expected frequency '{expected_frequency}'."
                )
        else:
            has_gaps = False
            detected_gaps = 0

    # 6. Physical bounds check (Negative consumption)
    valid_cons = cons_series.dropna()
    negative_cons_count = int((valid_cons < 0).sum())
    has_neg_cons = negative_cons_count > 0

    if has_neg_cons:
        msg = f"Found {negative_cons_count} negative consumption observation(s)."
        if allow_negative_consumption:
            warnings.append(msg)
        else:
            errors.append(msg)

    # Global validity status
    is_valid = len(errors) == 0

    return ValidationReport(
        total_rows=total_rows,
        missing_timestamps=missing_ts,
        invalid_timestamps=invalid_ts,
        missing_consumption=missing_cons,
        non_numeric_consumption=non_numeric_cons,
        duplicate_timestamps=dup_ts,
        is_monotonic_increasing=is_monotonic,
        expected_frequency=expected_frequency,
        has_gaps=has_gaps,
        detected_gaps_count=detected_gaps,
        negative_consumption_count=negative_cons_count,
        has_negative_consumption=has_neg_cons,
        warnings=tuple(warnings),
        errors=tuple(errors),
        is_valid=is_valid,
    )
