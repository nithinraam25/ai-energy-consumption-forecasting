"""Data preprocessing module for AI-Powered Energy Consumption Forecasting.

Provides deterministic cleaning, duplicate resolution policies, chronological sorting,
and sanitized dataset persistence without data leakage or silent aggregations.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import pandas as pd

from src.config import PROCESSED_DATA_DIR, DEFAULT_DATA_PATH, RAW_DATA_DIR

VALID_DUPLICATE_POLICIES = ("keep_first", "keep_last", "mean", "error")


@dataclass(frozen=True)
class CleaningReport:
    """Immutable audit report detailing records removed and policies applied."""

    initial_rows: int
    final_rows: int
    removed_missing_timestamps: int
    removed_missing_consumption: int
    removed_duplicates: int
    removed_negative_consumption: int
    duplicate_policy: str
    filter_negative_consumption: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a dictionary representation."""
        return asdict(self)


def preprocess_energy_data(
    df: pd.DataFrame,
    duplicate_policy: str = "keep_first",
    drop_missing_timestamps: bool = True,
    drop_missing_consumption: bool = True,
    filter_negative_consumption: bool = False,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean and sanitize an energy time-series DataFrame without mutating input.

    Strictly avoids silent zero-fills, implicit interpolations, or scaler transformations.

    Args:
        df: Input DataFrame containing at least 'timestamp' and 'consumption' columns.
        duplicate_policy: Policy for handling duplicate timestamps:
            'keep_first': Keep first chronological occurrence and drop subsequent.
            'keep_last': Keep last chronological occurrence and drop earlier.
            'error': Raise ValueError if any duplicate timestamp exists.
        drop_missing_timestamps: Whether to drop rows where timestamp is null or NaT.
        drop_missing_consumption: Whether to drop rows where consumption is null or NaN.
        filter_negative_consumption: If True, remove negative consumption values.
            Defaults to False to prevent silent assumption-making.

    Returns:
        tuple[pd.DataFrame, CleaningReport]: Cleaned DataFrame and execution report.

    Raises:
        ValueError: If required columns are missing, or duplicate_policy is violated.
    """
    if "timestamp" not in df.columns or "consumption" not in df.columns:
        raise ValueError(
            f"Input DataFrame must contain 'timestamp' and 'consumption' columns. Found: {list(df.columns)}"
        )

    if duplicate_policy not in VALID_DUPLICATE_POLICIES:
        raise ValueError(
            f"Invalid duplicate_policy '{duplicate_policy}'. Supported: {list(VALID_DUPLICATE_POLICIES)}"
        )

    # Make an explicit deep copy to guarantee caller's DataFrame is never mutated
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # Explicit type casting
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce", format="mixed")
    df_clean["consumption"] = pd.to_numeric(df_clean["consumption"], errors="coerce")

    # 1. Handle missing timestamps
    removed_missing_ts = 0
    if drop_missing_timestamps:
        mask_missing_ts = df_clean["timestamp"].isna()
        removed_missing_ts = int(mask_missing_ts.sum())
        df_clean = df_clean[~mask_missing_ts]

    # 2. Handle missing consumption
    removed_missing_cons = 0
    if drop_missing_consumption:
        mask_missing_cons = df_clean["consumption"].isna()
        removed_missing_cons = int(mask_missing_cons.sum())
        df_clean = df_clean[~mask_missing_cons]

    # 3. Handle negative consumption if explicitly requested
    removed_neg_cons = 0
    if filter_negative_consumption:
        mask_neg = df_clean["consumption"] < 0
        removed_neg_cons = int(mask_neg.sum())
        df_clean = df_clean[~mask_neg]

    # 4. Chronological ordering (stable sort preserving relative order of duplicate timestamps)
    df_clean = df_clean.sort_values(by="timestamp", kind="stable").reset_index(drop=True)

    # 5. Handle duplicates on timestamp
    num_duplicates = int(df_clean["timestamp"].duplicated().sum())
    removed_dups = 0
    if num_duplicates > 0:
        if duplicate_policy == "error":
            raise ValueError(
                f"Duplicate timestamps detected: {num_duplicates} duplicate record(s) "
                f"found under duplicate_policy='error'."
            )
        elif duplicate_policy == "keep_first":
            removed_dups = num_duplicates
            df_clean = df_clean.drop_duplicates(subset=["timestamp"], keep="first")
        elif duplicate_policy == "keep_last":
            removed_dups = num_duplicates
            df_clean = df_clean.drop_duplicates(subset=["timestamp"], keep="last")
        elif duplicate_policy == "mean":
            removed_dups = num_duplicates
            agg_dict = {
                col: "mean" if col == "consumption" else "first"
                for col in df_clean.columns
                if col != "timestamp"
            }
            df_clean = df_clean.groupby("timestamp", as_index=False, sort=True).agg(agg_dict)

    df_clean = df_clean.reset_index(drop=True)
    final_rows = len(df_clean)

    report = CleaningReport(
        initial_rows=initial_rows,
        final_rows=final_rows,
        removed_missing_timestamps=removed_missing_ts,
        removed_missing_consumption=removed_missing_cons,
        removed_duplicates=removed_dups,
        removed_negative_consumption=removed_neg_cons,
        duplicate_policy=duplicate_policy,
        filter_negative_consumption=filter_negative_consumption,
    )

    return df_clean, report


def save_cleaned_data(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    index: bool = False,
) -> Path:
    """Save cleaned energy dataset to CSV, preventing accidental raw data overwrites.

    Args:
        df: Cleaned energy DataFrame to persist.
        output_path: Destination path. Defaults to PROCESSED_DATA_DIR / 'energy_cleaned.csv'.
        index: Whether to write row index. Defaults to False.

    Returns:
        Path: Resolved absolute path where file was saved.

    Raises:
        ValueError: If output_path points directly to DEFAULT_DATA_PATH or raw data directory.
    """
    if output_path is None:
        target_path = PROCESSED_DATA_DIR / "energy_cleaned.csv"
    else:
        target_path = Path(output_path).resolve()

    # Safety guard: Prevent overwriting raw data
    resolved_default_raw = DEFAULT_DATA_PATH.resolve()
    resolved_raw_dir = RAW_DATA_DIR.resolve()

    if target_path == resolved_default_raw or (
        target_path.parent == resolved_raw_dir and target_path.name == resolved_default_raw.name
    ):
        raise ValueError(
            f"Safety guard triggered: Refusing to overwrite raw dataset file at '{target_path}'."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=index)
    return target_path
