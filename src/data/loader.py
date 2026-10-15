"""Data loading module for AI-Powered Energy Consumption Forecasting.

Provides robust CSV loading for energy time-series with automatic column
alias resolution, type conversions, and explicit error validation.
"""

from pathlib import Path
from typing import Sequence
import pandas as pd

from src.config import DEFAULT_DATA_PATH

# Standard supported aliases (evaluated case-insensitively and whitespace-stripped)
TIMESTAMP_ALIASES: tuple[str, ...] = ("timestamp", "datetime", "date", "time")
CONSUMPTION_ALIASES: tuple[str, ...] = ("consumption", "consumption_mw", "pjme_mw", "power")


def _find_column_match(
    columns: Sequence[str],
    explicit_name: str | None,
    aliases: Sequence[str],
    column_role: str,
) -> str:
    """Resolve the column name from explicit input or supported aliases.

    Raises:
        ValueError: If explicit column is not found, no aliases match, or multiple matches are ambiguous.
    """
    col_map = {str(col).strip().lower(): str(col) for col in columns}

    if explicit_name is not None:
        key = str(explicit_name).strip().lower()
        if key in col_map:
            return col_map[key]
        raise ValueError(
            f"Specified {column_role} column '{explicit_name}' was not found in CSV. "
            f"Available columns: {list(columns)}"
        )

    # Search against aliases
    matched = [col_map[alias] for alias in aliases if alias in col_map]
    if not matched:
        raise ValueError(
            f"Could not identify a {column_role} column in CSV. "
            f"Available columns: {list(columns)}. Supported {column_role} aliases: {list(aliases)}"
        )
    if len(matched) > 1:
        raise ValueError(
            f"Ambiguous {column_role} columns detected in CSV: {matched}. "
            f"Please specify '{column_role}_column' explicitly."
        )

    return matched[0]


def load_energy_data(
    file_path: Path | str | None = None,
    timestamp_column: str | None = None,
    consumption_column: str | None = None,
) -> pd.DataFrame:
    """Load an energy time-series CSV into a pandas DataFrame.

    Normalizes the time and consumption columns to 'timestamp' and 'consumption'.
    Converts timestamps to datetime objects and consumption values to numeric floats.
    Invalid rows are retained (as NaT or NaN) and not silently dropped.

    Args:
        file_path: Path to the energy CSV file. Defaults to DEFAULT_DATA_PATH from src.config.
        timestamp_column: Explicit name of the timestamp column in the CSV.
        consumption_column: Explicit name of the energy consumption column in the CSV.

    Returns:
        pd.DataFrame: Loaded DataFrame with normalized 'timestamp' and 'consumption' columns.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty, required columns are missing, or column resolution is ambiguous.
    """
    resolved_path = Path(file_path) if file_path is not None else DEFAULT_DATA_PATH

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Energy dataset file not found at: {resolved_path}")

    try:
        df = pd.read_csv(resolved_path)
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"Energy dataset at '{resolved_path}' is empty or contains no readable data.") from err

    if df.empty:
        raise ValueError(f"Energy dataset at '{resolved_path}' is empty.")

    actual_ts_col = _find_column_match(
        columns=df.columns,
        explicit_name=timestamp_column,
        aliases=TIMESTAMP_ALIASES,
        column_role="timestamp",
    )

    actual_cons_col = _find_column_match(
        columns=df.columns,
        explicit_name=consumption_column,
        aliases=CONSUMPTION_ALIASES,
        column_role="consumption",
    )

    # Prepare DataFrame and rename target columns
    rename_mapping = {actual_ts_col: "timestamp", actual_cons_col: "consumption"}
    df = df.rename(columns=rename_mapping)

    # Convert timestamps and consumption values explicitly
    # Invalid values are coerced to NaT / NaN rather than silently dropped
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")

    return df
