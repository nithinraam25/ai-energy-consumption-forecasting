"""Data loading and extraction module for NOAA ISD KPHL hourly weather observations.

Provides robust CSV loading, composite TMP parsing, continuous half-open windowing,
civil timezone localization, DST transition handling, and deterministic observation selection.
"""

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR

DEFAULT_WEATHER_DIR: Path = RAW_DATA_DIR / "weather"
KPHL_STATION_ID: str = "72408013739"


def parse_noaa_tmp(tmp_str: str) -> tuple[float, str, bool]:
    """Parse raw NOAA ISD TMP composite string '[+-]DDDD,Q'.

    Format definition (NOAA NCEI ISD Format Version 8):
        - DDDD: Air temperature in tenths of degrees Celsius.
        - Q: Quality code ('1' = Passed all checks, '5' = Passed checks with climatological limits).
        - '+9999': Sentinel for missing data.

    Returns:
        tuple[float, str, bool]: (temp_f, quality_code, is_valid).
    """
    if not isinstance(tmp_str, str) or "," not in tmp_str:
        return np.nan, "INVALID", False

    parts = tmp_str.split(",")
    if len(parts) != 2:
        return np.nan, "INVALID", False

    raw_val, qc = parts[0].strip(), parts[1].strip()
    if raw_val == "+9999" or qc not in {"1", "5"}:
        return np.nan, qc, False

    try:
        temp_c = int(raw_val) / 10.0
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        return temp_f, qc, True
    except (ValueError, TypeError):
        return np.nan, qc, False


def assign_nominal_utc_hour(dt_series: pd.Series) -> pd.Series:
    """Assign observations to continuous nominal UTC hourly windows.

    Continuous half-open interval definition:
        W(T) = [T - 30 minutes, T + 30 minutes).
        Mathematically: T = floor(t + 30 minutes).

    Args:
        dt_series: Datetime series with UTC timezone awareness.

    Returns:
        pd.Series: Series of floored nominal UTC hour timestamps.
    """
    return (dt_series + pd.Timedelta(minutes=30)).dt.floor("1h")


def load_raw_noaa_kphl(
    weather_dir: Path | str | None = None,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Load raw annual NOAA ISD CSV files for Philadelphia International Airport (KPHL).

    Args:
        weather_dir: Directory containing annual NOAA ISD CSVs. Defaults to data/raw/weather.
        years: Optional sequence of years to load. Defaults to 2002-2018 inclusive.

    Returns:
        pd.DataFrame: Combined raw weather dataframe.

    Raises:
        FileNotFoundError: If the directory or any expected annual CSV is missing.
        ValueError: If required columns ('DATE', 'REPORT_TYPE', 'TMP') are absent.
    """
    w_dir = Path(weather_dir) if weather_dir is not None else DEFAULT_WEATHER_DIR
    if not w_dir.is_dir():
        raise FileNotFoundError(f"Weather data directory not found: {w_dir}")

    target_years = list(years) if years is not None else list(range(2002, 2019))
    dfs = []
    required_cols = {"DATE", "REPORT_TYPE", "TMP"}

    for year in target_years:
        csv_file = w_dir / f"noaa_isd_kphl_{KPHL_STATION_ID}_{year}.csv"
        if not csv_file.is_file():
            raise FileNotFoundError(f"Required NOAA weather file missing: {csv_file}")

        df_year = pd.read_csv(
            csv_file,
            usecols=lambda c: c in required_cols or c in {"STATION", "QUALITY_CONTROL"},
            low_memory=False,
        )
        missing_cols = required_cols - set(df_year.columns)
        if missing_cols:
            raise ValueError(f"File {csv_file.name} is missing required columns: {missing_cols}")

        dfs.append(df_year)

    if not dfs:
        raise ValueError("No weather files were loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def select_hourly_kphl_observations(
    raw_df: pd.DataFrame,
    station_id: str = KPHL_STATION_ID,
) -> pd.DataFrame:
    """Filter routine METAR reports and select deterministic nearest hourly observations.

    Selection Protocol:
        1. Filter to routine FM-15 reports.
        2. Parse observation timestamps as UTC.
        3. Parse and validate composite TMP (scaling by 10 to °C, converting to °F).
        4. Group by continuous window W(T) = [T - 30m, T + 30m).
        5. For multiple reports in W(T), select nearest observation:
           - Smallest absolute offset |t - T|
           - Earlier timestamp (t < T) to preserve forecast causality
           - Higher quality code ('1' > '5')

    Args:
        raw_df: Raw NOAA dataframe containing 'DATE', 'REPORT_TYPE', 'TMP'.
        station_id: Expected station identifier.

    Returns:
        pd.DataFrame: Hourly dataframe with columns ['nominal_utc', 'dt_utc', 'temp_f', 'qc'].
    """
    df = raw_df[raw_df["REPORT_TYPE"] == "FM-15"].copy()
    if df.empty:
        return pd.DataFrame(columns=["nominal_utc", "dt_utc", "temp_f", "qc"])

    df["dt_utc"] = pd.to_datetime(df["DATE"], utc=True)

    parsed = [parse_noaa_tmp(val) for val in df["TMP"]]
    df["temp_f"] = [p[0] for p in parsed]
    df["qc"] = [p[1] for p in parsed]
    df["is_valid"] = [p[2] for p in parsed]

    # Retain valid observations only
    valid_df = df[df["is_valid"]].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["nominal_utc", "dt_utc", "temp_f", "qc"])

    # Continuous nominal UTC hour assignment
    valid_df["nominal_utc"] = assign_nominal_utc_hour(valid_df["dt_utc"])
    valid_df["offset"] = (valid_df["dt_utc"] - valid_df["nominal_utc"]).abs()

    # Deterministic ranking: offset (asc), dt_utc (asc), qc rank ('1' -> 1, '5' -> 2)
    qc_rank = {"1": 1, "5": 2}
    valid_df["qc_score"] = valid_df["qc"].map(qc_rank).fillna(99)

    sorted_df = valid_df.sort_values(
        by=["nominal_utc", "offset", "dt_utc", "qc_score"],
        ascending=[True, True, True, True],
    )

    selected = sorted_df.drop_duplicates(subset=["nominal_utc"], keep="first").copy()
    return selected[["nominal_utc", "dt_utc", "temp_f", "qc"]].reset_index(drop=True)


def build_hourly_kphl_weather_series(
    raw_df: pd.DataFrame,
    autumn_duplicate_policy: str = "mean",
    return_intermediate_utc: bool = False,
    start_year: int = 2002,
    end_year: int = 2018,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Construct complete hourly weather series aligned to legacy PJME naive timestamp grid.

    Protocol:
        1. Select deterministic hourly observations across continuous nominal UTC sequence.
        2. Create complete continuous UTC hourly grid from start_year to end_year.
        3. Project nominal UTC hours to Eastern Civil Time ('America/New_York').
        4. Spring Transition: Local 02:00 naturally omitted (matches Kaggle demand index).
        5. Autumn Transition: Local 01:00 occurs twice (05:00 UTC EDT, 06:00 UTC EST).
           - Both physical hours preserved in intermediate UTC table.
           - For schema compatibility with preprocess_energy_data(duplicate_policy='mean'),
             naive duplicate local hours are collapsed according to autumn_duplicate_policy:
             * 'mean': Synthetic arithmetic mean (temp_EDT + temp_EST) / 2.
             * 'edt': EDT observation only (05:00 UTC).
             * 'est': EST observation only (06:00 UTC).
        6. Strips timezone tag to yield naive local 'timestamp' column for join compatibility.

    Args:
        raw_df: Raw NOAA dataframe.
        autumn_duplicate_policy: Policy for resolving repeated autumn hour ('mean', 'edt', 'est').
        return_intermediate_utc: If True, also returns intermediate un-collapsed UTC table.
        start_year: Start year of full grid. Defaults to 2002.
        end_year: End year of full grid. Defaults to 2018.

    Returns:
        pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]:
            Cleaned naive weather DataFrame with columns ['timestamp', 'temp_f', 'nominal_utc'],
            and optionally intermediate UTC DataFrame.

    Raises:
        ValueError: If autumn_duplicate_policy is unrecognized.
    """
    valid_policies = {"mean", "edt", "est"}
    if autumn_duplicate_policy not in valid_policies:
        raise ValueError(
            f"Invalid autumn_duplicate_policy '{autumn_duplicate_policy}'. "
            f"Must be one of: {sorted(valid_policies)}"
        )

    # 1. Deterministic observation selection
    hourly_obs = select_hourly_kphl_observations(raw_df)

    # 2. Continuous nominal UTC grid
    start_utc = pd.Timestamp(f"{start_year}-01-01 00:00:00", tz="UTC")
    end_utc = pd.Timestamp(f"{end_year}-12-31 23:00:00", tz="UTC")
    full_utc_index = pd.date_range(start_utc, end_utc, freq="1h", tz="UTC")

    grid_df = pd.DataFrame({"nominal_utc": full_utc_index})
    grid_df = grid_df.merge(hourly_obs, on="nominal_utc", how="left")

    # 3. Project to Eastern Civil Time
    grid_df["local_dt"] = grid_df["nominal_utc"].dt.tz_convert("America/New_York")
    grid_df["is_dst"] = grid_df["local_dt"].apply(lambda x: bool(x.dst()))
    grid_df["naive_ts"] = grid_df["local_dt"].dt.tz_localize(None)

    intermediate_df = grid_df.copy()

    # 4. Handle Autumn Duplicate Naive Timestamps
    dup_mask = grid_df["naive_ts"].duplicated(keep=False)

    if not dup_mask.any():
        collapsed_df = grid_df.rename(columns={"naive_ts": "timestamp"})[
            ["timestamp", "temp_f", "nominal_utc"]
        ]
    else:
        non_dup = grid_df[~dup_mask].copy()
        dup_rows = grid_df[dup_mask].copy()

        if autumn_duplicate_policy == "mean":
            collapsed_dup = dup_rows.groupby("naive_ts", as_index=False).agg(
                {"temp_f": "mean", "nominal_utc": "first"}
            )
        elif autumn_duplicate_policy == "edt":
            # EDT corresponds to is_dst == True (05:00 UTC)
            collapsed_dup = dup_rows[dup_rows["is_dst"]].copy()
            collapsed_dup = collapsed_dup.drop_duplicates(subset=["naive_ts"], keep="first")
            collapsed_dup = collapsed_dup[["naive_ts", "temp_f", "nominal_utc"]]
        elif autumn_duplicate_policy == "est":
            # EST corresponds to is_dst == False (06:00 UTC)
            collapsed_dup = dup_rows[~dup_rows["is_dst"]].copy()
            collapsed_dup = collapsed_dup.drop_duplicates(subset=["naive_ts"], keep="first")
            collapsed_dup = collapsed_dup[["naive_ts", "temp_f", "nominal_utc"]]

        collapsed_df = pd.concat(
            [non_dup[["naive_ts", "temp_f", "nominal_utc"]], collapsed_dup],
            ignore_index=True,
        )
        collapsed_df = collapsed_df.sort_values(by="naive_ts").reset_index(drop=True)
        collapsed_df = collapsed_df.rename(columns={"naive_ts": "timestamp"})

    if return_intermediate_utc:
        return collapsed_df, intermediate_df
    return collapsed_df
