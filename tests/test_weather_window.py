"""
Unit tests for Phase 7 weather hourly selection window semantics, boundary behavior, and temperature parsing.
Ensures mathematical determinism and alignment protocol verification without modifying production code.
"""

import pytest
import pandas as pd
import numpy as np


def assign_nominal_hour(dt_series: pd.Series) -> pd.Series:
    """
    Explicit, continuous hourly window assignment:
    Every timestamp t belongs to exactly one nominal hour window W(T) = [T - 30m, T + 30m).
    Mathematically: T = floor(t + 30 minutes).
    """
    return (dt_series + pd.Timedelta(minutes=30)).dt.floor("1h")


def select_nearest_observation(df: pd.DataFrame, target_hour: pd.Timestamp) -> pd.Series:
    """
    Selects the single eligible observation closest to target_hour within W(T) = [T - 30m, T + 30m).
    Tie-breaking:
      1. Smallest absolute time offset |t - T|
      2. Earlier timestamp (t < T)
      3. Highest quality code ('1' > '5')
    """
    window_start = target_hour - pd.Timedelta(minutes=30)
    window_end = target_hour + pd.Timedelta(minutes=30)
    
    in_win = df[(df["datetime_utc"] >= window_start) & (df["datetime_utc"] < window_end)].copy()
    if in_win.empty:
        return pd.Series(dtype=object)
        
    in_win["abs_offset"] = (in_win["datetime_utc"] - target_hour).abs()
    
    # Priority rank: '1' is highest, '5' second
    qc_rank = {"1": 1, "5": 2}
    in_win["qc_score"] = in_win["qc"].map(qc_rank).fillna(99)
    
    # Sort deterministically
    sorted_df = in_win.sort_values(
        by=["abs_offset", "datetime_utc", "qc_score"],
        ascending=[True, True, True]
    )
    return sorted_df.iloc[0]


def parse_noaa_tmp(tmp_str: str) -> tuple[float, str, bool]:
    """
    Parses raw NOAA ISD TMP composite string '[+-]DDDD,Q'.
    Returns (temp_f, qc_code, is_valid).
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


# ==============================================================================
# UNIT TESTS
# ==============================================================================

def test_hourly_window_exact_boundaries():
    """Verify that W(T) = [T - 30m, T + 30m) partitions time without overlap or gaps."""
    # Target hour: 2002-01-01 14:00:00 UTC
    t_target = pd.Timestamp("2002-01-01 14:00:00", tz="UTC")
    
    timestamps = pd.Series([
        pd.Timestamp("2002-01-01 13:29:59", tz="UTC"),  # Just before lower boundary -> 13:00
        pd.Timestamp("2002-01-01 13:30:00", tz="UTC"),  # Exact lower boundary -> 14:00
        pd.Timestamp("2002-01-01 13:30:01", tz="UTC"),  # Just after lower boundary -> 14:00
        pd.Timestamp("2002-01-01 13:54:00", tz="UTC"),  # Standard METAR minute 54 -> 14:00
        pd.Timestamp("2002-01-01 14:00:00", tz="UTC"),  # Exact top of hour -> 14:00
        pd.Timestamp("2002-01-01 14:29:59", tz="UTC"),  # Just before upper boundary -> 14:00
        pd.Timestamp("2002-01-01 14:30:00", tz="UTC"),  # Exact upper boundary -> 15:00
        pd.Timestamp("2002-01-01 14:30:01", tz="UTC"),  # Just after upper boundary -> 15:00
    ])
    
    nominal_hours = assign_nominal_hour(timestamps)
    
    assert nominal_hours.iloc[0] == pd.Timestamp("2002-01-01 13:00:00", tz="UTC")
    assert nominal_hours.iloc[1] == t_target
    assert nominal_hours.iloc[2] == t_target
    assert nominal_hours.iloc[3] == t_target
    assert nominal_hours.iloc[4] == t_target
    assert nominal_hours.iloc[5] == t_target
    assert nominal_hours.iloc[6] == pd.Timestamp("2002-01-01 15:00:00", tz="UTC")
    assert nominal_hours.iloc[7] == pd.Timestamp("2002-01-01 15:00:00", tz="UTC")


def test_select_nearest_observation_in_window():
    """Verify that closest observation is selected when multiple reports exist in window."""
    t_target = pd.Timestamp("2002-01-01 14:00:00", tz="UTC")
    
    # Candidate reports:
    # 1. 13:30:00 (offset 30 min)
    # 2. 13:54:00 (offset 6 min)
    # 3. 14:20:00 (offset 20 min)
    df = pd.DataFrame([
        {"datetime_utc": pd.Timestamp("2002-01-01 13:30:00", tz="UTC"), "temp": 20.0, "qc": "1"},
        {"datetime_utc": pd.Timestamp("2002-01-01 13:54:00", tz="UTC"), "temp": 22.0, "qc": "5"},
        {"datetime_utc": pd.Timestamp("2002-01-01 14:20:00", tz="UTC"), "temp": 24.0, "qc": "1"},
    ])
    
    selected = select_nearest_observation(df, t_target)
    assert selected["datetime_utc"] == pd.Timestamp("2002-01-01 13:54:00", tz="UTC")
    assert selected["temp"] == 22.0


def test_select_nearest_observation_tie_breaking():
    """Verify deterministic tie-breaking when two observations have identical distance to T."""
    t_target = pd.Timestamp("2002-01-01 14:00:00", tz="UTC")
    
    # Two observations equidistant: 13:45:00 (15 min prior) and 14:15:00 (15 min after)
    df = pd.DataFrame([
        {"datetime_utc": pd.Timestamp("2002-01-01 14:15:00", tz="UTC"), "temp": 30.0, "qc": "1"},
        {"datetime_utc": pd.Timestamp("2002-01-01 13:45:00", tz="UTC"), "temp": 28.0, "qc": "1"},
    ])
    
    selected = select_nearest_observation(df, t_target)
    # Tie-breaker prefers earlier observation (13:45:00) for forecast causality
    assert selected["datetime_utc"] == pd.Timestamp("2002-01-01 13:45:00", tz="UTC")
    assert selected["temp"] == 28.0


def test_temperature_parsing_and_quality_masking():
    """Verify parsing of numeric temperature, unit conversion, and quality masking."""
    # 1. Valid positive: +0150,1 -> 15.0 C -> 59.0 F
    temp_f, qc, valid = parse_noaa_tmp("+0150,1")
    assert valid is True
    assert qc == "1"
    assert pytest.approx(temp_f, 0.01) == 59.0
    
    # 2. Valid negative: -0060,5 -> -6.0 C -> 21.2 F
    temp_f, qc, valid = parse_noaa_tmp("-0060,5")
    assert valid is True
    assert qc == "5"
    assert pytest.approx(temp_f, 0.01) == 21.2
    
    # 3. Sentinel missing: +9999,9 -> NaN
    temp_f, qc, valid = parse_noaa_tmp("+9999,9")
    assert valid is False
    assert qc == "9"
    assert np.isnan(temp_f)
    
    # 4. Erroneous QC flag: +0150,6 -> NaN
    temp_f, qc, valid = parse_noaa_tmp("+0150,6")
    assert valid is False
    assert qc == "6"
    assert np.isnan(temp_f)
    
    # 5. Suspect QC flag: +0150,2 -> NaN
    temp_f, qc, valid = parse_noaa_tmp("+0150,2")
    assert valid is False
    assert qc == "2"
    assert np.isnan(temp_f)
    
    # 6. Malformed input
    temp_f, qc, valid = parse_noaa_tmp("MALFORMED")
    assert valid is False
    assert np.isnan(temp_f)
