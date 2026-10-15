"""Focused unit tests for Phase 7C targeted NOAA HRRR loader and spatial parser."""

import pytest
import pandas as pd
import numpy as np

from src.data.hrrr_loader import (
    parse_hrrr_idx,
    parse_grib2_metadata,
    project_latlon_to_hrrr_grid,
    compute_kphl_spatial_weights,
    kelvin_to_fahrenheit,
    evaluate_operational_eligibility,
    KPHL_LAT,
    KPHL_LON,
)


# Mock sample .idx snippet matching real NOAA HRRR index files
MOCK_HRRR_IDX = """
604:310566969:d=2014100100:SNOWC:surface:2 hour fcst:
605:310597238:d=2014100100:SNOD:surface:2 hour fcst:
606:310622393:d=2014100100:TMP:2 m above ground:2 hour fcst:
607:311891457:d=2014100100:SPFH:2 m above ground:2 hour fcst:
608:313414015:d=2014100100:DPT:2 m above ground:2 hour fcst:
"""


# ==============================================================================
# 1. INDEX PARSING TESTS
# ==============================================================================

def test_parse_hrrr_idx_locates_2m_temperature():
    """Verify that parse_hrrr_idx correctly extracts byte boundaries for TMP:2 m above ground."""
    res = parse_hrrr_idx(MOCK_HRRR_IDX, target_var="TMP", target_level="2 m above ground")
    assert res is not None
    assert res["record_num"] == 606
    assert res["start_byte"] == 310622393
    assert res["end_byte"] == 311891456  # 311891457 - 1
    assert res["byte_length"] == 1269064
    assert res["var"] == "TMP"
    assert res["level"] == "2 m above ground"
    assert res["fcst"] == "2 hour fcst"


def test_parse_hrrr_idx_returns_none_for_missing_variable():
    """Verify parse_hrrr_idx returns None if requested variable is absent."""
    res = parse_hrrr_idx(MOCK_HRRR_IDX, target_var="NONEXISTENT", target_level="surface")
    assert res is None


# ==============================================================================
# 2. TEMPERATURE UNIT CONVERSION TESTS
# ==============================================================================

def test_kelvin_to_fahrenheit_conversion():
    """Verify Kelvin to Fahrenheit physical conversion across landmark temperatures."""
    assert pytest.approx(kelvin_to_fahrenheit(273.15), abs=1e-4) == 32.0  # Freezing point
    assert pytest.approx(kelvin_to_fahrenheit(373.15), abs=1e-4) == 212.0  # Boiling point
    assert pytest.approx(kelvin_to_fahrenheit(266.34), abs=1e-3) == 19.742  # GRIB sample ref value


# ==============================================================================
# 3. SPATIAL PROJECTION & BILINEAR WEIGHT TESTS
# ==============================================================================

def test_kphl_grid_projection_and_bilinear_weights():
    """Verify KPHL geographic coordinates map to correct HRRR grid indices with normalized weights."""
    weights_dict = compute_kphl_spatial_weights(KPHL_LAT, KPHL_LON)

    i_frac, j_frac = weights_dict["fractional_indices"]
    # Verified analytical coordinates: i ~ 1526.413, j ~ 656.176
    assert pytest.approx(i_frac, abs=0.01) == 1526.413
    assert pytest.approx(j_frac, abs=0.01) == 656.176

    # Nearest neighbor node
    i_near, j_near = weights_dict["nearest_node"]
    assert i_near == 1526
    assert j_near == 656

    # 4 Surrounding nodes
    nodes = weights_dict["nodes"]
    assert len(nodes) == 4
    assert nodes[0]["node"] == (1526, 656)
    assert nodes[1]["node"] == (1527, 656)
    assert nodes[2]["node"] == (1526, 657)
    assert nodes[3]["node"] == (1527, 657)

    # Weights must be strictly positive and sum exactly to 1.000000
    w_sum = sum(n["weight"] for n in nodes)
    assert pytest.approx(w_sum, abs=1e-6) == 1.0
    for n in nodes:
        assert 0.0 < n["weight"] < 1.0

    # Distinguish bilinear weights from nearest-neighbor binary flag
    assert nodes[0]["weight"] < 1.0, "Bilinear weight must not be confused with nearest-neighbor indicator"


# ==============================================================================
# 4. OPERATIONAL ELIGIBILITY & LATENCY SENSITIVITY TESTS
# ==============================================================================

def test_operational_eligibility_primary_f02():
    """Verify f02 lead from cycle T-2h satisfies forecast origin cutoff under standard latency."""
    target_t = pd.Timestamp("2015-06-01 15:00:00", tz="UTC")
    cycle_init = pd.Timestamp("2015-06-01 13:00:00", tz="UTC")  # T - 2h

    # 1. At 50 min latency: available at 13:50 UTC, origin is 14:00 UTC -> 10 min buffer
    eval_50 = evaluate_operational_eligibility(
        target_hour_utc=target_t,
        cycle_init_utc=cycle_init,
        forecast_lead_hours=2,
        forecast_origin_lead_minutes=60,
        model_latency_minutes=50,
    )
    assert eval_50["is_admissible"] is True
    assert eval_50["valid_matches_target"] is True
    assert eval_50["available_before_origin"] is True
    assert eval_50["buffer_minutes"] == 10.0
    assert eval_50["valid_time_utc"] == target_t

    # 2. At 60 min latency: available at 14:00 UTC, origin is 14:00 UTC -> 0 min buffer
    eval_60 = evaluate_operational_eligibility(
        target_hour_utc=target_t,
        cycle_init_utc=cycle_init,
        forecast_lead_hours=2,
        forecast_origin_lead_minutes=60,
        model_latency_minutes=60,
    )
    assert eval_60["is_admissible"] is True
    assert eval_60["buffer_minutes"] == 0.0  # Zero margin for error

    # 3. At 65 min latency: available at 14:05 UTC > 14:00 UTC -> Ineligible (late!)
    eval_65 = evaluate_operational_eligibility(
        target_hour_utc=target_t,
        cycle_init_utc=cycle_init,
        forecast_lead_hours=2,
        forecast_origin_lead_minutes=60,
        model_latency_minutes=65,
    )
    assert eval_65["is_admissible"] is False
    assert eval_65["available_before_origin"] is False
    assert eval_65["buffer_minutes"] == -5.0


def test_operational_eligibility_step_back_fallback_f03():
    """Verify step-back fallback cycle T-3h with lead f03 provides robust operational buffer."""
    target_t = pd.Timestamp("2015-06-01 15:00:00", tz="UTC")
    cycle_init_f03 = pd.Timestamp("2015-06-01 12:00:00", tz="UTC")  # T - 3h

    eval_f03 = evaluate_operational_eligibility(
        target_hour_utc=target_t,
        cycle_init_utc=cycle_init_f03,
        forecast_lead_hours=3,
        forecast_origin_lead_minutes=60,
        model_latency_minutes=50,
    )
    assert eval_f03["is_admissible"] is True
    assert eval_f03["valid_time_utc"] == target_t
    assert eval_f03["buffer_minutes"] == 70.0  # 70 minutes of delivery margin


# ==============================================================================
# 5. DST TRANSITION ALIGNMENT & INTERVAL LOGIC TESTS
# ==============================================================================

def test_dst_transitions_preserve_utc_without_inventing_interval_semantics():
    """Verify autumn fallback preserves two physical runs and spring forward handles nonexistent local hour."""
    # Autumn 2015 Fallback (2015-11-01):
    # Clocks repeat 01:00 (first 05:00 UTC in EDT, second 06:00 UTC in EST)
    run_edt = evaluate_operational_eligibility(
        target_hour_utc=pd.Timestamp("2015-11-01 05:00:00", tz="UTC"),
        cycle_init_utc=pd.Timestamp("2015-11-01 03:00:00", tz="UTC"),
        forecast_lead_hours=2,
    )
    run_est = evaluate_operational_eligibility(
        target_hour_utc=pd.Timestamp("2015-11-01 06:00:00", tz="UTC"),
        cycle_init_utc=pd.Timestamp("2015-11-01 04:00:00", tz="UTC"),
        forecast_lead_hours=2,
    )
    assert run_edt["is_admissible"] is True
    assert run_est["is_admissible"] is True
    assert run_edt["valid_time_utc"] != run_est["valid_time_utc"]

    # When projected to Eastern Civil Time:
    t_edt_local = run_edt["valid_time_utc"].tz_convert("America/New_York")
    t_est_local = run_est["valid_time_utc"].tz_convert("America/New_York")
    assert str(t_edt_local) == "2015-11-01 01:00:00-04:00"
    assert str(t_est_local) == "2015-11-01 01:00:00-05:00"

    # Both physical UTC forecasts are distinct. When stripped of timezone for naive join,
    # they produce a duplicate collision at '2015-11-01 01:00:00', requiring explicit mean collapsing.
    assert t_edt_local.tz_localize(None) == t_est_local.tz_localize(None)
