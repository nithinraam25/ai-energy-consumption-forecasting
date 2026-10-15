"""Unit tests for forecast error analysis module and daylight saving time edge cases."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    compute_error_metrics_by_hour,
    compute_error_metrics_by_weekday,
    compute_error_metrics_by_demand_band,
    identify_largest_errors,
    plot_actual_vs_predicted,
    plot_residuals_over_time,
    plot_error_distribution,
    plot_mae_by_hour,
    plot_mae_by_weekday,
    plot_mae_by_demand_band,
)
from src.features.temporal_features import add_lag_features, add_target, create_temporal_features
from src.data.preprocessor import preprocess_energy_data


@pytest.fixture
def mock_predictions_df() -> pd.DataFrame:
    """Deterministic synthetic test prediction DataFrame over 168 hours (7 full days)."""
    ts = pd.date_range("2026-06-01 00:00:00", periods=168, freq="h")
    # Base diurnal wave between 20k and 35k MW
    actual = 25000.0 + 8000.0 * np.sin(2 * np.pi * ts.hour / 24.0) + (ts.dayofweek * 500.0)
    # Predicted with small synthetic error (500 MW noise)
    noise = np.where(ts.hour % 2 == 0, 400.0, -300.0)
    pred = actual + noise

    return pd.DataFrame({
        "target_timestamp": ts,
        "target_next_hour": actual,
        "pred_EnergyTree_HistGB_Tuned_Refit": pred,
        "pred_tree": pred + 200.0,
    })


def test_compute_error_metrics_by_hour(mock_predictions_df: pd.DataFrame) -> None:
    """Verify hourly error disaggregation produces 24 hours with exact sample counts."""
    res = compute_error_metrics_by_hour(
        mock_predictions_df,
        target_col="target_next_hour",
        pred_col="pred_EnergyTree_HistGB_Tuned_Refit",
    )
    assert len(res) == 24
    assert list(res["hour"]) == list(range(24))
    assert res["sample_count"].sum() == len(mock_predictions_df)
    assert (res["mae"] > 0).all()
    assert (res["rmse"] >= res["mae"]).all()


def test_compute_error_metrics_by_weekday(mock_predictions_df: pd.DataFrame) -> None:
    """Verify weekday error disaggregation produces 7 days with accurate names."""
    res = compute_error_metrics_by_weekday(
        mock_predictions_df,
        target_col="target_next_hour",
        pred_col="pred_EnergyTree_HistGB_Tuned_Refit",
    )
    assert len(res) == 7
    expected_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert list(res["day_name"]) == expected_days
    assert res["sample_count"].sum() == len(mock_predictions_df)
    assert (res["mae"] > 0).all()


def test_compute_error_metrics_by_demand_band(mock_predictions_df: pd.DataFrame) -> None:
    """Verify demand regime error grouping partitions observations accurately."""
    res = compute_error_metrics_by_demand_band(
        mock_predictions_df,
        target_col="target_next_hour",
        pred_col="pred_EnergyTree_HistGB_Tuned_Refit",
    )
    assert len(res) > 0
    assert "demand_band" in res.columns
    assert "mae" in res.columns
    assert res["sample_count"].sum() == len(mock_predictions_df)


def test_identify_largest_errors(mock_predictions_df: pd.DataFrame) -> None:
    """Verify largest error identification ranks instances by absolute residual descending."""
    top_5 = identify_largest_errors(
        mock_predictions_df,
        target_col="target_next_hour",
        pred_col="pred_EnergyTree_HistGB_Tuned_Refit",
        top_n=5,
    )
    assert len(top_5) == 5
    # Must be monotonically non-increasing
    for i in range(len(top_5) - 1):
        assert top_5.iloc[i]["abs_error_mw"] >= top_5.iloc[i + 1]["abs_error_mw"]


def test_error_analysis_plots(mock_predictions_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify diagnostic plotting routines generate valid PNG files without raising exceptions."""
    p1 = plot_actual_vs_predicted(
        mock_predictions_df,
        save_path=tmp_path / "actual_vs_pred.png",
        start_time="2026-06-01",
        end_time="2026-06-03",
    )
    assert p1.is_file() and p1.stat().st_size > 1000

    p2 = plot_residuals_over_time(mock_predictions_df, save_path=tmp_path / "residuals.png")
    assert p2.is_file() and p2.stat().st_size > 1000

    p3 = plot_error_distribution(mock_predictions_df, save_path=tmp_path / "error_dist.png")
    assert p3.is_file() and p3.stat().st_size > 1000

    hourly_df = compute_error_metrics_by_hour(mock_predictions_df)
    p4 = plot_mae_by_hour(hourly_df, save_path=tmp_path / "mae_hour.png")
    assert p4.is_file() and p4.stat().st_size > 1000

    weekday_df = compute_error_metrics_by_weekday(mock_predictions_df)
    p5 = plot_mae_by_weekday(weekday_df, save_path=tmp_path / "mae_weekday.png")
    assert p5.is_file() and p5.stat().st_size > 1000

    demand_df = compute_error_metrics_by_demand_band(mock_predictions_df)
    p6 = plot_mae_by_demand_band(demand_df, save_path=tmp_path / "mae_demand.png")
    assert p6.is_file() and p6.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Focused Daylight Saving Time & Timestamp Semantics Tests
# ---------------------------------------------------------------------------


def test_autumn_repeated_hour_aggregation_preserves_mean() -> None:
    """Verify duplicate 02:00 autumn clock shift aggregation under mean policy."""
    raw_df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2016-11-06 01:00:00",
            "2016-11-06 02:00:00",
            "2016-11-06 02:00:00",  # Second physical 02:00 reading
            "2016-11-06 03:00:00",
        ]),
        "consumption": [20000.0, 19000.0, 21000.0, 22000.0],
    })

    clean_df, report = preprocess_energy_data(raw_df, duplicate_policy="mean")
    assert len(clean_df) == 3
    assert report.removed_duplicates == 1

    # The 02:00 row must have mean consumption = (19000 + 21000) / 2 = 20000.0
    row_02 = clean_df[clean_df["timestamp"] == pd.Timestamp("2016-11-06 02:00:00")]
    assert len(row_02) == 1
    assert row_02["consumption"].iloc[0] == 20000.0


def test_spring_forward_gap_prevents_false_lag_association() -> None:
    """Verify 24-hour lag does not associate mismatched diurnal hours across spring gap."""
    # 2-day sequence with spring forward gap on day 2 at 03:00
    # Day 1: 2026-03-07 00:00 to 23:00 (24 hours)
    day1 = pd.date_range("2026-03-07 00:00:00", periods=24, freq="h")
    # Day 2: 2026-03-08 with 03:00 missing
    day2_p1 = pd.date_range("2026-03-08 00:00:00", "2026-03-08 02:00:00", freq="h")
    day2_p2 = pd.date_range("2026-03-08 04:00:00", "2026-03-08 23:00:00", freq="h")
    all_ts = day1.union(day2_p1).union(day2_p2)

    df = pd.DataFrame({
        "timestamp": all_ts,
        "consumption": [1000.0 + i for i in range(len(all_ts))],
    })

    res = add_lag_features(df, lags=(24,))

    # For 2026-03-08 03:00, it does not exist in df.
    # For 2026-03-08 04:00, 24 hours prior is 2026-03-07 04:00, which DOES exist!
    val_day1_04 = df[df["timestamp"] == pd.Timestamp("2026-03-07 04:00:00")]["consumption"].iloc[0]
    lag_day2_04 = res[res["timestamp"] == pd.Timestamp("2026-03-08 04:00:00")]["lag_24"].iloc[0]
    assert lag_day2_04 == val_day1_04


def test_identify_largest_errors_zero_actual_handling() -> None:
    """Verify identify_largest_errors handles zero actual demand without ZeroDivisionError."""
    df = pd.DataFrame({
        "target_timestamp": pd.date_range("2026-01-01 00:00:00", periods=3, freq="h"),
        "target_next_hour": [0.0, 100.0, 200.0],
        "pred": [50.0, 120.0, 190.0],
    })
    res = identify_largest_errors(df, target_col="target_next_hour", pred_col="pred", top_n=3)
    assert len(res) == 3
    # First row has actual 0.0, so pct_error must be None/NaN
    row_zero = res[res["actual_mw"] == 0.0].iloc[0]
    assert row_zero["pct_error"] is None or np.isnan(row_zero["pct_error"])


def test_demand_bands_all_samples_partitioned_unambiguously() -> None:
    """Verify demand band partitioning covers every observation and matches total count."""
    df = pd.DataFrame({
        "target_next_hour": [15000.0, 20000.0, 25000.0, 32000.0, 40000.0, 50000.0],
        "pred": [15100.0, 19900.0, 25050.0, 31900.0, 40200.0, 49800.0],
    })
    bands_df = compute_error_metrics_by_demand_band(df, target_col="target_next_hour", pred_col="pred")
    assert bands_df["sample_count"].sum() == len(df)
