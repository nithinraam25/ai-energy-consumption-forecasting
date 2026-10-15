"""Forecast error analysis module for AI-Powered Energy Consumption Forecasting.

Generates reproducible residual diagnostics, temporal error disaggregations
(by hour of day, day of week, demand quantile/band), and publication-quality
diagnostic visualizations.
"""

from pathlib import Path
from typing import Any, Sequence
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FORECASTS_DIR, METRICS_DIR, PLOTS_DIR, ensure_directories
from src.evaluation.metrics import calculate_metrics, MetricResults


def compute_error_metrics_by_hour(
    df: pd.DataFrame,
    target_col: str = "target_next_hour",
    pred_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    timestamp_col: str = "target_timestamp",
) -> pd.DataFrame:
    """Disaggregate forecast accuracy across the 24 hours of the diurnal cycle.

    Args:
        df: DataFrame containing timestamps, actual targets, and model predictions.
        target_col: True target demand column name.
        pred_col: Forecast prediction column name.
        timestamp_col: Target evaluation timestamp column name.

    Returns:
        pd.DataFrame: Hourly error breakdown with sample counts, MAE, RMSE, R2, MAPE, and mean residual.
    """
    work_df = df.copy()
    work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col])
    work_df["hour"] = work_df[timestamp_col].dt.hour
    work_df["residual"] = work_df[target_col] - work_df[pred_col]

    records = []
    for hour, grp in work_df.groupby("hour"):
        y_true = grp[target_col].to_numpy()
        y_pred = grp[pred_col].to_numpy()
        m = calculate_metrics(y_true, y_pred)
        mean_res = float(np.mean(grp["residual"]))
        records.append({
            "hour": int(hour),
            "sample_count": m.sample_count,
            "mae": m.mae,
            "rmse": m.rmse,
            "r2": m.r2,
            "mape": m.mape,
            "mean_residual": mean_res,
        })

    return pd.DataFrame(records).sort_values("hour").reset_index(drop=True)


def compute_error_metrics_by_weekday(
    df: pd.DataFrame,
    target_col: str = "target_next_hour",
    pred_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    timestamp_col: str = "target_timestamp",
) -> pd.DataFrame:
    """Disaggregate forecast accuracy across the 7 days of the week.

    Args:
        df: DataFrame containing timestamps, actual targets, and model predictions.
        target_col: True target demand column name.
        pred_col: Forecast prediction column name.
        timestamp_col: Target evaluation timestamp column name.

    Returns:
        pd.DataFrame: Weekday error breakdown with sample counts, MAE, RMSE, R2, MAPE, and mean residual.
    """
    work_df = df.copy()
    work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col])
    work_df["day_of_week"] = work_df[timestamp_col].dt.dayofweek
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    work_df["day_name"] = work_df["day_of_week"].map(lambda d: day_names[d])
    work_df["residual"] = work_df[target_col] - work_df[pred_col]

    records = []
    for dow, grp in work_df.groupby("day_of_week"):
        y_true = grp[target_col].to_numpy()
        y_pred = grp[pred_col].to_numpy()
        m = calculate_metrics(y_true, y_pred)
        mean_res = float(np.mean(grp["residual"]))
        records.append({
            "day_of_week": int(dow),
            "day_name": day_names[dow],
            "sample_count": m.sample_count,
            "mae": m.mae,
            "rmse": m.rmse,
            "r2": m.r2,
            "mape": m.mape,
            "mean_residual": mean_res,
        })

    return pd.DataFrame(records).sort_values("day_of_week").reset_index(drop=True)


def compute_error_metrics_by_demand_band(
    df: pd.DataFrame,
    target_col: str = "target_next_hour",
    pred_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    custom_bins: Sequence[float] | None = None,
    bin_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Disaggregate forecast accuracy across demand magnitude regimes.

    Default partitions demand into 4 intuitive operational regimes:
    - Low (<25,000 MW): Off-peak / overnight baseload.
    - Moderate (25,000 - 32,000 MW): Normal daytime / shoulder demand.
    - High (32,000 - 40,000 MW): Peak daytime / heating & cooling surge.
    - Extreme Peak (>40,000 MW): Grid stress / heatwave or deep freeze events.

    Args:
        df: DataFrame containing actual targets and model predictions.
        target_col: True target demand column name.
        pred_col: Forecast prediction column name.
        custom_bins: Optional custom numerical bin edges.
        bin_labels: Optional labels for each bin.

    Returns:
        pd.DataFrame: Demand regime error breakdown.
    """
    work_df = df.copy()
    min_val = float(work_df[target_col].min())
    max_val = float(work_df[target_col].max())

    bins = list(custom_bins) if custom_bins is not None else [
        min(min_val - 1.0, 15000.0),
        25000.0,
        32000.0,
        40000.0,
        max(max_val + 1.0, 55000.0),
    ]
    labels = list(bin_labels) if bin_labels is not None else [
        "Low (<25k MW)",
        "Moderate (25k-32k MW)",
        "High (32k-40k MW)",
        "Extreme Peak (>40k MW)",
    ]

    work_df["demand_band"] = pd.cut(work_df[target_col], bins=bins, labels=labels, include_lowest=True)
    work_df["residual"] = work_df[target_col] - work_df[pred_col]

    records = []
    for idx, label in enumerate(labels):
        grp = work_df[work_df["demand_band"] == label]
        if grp.empty:
            continue
        y_true = grp[target_col].to_numpy()
        y_pred = grp[pred_col].to_numpy()
        m = calculate_metrics(y_true, y_pred)
        mean_res = float(np.mean(grp["residual"]))
        records.append({
            "demand_band": label,
            "min_demand_mw": float(grp[target_col].min()),
            "max_demand_mw": float(grp[target_col].max()),
            "sample_count": m.sample_count,
            "mae": m.mae,
            "rmse": m.rmse,
            "r2": m.r2,
            "mape": m.mape,
            "mean_residual": mean_res,
        })

    return pd.DataFrame(records)


def identify_largest_errors(
    df: pd.DataFrame,
    target_col: str = "target_next_hour",
    pred_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    top_n: int = 20,
    timestamp_col: str = "target_timestamp",
) -> pd.DataFrame:
    """Identify the largest absolute forecast error instances for diagnostic scrutiny.

    Args:
        df: DataFrame containing timestamps, actual targets, and model predictions.
        target_col: True target demand column name.
        pred_col: Forecast prediction column name.
        top_n: Number of outlier errors to return. Defaults to 20.
        timestamp_col: Target timestamp column name.

    Returns:
        pd.DataFrame: Table of largest prediction errors sorted by absolute error descending.
    """
    work_df = df.copy()
    work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col])
    work_df["residual"] = work_df[target_col] - work_df[pred_col]
    work_df["abs_error"] = work_df["residual"].abs()
    # Guard against division by zero matching calculate_metrics threshold
    work_df["pct_error"] = np.where(
        work_df[target_col].abs() > 1e-5,
        (work_df["abs_error"] / work_df[target_col]) * 100.0,
        np.nan,
    )

    top_df = work_df.sort_values("abs_error", ascending=False).head(top_n).copy()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    results = []
    for _, row in top_df.iterrows():
        ts = row[timestamp_col]
        results.append({
            "target_timestamp": str(ts),
            "actual_mw": float(row[target_col]),
            "predicted_mw": float(row[pred_col]),
            "residual_mw": float(row["residual"]),
            "abs_error_mw": float(row["abs_error"]),
            "pct_error": float(row["pct_error"]) if pd.notna(row["pct_error"]) else None,
            "hour": int(ts.hour),
            "day_name": day_names[ts.dayofweek],
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualization Generation Functions
# ---------------------------------------------------------------------------


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    target_col: str = "target_next_hour",
    pred_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    baseline_col: str | None = "pred_tree",
    timestamp_col: str = "target_timestamp",
    start_time: str | None = None,
    end_time: str | None = None,
    save_path: Path | str | None = None,
    title: str = "Actual vs. Predicted Electrical Demand (Contiguous Test Window)",
) -> Path:
    """Plot contiguous time-series comparison between actual and predicted demand."""
    work_df = df.copy()
    work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col])
    work_df = work_df.sort_values(timestamp_col).reset_index(drop=True)

    # Default to a representative 3-week contiguous test window in July 2018 (summer peak)
    if start_time is None:
        start_time = "2018-07-01 00:00:00"
    if end_time is None:
        end_time = "2018-07-21 23:00:00"

    mask = (work_df[timestamp_col] >= pd.Timestamp(start_time)) & (work_df[timestamp_col] <= pd.Timestamp(end_time))
    window = work_df[mask].copy()
    if window.empty:
        # Fall back to first 500 rows if specified range is empty
        window = work_df.iloc[:500].copy()

    plt.figure(figsize=(14, 6), dpi=150)
    plt.plot(window[timestamp_col], window[target_col], label="Actual Demand (PJME_MW)", color="#1f77b4", linewidth=1.8)
    plt.plot(window[timestamp_col], window[pred_col], label="Tuned Model Forecast (HistGB Refit)", color="#d62728", linestyle="--", linewidth=1.5, alpha=0.9)
    if baseline_col and baseline_col in window.columns:
        plt.plot(window[timestamp_col], window[baseline_col], label="Frozen Baseline Forecast (HistGB)", color="#7f7f7f", linestyle=":", linewidth=1.2, alpha=0.7)

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Target Timestamp (Local Hour Ending)", fontsize=11, labelpad=8)
    plt.ylabel("Power Demand (Megawatts - MW)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=25)
    plt.tight_layout()

    dest = Path(save_path) if save_path else PLOTS_DIR / "pjm_actual_vs_predicted_timeseries.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()
    return dest


def plot_residuals_over_time(
    df: pd.DataFrame,
    target_col: str = "target_next_hour",
    pred_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    timestamp_col: str = "target_timestamp",
    save_path: Path | str | None = None,
    title: str = "Forecasting Residuals Over Time (Actual - Predicted Demand)",
) -> Path:
    """Plot prediction residuals across the complete holdout test partition."""
    work_df = df.copy()
    work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col])
    work_df = work_df.sort_values(timestamp_col).reset_index(drop=True)
    residuals = work_df[target_col] - work_df[pred_col]

    plt.figure(figsize=(14, 5), dpi=150)
    plt.scatter(work_df[timestamp_col], residuals, color="#2ca02c", alpha=0.25, s=6, label="Residuals ($e = y - \\hat{y}$)")
    plt.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)

    # 3-sigma bands
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    plt.axhline(mean_res + 3 * std_res, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label=f"+3σ ({mean_res + 3*std_res:.0f} MW)")
    plt.axhline(mean_res - 3 * std_res, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label=f"-3σ ({mean_res - 3*std_res:.0f} MW)")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Target Timestamp (2016-2018 Test Period)", fontsize=11, labelpad=8)
    plt.ylabel("Residual (MW)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=20)
    plt.tight_layout()

    dest = Path(save_path) if save_path else PLOTS_DIR / "pjm_residuals_over_time.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()
    return dest


def plot_error_distribution(
    df: pd.DataFrame,
    target_col: str = "target_next_hour",
    pred_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    save_path: Path | str | None = None,
    title: str = "Residual Error Distribution (Actual - Predicted Demand)",
) -> Path:
    """Plot histogram and density distribution of forecast residuals."""
    residuals = df[target_col] - df[pred_col]
    mean_res = float(np.mean(residuals))
    median_res = float(np.median(residuals))
    std_res = float(np.std(residuals))

    plt.figure(figsize=(10, 5.5), dpi=150)
    counts, bins, patches = plt.hist(residuals, bins=80, density=True, color="#3b528b", alpha=0.7, edgecolor="white", linewidth=0.5)

    # Plot normal fit curve for comparison
    x = np.linspace(residuals.min(), residuals.max(), 300)
    p = (1 / (std_res * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_res) / std_res) ** 2)
    plt.plot(x, p, "k--", linewidth=1.5, label=f"Gaussian Fit (μ={mean_res:.1f}, σ={std_res:.1f})")

    plt.axvline(mean_res, color="red", linestyle="-", linewidth=1.2, label=f"Mean: {mean_res:.1f} MW")
    plt.axvline(median_res, color="orange", linestyle=":", linewidth=1.5, label=f"Median: {median_res:.1f} MW")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Residual Error (MW)", fontsize=11, labelpad=8)
    plt.ylabel("Probability Density", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.xlim(-4000, 4000)
    plt.tight_layout()

    dest = Path(save_path) if save_path else PLOTS_DIR / "pjm_error_distribution.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()
    return dest


def plot_mae_by_hour(
    hourly_df: pd.DataFrame,
    save_path: Path | str | None = None,
    title: str = "Mean Absolute Error (MAE) by Hour of Day",
) -> Path:
    """Plot bar chart of MAE disaggregated across diurnal hours (0-23)."""
    plt.figure(figsize=(11, 5), dpi=150)
    bars = plt.bar(hourly_df["hour"], hourly_df["mae"], color="#1f77b4", edgecolor="#0e4b75", alpha=0.85, width=0.7)

    # Annotate overall mean
    mean_mae = float(hourly_df["mae"].mean())
    plt.axhline(mean_mae, color="red", linestyle="--", linewidth=1.2, label=f"Diurnal Average MAE: {mean_mae:.1f} MW")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Hour of Day (00:00 - 23:00 Local Hour Ending)", fontsize=11, labelpad=8)
    plt.ylabel("MAE (Megawatts - MW)", fontsize=11, labelpad=8)
    plt.xticks(hourly_df["hour"])
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()

    dest = Path(save_path) if save_path else PLOTS_DIR / "pjm_mae_by_hour_of_day.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()
    return dest


def plot_mae_by_weekday(
    weekday_df: pd.DataFrame,
    save_path: Path | str | None = None,
    title: str = "Mean Absolute Error (MAE) by Day of Week",
) -> Path:
    """Plot bar chart of MAE disaggregated across Monday-Sunday."""
    plt.figure(figsize=(9, 5), dpi=150)
    plt.bar(weekday_df["day_name"], weekday_df["mae"], color="#5c2751", edgecolor="#391632", alpha=0.85, width=0.6)

    mean_mae = float(weekday_df["mae"].mean())
    plt.axhline(mean_mae, color="red", linestyle="--", linewidth=1.2, label=f"Weekly Average MAE: {mean_mae:.1f} MW")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Day of Week", fontsize=11, labelpad=8)
    plt.ylabel("MAE (Megawatts - MW)", fontsize=11, labelpad=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()

    dest = Path(save_path) if save_path else PLOTS_DIR / "pjm_mae_by_day_of_week.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()
    return dest


def plot_mae_by_demand_band(
    demand_df: pd.DataFrame,
    save_path: Path | str | None = None,
    title: str = "Mean Absolute Error (MAE) by Demand Regime",
) -> Path:
    """Plot bar chart of MAE disaggregated across demand bands."""
    plt.figure(figsize=(9.5, 5), dpi=150)
    plt.bar(demand_df["demand_band"], demand_df["mae"], color="#d95f02", edgecolor="#a64200", alpha=0.85, width=0.55)

    mean_mae = float(demand_df["mae"].mean())
    plt.axhline(mean_mae, color="black", linestyle="--", linewidth=1.2, label=f"Average Across Regimes: {mean_mae:.1f} MW")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Demand Magnitude Regime", fontsize=11, labelpad=8)
    plt.ylabel("MAE (Megawatts - MW)", fontsize=11, labelpad=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
    plt.tight_layout()

    dest = Path(save_path) if save_path else PLOTS_DIR / "pjm_mae_by_demand_band.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    plt.close()
    return dest


# ---------------------------------------------------------------------------
# End-to-End Analysis Workflow Execution
# ---------------------------------------------------------------------------


def run_forecast_error_analysis(
    predictions_path: Path | str | None = None,
    model_col: str = "pred_EnergyTree_HistGB_Tuned_Refit",
    baseline_col: str = "pred_tree",
    target_col: str = "target_next_hour",
    timestamp_col: str = "target_timestamp",
) -> dict[str, Any]:
    """Execute end-to-end forecast error analysis on saved test holdout predictions.

    Loads approved test predictions, computes temporal and regime disaggregations,
    saves tabular summaries, and renders publication-ready diagnostic charts.

    Returns:
        dict[str, Any]: Summary metadata with paths and metrics.
    """
    ensure_directories()
    pred_file = Path(predictions_path) if predictions_path else FORECASTS_DIR / "pjm_test_predictions_phase5a.csv"
    if not pred_file.is_file():
        raise FileNotFoundError(f"Predictions file not found at: {pred_file}")

    df = pd.read_csv(pred_file)
    if model_col not in df.columns:
        raise ValueError(f"Specified model column '{model_col}' missing from predictions CSV.")

    print("=" * 75)
    print("  EXECUTING COMPREHENSIVE FORECAST ERROR ANALYSIS")
    print(f"  Source File: {pred_file}")
    print(f"  Target Column: {target_col} (Megawatts - MW)")
    print(f"  Evaluated Model: {model_col}")
    print(f"  Test Samples: {len(df):,}")
    print("=" * 75)

    # 1. Compute Tabular Breakdowns
    hourly_df = compute_error_metrics_by_hour(df, target_col, model_col, timestamp_col)
    weekday_df = compute_error_metrics_by_weekday(df, target_col, model_col, timestamp_col)
    demand_df = compute_error_metrics_by_demand_band(df, target_col, model_col)
    largest_errors_df = identify_largest_errors(df, target_col, model_col, top_n=20, timestamp_col=timestamp_col)

    # 2. Save Tabular Summaries
    hourly_csv = METRICS_DIR / "pjm_error_analysis_hourly.csv"
    hourly_json = METRICS_DIR / "pjm_error_analysis_hourly.json"
    hourly_df.to_csv(hourly_csv, index=False)
    hourly_df.to_json(hourly_json, orient="records", indent=2)

    weekday_csv = METRICS_DIR / "pjm_error_analysis_weekday.csv"
    weekday_json = METRICS_DIR / "pjm_error_analysis_weekday.json"
    weekday_df.to_csv(weekday_csv, index=False)
    weekday_df.to_json(weekday_json, orient="records", indent=2)

    demand_csv = METRICS_DIR / "pjm_error_analysis_demand_bands.csv"
    demand_json = METRICS_DIR / "pjm_error_analysis_demand_bands.json"
    demand_df.to_csv(demand_csv, index=False)
    demand_df.to_json(demand_json, orient="records", indent=2)

    largest_csv = METRICS_DIR / "pjm_largest_errors.csv"
    largest_json = METRICS_DIR / "pjm_largest_errors.json"
    largest_errors_df.to_csv(largest_csv, index=False)
    largest_errors_df.to_json(largest_json, orient="records", indent=2)

    # 3. Render Visualizations
    p1 = plot_actual_vs_predicted(df, target_col, model_col, baseline_col, timestamp_col)
    p2 = plot_residuals_over_time(df, target_col, model_col, timestamp_col)
    p3 = plot_error_distribution(df, target_col, model_col)
    p4 = plot_mae_by_hour(hourly_df)
    p5 = plot_mae_by_weekday(weekday_df)
    p6 = plot_mae_by_demand_band(demand_df)

    print("\n[Artifacts Generated]")
    print(f"  Tabular - Hourly Breakdown:      {hourly_csv}")
    print(f"  Tabular - Weekday Breakdown:     {weekday_csv}")
    print(f"  Tabular - Demand Band Breakdown: {demand_csv}")
    print(f"  Tabular - Largest Errors:        {largest_csv}")
    print(f"  Plot 1 - Actual vs Predicted:    {p1}")
    print(f"  Plot 2 - Residuals Over Time:    {p2}")
    print(f"  Plot 3 - Error Distribution:     {p3}")
    print(f"  Plot 4 - MAE by Hour of Day:     {p4}")
    print(f"  Plot 5 - MAE by Day of Week:     {p5}")
    print(f"  Plot 6 - MAE by Demand Band:     {p6}")
    print("=" * 75)

    return {
        "status": "success",
        "hourly_df": hourly_df,
        "weekday_df": weekday_df,
        "demand_df": demand_df,
        "largest_errors_df": largest_errors_df,
        "plots": [str(p1), str(p2), str(p3), str(p4), str(p5), str(p6)],
    }


if __name__ == "__main__":
    run_forecast_error_analysis()
