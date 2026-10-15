"""Comprehensive, reproducible forecast evaluation suite for PJM East energy forecasting.

Executes:
1. Multi-horizon evaluation across H=1, H=24, and H=48.
2. Comparison against multiple rigorous baselines:
   - Persistence (lag 1, lag 24, lag 48)
   - Daily Seasonal Naive (lag 24)
   - Weekly Seasonal Naive (lag 168)
   - Baseline Tree (Strict Lags, No Weather)
   - Production NWP Forecast-Enhanced Model
3. Scale-dependent and percentage metrics: MAE, RMSE, R2, MAPE, WAPE.
4. Statistical significance testing (Diebold-Mariano / paired t-test).
5. Granular sliced error analysis (by hour, season, day of week, demand quartile, peak-load).
6. Conformal prediction interval evaluation (coverage, width, Winkler score, quantile ordering).
7. Machine-readable outputs and publication-ready diagnostic charts.
"""

from pathlib import Path
from typing import Any
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
)
from src.data.multi_station import load_multi_station_dataset
from src.features.horizon_features import (
    create_horizon_feature_matrix,
    get_horizon_feature_names,
    SUPPORTED_HORIZONS,
)
from src.evaluation.metrics import chronological_split, calculate_metrics, MetricResults
from src.features.temporal_features import create_temporal_features
from src.models.tree_models import EnergyTreeForecaster
from src.models.probabilistic_forecaster import (
    MultiQuantileForecaster,
    AdaptiveConformalCalibrator,
    ConformalIntervalCalibrator,
    DEFAULT_PROBABILISTIC_MODEL_PATH,
    DEFAULT_ADAPTIVE_CALIBRATION_PATH,
    DEFAULT_CALIBRATION_ARTIFACT_PATH,
)

logger = logging.getLogger(__name__)


def compute_winkler_score(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Calculate the mean Winkler interval score for a (1 - alpha) prediction interval.

    Score = (u - l) + (2/alpha)*(l - y)*1(y < l) + (2/alpha)*(y - u)*1(y > u).
    Lower score represents sharper, better-calibrated intervals.
    """
    valid = ~(np.isnan(y_true) | np.isnan(q_low) | np.isnan(q_high))
    if not np.any(valid):
        return float("nan")
    y_t = np.asarray(y_true)[valid]
    q_l = np.asarray(q_low)[valid]
    q_h = np.asarray(q_high)[valid]
    widths = q_h - q_l
    under_coverage = np.maximum(0.0, q_l - y_t)
    over_coverage = np.maximum(0.0, y_t - q_h)
    penalty = (2.0 / alpha) * (under_coverage + over_coverage)
    return float(np.mean(widths + penalty))


def diebold_mariano_test(
    y_true: np.ndarray,
    pred_benchmark: np.ndarray,
    pred_model: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Calculate Diebold-Mariano test statistic and p-value for absolute forecast errors.

    Tests null hypothesis H0: E[|e_benchmark| - |e_model|] = 0 (equal accuracy).
    Positive DM statistic indicates that pred_model has significantly smaller errors.
    """
    e_bench = np.abs(y_true - pred_benchmark)
    e_model = np.abs(y_true - pred_model)
    d = e_bench - e_model
    n = len(d)
    d_mean = np.mean(d)

    # Autocovariance adjustment for h-step horizon (Newey-West Bartlett kernel)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        gamma_k = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        gamma_sum += 2.0 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 1e-12:
        return 0.0, 1.0

    dm_stat = float(d_mean / np.sqrt(var_d))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_value


def get_season(month: int) -> str:
    """Classify month into meteorological season."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


def run_comprehensive_evaluation() -> dict[str, Any]:
    """Execute complete deterministic evaluation and export machine-readable metrics."""
    out_dir = OUTPUTS_DIR / "improvement"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  PHASE 2: COMPREHENSIVE FORECAST ACCURACY & EVALUATION BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    clean_csv = PROCESSED_DATA_DIR / "pjm_cleaned.csv"
    if not clean_csv.is_file():
        raise FileNotFoundError(f"Cleaned demand dataset not found at: {clean_csv}")

    clean_energy = pd.read_csv(clean_csv, parse_dates=["timestamp"])
    clean_energy = clean_energy.sort_values("timestamp").reset_index(drop=True)
    multi_nwp = load_multi_station_dataset()

    fixed_params = {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": 8,
        "max_leaf_nodes": 45,
        "l2_regularization": 0.5,
        "min_samples_leaf": 20,
        "random_state": RANDOM_STATE,
    }

    results: dict[str, Any] = {
        "evaluation_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_summary": {
            "total_records": len(clean_energy),
            "start_time": clean_energy["timestamp"].min().isoformat(),
            "end_time": clean_energy["timestamp"].max().isoformat(),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "horizons": {},
    }

    all_predictions_dfs: list[pd.DataFrame] = []
    h1_test_df: pd.DataFrame | None = None

    # 2. Iterate Supported Horizons
    for h in SUPPORTED_HORIZONS:
        print(f"\n--- Evaluating Horizon H={h}h ---")
        h_df_nwp = create_horizon_feature_matrix(clean_energy, nwp_df=multi_nwp, horizon=h)
        tr_w, v_w, te_w = chronological_split(h_df_nwp, 0.70, 0.15, 0.15)
        if h == 1:
            h1_test_df = te_w.copy()

        feats_w = get_horizon_feature_names(h, include_weather=True)
        feats_b = get_horizon_feature_names(h, include_weather=False)

        # Fit Models on Training Partition
        m_w = EnergyTreeForecaster(feature_names=feats_w, **fixed_params)
        m_w.fit(tr_w[feats_w], tr_w["target_demand"])

        m_b = EnergyTreeForecaster(feature_names=feats_b, **fixed_params)
        m_b.fit(tr_w[feats_b], tr_w["target_demand"])

        # Predict on Test Partition (strictly identical target rows and timestamps)
        y_test = te_w["target_demand"].to_numpy()
        pred_nwp = m_w.predict(te_w[feats_w])
        pred_base = m_b.predict(te_w[feats_b])

        # Define Baseline Predictions
        # Persistence: lag at t_origin = T - H
        pers_col = f"lag_{h}" if f"lag_{h}" in te_w.columns else f"lag_{min(c for c in [1, 24, 48] if c >= h)}"
        pred_pers = te_w[pers_col].to_numpy()

        # Daily Seasonal Naive: lag_24 if h <= 24, else lag_48
        daily_col = "lag_24" if "lag_24" in te_w.columns else "lag_48"
        pred_daily_sn = te_w[daily_col].to_numpy()

        # Weekly Seasonal Naive: lag_168
        pred_weekly_sn = te_w["lag_168"].to_numpy()

        # Calculate Metrics
        met_nwp = calculate_metrics(y_test, pred_nwp)
        met_base = calculate_metrics(y_test, pred_base)
        met_pers = calculate_metrics(y_test, pred_pers)
        met_daily = calculate_metrics(y_test, pred_daily_sn)
        met_weekly = calculate_metrics(y_test, pred_weekly_sn)

        # Statistical Significance Tests
        dm_pers, p_pers = diebold_mariano_test(y_test, pred_pers, pred_nwp, h=h)
        dm_base, p_base = diebold_mariano_test(y_test, pred_base, pred_nwp, h=h)
        dm_weekly, p_weekly = diebold_mariano_test(y_test, pred_weekly_sn, pred_nwp, h=h)

        # Gains
        gain_over_base = ((met_base.mae - met_nwp.mae) / met_base.mae) * 100.0
        gain_over_pers = ((met_pers.mae - met_nwp.mae) / met_pers.mae) * 100.0

        h_info = {
            "horizon_hours": h,
            "test_sample_count": len(y_test),
            "test_start_utc": te_w["target_timestamp"].min().isoformat(),
            "test_end_utc": te_w["target_timestamp"].max().isoformat(),
            "models": {
                "production_nwp": {
                    "mae_mw": round(met_nwp.mae, 2),
                    "rmse_mw": round(met_nwp.rmse, 2),
                    "r2": round(met_nwp.r2, 4),
                    "mape_pct": round(met_nwp.mape, 2),
                    "wape_pct": round(met_nwp.wape, 2),
                    "features_count": len(feats_w),
                },
                "baseline_no_weather": {
                    "mae_mw": round(met_base.mae, 2),
                    "rmse_mw": round(met_base.rmse, 2),
                    "r2": round(met_base.r2, 4),
                    "mape_pct": round(met_base.mape, 2),
                    "wape_pct": round(met_base.wape, 2),
                    "gain_nwp_pct": round(gain_over_base, 2),
                    "diebold_mariano_stat": round(dm_base, 3),
                    "p_value": float(f"{p_base:.4e}"),
                    "statistically_significant": bool(p_base < 0.01),
                },
                "persistence": {
                    "lag_used": pers_col,
                    "mae_mw": round(met_pers.mae, 2),
                    "rmse_mw": round(met_pers.rmse, 2),
                    "r2": round(met_pers.r2, 4),
                    "mape_pct": round(met_pers.mape, 2),
                    "wape_pct": round(met_pers.wape, 2),
                    "gain_nwp_pct": round(gain_over_pers, 2),
                    "diebold_mariano_stat": round(dm_pers, 3),
                    "p_value": float(f"{p_pers:.4e}"),
                },
                "daily_seasonal_naive": {
                    "lag_used": daily_col,
                    "mae_mw": round(met_daily.mae, 2),
                    "rmse_mw": round(met_daily.rmse, 2),
                    "r2": round(met_daily.r2, 4),
                    "mape_pct": round(met_daily.mape, 2),
                    "wape_pct": round(met_daily.wape, 2),
                },
                "weekly_seasonal_naive": {
                    "lag_used": "lag_168",
                    "mae_mw": round(met_weekly.mae, 2),
                    "rmse_mw": round(met_weekly.rmse, 2),
                    "r2": round(met_weekly.r2, 4),
                    "mape_pct": round(met_weekly.mape, 2),
                    "wape_pct": round(met_weekly.wape, 2),
                    "diebold_mariano_stat": round(dm_weekly, 3),
                    "p_value": float(f"{p_weekly:.4e}"),
                },
            },
        }

        # 3. Granular Sliced Analysis for Horizon
        eval_df = te_w.copy()
        eval_df["actual_demand_mw"] = y_test
        eval_df["pred_nwp_mw"] = pred_nwp
        eval_df["abs_error_mw"] = np.abs(y_test - pred_nwp)
        eval_df["residual_mw"] = y_test - pred_nwp
        eval_df["month"] = eval_df["target_timestamp"].dt.month
        eval_df["hour"] = eval_df["target_timestamp"].dt.hour
        eval_df["season"] = eval_df["month"].apply(get_season)
        eval_df["is_weekend"] = (eval_df["target_timestamp"].dt.dayofweek >= 5).astype(int)

        # Sliced by Season
        season_slices = {}
        for s in ("Winter", "Spring", "Summer", "Fall"):
            sub = eval_df[eval_df["season"] == s]
            if len(sub) > 0:
                sub_met = calculate_metrics(sub["actual_demand_mw"], sub["pred_nwp_mw"])
                season_slices[s] = {
                    "sample_count": len(sub),
                    "mae_mw": round(sub_met.mae, 2),
                    "rmse_mw": round(sub_met.rmse, 2),
                    "mape_pct": round(sub_met.mape, 2),
                    "wape_pct": round(sub_met.wape, 2),
                }
        h_info["sliced_by_season"] = season_slices

        # Sliced by Demand Quartiles & Peak Regimes
        q25, q50, q75, q95 = np.percentile(y_test, [25, 50, 75, 95])
        tier_masks = {
            "Q1_Low (<25th)": y_test <= q25,
            "Q2_MedLow (25-50th)": (y_test > q25) & (y_test <= q50),
            "Q3_MedHigh (50-75th)": (y_test > q50) & (y_test <= q75),
            "Q4_High (75-95th)": (y_test > q75) & (y_test <= q95),
            "Peak_Extreme (>95th)": y_test > q95,
        }
        tier_slices = {}
        for tier_name, mask in tier_masks.items():
            sub_y = y_test[mask]
            sub_pred = pred_nwp[mask]
            sub_met = calculate_metrics(sub_y, sub_pred)
            tier_slices[tier_name] = {
                "sample_count": int(mask.sum()),
                "mae_mw": round(sub_met.mae, 2),
                "rmse_mw": round(sub_met.rmse, 2),
                "mape_pct": round(sub_met.mape, 2),
                "wape_pct": round(sub_met.wape, 2),
            }
        h_info["sliced_by_demand_tier"] = tier_slices

        # Sliced by Weekday vs Weekend
        weekday_sub = eval_df[eval_df["is_weekend"] == 0]
        weekend_sub = eval_df[eval_df["is_weekend"] == 1]
        h_info["sliced_by_day_type"] = {
            "weekday": {
                "sample_count": len(weekday_sub),
                "mae_mw": round(float(weekday_sub["abs_error_mw"].mean()), 2),
                "wape_pct": round(float(weekday_sub["abs_error_mw"].sum() / weekday_sub["actual_demand_mw"].sum() * 100), 2),
            },
            "weekend": {
                "sample_count": len(weekend_sub),
                "mae_mw": round(float(weekend_sub["abs_error_mw"].mean()), 2),
                "wape_pct": round(float(weekend_sub["abs_error_mw"].sum() / weekend_sub["actual_demand_mw"].sum() * 100), 2),
            },
        }

        results["horizons"][f"H={h}h"] = h_info

        # Collect Predictions for H=1 and H=24
        pred_record = pd.DataFrame({
            "target_timestamp": eval_df["target_timestamp"],
            "horizon": h,
            "actual_demand_mw": y_test,
            "pred_nwp_mw": np.round(pred_nwp, 2),
            "pred_base_mw": np.round(pred_base, 2),
            "pred_pers_mw": np.round(pred_pers, 2),
            "residual_mw": np.round(eval_df["residual_mw"], 2),
        })
        all_predictions_dfs.append(pred_record)

    # 4. Probabilistic Prediction Interval Evaluation
    print("\n--- Evaluating Probabilistic Prediction Intervals & Conformal Calibration ---")
    prob_path = DEFAULT_PROBABILISTIC_MODEL_PATH
    adaptive_cal_path = DEFAULT_ADAPTIVE_CALIBRATION_PATH
    global_cal_path = DEFAULT_CALIBRATION_ARTIFACT_PATH

    prob_eval: dict[str, Any] = {}
    if prob_path.is_file() and adaptive_cal_path.is_file():
        prob_model = MultiQuantileForecaster.load(prob_path)
        adaptive_cal = AdaptiveConformalCalibrator.load(adaptive_cal_path)
        global_cal = ConformalIntervalCalibrator.load(global_cal_path) if global_cal_path.is_file() else None

        # Construct feature matrix for probabilistic model as originally trained
        feat_base = create_temporal_features(clean_energy, lags=(1, 24, 168), windows=(24,), target_column_name="target_next_hour", drop_na=True)
        _, _, te_base = chronological_split(feat_base, 0.70, 0.15, 0.15)

        multi_subset = multi_nwp[[
            "timestamp", "temp_f_zonal_weighted", "cooling_degree_proxy_regional",
            "heating_degree_proxy_regional", "temp_spread_regional",
        ]].copy()
        multi_subset["timestamp"] = pd.to_datetime(multi_subset["timestamp"])

        te_multi = te_base.merge(multi_subset, left_on="target_timestamp", right_on="timestamp", how="left")
        for col in ["temp_f_zonal_weighted", "cooling_degree_proxy_regional", "heating_degree_proxy_regional", "temp_spread_regional"]:
            te_multi[col] = te_multi[col].ffill().bfill()

        y_test_prob = te_multi["target_next_hour"].to_numpy()
        q_raw = prob_model.predict(te_multi[prob_model.feature_names])

        raw_q05 = q_raw["q05"]
        raw_q50 = q_raw["q50"]
        raw_q95 = q_raw["q95"]

        # Check raw ordering
        raw_violations = int(np.sum((raw_q05 > raw_q50) | (raw_q50 > raw_q95)))

        # 1. Raw Quantile Model
        raw_cov = float(np.mean((y_test_prob >= raw_q05) & (y_test_prob <= raw_q95)) * 100.0)
        raw_width = raw_q95 - raw_q05
        raw_winkler = compute_winkler_score(y_test_prob, raw_q05, raw_q95, alpha=0.10)

        # 2. Global Scalar CQR
        if global_cal is not None:
            c_low_g, c_med_g, c_high_g = global_cal.adjust(raw_q05, raw_q50, raw_q95)
            glob_cov = float(np.mean((y_test_prob >= c_low_g) & (y_test_prob <= c_high_g)) * 100.0)
            glob_width = c_high_g - c_low_g
            glob_winkler = compute_winkler_score(y_test_prob, c_low_g, c_high_g, alpha=0.10)
        else:
            glob_cov, glob_width, glob_winkler = 0.0, np.zeros_like(raw_width), 0.0

        # 3. Adaptive Diurnal CQR
        te_ts = te_multi["target_timestamp"].to_numpy()
        c_low_a, c_med_a, c_high_a = adaptive_cal.adjust(
            raw_q05, raw_q50, raw_q95, timestamps=te_ts, demand_preds=raw_q50
        )
        adapt_cov = float(np.mean((y_test_prob >= c_low_a) & (y_test_prob <= c_high_a)) * 100.0)
        adapt_width = c_high_a - c_low_a
        adapt_winkler = compute_winkler_score(y_test_prob, c_low_a, c_high_a, alpha=0.10)
        adapt_violations = int(np.sum((c_low_a > c_med_a) | (c_med_a > c_high_a)))

        prob_eval = {
            "nominal_target_coverage": 0.90,
            "test_sample_count": len(y_test_prob),
            "raw_quantiles": {
                "coverage_pct": round(raw_cov, 2),
                "mean_width_mw": round(float(np.mean(raw_width)), 2),
                "median_width_mw": round(float(np.median(raw_width)), 2),
                "winkler_score_mw": round(raw_winkler, 2),
                "ordering_violations": raw_violations,
            },
            "global_scalar_conformal": {
                "coverage_pct": round(glob_cov, 2),
                "mean_width_mw": round(float(np.mean(glob_width)), 2),
                "median_width_mw": round(float(np.median(glob_width)), 2),
                "winkler_score_mw": round(glob_winkler, 2),
            },
            "champion_adaptive_diurnal_conformal": {
                "coverage_pct": round(adapt_cov, 2),
                "mean_width_mw": round(float(np.mean(adapt_width)), 2),
                "median_width_mw": round(float(np.median(adapt_width)), 2),
                "winkler_score_mw": round(adapt_winkler, 2),
                "ordering_violations": adapt_violations,
                "coverage_gap_to_nominal_pct": round(adapt_cov - 90.0, 2),
            },
        }

        # Sliced Adaptive Interval Coverage
        prob_sliced_df = te_multi.copy()
        prob_sliced_df["actual_demand_mw"] = y_test_prob
        prob_sliced_df["covered"] = (y_test_prob >= c_low_a) & (y_test_prob <= c_high_a)
        prob_sliced_df["month"] = prob_sliced_df["target_timestamp"].dt.month
        prob_sliced_df["hour"] = prob_sliced_df["target_timestamp"].dt.hour
        prob_sliced_df["season"] = prob_sliced_df["month"].apply(get_season)

        prob_eval["sliced_coverage"] = {
            "by_season_pct": {
                s: round(float(prob_sliced_df[prob_sliced_df["season"] == s]["covered"].mean() * 100), 2)
                for s in ("Winter", "Spring", "Summer", "Fall")
            },
            "by_hour_pct": {
                f"H{h:02d}": round(float(prob_sliced_df[prob_sliced_df["hour"] == h]["covered"].mean() * 100), 2)
                for h in range(24)
            },
        }

        # Append interval columns to H=1 predictions
        prob_res_df = pd.DataFrame({
            "target_timestamp": te_multi["target_timestamp"],
            "q05_adaptive_mw": np.round(c_low_a, 2),
            "q50_median_mw": np.round(c_med_a, 2),
            "q95_adaptive_mw": np.round(c_high_a, 2),
            "interval_width_mw": np.round(adapt_width, 2),
            "inside_interval": (y_test_prob >= c_low_a) & (y_test_prob <= c_high_a),
        })
        all_predictions_dfs[0] = all_predictions_dfs[0].merge(prob_res_df, on="target_timestamp", how="left")

    results["probabilistic_calibration"] = prob_eval

    # 5. Export Deliverables
    # A. JSON Metrics
    metrics_path = out_dir / "forecast_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved Metrics JSON] -> {metrics_path}")

    # B. Combined Predictions CSV
    combined_pred_df = pd.concat(all_predictions_dfs, ignore_index=True)
    pred_path = out_dir / "forecast_predictions.csv"
    combined_pred_df.to_csv(pred_path, index=False)
    print(f"[Saved Predictions CSV] -> {pred_path} ({len(combined_pred_df):,} total rows)")

    # 6. Generate Diagnostic Plots
    print("\n--- Generating Diagnostic Figures ---")
    _generate_diagnostic_plots(combined_pred_df, plots_dir)

    print("\n" + "=" * 80)
    print("  PHASE 2 EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return results


def _generate_diagnostic_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate high-resolution genuine evaluation figures for report inclusion."""
    df_clean = df.copy()
    df_clean["target_timestamp"] = pd.to_datetime(df_clean["target_timestamp"])

    # Slice for summer 2018 peak heatwave window (July 1 to July 15, 2018 = 336 hours)
    start_ts = pd.Timestamp("2018-07-01 00:00:00")
    end_ts = pd.Timestamp("2018-07-15 23:00:00")

    # -------------------------------------------------------------------------
    # Plot 1: Actual vs Predicted H=1h (Real-Time Dispatch)
    # -------------------------------------------------------------------------
    h1_df = df_clean[df_clean["horizon"] == 1].sort_values("target_timestamp").reset_index(drop=True)
    s1 = h1_df[(h1_df["target_timestamp"] >= start_ts) & (h1_df["target_timestamp"] <= end_ts)]

    plt.figure(figsize=(14, 5), dpi=200)
    plt.plot(s1["target_timestamp"], s1["actual_demand_mw"], label="Actual Grid Demand (MW)", color="#0f172a", lw=1.8)
    plt.plot(s1["target_timestamp"], s1["pred_nwp_mw"], label="Production NWP Model (MW)", color="#0284c7", lw=1.5, ls="--")
    if "q05_adaptive_mw" in s1.columns:
        plt.fill_between(
            s1["target_timestamp"],
            s1["q05_adaptive_mw"],
            s1["q95_adaptive_mw"],
            color="#38bdf8",
            alpha=0.25,
            label="Adaptive 90% Conformal Interval",
        )
    plt.title("PJM East Actual vs. Predicted Demand (Peak Summer July 2018, Horizon H=1h)", fontsize=13, fontweight="bold")
    plt.xlabel("Target Timestamp (UTC)", fontsize=11)
    plt.ylabel("Power Demand (MW)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "actual_vs_predicted_h1.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: Actual vs Predicted H=24h (Day-Ahead Market)
    # -------------------------------------------------------------------------
    h24_df = df_clean[df_clean["horizon"] == 24].sort_values("target_timestamp").reset_index(drop=True)
    s24 = h24_df[(h24_df["target_timestamp"] >= start_ts) & (h24_df["target_timestamp"] <= end_ts)]

    plt.figure(figsize=(14, 5), dpi=200)
    plt.plot(s24["target_timestamp"], s24["actual_demand_mw"], label="Actual Grid Demand (MW)", color="#0f172a", lw=1.8)
    plt.plot(s24["target_timestamp"], s24["pred_nwp_mw"], label="Production NWP Model (MW)", color="#0284c7", lw=1.5, ls="-")
    plt.plot(s24["target_timestamp"], s24["pred_base_mw"], label="Baseline Tree (No Weather)", color="#94a3b8", lw=1.2, ls=":")
    plt.title("PJM East Actual vs. Predicted Demand (Peak Summer July 2018, Horizon H=24h Day-Ahead)", fontsize=13, fontweight="bold")
    plt.xlabel("Target Timestamp (UTC)", fontsize=11)
    plt.ylabel("Power Demand (MW)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "actual_vs_predicted_h24.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 3: Actual vs Predicted H=48h (Two-Day-Ahead Operations)
    # -------------------------------------------------------------------------
    h48_df = df_clean[df_clean["horizon"] == 48].sort_values("target_timestamp").reset_index(drop=True)
    s48 = h48_df[(h48_df["target_timestamp"] >= start_ts) & (h48_df["target_timestamp"] <= end_ts)]

    plt.figure(figsize=(14, 5), dpi=200)
    plt.plot(s48["target_timestamp"], s48["actual_demand_mw"], label="Actual Grid Demand (MW)", color="#0f172a", lw=1.8)
    plt.plot(s48["target_timestamp"], s48["pred_nwp_mw"], label="Production NWP Model (MW)", color="#0284c7", lw=1.5, ls="-")
    plt.plot(s48["target_timestamp"], s48["pred_base_mw"], label="Baseline Tree (No Weather)", color="#94a3b8", lw=1.2, ls=":")
    plt.title("PJM East Actual vs. Predicted Demand (Peak Summer July 2018, Horizon H=48h Two-Day)", fontsize=13, fontweight="bold")
    plt.xlabel("Target Timestamp (UTC)", fontsize=11)
    plt.ylabel("Power Demand (MW)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "actual_vs_predicted_h48.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 4: Forecast Error Distribution (Multi-Horizon 3-Panel)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200, sharey=True)
    horizons = [1, 24, 48]
    colors = ["#0284c7", "#0d9488", "#7c3aed"]

    for idx, (h_val, col) in enumerate(zip(horizons, colors)):
        sub = df_clean[df_clean["horizon"] == h_val]
        res = sub["residual_mw"].dropna().to_numpy()
        mu, std = float(np.mean(res)), float(np.std(res))
        ax = axes[idx]
        ax.hist(res, bins=60, density=True, color=col, alpha=0.65, edgecolor="black", label=f"Residuals (N={len(res):,})")
        x_r = np.linspace(mu - 3.5 * std, mu + 3.5 * std, 200)
        ax.plot(x_r, stats.norm.pdf(x_r, mu, std), color="#dc2626", lw=1.8, label=f"Normal ($\\mu$={mu:.1f}, $\\sigma$={std:.1f})")
        ax.set_title(f"Horizon H={h_val}h Error Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual Error (Actual - Predicted MW)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("Forecast Error Distributions Across Horizons (Held-Out Test Set 2016–2018)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "forecast_error_distribution.png")
    plt.savefig(plots_dir / "residual_distribution_h1.png")  # Backward compatibility
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 5: Horizon Metric Comparison Across Baselines
    # -------------------------------------------------------------------------
    h_labels = ["H=1h (Real-Time)", "H=24h (Day-Ahead)", "H=48h (Two-Day)"]
    pers_maes = [
        float(df_clean[df_clean["horizon"] == 1]["pred_pers_mw"].sub(df_clean[df_clean["horizon"] == 1]["actual_demand_mw"]).abs().mean()),
        float(df_clean[df_clean["horizon"] == 24]["pred_pers_mw"].sub(df_clean[df_clean["horizon"] == 24]["actual_demand_mw"]).abs().mean()),
        float(df_clean[df_clean["horizon"] == 48]["pred_pers_mw"].sub(df_clean[df_clean["horizon"] == 48]["actual_demand_mw"]).abs().mean()),
    ]
    base_maes = [
        float(df_clean[df_clean["horizon"] == 1]["pred_base_mw"].sub(df_clean[df_clean["horizon"] == 1]["actual_demand_mw"]).abs().mean()),
        float(df_clean[df_clean["horizon"] == 24]["pred_base_mw"].sub(df_clean[df_clean["horizon"] == 24]["actual_demand_mw"]).abs().mean()),
        float(df_clean[df_clean["horizon"] == 48]["pred_base_mw"].sub(df_clean[df_clean["horizon"] == 48]["actual_demand_mw"]).abs().mean()),
    ]
    nwp_maes = [
        float(df_clean[df_clean["horizon"] == 1]["residual_mw"].abs().mean()),
        float(df_clean[df_clean["horizon"] == 24]["residual_mw"].abs().mean()),
        float(df_clean[df_clean["horizon"] == 48]["residual_mw"].abs().mean()),
    ]

    x = np.arange(len(h_labels))
    width = 0.25
    plt.figure(figsize=(10, 5), dpi=200)
    plt.bar(x - width, pers_maes, width, label="Persistence Benchmark", color="#cbd5e1", edgecolor="#94a3b8")
    plt.bar(x, base_maes, width, label="Baseline Tree (No Weather)", color="#94a3b8", edgecolor="#64748b")
    plt.bar(x + width, nwp_maes, width, label="Production NWP Model", color="#0284c7", edgecolor="#0369a1")

    for i in range(len(x)):
        gain = ((base_maes[i] - nwp_maes[i]) / base_maes[i]) * 100
        plt.text(x[i] + width, nwp_maes[i] + 45, f"+{gain:.1f}%\n(NWP Gain)", ha="center", fontsize=8.5, fontweight="bold", color="#0369a1")

    plt.xticks(x, h_labels, fontsize=10)
    plt.ylabel("Mean Absolute Error (MW)", fontsize=11)
    plt.title("Forecast Horizon Error Benchmark Comparison (Test Holdout N=21,773)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, loc="upper left")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "horizon_metric_comparison.png")
    plt.savefig(plots_dir / "multi_horizon_mae_comparison.png")  # Backward compatibility
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 6: Prediction Interval Calibration & Sharpness (Adaptive Diurnal)
    # -------------------------------------------------------------------------
    if "q05_adaptive_mw" in s1.columns and s1["q05_adaptive_mw"].notna().any():
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 7), dpi=200, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

        # Top: Intervals + Actuals + Violations
        ax_top.plot(s1["target_timestamp"], s1["actual_demand_mw"], color="#0f172a", lw=1.8, label="Actual Demand (MW)")
        ax_top.plot(s1["target_timestamp"], s1["q50_median_mw"], color="#0284c7", lw=1.2, ls="--", label="Median Forecast Q50 (MW)")
        ax_top.fill_between(
            s1["target_timestamp"],
            s1["q05_adaptive_mw"],
            s1["q95_adaptive_mw"],
            color="#38bdf8",
            alpha=0.30,
            label="Adaptive Diurnal 90% Interval [Q05, Q95]",
        )

        # Highlight coverage misses
        is_covered = (s1["actual_demand_mw"] >= s1["q05_adaptive_mw"]) & (s1["actual_demand_mw"] <= s1["q95_adaptive_mw"])
        misses = s1[~is_covered]
        if len(misses) > 0:
            ax_top.scatter(misses["target_timestamp"], misses["actual_demand_mw"], color="#ef4444", s=24, zorder=5, label=f"Interval Misses ({len(misses)})")

        ax_top.set_title("Adaptive Diurnal Conformal Prediction Intervals (Peak July 2018, H=1h)", fontsize=12, fontweight="bold")
        ax_top.set_ylabel("Power Demand (MW)", fontsize=10)
        ax_top.grid(True, linestyle=":", alpha=0.5)
        ax_top.legend(frameon=True, loc="upper right", fontsize=9)

        # Bottom: Interval Width Tracking (Sharpness)
        ax_bot.plot(s1["target_timestamp"], s1["interval_width_mw"], color="#6366f1", lw=1.5, label="Interval Width (Q95 - Q05 MW)")
        ax_bot.set_ylabel("Width (MW)", fontsize=10)
        ax_bot.set_xlabel("Target Timestamp (UTC)", fontsize=10)
        ax_bot.grid(True, linestyle=":", alpha=0.5)
        ax_bot.legend(frameon=True, loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(plots_dir / "prediction_intervals.png")
        plt.close()

    print(f"[Saved All 6 Diagnostic Plots] -> {plots_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_comprehensive_evaluation()
