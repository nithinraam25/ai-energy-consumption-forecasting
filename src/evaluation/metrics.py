"""Evaluation metrics and chronological validation utilities.

Provides robust regression metrics (MAE, RMSE, R2, MAPE), chronological data splits,
and standardized multi-model comparison workflows.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.baseline import PersistenceForecaster, SeasonalNaiveForecaster
from src.models.tree_models import EnergyTreeForecaster


@dataclass(frozen=True)
class MetricResults:
    """Immutable evaluation metrics report for energy consumption forecasting."""

    mae: float
    rmse: float
    r2: float
    mape: float
    sample_count: int
    mape_excluded_count: int
    notes: tuple[str, ...]

    def to_dict(self, clean_nan: bool = False) -> dict[str, Any]:
        """Convert results to a standard dictionary.

        Args:
            clean_nan: If True, convert float NaN values to None for RFC 8259
                standards-compliant JSON serialization (null).
        """
        d = asdict(self)
        if clean_nan:
            return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}
        return d


def calculate_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    eps: float = 1e-5,
) -> MetricResults:
    """Calculate MAE, RMSE, R2, and MAPE with rigorous edge-case handling.

    EDGE-CASE POLICIES:
        1. Non-finite values: Raises ValueError if any NaN or Inf is passed.
        2. Empty inputs: Raises ValueError if arrays are empty.
        3. Length mismatch: Raises ValueError if lengths differ.
        4. Constant target series for R2:
           If y_true has zero variance (SS_tot == 0):
               - If y_pred equals y_true perfectly: R2 is set to 1.0.
               - Otherwise: R2 is set to float('nan') with an explicit note.
        5. Zero / Near-Zero Actuals for MAPE:
           Division by zero is prevented by excluding points where |y_true| <= eps.
           Excluded counts are recorded in mape_excluded_count.
           If all points are near zero, MAPE returns float('nan').

    Args:
        y_true: Ground truth observed consumption values.
        y_pred: Forecasted consumption values.
        eps: Small threshold below which actual values are considered zero for MAPE.

    Returns:
        MetricResults: Typed container of evaluated regression metrics.

    Raises:
        ValueError: On length mismatches, empty inputs, or non-finite numbers.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Array length mismatch: y_true has {len(y_t)} items, y_pred has {len(y_p)} items.")

    if len(y_t) == 0:
        raise ValueError("Cannot calculate metrics on empty input arrays.")

    if not np.isfinite(y_t).all() or not np.isfinite(y_p).all():
        raise ValueError("Input arrays contain non-finite numbers (NaN or Inf). Clean inputs before evaluation.")

    notes: list[str] = []

    # 1. MAE
    mae = float(mean_absolute_error(y_t, y_p))

    # 2. RMSE
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # 3. R2 with constant target handling
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum((y_t - y_p) ** 2))

    if np.isclose(ss_tot, 0.0, atol=1e-10):
        if np.isclose(ss_res, 0.0, atol=1e-10):
            r2 = 1.0
            notes.append("Constant target series predicted perfectly (R2=1.0).")
        else:
            r2 = float("nan")
            notes.append("Constant target series has non-zero residual; R2 is undefined (NaN).")
    else:
        r2 = float(r2_score(y_t, y_p))

    # 4. MAPE with zero-value exclusion policy
    non_zero_mask = np.abs(y_t) > eps
    mape_excluded = int((~non_zero_mask).sum())

    if mape_excluded == len(y_t):
        mape = float("nan")
        notes.append("All actual values are zero or near-zero; MAPE is undefined (NaN).")
    else:
        mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0)
        if mape_excluded > 0:
            notes.append(f"Excluded {mape_excluded} near-zero observation(s) from MAPE calculation.")

    return MetricResults(
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        sample_count=len(y_t),
        mape_excluded_count=mape_excluded,
        notes=tuple(notes),
    )


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into training, validation, and test partitions.

    LEAKAGE-SAFE TEMPORAL ISOLATION:
        Never shuffles rows. Strict ordering is maintained:
        train: [0 : N_train]
        val:   [N_train : N_train + N_val]
        test:  [N_train + N_val : N]

    Args:
        df: Input DataFrame containing chronological time series.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        timestamp_column: Column name of datetime index for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios do not sum to 1.0, df is empty, or partitions are empty.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(f"Ratios must sum to 1.0. Got: train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total_ratio}).")

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if timestamp_column in df.columns:
        if not df[timestamp_column].is_monotonic_increasing:
            raise ValueError("Input DataFrame timestamps must be monotonically increasing before chronological split.")

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Insufficient samples ({n}) to produce non-empty partitions. "
            f"Resulting counts: train={n_train}, val={n_val}, test={n_test}."
        )

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def compare_models(
    eval_df: pd.DataFrame,
    tree_model: EnergyTreeForecaster,
    feature_names: Sequence[str],
    target_column: str = "target_next_hour",
    timestamp_column: str = "timestamp",
    target_timestamp_column: str = "target_timestamp",
    historical_df: pd.DataFrame | None = None,
    save_metrics_path: Path | str | None = None,
    save_forecasts_path: Path | str | None = None,
    extra_models: Sequence[tuple[str, EnergyTreeForecaster]] | None = None,
) -> pd.DataFrame:
    """Compare Persistence, Seasonal-Naive, and Tree models on identical target timestamps.

    Strictly aligns all predictions to the identical timestamp subset before computing metrics.

    Args:
        eval_df: Evaluation feature matrix containing features, target, and timestamp columns.
        tree_model: Fitted EnergyTreeForecaster instance (primary baseline tree).
        feature_names: Predictor feature column names for the tree model.
        target_column: Name of true target column. Defaults to 'target_next_hour'.
        timestamp_column: Forecast origin timestamp column.
        target_timestamp_column: Target evaluation timestamp column.
        historical_df: Optional full historical DataFrame (including preceding train/val periods)
            allowing Seasonal-Naive and Persistence to look up observations prior to the eval split boundary.
        save_metrics_path: Optional path to save summary metrics CSV / JSON.
        save_forecasts_path: Optional path to save aligned predictions CSV.
        extra_models: Optional sequence of (model_name, fitted_model_instance) to evaluate
            alongside the baseline tree on the identical timestamp subset.

    Returns:
        pd.DataFrame: Metric comparison table across all models.

    Raises:
        ValueError: If required columns are missing or aligned set is empty.
    """
    if target_column not in eval_df.columns:
        raise ValueError(f"Target column '{target_column}' missing from evaluation DataFrame.")

    # 1. Clean valid evaluation rows
    clean_eval = eval_df.dropna(subset=[target_column, timestamp_column, target_timestamp_column]).copy()
    if clean_eval.empty:
        raise ValueError("No valid rows remaining in evaluation DataFrame after dropping target NaNs.")

    # Source history for baselines (allows lookup across train/val split boundaries)
    history_source = historical_df if historical_df is not None else clean_eval

    # 2. Tree model predictions
    tree_preds = tree_model.predict(clean_eval[list(feature_names)])
    clean_eval["pred_tree"] = tree_preds

    # Additional model predictions
    extra_pred_cols: list[str] = []
    if extra_models is not None:
        for extra_name, extra_model in extra_models:
            col_name = f"pred_{extra_name}"
            model_feats = list(extra_model.feature_names) if getattr(extra_model, "feature_names", None) else list(feature_names)
            clean_eval[col_name] = extra_model.predict(clean_eval[model_feats])
            extra_pred_cols.append(col_name)

    # 3. Persistence predictions
    pers_forecaster = PersistenceForecaster(missing_policy="exclude")
    pers_df = pers_forecaster.predict(
        history_source,
        target_timestamps=list(clean_eval[target_timestamp_column]),
        consumption_column="consumption",
        timestamp_column=timestamp_column,
    )
    clean_eval = clean_eval.merge(
        pers_df[["target_timestamp", "predicted_consumption"]].rename(columns={"predicted_consumption": "pred_persistence"}),
        on="target_timestamp",
        how="left",
    )

    # 4. Seasonal-Naive predictions (24-hour seasonal cycle)
    snaive_forecaster = SeasonalNaiveForecaster(season_hours=24, missing_policy="exclude")
    snaive_df = snaive_forecaster.predict(
        history_source,
        target_timestamps=list(clean_eval[target_timestamp_column]),
        consumption_column="consumption",
        timestamp_column=timestamp_column,
    )
    clean_eval = clean_eval.merge(
        snaive_df[["target_timestamp", "predicted_consumption"]].rename(columns={"predicted_consumption": "pred_seasonal_naive"}),
        on="target_timestamp",
        how="left",
    )

    # 5. Extract strictly aligned common evaluation subset
    subset_cols = ["pred_tree", "pred_persistence", "pred_seasonal_naive", target_column] + extra_pred_cols
    aligned = clean_eval.dropna(subset=subset_cols).copy()

    if aligned.empty:
        raise ValueError(
            "Aligned evaluation subset is empty. Seasonal-Naive required 24 hours of earlier history "
            "that was not available in the provided evaluation slice. Supply historical_df to enable cross-boundary lookups."
        )

    y_true = aligned[target_column].to_numpy()

    # Compute metrics for each model on the EXACT same sample subset
    metrics_summary: list[dict[str, Any]] = []
    models_to_eval = [
        ("Persistence", aligned["pred_persistence"].to_numpy()),
        ("SeasonalNaive_24h", aligned["pred_seasonal_naive"].to_numpy()),
        ("EnergyTree_HistGB", aligned["pred_tree"].to_numpy()),
    ]
    if extra_models is not None:
        for extra_name, _ in extra_models:
            models_to_eval.append((extra_name, aligned[f"pred_{extra_name}"].to_numpy()))

    for name, y_pred in models_to_eval:
        m = calculate_metrics(y_true, y_pred)
        metrics_summary.append({
            "model": name,
            "sample_count": m.sample_count,
            "mae": m.mae,
            "rmse": m.rmse,
            "r2": m.r2,
            "mape": m.mape,
            "mape_excluded_count": m.mape_excluded_count,
        })

    summary_df = pd.DataFrame(metrics_summary)

    # Save optional outputs
    if save_metrics_path is not None:
        p_met = Path(save_metrics_path)
        p_met.parent.mkdir(parents=True, exist_ok=True)
        if p_met.suffix.lower() == ".json":
            # RFC 8259 compliance: map NaN values to None so they serialize as valid JSON null
            json_safe_metrics = [
                {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
                for row in metrics_summary
            ]
            p_met.write_text(json.dumps(json_safe_metrics, indent=2))
        else:
            summary_df.to_csv(p_met, index=False)

    if save_forecasts_path is not None:
        p_fc = Path(save_forecasts_path)
        p_fc.parent.mkdir(parents=True, exist_ok=True)
        save_cols = [target_timestamp_column, target_column, "pred_persistence", "pred_seasonal_naive", "pred_tree"] + extra_pred_cols
        aligned[save_cols].to_csv(p_fc, index=False)

    return summary_df
