"""Validation-based hyperparameter tuning and model selection for EnergyTreeForecaster.

Ensures strict chronological isolation: searches and tunes hyperparameters using
training and validation partitions only, keeping test holdouts completely untouched.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, NamedTuple, Sequence
import json
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.models.tree_models import EnergyTreeForecaster


class TuningResult(NamedTuple):
    """Container for model selection tuning outputs.

    Provides tuple unpacking (best_config, results_df) alongside structured attributes.
    """
    best_config: dict[str, Any]
    results_df: pd.DataFrame

    @property
    def summary_df(self) -> pd.DataFrame:
        """Alias for results_df."""
        return self.results_df

    @property
    def best_metrics(self) -> Any:
        """Metric results corresponding to the top-ranked candidate."""
        from src.evaluation.metrics import MetricResults
        row = self.results_df.iloc[0]
        return MetricResults(
            mae=float(row["val_mae"]),
            rmse=float(row["val_rmse"]),
            r2=float(row["val_r2"]),
            mape=float(row["val_mape"]),
            sample_count=int(row["sample_count"]),
            mape_excluded_count=int(row.get("mape_excluded_count", 0)),
            notes=("Selected validation configuration",),
        )

    @property
    def baseline_metrics(self) -> Any:
        """Metric results for the baseline configuration if present, or least optimal."""
        from src.evaluation.metrics import MetricResults
        base_rows = self.results_df[self.results_df["config_name"] == "config_0_baseline"]
        row = base_rows.iloc[0] if not base_rows.empty else self.results_df.iloc[-1]
        return MetricResults(
            mae=float(row["val_mae"]),
            rmse=float(row["val_rmse"]),
            r2=float(row["val_r2"]),
            mape=float(row["val_mape"]),
            sample_count=int(row["sample_count"]),
            mape_excluded_count=int(row.get("mape_excluded_count", 0)),
            notes=("Baseline reference configuration",),
        )


DEFAULT_CANDIDATE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "config_0_baseline",
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "min_samples_leaf": 20,
    },
    {
        "name": "config_1_l2_reg",
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "min_samples_leaf": 20,
    },
    {
        "name": "config_2_deeper_leaves",
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_depth": 8,
        "max_leaf_nodes": 45,
        "l2_regularization": 0.0,
        "min_samples_leaf": 20,
    },
    {
        "name": "config_3_lower_lr_more_trees",
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "min_samples_leaf": 20,
    },
    {
        "name": "config_4_lower_lr_deeper_leaves",
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": 8,
        "max_leaf_nodes": 45,
        "l2_regularization": 0.5,
        "min_samples_leaf": 20,
    },
    {
        "name": "config_5_conservative_leaf_size",
        "learning_rate": 0.08,
        "max_iter": 150,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.1,
        "min_samples_leaf": 30,
    },
)


def tune_tree_model_on_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: Sequence[str],
    candidate_configs: Sequence[dict[str, Any]] | None = None,
    target_column: str = "target_next_hour",
    metric_to_optimize: str = "mae",
    random_state: int = RANDOM_STATE,
    save_tuning_path: Path | str | None = None,
    candidates: Sequence[dict[str, Any]] | None = None,
) -> TuningResult:
    """Execute validation-based model selection over candidate tree configurations.

    STRICT ISOLATION CONTRACT:
        - Estimator candidates are fitted strictly on train_df[feature_names].
        - Evaluation is performed strictly on val_df[feature_names].
        - Test partition data is never accepted, passed, or examined.

    Args:
        train_df: Chronological training partition DataFrame.
        val_df: Chronological validation partition DataFrame.
        feature_names: Predictor column names.
        candidate_configs: Sequence of parameter dicts to evaluate. Defaults to DEFAULT_CANDIDATE_CONFIGS.
        target_column: Ground truth target column. Defaults to 'target_next_hour'.
        metric_to_optimize: Selection criterion ('mae', 'rmse', 'r2', 'mape'). Defaults to 'mae'.
        random_state: Seed for reproducibility.
        save_tuning_path: Optional path to save tuning results (CSV or JSON).
        candidates: Optional alias for candidate_configs.

    Returns:
        TuningResult: NamedTuple with best_config and results_df, supporting tuple unpacking.

    Raises:
        ValueError: If input partitions are empty or target/features missing, or candidates empty.
    """
    if train_df.empty or val_df.empty:
        raise ValueError("Cannot perform model selection on empty training or validation DataFrames.")

    if target_column not in train_df.columns or target_column not in val_df.columns:
        raise ValueError(f"Target column '{target_column}' missing from training or validation partition.")

    configs = candidate_configs if candidate_configs is not None else candidates
    configs_to_evaluate = configs if configs is not None else DEFAULT_CANDIDATE_CONFIGS

    if not configs_to_evaluate:
        raise ValueError("At least one candidate configuration must be provided.")

    results: list[dict[str, Any]] = []

    X_train = train_df[list(feature_names)]
    y_train = train_df[target_column]
    X_val = val_df[list(feature_names)]
    y_val = val_df[target_column]

    from src.evaluation.metrics import calculate_metrics

    for config in configs_to_evaluate:
        cfg = config.copy()
        cfg_name = cfg.pop("name", f"config_{len(results)}")
        # If parameters are nested under 'params', extract them
        params = cfg.pop("params", cfg)

        forecaster = EnergyTreeForecaster(
            feature_names=feature_names,
            random_state=random_state,
            **params,
        )
        forecaster.fit(X_train, y_train)
        y_pred = forecaster.predict(X_val)

        metrics = calculate_metrics(y_val, y_pred)

        record: dict[str, Any] = {
            "config_name": cfg_name,
            "val_mae": metrics.mae,
            "val_rmse": metrics.rmse,
            "val_r2": metrics.r2,
            "val_mape": metrics.mape,
            "sample_count": metrics.sample_count,
            "learning_rate": params.get("learning_rate", 0.1),
            "max_iter": params.get("max_iter", 100),
            "max_depth": params.get("max_depth", 6),
            "max_leaf_nodes": params.get("max_leaf_nodes", 31),
            "l2_regularization": params.get("l2_regularization", 0.0),
            "min_samples_leaf": params.get("min_samples_leaf", 20),
        }
        results.append(record)

    summary_df = pd.DataFrame(results)

    # Sort results
    ascending = metric_to_optimize.lower() in ("mae", "rmse", "mape", "val_mae", "val_rmse", "val_mape")
    sort_col = f"val_{metric_to_optimize}" if not metric_to_optimize.startswith("val_") else metric_to_optimize
    summary_df = summary_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

    best_record = summary_df.iloc[0].to_dict()

    # Reconstruct best config dictionary
    best_params = {
        "learning_rate": float(best_record["learning_rate"]),
        "max_iter": int(best_record["max_iter"]),
        "max_depth": int(best_record["max_depth"]) if pd.notna(best_record["max_depth"]) else None,
        "max_leaf_nodes": int(best_record["max_leaf_nodes"]) if pd.notna(best_record["max_leaf_nodes"]) else None,
        "l2_regularization": float(best_record["l2_regularization"]),
        "min_samples_leaf": int(best_record["min_samples_leaf"]),
    }
    best_config: dict[str, Any] = {
        "name": best_record["config_name"],
        "params": best_params,
        **best_params,
    }

    if save_tuning_path is not None:
        p = Path(save_tuning_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            # RFC 8259 compliance
            json_safe = [
                {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
                for row in results
            ]
            p.write_text(json.dumps(json_safe, indent=2))
        else:
            summary_df.to_csv(p, index=False)

    return TuningResult(best_config=best_config, results_df=summary_df)
