"""Unit tests for forecasting baselines, tree model, and chronological evaluation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import PersistenceForecaster, SeasonalNaiveForecaster
from src.models.tree_models import EnergyTreeForecaster
from src.evaluation.metrics import (
    calculate_metrics,
    chronological_split,
    compare_models,
    MetricResults,
)


@pytest.fixture
def synthetic_energy_df() -> pd.DataFrame:
    """Deterministic synthetic hourly energy dataset over 200 hours."""
    periods = 200
    ts = pd.date_range("2026-01-01 00:00:00", periods=periods, freq="h")
    # Diurnal wave + trend
    consumption = 100.0 + (np.arange(periods) * 0.1) + 20.0 * np.sin(2 * np.pi * ts.hour / 24.0)
    return pd.DataFrame({"timestamp": ts, "consumption": consumption})


# ---------------------------------------------------------------------------
# 1. Baseline Model Tests
# ---------------------------------------------------------------------------


def test_persistence_predictions() -> None:
    """Test 1 & 3: Correct persistence predictions and timestamp alignment."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:00", periods=4, freq="h"),
        "consumption": [100.0, 105.0, 110.0, 115.0],
    })
    forecaster = PersistenceForecaster()
    res = forecaster.predict(df)

    assert len(res) == 3
    # Target 01:00 should predict 100.0 (consumption from 00:00)
    assert res["target_timestamp"].iloc[0] == pd.Timestamp("2026-01-01 01:00")
    assert res["predicted_consumption"].iloc[0] == 100.0
    assert res["actual_consumption"].iloc[0] == 105.0

    # Target 03:00 should predict 110.0 (consumption from 02:00)
    assert res["target_timestamp"].iloc[2] == pd.Timestamp("2026-01-01 03:00")
    assert res["predicted_consumption"].iloc[2] == 110.0
    assert res["actual_consumption"].iloc[2] == 115.0


def test_seasonal_naive_predictions_hourly(synthetic_energy_df: pd.DataFrame) -> None:
    """Test 2 & 3: Correct daily seasonal-naive predictions on hourly series."""
    forecaster = SeasonalNaiveForecaster(season_hours=24)
    res = forecaster.predict(synthetic_energy_df)

    # First predicted target should be at 2026-01-02 00:00 (24h after start)
    first_row = res.iloc[0]
    assert first_row["target_timestamp"] == pd.Timestamp("2026-01-02 00:00:00")
    assert first_row["lookup_timestamp"] == pd.Timestamp("2026-01-01 00:00:00")
    # Predicted value should match 2026-01-01 00:00 consumption
    expected_pred = synthetic_energy_df.loc[synthetic_energy_df["timestamp"] == pd.Timestamp("2026-01-01 00:00:00"), "consumption"].iloc[0]
    assert first_row["predicted_consumption"] == expected_pred


def test_seasonal_naive_missing_history_handling() -> None:
    """Test 4: Behavior when historical observations are missing."""
    # Only 5 hours of data, asking for 24h seasonal naive prediction
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:00", periods=5, freq="h"),
        "consumption": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    # Policy 'exclude' -> returns NaN
    forecaster_exclude = SeasonalNaiveForecaster(season_hours=24, missing_policy="exclude")
    res_ex = forecaster_exclude.predict(df, target_timestamps=[pd.Timestamp("2026-01-01 04:00")])
    assert pd.isna(res_ex["predicted_consumption"].iloc[0])

    # Policy 'error' -> raises ValueError
    forecaster_err = SeasonalNaiveForecaster(season_hours=24, missing_policy="error")
    with pytest.raises(ValueError, match="Missing historical observation"):
        forecaster_err.predict(df, target_timestamps=[pd.Timestamp("2026-01-01 04:00")])


# ---------------------------------------------------------------------------
# 2. Chronological Split Tests
# ---------------------------------------------------------------------------


def test_chronological_split(synthetic_energy_df: pd.DataFrame) -> None:
    """Test 5 & 6: Correct chronological train/val/test split without shuffling."""
    train, val, test = chronological_split(synthetic_energy_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # 200 rows: 70% = 140, 15% = 30, 15% = 30
    assert len(train) == 140
    assert len(val) == 30
    assert len(test) == 30

    # Verify strict non-overlapping temporal continuity
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()

    # Verify no shuffling
    assert train["timestamp"].is_monotonic_increasing
    assert val["timestamp"].is_monotonic_increasing
    assert test["timestamp"].is_monotonic_increasing


def test_chronological_split_insufficient_samples() -> None:
    """Test 7: Insufficient samples or empty DataFrame raises ValueError."""
    small_df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")], "consumption": [10.0]})
    with pytest.raises(ValueError, match="Insufficient samples"):
        chronological_split(small_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    with pytest.raises(ValueError, match="empty DataFrame"):
        chronological_split(pd.DataFrame(), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


# ---------------------------------------------------------------------------
# 3. Metrics Calculation Tests
# ---------------------------------------------------------------------------


def test_calculate_metrics_values() -> None:
    """Test 8, 9, 10: Correct MAE, RMSE, and R2 calculation."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])
    # Errors: [2, -2, 3, -2] -> Absolute errors: [2, 2, 3, 2] -> MAE = 9/4 = 2.25
    # Squared errors: [4, 4, 9, 4] -> MSE = 21/4 = 5.25 -> RMSE = sqrt(5.25) ~ 2.291288
    metrics = calculate_metrics(y_true, y_pred)
    assert isinstance(metrics, MetricResults)
    assert np.isclose(metrics.mae, 2.25)
    assert np.isclose(metrics.rmse, np.sqrt(5.25))
    assert 0.9 < metrics.r2 <= 1.0


def test_calculate_metrics_constant_target_r2() -> None:
    """Test 11: Explicit handling of constant targets for R2."""
    y_true = np.array([50.0, 50.0, 50.0])
    # Exact prediction: R2 = 1.0
    metrics_exact = calculate_metrics(y_true, np.array([50.0, 50.0, 50.0]))
    assert metrics_exact.r2 == 1.0

    # Inexact prediction on constant target: R2 is undefined (NaN)
    metrics_inexact = calculate_metrics(y_true, np.array([51.0, 49.0, 50.0]))
    assert np.isnan(metrics_inexact.r2)
    assert any("constant" in note.lower() and "target" in note.lower() for note in metrics_inexact.notes)


def test_calculate_metrics_mape_with_zeros() -> None:
    """Test 12: Explicit MAPE behavior when actual values are zero."""
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([5.0, 110.0, 190.0])
    metrics = calculate_metrics(y_true, y_pred)
    # 0.0 is excluded from percentage calculation
    assert metrics.mape_excluded_count == 1
    # Residuals for 100 (10%) and 200 (5%) -> mean = 7.5%
    assert np.isclose(metrics.mape, 7.5)


def test_calculate_metrics_validation_errors() -> None:
    """Test 13, 14, 15: Rejection of mismatched lengths, empty arrays, non-finite values."""
    # Length mismatch
    with pytest.raises(ValueError, match="length mismatch"):
        calculate_metrics([1.0, 2.0], [1.0])

    # Empty inputs
    with pytest.raises(ValueError, match="empty input arrays"):
        calculate_metrics([], [])

    # Non-finite values
    with pytest.raises(ValueError, match="non-finite numbers"):
        calculate_metrics([1.0, np.nan], [1.0, 2.0])


# ---------------------------------------------------------------------------
# 4. Tree Model Tests
# ---------------------------------------------------------------------------


def test_tree_model_fitting_and_predictions(synthetic_energy_df: pd.DataFrame) -> None:
    """Test 16, 18: Tree-model fitting on synthetic data, output length and finiteness."""
    from src.features.temporal_features import create_temporal_features

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1, 24), windows=(12,), expected_frequency="h")
    clean_df = feat_df.dropna().reset_index(drop=True)

    features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_12"]
    model = EnergyTreeForecaster(feature_names=features, max_iter=20)

    X = clean_df[features]
    y = clean_df["target_next_hour"]

    model.fit(X, y)
    assert model.is_fitted

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert np.isfinite(preds).all()


def test_tree_model_feature_validation() -> None:
    """Test 17: Feature-column validation in DataFrame input."""
    model = EnergyTreeForecaster(feature_names=["lag_1", "lag_2"])
    X_missing = pd.DataFrame({"lag_1": [1.0, 2.0]})  # lag_2 missing

    with pytest.raises(ValueError, match="missing required feature columns"):
        model.fit(X_missing, np.array([1.0, 2.0]))


def test_tree_model_prohibits_target_in_features() -> None:
    """Test 21: Anti-leakage check: target column cannot be included in features."""
    with pytest.raises(ValueError, match="Anti-leakage violation"):
        EnergyTreeForecaster(feature_names=["lag_1", "target_next_hour"])

    with pytest.raises(ValueError, match="Anti-leakage violation"):
        EnergyTreeForecaster(feature_names=["target_timestamp", "hour"])


def test_tree_model_save_and_load(tmp_path: Path) -> None:
    """Test 19: Model save-and-load round trip with joblib."""
    X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [10.0, 20.0, 30.0, 40.0]})
    y = np.array([1.1, 2.1, 3.1, 4.1])

    model = EnergyTreeForecaster(feature_names=["feat_a", "feat_b"], max_iter=10)
    model.fit(X, y)
    original_preds = model.predict(X)

    save_dest = tmp_path / "models" / "test_tree.joblib"
    saved_path = model.save(save_dest)
    assert saved_path == save_dest.resolve()
    assert saved_path.is_file()

    loaded_model = EnergyTreeForecaster.load(save_dest)
    assert loaded_model.is_fitted
    assert loaded_model.feature_names == ("feat_a", "feat_b")
    loaded_preds = loaded_model.predict(X)

    np.testing.assert_allclose(original_preds, loaded_preds)


def test_training_partition_isolation(synthetic_energy_df: pd.DataFrame) -> None:
    """Test 20: Confirmation that fitting uses only the training partition."""
    from src.features.temporal_features import create_temporal_features

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1,), windows=(2,), drop_na=True)
    train, val, test = chronological_split(feat_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    features = ["hour", "lag_1", "rolling_mean_2"]
    model = EnergyTreeForecaster(feature_names=features, max_iter=10)

    # Fit strictly on train
    model.fit(train[features], train["target_next_hour"])
    assert model.metadata_["n_samples_fit"] == len(train)
    assert model.metadata_["n_samples_fit"] < len(feat_df)

    # Predict on test
    test_preds = model.predict(test[features])
    assert len(test_preds) == len(test)


# ---------------------------------------------------------------------------
# 5. Model Comparison Workflow Test
# ---------------------------------------------------------------------------


def test_compare_models_workflow(synthetic_energy_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test: Reusable comparison workflow on identical test timestamps."""
    from src.features.temporal_features import create_temporal_features

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1, 24), windows=(24,), expected_frequency="h")
    clean_df = feat_df.dropna().reset_index(drop=True)

    features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_24"]
    train, val, test = chronological_split(clean_df, 0.7, 0.15, 0.15)

    model = EnergyTreeForecaster(feature_names=features, max_iter=20)
    model.fit(train[features], train["target_next_hour"])

    metrics_csv = tmp_path / "metrics.csv"
    forecasts_csv = tmp_path / "forecasts.csv"

    comp_df = compare_models(
        eval_df=test,
        tree_model=model,
        feature_names=features,
        target_column="target_next_hour",
        save_metrics_path=metrics_csv,
        save_forecasts_path=forecasts_csv,
    )

    assert len(comp_df) == 3
    assert set(comp_df["model"]) == {"Persistence", "SeasonalNaive_24h", "EnergyTree_HistGB"}
    assert metrics_csv.is_file()
    assert forecasts_csv.is_file()


def test_persistence_duplicate_and_gap_handling() -> None:
    """Verify PersistenceForecaster rejects duplicates and handles gaps explicitly."""
    # Duplicate timestamps
    dup_df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 00:00")],
        "consumption": [10.0, 20.0],
    })
    forecaster = PersistenceForecaster()
    with pytest.raises(ValueError, match="Duplicate timestamps detected in PersistenceForecaster"):
        forecaster.predict(dup_df)

    # Missing anchor with missing_policy='error'
    gap_df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 05:00")],
        "consumption": [10.0, 20.0],
    })
    forecaster_err = PersistenceForecaster(step_hours=1, missing_policy="error")
    with pytest.raises(ValueError, match="Missing historical observation for persistence lookup"):
        forecaster_err.predict(gap_df, target_timestamps=[pd.Timestamp("2026-01-01 05:00")])


def test_seasonal_naive_duplicate_handling() -> None:
    """Verify SeasonalNaiveForecaster rejects duplicate timestamps."""
    dup_df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 00:00")],
        "consumption": [10.0, 20.0],
    })
    forecaster = SeasonalNaiveForecaster(season_hours=24)
    with pytest.raises(ValueError, match="Duplicate timestamps detected in SeasonalNaiveForecaster"):
        forecaster.predict(dup_df)


def test_compare_models_json_export_null_for_nan(synthetic_energy_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify JSON export converts NaN values to valid RFC 8259 null."""
    import json
    from src.features.temporal_features import create_temporal_features

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1, 24), windows=(24,), expected_frequency="h", drop_na=True)
    features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_24"]
    train, val, test = chronological_split(feat_df, 0.7, 0.15, 0.15)

    model = EnergyTreeForecaster(feature_names=features, max_iter=10)
    model.fit(train[features], train["target_next_hour"])

    json_path = tmp_path / "metrics.json"
    _ = compare_models(
        eval_df=test,
        tree_model=model,
        feature_names=features,
        target_column="target_next_hour",
        save_metrics_path=json_path,
        historical_df=feat_df,
    )

    assert json_path.is_file()
    # Ensure JSON loads with strict RFC parser and no literal NaN string
    raw_text = json_path.read_text()
    assert "NaN" not in raw_text, "Found invalid literal 'NaN' in JSON export; expected 'null'"
    parsed = json.loads(raw_text)
    assert len(parsed) == 3


def test_compare_models_cross_boundary_history(synthetic_energy_df: pd.DataFrame) -> None:
    """Verify compare_models resolves cross-boundary history using external historical_df."""
    from src.features.temporal_features import create_temporal_features

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1, 24), windows=(24,), drop_na=True)
    features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_24"]
    train, val, test = chronological_split(feat_df, 0.7, 0.15, 0.15)

    model = EnergyTreeForecaster(feature_names=features, max_iter=10)
    model.fit(train[features], train["target_next_hour"])

    # Provide synthetic_energy_df as full history
    comp_df = compare_models(
        eval_df=test,
        tree_model=model,
        feature_names=features,
        target_column="target_next_hour",
        historical_df=synthetic_energy_df,
    )
    assert len(comp_df) == 3
    for count in comp_df["sample_count"]:
        assert count == len(test)


# ---------------------------------------------------------------------------
# 5. Model Selection & Tuning Tests (Phase 5A)
# ---------------------------------------------------------------------------


def test_tune_tree_model_on_validation(synthetic_energy_df: pd.DataFrame) -> None:
    """Verify validation-based hyperparameter tuning procedure and result ranking."""
    from src.features.temporal_features import create_temporal_features
    from src.models.model_selection import tune_tree_model_on_validation

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1, 24), windows=(24,), drop_na=True)
    features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_24"]
    train, val, test = chronological_split(feat_df, 0.7, 0.15, 0.15)

    mini_candidates = [
        {"name": "cand_a", "params": {"max_iter": 10, "learning_rate": 0.1, "max_depth": 4}},
        {"name": "cand_b", "params": {"max_iter": 20, "learning_rate": 0.05, "max_depth": 6}},
    ]

    result = tune_tree_model_on_validation(
        train_df=train,
        val_df=val,
        feature_names=features,
        target_column="target_next_hour",
        candidates=mini_candidates,
    )

    assert result.best_config is not None
    assert result.best_config["name"] in ("cand_a", "cand_b")
    assert len(result.results_df) == 2
    assert "val_mae" in result.results_df.columns
    assert "val_rmse" in result.results_df.columns
    assert "val_r2" in result.results_df.columns

    # Verify ranking: first row in results_df must have lowest MAE
    assert result.results_df.iloc[0]["val_mae"] <= result.results_df.iloc[1]["val_mae"]
    assert result.best_metrics.mae == result.results_df.iloc[0]["val_mae"]


def test_model_selection_isolation_no_test_leakage(synthetic_energy_df: pd.DataFrame) -> None:
    """Verify test partition isolation: model selection does not touch test data."""
    from src.features.temporal_features import create_temporal_features
    from src.models.model_selection import tune_tree_model_on_validation

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1, 24), windows=(24,), drop_na=True)
    features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_24"]
    train, val, test = chronological_split(feat_df, 0.7, 0.15, 0.15)

    candidates = [
        {"name": "c1", "params": {"max_iter": 5, "learning_rate": 0.1}},
    ]

    # Only train and val are supplied to tuning
    result = tune_tree_model_on_validation(
        train_df=train,
        val_df=val,
        feature_names=features,
        target_column="target_next_hour",
        candidates=candidates,
    )

    # Standard fitting uses train_df only
    selected_model = EnergyTreeForecaster(
        feature_names=features,
        **result.best_config["params"],
    )
    selected_model.fit(train[features], train["target_next_hour"])
    assert selected_model.is_fitted

    # Documented refit procedure uses train + val only (never test)
    train_val = pd.concat([train, val], ignore_index=True)
    refit_model = EnergyTreeForecaster(
        feature_names=features,
        **result.best_config["params"],
    )
    refit_model.fit(train_val[features], train_val["target_next_hour"])
    assert refit_model.is_fitted

    # Evaluate test holdout strictly after fitting
    test_preds = refit_model.predict(test[features])
    assert len(test_preds) == len(test)
    assert not test["timestamp"].isin(train_val["timestamp"]).any()


def test_compare_models_with_extra_models(synthetic_energy_df: pd.DataFrame) -> None:
    """Verify compare_models benchmark with extra tuned model variants."""
    from src.features.temporal_features import create_temporal_features

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1, 24), windows=(24,), drop_na=True)
    features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_24"]
    train, val, test = chronological_split(feat_df, 0.7, 0.15, 0.15)

    base_model = EnergyTreeForecaster(feature_names=features, max_iter=10)
    base_model.fit(train[features], train["target_next_hour"])

    extra_model = EnergyTreeForecaster(feature_names=features, max_iter=15, learning_rate=0.05)
    extra_model.fit(train[features], train["target_next_hour"])

    comp_df = compare_models(
        eval_df=test,
        tree_model=base_model,
        feature_names=features,
        target_column="target_next_hour",
        historical_df=synthetic_energy_df,
        extra_models=[("EnergyTree_Tuned_Test", extra_model)],
    )

    assert len(comp_df) == 4
    model_names = comp_df["model"].tolist()
    assert "Persistence" in model_names
    assert "SeasonalNaive_24h" in model_names
    assert "EnergyTree_HistGB" in model_names
    assert "EnergyTree_Tuned_Test" in model_names
    for count in comp_df["sample_count"]:
        assert count == len(test)


def test_tune_tree_model_rejects_empty_candidates(synthetic_energy_df: pd.DataFrame) -> None:
    """Verify tune_tree_model_on_validation raises ValueError on empty candidate list."""
    from src.features.temporal_features import create_temporal_features
    from src.models.model_selection import tune_tree_model_on_validation

    feat_df = create_temporal_features(synthetic_energy_df, lags=(1,), windows=(2,), drop_na=True)
    train, val, _ = chronological_split(feat_df, 0.7, 0.15, 0.15)

    with pytest.raises(ValueError, match="At least one candidate"):
        tune_tree_model_on_validation(
            train_df=train,
            val_df=val,
            feature_names=["hour", "lag_1"],
            target_column="target_next_hour",
            candidates=[],
        )
