import sys
import tempfile
from pathlib import Path

# Add project root to sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data.loader import load_energy_data
from src.data.validator import validate_energy_data
from src.data.preprocessor import preprocess_energy_data, save_cleaned_data


def run_smoke_test() -> None:
    """Execute end-to-end smoke test on a synthetic temporary CSV fixture."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_csv = tmp_path / "smoke_raw_energy.csv"

        # Raw fixture with known edge cases:
        # - Unsorted chronological order
        # - One duplicate timestamp (2026-01-01 01:00)
        # - One missing consumption value (None)
        # - One negative consumption value (-50.0)
        # - Hourly gap between 03:00 and 05:00 (04:00 missing)
        raw_content = (
            "Datetime,PJME_MW\n"
            "2026-01-01 03:00:00,13500.5\n"
            "2026-01-01 01:00:00,12000.0\n"
            "2026-01-01 01:00:00,12100.0\n"
            "2026-01-01 02:00:00,\n"
            "2026-01-01 05:00:00,-50.0\n"
            "2026-01-01 06:00:00,14000.0\n"
        )
        raw_csv.write_text(raw_content)

        # 1. Loader test
        df_loaded = load_energy_data(raw_csv)
        assert list(df_loaded.columns) == ["timestamp", "consumption"], "Column normalization failed"
        assert len(df_loaded) == 6, "Row count mismatch in loader"
        print("[Pass] 1. Loader normalized columns and parsed records.")

        # 2. Validator test
        val_report = validate_energy_data(df_loaded, expected_frequency="h", allow_negative_consumption=True)
        assert val_report.total_rows == 6
        assert val_report.missing_consumption == 1
        assert val_report.duplicate_timestamps == 1
        assert val_report.has_negative_consumption is True
        assert val_report.has_gaps is True
        assert val_report.detected_gaps_count >= 1  # 00:00, 04:00 gaps
        print(f"[Pass] 2. Validator identified: {val_report.duplicate_timestamps} duplicate(s), "
              f"{val_report.missing_consumption} missing consumption, {val_report.detected_gaps_count} gap(s).")

        # Snapshot before preprocessing to check immutability
        df_before = df_loaded.copy(deep=True)

        # 3 & 4. Preprocessor test (chronological output, duplicate and missing policies)
        df_cleaned, clean_report = preprocess_energy_data(
            df_loaded,
            duplicate_policy="keep_first",
            drop_missing_timestamps=True,
            drop_missing_consumption=True,
            filter_negative_consumption=True,
        )

        # Confirm original DataFrame was not mutated
        pd.testing.assert_frame_equal(df_loaded, df_before)
        print("[Pass] 3. Original DataFrame remained completely unchanged.")

        # Verify chronological sorting and cleaning results
        assert clean_report.initial_rows == 6
        assert clean_report.removed_missing_consumption == 1  # 02:00
        assert clean_report.removed_duplicates == 1           # 01:00 second entry
        assert clean_report.removed_negative_consumption == 1 # 05:00 (-50.0)
        assert clean_report.final_rows == 3                  # 01:00, 03:00, 06:00
        assert df_cleaned["timestamp"].is_monotonic_increasing, "Cleaned data is not chronologically sorted"
        assert (df_cleaned["consumption"] > 0).all(), "Negative consumption values were not removed"
        print(f"[Pass] 4. Preprocessor sanitized dataset ({clean_report.initial_rows} -> {clean_report.final_rows} rows).")

        # 5. Save cleaned output to temp directory and read back
        out_path = tmp_path / "cleaned_output.csv"
        saved_file = save_cleaned_data(df_cleaned, output_path=out_path)
        assert saved_file == out_path.resolve()
        assert saved_file.is_file()

        df_readback = pd.read_csv(saved_file)
        assert len(df_readback) == 3
        print(f"[Pass] 5. Cleaned output saved and successfully read back from: {saved_file.name}")


def run_features_smoke_test() -> None:
    """Execute end-to-end smoke test for Phase 3 feature engineering and sequences."""
    from src.features.temporal_features import create_temporal_features
    from src.features.sequence_builder import build_sliding_windows

    # Create contiguous 48-hour synthetic series
    ts = pd.date_range("2026-01-01 00:00:00", periods=48, freq="h")
    consumption = 50.0 + np.arange(48, dtype=float) + 5.0 * np.sin(2 * np.pi * ts.hour / 24.0)
    df = pd.DataFrame({"timestamp": ts, "consumption": consumption})

    # 1. Feature generation
    df_feat = create_temporal_features(df, lags=(1, 24), windows=(12,), expected_frequency="h")
    assert "hour_sin" in df_feat.columns
    assert "hour_cos" in df_feat.columns
    assert "lag_1" in df_feat.columns
    assert "lag_24" in df_feat.columns
    assert "rolling_mean_12" in df_feat.columns
    assert "target_next_hour" in df_feat.columns
    print(f"[Pass] 6. Feature pipeline generated {len(df_feat.columns)} columns across 48 records.")

    # 2. Sliding window generation
    dataset = build_sliding_windows(
        df,
        sequence_length=24,
        prediction_horizon=1,
        expected_frequency="h",
    )
    # 48 - 24 - 1 + 1 = 24 windows
    assert dataset.num_samples == 24
    assert dataset.sequence_length == 24
    assert dataset.X.shape == (24, 24, 1)
    assert dataset.y.shape == (24,)
    # Verify input window strictly ends 1 hour before target
    assert dataset.target_timestamps[0] == pd.Timestamp("2026-01-02 00:00:00")
    assert dataset.window_end_timestamps[0] == pd.Timestamp("2026-01-01 23:00:00")
    print(f"[Pass] 7. Sliding-window sequence generator produced tensor shape {dataset.X.shape} without leakage.")


def run_models_smoke_test() -> None:
    """Execute end-to-end smoke test for Phase 4 baselines, tree model, and metrics."""
    from src.features.temporal_features import create_temporal_features
    from src.models.baseline import PersistenceForecaster, SeasonalNaiveForecaster
    from src.models.tree_models import EnergyTreeForecaster
    from src.evaluation.metrics import calculate_metrics, chronological_split, compare_models

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Deterministic synthetic hourly series (100 hours)
        periods = 100
        ts = pd.date_range("2026-01-01 00:00:00", periods=periods, freq="h")
        consumption = 100.0 + (np.arange(periods) * 0.2) + 15.0 * np.sin(2 * np.pi * ts.hour / 24.0)
        raw_df = pd.DataFrame({"timestamp": ts, "consumption": consumption})

        # 2. Feature matrix
        feat_df = create_temporal_features(raw_df, lags=(1, 24), windows=(12,), expected_frequency="h", drop_na=True)
        features = ["hour", "hour_sin", "hour_cos", "lag_1", "lag_24", "rolling_mean_12"]

        # 3. Chronological split (70/15/15)
        train_df, val_df, test_df = chronological_split(feat_df, 0.70, 0.15, 0.15)
        assert len(train_df) + len(val_df) + len(test_df) == len(feat_df)
        print(f"[Pass] 8. Chronological split produced partitions: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}.")

        # 4. Tree model fitting
        model = EnergyTreeForecaster(feature_names=features, max_iter=25)
        model.fit(train_df[features], train_df["target_next_hour"])
        assert model.is_fitted
        print("[Pass] 9. EnergyTreeForecaster successfully fitted on training partition.")

        # 5. Serialization and reload round trip
        saved_model_path = model.save(tmp_path / "smoke_tree.joblib")
        assert saved_model_path.is_file()
        reloaded_model = EnergyTreeForecaster.load(saved_model_path)
        assert reloaded_model.is_fitted
        print(f"[Pass] 10. Model serialized and reloaded successfully from {saved_model_path.name}.")

        # 6. Evaluation metrics calculation
        test_preds = reloaded_model.predict(test_df[features])
        metrics = calculate_metrics(test_df["target_next_hour"], test_preds)
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.r2 <= 1.0
        print(f"[Pass] 11. Metrics evaluated: MAE={metrics.mae:.3f}, RMSE={metrics.rmse:.3f}, R2={metrics.r2:.3f}, MAPE={metrics.mape:.2f}%.")

        # 7. Model comparison workflow
        comp_df = compare_models(
            eval_df=test_df,
            tree_model=reloaded_model,
            feature_names=features,
            target_column="target_next_hour",
            historical_df=feat_df,
        )
        assert len(comp_df) == 3
        print(f"[Pass] 12. Model comparison executed across {len(comp_df)} models on identical test timestamps.")


def test_smoke_phase2() -> None:
    """Pytest hook for Phase 2 smoke test."""
    run_smoke_test()


def test_smoke_phase3() -> None:
    """Pytest hook for Phase 3 smoke test."""
    run_features_smoke_test()


def test_smoke_phase4() -> None:
    """Pytest hook for Phase 4 smoke test."""
    run_models_smoke_test()


if __name__ == "__main__":
    run_smoke_test()
    run_features_smoke_test()
    run_models_smoke_test()
    print("\nAll Phase 2, Phase 3, and Phase 4 smoke tests completed successfully!")

