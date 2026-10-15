"""Unit tests for the data package (loader, validator, and preprocessor)."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.loader import load_energy_data
from src.data.validator import validate_energy_data, ValidationReport
from src.data.preprocessor import preprocess_energy_data, save_cleaned_data, CleaningReport
from src.config import DEFAULT_DATA_PATH


# ---------------------------------------------------------------------------
# Loader Tests
# ---------------------------------------------------------------------------


def test_load_energy_data_standard_columns(tmp_path: Path) -> None:
    """Test 1: Successful loading with standard column names (timestamp, consumption)."""
    csv_file = tmp_path / "standard.csv"
    csv_file.write_text("timestamp,consumption\n2026-01-01 00:00:00,150.5\n2026-01-01 01:00:00,142.3\n")

    df = load_energy_data(csv_file)
    assert list(df.columns) == ["timestamp", "consumption"]
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_numeric_dtype(df["consumption"])
    assert df["consumption"].iloc[0] == 150.5


def test_load_energy_data_supported_aliases(tmp_path: Path) -> None:
    """Test 2: Successful loading with supported column aliases (e.g., Datetime, PJME_MW)."""
    csv_file = tmp_path / "pjm_format.csv"
    csv_file.write_text("Datetime,PJME_MW\n2026-01-01 00:00:00,26404.0\n2026-01-01 01:00:00,25000.0\n")

    df = load_energy_data(csv_file)
    assert "timestamp" in df.columns
    assert "consumption" in df.columns
    assert len(df) == 2
    assert df["consumption"].iloc[0] == 26404.0


def test_load_energy_data_missing_file(tmp_path: Path) -> None:
    """Test 3: Missing input file raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_energy_data(non_existent)


def test_load_energy_data_missing_required_columns(tmp_path: Path) -> None:
    """Test 4: Missing required columns raises ValueError."""
    csv_file = tmp_path / "invalid_columns.csv"
    csv_file.write_text("unrelated_col,another_col\n1,2\n")

    with pytest.raises(ValueError, match="Could not identify a timestamp column"):
        load_energy_data(csv_file)


def test_load_energy_data_empty_csv(tmp_path: Path) -> None:
    """Test 5: Empty CSV handling raises ValueError."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")

    with pytest.raises(ValueError, match="empty"):
        load_energy_data(csv_file)


def test_load_energy_data_invalid_timestamps(tmp_path: Path) -> None:
    """Test 6: Invalid timestamps are coerced to NaT rather than silently dropped."""
    csv_file = tmp_path / "invalid_ts.csv"
    csv_file.write_text("timestamp,consumption\nnot-a-date,100.0\n2026-01-01 01:00:00,105.0\n")

    df = load_energy_data(csv_file)
    assert len(df) == 2
    assert pd.isna(df["timestamp"].iloc[0])
    assert pd.notna(df["timestamp"].iloc[1])


def test_load_energy_data_non_numeric_consumption(tmp_path: Path) -> None:
    """Test 7: Non-numeric consumption values are coerced to NaN rather than silently dropped."""
    csv_file = tmp_path / "non_numeric.csv"
    csv_file.write_text("timestamp,consumption\n2026-01-01 00:00:00,invalid_number\n2026-01-01 01:00:00,105.0\n")

    df = load_energy_data(csv_file)
    assert len(df) == 2
    assert pd.isna(df["consumption"].iloc[0])
    assert df["consumption"].iloc[1] == 105.0


# ---------------------------------------------------------------------------
# Validator Tests
# ---------------------------------------------------------------------------


def test_validator_missing_consumption() -> None:
    """Test 8: Missing consumption values detection."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
        "consumption": [100.0, None, 110.0],
    })
    report = validate_energy_data(df)
    assert isinstance(report, ValidationReport)
    assert report.total_rows == 3
    assert report.missing_consumption == 1
    assert any("missing consumption" in w for w in report.warnings)


def test_validator_duplicate_timestamps() -> None:
    """Test 9: Duplicate timestamps detection."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 01:00"]),
        "consumption": [100.0, 105.0, 110.0],
    })
    report = validate_energy_data(df)
    assert report.duplicate_timestamps == 1
    assert any("duplicate timestamp" in w for w in report.warnings)


def test_validator_unsorted_timestamps() -> None:
    """Test 10: Unsorted timestamps detection."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 02:00", "2026-01-01 00:00", "2026-01-01 01:00"]),
        "consumption": [100.0, 105.0, 110.0],
    })
    report = validate_energy_data(df)
    assert not report.is_monotonic_increasing
    assert any("not strictly monotonically increasing" in w for w in report.warnings)


def test_validator_expected_frequency_gap_detection() -> None:
    """Test 11: Expected-frequency gap detection when gaps exist."""
    # 00:00 and 02:00 (01:00 missing in hourly freq)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 02:00"]),
        "consumption": [100.0, 120.0],
    })
    report = validate_energy_data(df, expected_frequency="h")
    assert report.has_gaps is True
    assert report.detected_gaps_count == 1
    assert any("missing interval step" in w for w in report.warnings)


def test_validator_no_gap_assertion_without_frequency() -> None:
    """Test 12: No gap assertion when expected_frequency is omitted."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 05:00"]),
        "consumption": [100.0, 120.0],
    })
    report = validate_energy_data(df, expected_frequency=None)
    assert report.has_gaps is None
    assert report.detected_gaps_count == 0


# ---------------------------------------------------------------------------
# Preprocessor Tests
# ---------------------------------------------------------------------------


def test_preprocessor_explicit_duplicate_policies() -> None:
    """Test 13: Explicit duplicate policies (keep_first, keep_last, error)."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00"]),
        "consumption": [100.0, 200.0],
    })

    # Policy: keep_first
    df_first, rep_first = preprocess_energy_data(df, duplicate_policy="keep_first")
    assert len(df_first) == 1
    assert df_first["consumption"].iloc[0] == 100.0
    assert rep_first.removed_duplicates == 1

    # Policy: keep_last
    df_last, rep_last = preprocess_energy_data(df, duplicate_policy="keep_last")
    assert len(df_last) == 1
    assert df_last["consumption"].iloc[0] == 200.0
    assert rep_last.removed_duplicates == 1

    # Policy: mean (aggregates repeated clock hours, e.g. DST transitions)
    df_mean, rep_mean = preprocess_energy_data(df, duplicate_policy="mean")
    assert len(df_mean) == 1
    assert df_mean["consumption"].iloc[0] == 150.0
    assert rep_mean.removed_duplicates == 1

    # Policy: error
    with pytest.raises(ValueError, match="Duplicate timestamps detected"):
        preprocess_energy_data(df, duplicate_policy="error")


def test_preprocessor_accurate_cleaning_row_counts() -> None:
    """Test 14: Accurate cleaning row counts in CleaningReport."""
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:00", "invalid_ts", "2026-01-01 01:00", "2026-01-01 01:00", "2026-01-01 02:00"],
        "consumption": [100.0, 105.0, None, 115.0, -10.0],
    })

    df_cleaned, report = preprocess_energy_data(
        df,
        duplicate_policy="keep_first",
        drop_missing_timestamps=True,
        drop_missing_consumption=True,
        filter_negative_consumption=True,
    )

    assert isinstance(report, CleaningReport)
    assert report.initial_rows == 5
    assert report.removed_missing_timestamps == 1
    assert report.removed_missing_consumption == 1
    assert report.removed_negative_consumption == 1
    assert report.removed_duplicates == 0  # Duplicate timestamp had None consumption, dropped before dup check
    assert report.final_rows == 2
    assert len(df_cleaned) == 2


def test_save_cleaned_data_to_temp_path(tmp_path: Path) -> None:
    """Test 15: Saving cleaned output to a temporary path without touching raw paths."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
        "consumption": [100.0, 110.0],
    })
    dest_path = tmp_path / "subfolder" / "cleaned_output.csv"
    saved = save_cleaned_data(df, output_path=dest_path)

    assert saved == dest_path.resolve()
    assert saved.is_file()
    loaded_saved = pd.read_csv(saved)
    assert len(loaded_saved) == 2


def test_preprocessor_does_not_mutate_original_dataframe() -> None:
    """Test 16: Confirming the original DataFrame is not unexpectedly mutated."""
    original_df = pd.DataFrame({
        "timestamp": ["2026-01-01 02:00", "2026-01-01 01:00"],
        "consumption": ["100.5", "200.5"],
    })
    copy_before = original_df.copy(deep=True)

    df_cleaned, _ = preprocess_energy_data(original_df)

    # Verify original_df has unchanged column values and types
    pd.testing.assert_frame_equal(original_df, copy_before)
    # Verify cleaned df is sorted and properly typed
    assert df_cleaned["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 01:00")
    assert pd.api.types.is_numeric_dtype(df_cleaned["consumption"])


def test_save_cleaned_data_refuses_overwriting_raw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Safety check: Refuses to overwrite DEFAULT_DATA_PATH."""
    fake_raw = tmp_path / "raw" / "energy.csv"
    fake_raw.parent.mkdir(parents=True, exist_ok=True)
    fake_raw.write_text("raw_content")

    monkeypatch.setattr("src.data.preprocessor.DEFAULT_DATA_PATH", fake_raw)

    df = pd.DataFrame({"timestamp": [1], "consumption": [2]})
    with pytest.raises(ValueError, match="Refusing to overwrite raw dataset"):
        save_cleaned_data(df, output_path=fake_raw)
