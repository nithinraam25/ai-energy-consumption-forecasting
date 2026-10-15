"""Tests for the centralized configuration module (src.config)."""

from pathlib import Path
import pytest

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SIMULATED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    FORECASTS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    LOGS_DIR,
    TESTS_DIR,
    DOCS_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    RANDOM_STATE,
    REQUIRED_DIRECTORIES,
    ensure_directories,
)


def test_project_root_is_absolute() -> None:
    """Verify PROJECT_ROOT is an absolute path."""
    assert PROJECT_ROOT.is_absolute(), f"Expected absolute path, got {PROJECT_ROOT}"


def test_project_root_contains_src_config() -> None:
    """Verify PROJECT_ROOT corresponds to the repository containing src/config.py."""
    config_file = PROJECT_ROOT / "src" / "config.py"
    assert config_file.is_file(), f"Expected {config_file} to exist as a file"


def test_configured_directory_paths_resolve_under_project_root() -> None:
    """Verify all configured directory paths resolve under PROJECT_ROOT."""
    for directory in REQUIRED_DIRECTORIES:
        assert directory.is_absolute(), f"{directory} is not absolute"
        assert PROJECT_ROOT in directory.parents or directory == PROJECT_ROOT, (
            f"Directory {directory} does not resolve under PROJECT_ROOT {PROJECT_ROOT}"
        )


def test_default_data_path() -> None:
    """Verify DEFAULT_DATA_PATH points to data/raw/energy.csv."""
    expected = RAW_DATA_DIR / "energy.csv"
    assert DEFAULT_DATA_PATH == expected
    assert DEFAULT_DATA_PATH == PROJECT_ROOT / "data" / "raw" / "energy.csv"


def test_default_model_path() -> None:
    """Verify DEFAULT_MODEL_PATH points to models/energy_forecaster.joblib."""
    expected = MODELS_DIR / "energy_forecaster.joblib"
    assert DEFAULT_MODEL_PATH == expected
    assert DEFAULT_MODEL_PATH == PROJECT_ROOT / "models" / "energy_forecaster.joblib"


def test_random_state_value() -> None:
    """Verify RANDOM_STATE equals 42."""
    assert RANDOM_STATE == 42


def test_ensure_directories_creates_all_required_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ensure_directories creates all required runtime directories."""
    # Test with monkeypatched temporary project root to verify creation logic in isolation
    fake_root = tmp_path / "fake_project"
    fake_dirs = (
        fake_root / "data" / "raw",
        fake_root / "data" / "processed",
        fake_root / "data" / "simulated",
        fake_root / "models",
        fake_root / "outputs" / "forecasts",
        fake_root / "outputs" / "metrics",
        fake_root / "outputs" / "plots",
        fake_root / "outputs" / "logs",
        fake_root / "tests",
        fake_root / "docs",
    )
    monkeypatch.setattr("src.config.REQUIRED_DIRECTORIES", fake_dirs)

    # Confirm directories do not exist yet
    for d in fake_dirs:
        assert not d.exists()

    # Call ensure_directories
    ensure_directories()

    # Confirm all were created
    for d in fake_dirs:
        assert d.is_dir()


def test_ensure_directories_idempotency() -> None:
    """Verify calling ensure_directories twice succeeds without error."""
    # First call
    ensure_directories()
    # Second call (must be safe and idempotent)
    ensure_directories()

    for directory in REQUIRED_DIRECTORIES:
        assert directory.is_dir()
