"""Centralized configuration module for AI-Powered Energy Consumption Forecasting.

All project paths are derived dynamically from the location of this file,
ensuring OS independence and eliminating reliance on current working directory.
"""

from pathlib import Path

# Project root derived relative to this file: <PROJECT_ROOT>/src/config.py -> parent.parent
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Core data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SIMULATED_DATA_DIR: Path = DATA_DIR / "simulated"

# Models and outputs directories
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FORECASTS_DIR: Path = OUTPUTS_DIR / "forecasts"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

# Tests and documentation directories
TESTS_DIR: Path = PROJECT_ROOT / "tests"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Default file paths
DEFAULT_DATA_PATH: Path = RAW_DATA_DIR / "energy.csv"
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "energy_forecaster.joblib"

# Reproducibility seed
RANDOM_STATE: int = 42

# List of runtime directories that need to exist for data pipelines, modeling, and output logging
REQUIRED_DIRECTORIES: tuple[Path, ...] = (
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
)


def ensure_directories() -> None:
    """Create all required runtime directories if they do not already exist.

    This function is idempotent and safe to call multiple times.
    Importing src.config does not execute this function automatically.
    """
    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
