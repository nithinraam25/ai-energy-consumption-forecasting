"""Models package for AI-Powered Energy Consumption Forecasting.

Exports base forecaster interface, baseline models, and tree regressor.
"""

from src.models.base_model import BaseForecaster
from src.models.baseline import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from src.models.tree_models import EnergyTreeForecaster
from src.models.inference import (
    EnergyInferenceEngine,
    DEFAULT_PRODUCTION_MODEL_PATH,
    DEFAULT_FEATURE_SCHEMA_PATH,
)

__all__ = [
    "BaseForecaster",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "EnergyTreeForecaster",
    "tune_tree_model_on_validation",
    "DEFAULT_CANDIDATE_CONFIGS",
    "EnergyInferenceEngine",
    "DEFAULT_PRODUCTION_MODEL_PATH",
    "DEFAULT_FEATURE_SCHEMA_PATH",
]
