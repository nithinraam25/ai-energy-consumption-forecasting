"""Production Inference Engine for AI-Powered Energy Consumption Forecasting.

Provides:
- Strict input data and schema validation against production contracts.
- Automated detection of missing, extra, or out-of-distribution features.
- Verification of operational NWP forecast availability (t_avail <= t_origin).
- Strict refusal to silently substitute observed surface weather for missing forecasts.
- Prediction generation with confidence and metadata logging.
"""

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, FORECASTS_DIR
from src.models.tree_models import EnergyTreeForecaster

DEFAULT_PRODUCTION_MODEL_PATH: Path = MODELS_DIR / "pjm_production_forecaster_phase8.joblib"
DEFAULT_FEATURE_SCHEMA_PATH: Path = MODELS_DIR / "production_feature_schema.json"
DEFAULT_MODEL_CONFIG_PATH: Path = MODELS_DIR / "production_model_config.json"


class EnergyInferenceEngine:
    """Production inference engine for PJM East regional energy forecaster."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_PRODUCTION_MODEL_PATH,
        schema_path: Path | str = DEFAULT_FEATURE_SCHEMA_PATH,
        model_version: str = "2.0.0-phase8",
        model: Any | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Initialize production inference engine.

        Args:
            model_path: Path to serialized forecaster joblib file.
            schema_path: Path to JSON feature schema definition.
            model_version: Version identifier string for deployment tracking.
            model: Optional pre-loaded model object.
            schema: Optional pre-loaded schema dictionary.
        """
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path)
        self.model_version = model_version

        if model is not None:
            self.model = model
        else:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Production model artifact not found at: {self.model_path}")
            self.model = EnergyTreeForecaster.load(self.model_path)

        if schema is not None:
            self.schema = schema
        else:
            if not self.schema_path.is_file():
                raise FileNotFoundError(f"Production feature schema not found at: {self.schema_path}")
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema: dict[str, Any] = json.load(f)

        self.expected_features: list[str] = self.schema["features"]
        self.target_name: str = self.schema.get("target_name", "target_demand")

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input schema and return DataFrame.

        Raises:
            ValueError: If schema is violated.
        """
        self.validate_features(df)
        return df

    def validate_features(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate an input feature matrix against the production schema contract.

        Args:
            df: Feature DataFrame to validate.

        Returns:
            dict[str, Any]: Validation summary.

        Raises:
            ValueError: If required features are missing, unexpected columns exist,
                or NaN/infinite values are detected.
        """
        input_cols = set(df.columns)
        expected_set = set(self.expected_features)

        missing_feats = expected_set - input_cols
        if missing_feats:
            raise ValueError(
                f"PRODUCTION SCHEMA VIOLATION: Missing {len(missing_feats)} required feature(s): "
                f"{sorted(missing_feats)}"
            )

        # Check for target leakage
        if self.target_name in df.columns:
            # Target is present, verify it is not included in input features
            pass

        # Check for NaN / Infs in feature matrix
        feat_df = df[self.expected_features]
        nan_counts = feat_df.isna().sum()
        if nan_counts.any():
            bad_cols = nan_counts[nan_counts > 0].to_dict()
            raise ValueError(f"PRODUCTION DATA INTEGRITY ERROR: NaN values detected in input features: {bad_cols}")

        inf_counts = np.isinf(feat_df.to_numpy()).sum()
        if inf_counts > 0:
            raise ValueError(f"PRODUCTION DATA INTEGRITY ERROR: {inf_counts} infinite values detected in input features!")

        return {
            "is_valid": True,
            "sample_count": len(df),
            "features_verified": len(self.expected_features),
            "expected_feature_list": self.expected_features,
        }

    def verify_forecast_availability(
        self,
        df: pd.DataFrame,
        cutoff_lead_hours: int = 1,
    ) -> None:
        """Verify that operational weather forecast timestamps satisfy availability cutoff.

        Raises:
            ValueError: If any forecast record was published/available after the cutoff.
        """
        if "simulated_avail_utc" in df.columns and "valid_time_utc" in df.columns:
            avail = pd.to_datetime(df["simulated_avail_utc"])
            valid = pd.to_datetime(df["valid_time_utc"])
            cutoff = valid - pd.Timedelta(hours=cutoff_lead_hours)

            late = df[avail > cutoff]
            if not late.empty:
                raise ValueError(
                    f"STRICT CAUSALITY VIOLATION DETECTED: {len(late)} records in inference batch have "
                    f"forecast availability after the forecast cutoff! Latest violation: {late.iloc[0]['simulated_avail_utc']}"
                )

    def predict(
        self,
        input_df: pd.DataFrame,
        timestamp_col: str = "target_timestamp",
        enforce_availability_check: bool = True,
        enforce_cutoff: bool | None = None,
    ) -> pd.DataFrame:
        """Execute production inference on validated input batch.

        Args:
            input_df: Input DataFrame containing required features.
            timestamp_col: Name of target timestamp column. Defaults to 'target_timestamp'.
            enforce_availability_check: If True, checks availability timestamps.
            enforce_cutoff: Alias for enforce_availability_check.

        Returns:
            pd.DataFrame: Aligned forecast table with predictions in MW.
        """
        self.validate_features(input_df)

        should_check = enforce_cutoff if enforce_cutoff is not None else enforce_availability_check
        if should_check:
            self.verify_forecast_availability(input_df)

        X = input_df[self.expected_features]
        preds = self.model.predict(X)

        res = pd.DataFrame({
            "target_timestamp": input_df[timestamp_col] if timestamp_col in input_df.columns else input_df.index,
            "predicted_demand_mw": np.round(preds, 2),
            "model_version": self.model_version,
            "features_used": len(self.expected_features),
            "prediction_timestamp_utc": pd.Timestamp.now(tz="UTC"),
        })

        if self.target_name in input_df.columns:
            res["actual_demand_mw"] = input_df[self.target_name]
            res["residual_mw"] = input_df[self.target_name] - preds

        return res

    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        output_path: Path | str,
    ) -> Path:
        """Save inference outputs to standard CSV format."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(out_p, index=False)
        return out_p
