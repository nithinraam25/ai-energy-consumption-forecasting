"""Causal weather feature engineering and imputation for AI-Powered Energy Consumption Forecasting.

Implements degree-based comfort proxies, strictly causal forward-fill with physical-elapsed-time
gap limits, and a training-only 3-tier diurnal fallback hierarchy with zero future lookahead.
"""

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd


def compute_degree_proxies(
    temp_series: pd.Series,
    base_temp: float = 65.0,
) -> tuple[pd.Series, pd.Series]:
    """Calculate hourly cooling and heating degree proxies from dry-bulb temperature.

    Mathematical formulations:
        cooling_degree_proxy_f = max(0.0, temp_f - base_temp)
        heating_degree_proxy_f = max(0.0, base_temp - temp_f)

    These piecewise-linear formulations capture non-linear space-conditioning demand
    above and below the standard utility comfort neutral band (65.0°F / 18.33°C).

    Args:
        temp_series: Pandas series containing temperatures in degrees Fahrenheit.
        base_temp: Reference base comfort temperature. Defaults to 65.0°F.

    Returns:
        tuple[pd.Series, pd.Series]: (cooling_degree_proxy_f, heating_degree_proxy_f).
    """
    vals = temp_series.to_numpy(dtype=float)
    cooling = np.maximum(0.0, vals - float(base_temp))
    heating = np.maximum(0.0, float(base_temp) - vals)

    cooling_series = pd.Series(cooling, index=temp_series.index, name="cooling_degree_proxy_f")
    heating_series = pd.Series(heating, index=temp_series.index, name="heating_degree_proxy_f")
    return cooling_series, heating_series


class WeatherImputer:
    """Causal, zero-leakage imputer for hourly weather time-series.

    Enforces strict temporal ordering and causal constraints:
        1. Short Gaps (Delta t <= max_forward_fill_hours):
           Forward-fill from the last known valid physical observation.
        2. Extended Gaps (Delta t > max_forward_fill_hours) or Start of Series:
           3-Tier fallback hierarchy fitted exclusively on the training partition:
           - Tier 1: Diurnal-seasonal lookup table mu_train(month, hour) [12 x 24 cells].
           - Tier 2: Monthly marginal mean mu_train(month) [12 cells].
           - Tier 3: Global training mean mu_train_global [1 scalar].
        3. Zero Lookahead:
           Backward-fill, linear interpolation across time, and test/validation statistics
           are strictly prohibited.
    """

    def __init__(self, max_forward_fill_hours: int = 2) -> None:
        """Initialize WeatherImputer.

        Args:
            max_forward_fill_hours: Maximum elapsed hours eligible for forward-fill. Defaults to 2.
        """
        self.max_forward_fill_hours = int(max_forward_fill_hours)
        self.diurnal_table_: dict[str, float] = {}
        self.monthly_means_: dict[str, float] = {}
        self.global_mean_: float | None = None
        self.metadata_: dict[str, Any] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        timestamp_col: str = "target_timestamp",
        temp_col: str = "temp_f",
    ) -> "WeatherImputer":
        """Fit imputation hierarchy parameters exclusively on training partition rows.

        Args:
            train_df: Training partition dataframe.
            timestamp_col: Name of datetime timestamp column. Defaults to 'target_timestamp'.
            temp_col: Name of temperature column. Defaults to 'temp_f'.

        Returns:
            WeatherImputer: Fitted instance.

        Raises:
            ValueError: If required columns are missing from train_df.
            RuntimeError: If zero valid temperature observations exist in train_df.
        """
        if timestamp_col not in train_df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in training DataFrame.")
        if temp_col not in train_df.columns:
            raise ValueError(f"Temperature column '{temp_col}' not found in training DataFrame.")

        df = train_df[[timestamp_col, temp_col]].copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        valid_df = df.dropna(subset=[temp_col]).copy()
        if valid_df.empty:
            raise RuntimeError(
                "Zero valid temperature observations found in training partition. "
                "Cannot fit WeatherImputer fallback statistics on empty data."
            )

        # Tier 3: Global Training Scalar Mean
        self.global_mean_ = float(valid_df[temp_col].mean())

        # Tier 2: Monthly Marginal Means (12 values)
        valid_df["month"] = valid_df[timestamp_col].dt.month
        monthly_grouped = valid_df.groupby("month")[temp_col].mean()
        self.monthly_means_ = {}
        for m in range(1, 13):
            if m in monthly_grouped.index and not np.isnan(monthly_grouped.loc[m]):
                self.monthly_means_[str(m)] = float(monthly_grouped.loc[m])
            else:
                self.monthly_means_[str(m)] = self.global_mean_

        # Tier 1: Diurnal-Seasonal Lookup Table (12 months x 24 hours = 288 cells)
        valid_df["hour"] = valid_df[timestamp_col].dt.hour
        cell_grouped = valid_df.groupby(["month", "hour"])[temp_col].mean()

        self.diurnal_table_ = {}
        for m in range(1, 13):
            for h in range(24):
                key = f"{m}_{h}"
                if (m, h) in cell_grouped.index and not np.isnan(cell_grouped.loc[(m, h)]):
                    self.diurnal_table_[key] = float(cell_grouped.loc[(m, h)])
                else:
                    self.diurnal_table_[key] = self.monthly_means_[str(m)]

        self.metadata_ = {
            "n_training_samples": len(train_df),
            "n_valid_training_samples": len(valid_df),
            "n_missing_training_samples": len(train_df) - len(valid_df),
            "global_training_mean_f": round(self.global_mean_, 4),
            "training_start": str(df[timestamp_col].min()),
            "training_end": str(df[timestamp_col].max()),
            "max_forward_fill_hours": self.max_forward_fill_hours,
        }
        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "target_timestamp",
        temp_col: str = "temp_f",
        utc_col: str | None = "nominal_utc",
        initial_state: tuple[pd.Timestamp, float] | None = None,
    ) -> tuple[pd.DataFrame, tuple[pd.Timestamp, float] | None]:
        """Causally impute missing temperature values with zero future lookahead.

        Args:
            df: DataFrame to impute (train, validation, or test).
            timestamp_col: Nominal local timestamp column. Defaults to 'target_timestamp'.
            temp_col: Temperature column to impute in-place or generate. Defaults to 'temp_f'.
            utc_col: Optional nominal UTC timestamp column for physical elapsed-time gap calculation.
            initial_state: Optional (last_valid_timestamp, last_valid_temp) carried over from
                the immediately preceding chronological partition boundary.

        Returns:
            tuple[pd.DataFrame, tuple[pd.Timestamp, float] | None]:
                (DataFrame with imputed 'temp_f', final (last_valid_timestamp, last_valid_temp) state).

        Raises:
            RuntimeError: If imputer has not been fitted.
            ValueError: If required columns are missing.
        """
        if not self.is_fitted_:
            raise RuntimeError("WeatherImputer must be fitted on training data before calling transform().")

        if timestamp_col not in df.columns:
            raise ValueError(f"Column '{timestamp_col}' not found in DataFrame.")
        if temp_col not in df.columns:
            raise ValueError(f"Column '{temp_col}' not found in DataFrame.")

        df_out = df.copy()
        ts_series = pd.to_datetime(df_out[timestamp_col])

        # Use UTC timestamp if available for physical elapsed time; otherwise fallback to ts_series
        if utc_col is not None and utc_col in df_out.columns:
            elapsed_ts_series = pd.to_datetime(df_out[utc_col])
        else:
            elapsed_ts_series = ts_series

        raw_temps = df_out[temp_col].to_numpy(dtype=float)
        n = len(df_out)

        last_valid_ts: pd.Timestamp | None = initial_state[0] if initial_state is not None else None
        last_valid_val: float | None = initial_state[1] if initial_state is not None else None

        imputed_temps = np.empty(n, dtype=float)
        max_ffill_delta = pd.Timedelta(hours=self.max_forward_fill_hours)

        for i in range(n):
            v = raw_temps[i]
            t = elapsed_ts_series.iloc[i]
            local_t = ts_series.iloc[i]

            if not np.isnan(v):
                last_valid_ts = t
                last_valid_val = float(v)
                imputed_temps[i] = float(v)
            else:
                if last_valid_ts is not None and last_valid_val is not None:
                    delta_t = t - last_valid_ts
                    if pd.Timedelta(0) <= delta_t <= max_ffill_delta:
                        # Causal short-gap forward fill
                        imputed_temps[i] = last_valid_val
                    else:
                        # Extended gap: Tier 1 diurnal fallback
                        m, h = local_t.month, local_t.hour
                        imputed_temps[i] = self.diurnal_table_[f"{m}_{h}"]
                else:
                    # No prior valid observation: Tier 1 diurnal fallback
                    m, h = local_t.month, local_t.hour
                    imputed_temps[i] = self.diurnal_table_[f"{m}_{h}"]

        df_out[temp_col] = imputed_temps
        final_state = (last_valid_ts, last_valid_val) if last_valid_ts is not None else None
        return df_out, final_state

    def save(self, file_path: Path | str) -> Path:
        """Persist imputer statistics to JSON file.

        Args:
            file_path: Target path for JSON serialization.

        Returns:
            Path: Absolute path to saved JSON file.

        Raises:
            RuntimeError: If imputer is not fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted WeatherImputer.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": self.metadata_,
            "global_mean": self.global_mean_,
            "monthly_means": self.monthly_means_,
            "diurnal_table": self.diurnal_table_,
            "max_forward_fill_hours": self.max_forward_fill_hours,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return path

    @classmethod
    def load(cls, file_path: Path | str) -> "WeatherImputer":
        """Load frozen WeatherImputer parameters from JSON file.

        Args:
            file_path: Path to serialized JSON file.

        Returns:
            WeatherImputer: Loaded and ready instance.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Weather imputer parameter file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        imputer = cls(max_forward_fill_hours=payload.get("max_forward_fill_hours", 2))
        imputer.global_mean_ = float(payload["global_mean"])
        imputer.monthly_means_ = {str(k): float(v) for k, v in payload["monthly_means"].items()}
        imputer.diurnal_table_ = {str(k): float(v) for k, v in payload["diurnal_table"].items()}
        imputer.metadata_ = payload.get("metadata", {})
        imputer.is_fitted_ = True
        return imputer


def add_weather_features(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    imputer: WeatherImputer,
    target_timestamp_col: str = "target_timestamp",
    weather_timestamp_col: str = "timestamp",
    initial_state: tuple[pd.Timestamp, float] | None = None,
    alignment_hypothesis: str = "A",
) -> tuple[pd.DataFrame, tuple[pd.Timestamp, float] | None]:
    """Merge weather series, causally impute missing values, and generate degree proxies.

    Alignment Hypotheses:
        - 'A' (Pre-registered primary): Realized weather at target hour T (contemporaneous).
        - 'B' (Validation diagnostic): Weather at T - 1 hour (Hour-Ending driver).
        - 'C' (Validation diagnostic): Two-sample mean [temp(T-1) + temp(T)] / 2.

    Args:
        demand_df: Feature DataFrame containing target_timestamp_col.
        weather_df: Weather DataFrame containing weather_timestamp_col and 'temp_f'.
        imputer: Fitted WeatherImputer instance.
        target_timestamp_col: Column name in demand_df to join on. Defaults to 'target_timestamp'.
        weather_timestamp_col: Column name in weather_df to match. Defaults to 'timestamp'.
        initial_state: State carried over across partition boundaries.
        alignment_hypothesis: Weather alignment hypothesis ('A', 'B', or 'C'). Defaults to 'A'.

    Returns:
        tuple[pd.DataFrame, tuple[pd.Timestamp, float] | None]:
            (Augmented DataFrame with ['temp_f', 'cooling_degree_proxy_f', 'heating_degree_proxy_f'], final_state).

    Raises:
        ValueError: If hypothesis is unrecognized or required columns are missing.
    """
    valid_hypotheses = {"A", "B", "C"}
    if alignment_hypothesis not in valid_hypotheses:
        raise ValueError(
            f"Unrecognized alignment_hypothesis '{alignment_hypothesis}'. "
            f"Must be one of: {sorted(valid_hypotheses)}"
        )

    if target_timestamp_col not in demand_df.columns:
        raise ValueError(f"Target timestamp column '{target_timestamp_col}' not found in demand DataFrame.")
    if weather_timestamp_col not in weather_df.columns:
        raise ValueError(f"Weather timestamp column '{weather_timestamp_col}' not found in weather DataFrame.")
    if "temp_f" not in weather_df.columns:
        raise ValueError("Weather DataFrame must contain 'temp_f' column.")

    w_df = weather_df.copy()
    w_df[weather_timestamp_col] = pd.to_datetime(w_df[weather_timestamp_col])

    # Construct alignment mapping
    cols_to_pull = [weather_timestamp_col, "temp_f"]
    if "nominal_utc" in w_df.columns:
        cols_to_pull.append("nominal_utc")

    w_subset = w_df[cols_to_pull].copy()

    if alignment_hypothesis == "A":
        # Contemporaneous: match weather(T) to target(T)
        pass
    elif alignment_hypothesis == "B":
        # T - 1 hour: shift weather timestamp forward so weather(T-1) matches target(T)
        w_subset[weather_timestamp_col] = w_subset[weather_timestamp_col] + pd.Timedelta(hours=1)
    elif alignment_hypothesis == "C":
        # Two-sample average: [temp(T-1) + temp(T)] / 2
        w_shifted = w_subset.copy()
        w_shifted[weather_timestamp_col] = w_shifted[weather_timestamp_col] + pd.Timedelta(hours=1)
        w_merged_samples = w_subset.merge(
            w_shifted[[weather_timestamp_col, "temp_f"]],
            on=weather_timestamp_col,
            suffixes=("", "_prev"),
            how="left",
        )
        w_merged_samples["temp_f"] = (w_merged_samples["temp_f"] + w_merged_samples["temp_f_prev"]) / 2.0
        w_subset = w_merged_samples[cols_to_pull].copy()

    # Join onto demand DataFrame
    merged = demand_df.merge(
        w_subset,
        left_on=target_timestamp_col,
        right_on=weather_timestamp_col,
        how="left",
    )
    if weather_timestamp_col != target_timestamp_col and weather_timestamp_col in merged.columns:
        merged = merged.drop(columns=[weather_timestamp_col])

    # Causally impute
    utc_col = "nominal_utc" if "nominal_utc" in merged.columns else None
    imputed_df, final_state = imputer.transform(
        merged,
        timestamp_col=target_timestamp_col,
        temp_col="temp_f",
        utc_col=utc_col,
        initial_state=initial_state,
    )

    # Compute piecewise degree proxies
    cooling, heating = compute_degree_proxies(imputed_df["temp_f"], base_temp=65.0)
    imputed_df["cooling_degree_proxy_f"] = cooling
    imputed_df["heating_degree_proxy_f"] = heating

    # Verify 100% finite values
    for col in ["temp_f", "cooling_degree_proxy_f", "heating_degree_proxy_f"]:
        if imputed_df[col].isna().any():
            nan_cnt = int(imputed_df[col].isna().sum())
            raise RuntimeError(f"NaNs detected in weather feature column '{col}' after imputation: {nan_cnt}")
        if np.isinf(imputed_df[col]).any():
            inf_cnt = int(np.isinf(imputed_df[col]).sum())
            raise RuntimeError(f"Infinities detected in weather feature column '{col}': {inf_cnt}")

    return imputed_df, final_state


def add_operational_nwp_features(
    demand_df: pd.DataFrame,
    nwp_df: pd.DataFrame,
    imputer: WeatherImputer,
    target_timestamp_col: str = "target_timestamp",
    nwp_timestamp_col: str = "timestamp",
    initial_state: tuple[pd.Timestamp, float] | None = None,
    enforce_causality: bool = True,
    forecast_origin_lead_hours: int = 1,
) -> tuple[pd.DataFrame, tuple[pd.Timestamp, float] | None]:
    """Merge operational NWP forecasts, enforce forecast cutoff causality, and generate degree proxies.

    Guarantees:
        1. Operational Cutoff Causality:
           For every forecast row, verifies:
               simulated_avail_utc <= energy_prediction_cutoff_utc (target_utc - 1h).
           Rejects forecasts issued or available after the energy forecast origin.
        2. Zero Realized-Weather Substitution:
           Missing NWP forecasts are imputed strictly causally via the training-only
           WeatherImputer hierarchy, never substituted with observed surface measurements.
        3. Piecewise Space-Conditioning Proxies:
           cooling_degree_proxy_fcst = max(0.0, temp_f_fcst - 65.0)
           heating_degree_proxy_fcst = max(0.0, 65.0 - temp_f_fcst)

    Args:
        demand_df: Feature DataFrame containing target_timestamp_col.
        nwp_df: NWP forecast DataFrame containing nwp_timestamp_col and 'temp_f_fcst'.
        imputer: Fitted WeatherImputer instance.
        target_timestamp_col: Target timestamp column in demand_df. Defaults to 'target_timestamp'.
        nwp_timestamp_col: Timestamp column in nwp_df. Defaults to 'timestamp'.
        initial_state: Imputation state carried over across partition boundaries.
        enforce_causality: If True, asserts zero forecasts arrive after forecast origin.
        forecast_origin_lead_hours: Lead time of energy prediction cutoff (1 hour).

    Returns:
        tuple[pd.DataFrame, tuple[pd.Timestamp, float] | None]:
            (Augmented DataFrame with ['temp_f_fcst', 'cooling_degree_proxy_fcst', 'heating_degree_proxy_fcst'], final_state).

    Raises:
        ValueError: If required columns are missing or a causality violation is detected.
    """
    if target_timestamp_col not in demand_df.columns:
        raise ValueError(f"Target timestamp column '{target_timestamp_col}' not found in demand DataFrame.")
    if nwp_timestamp_col not in nwp_df.columns:
        raise ValueError(f"NWP timestamp column '{nwp_timestamp_col}' not found in NWP DataFrame.")
    if "temp_f_fcst" not in nwp_df.columns:
        raise ValueError("NWP DataFrame must contain 'temp_f_fcst' column.")

    n_df = nwp_df.copy()
    n_df[nwp_timestamp_col] = pd.to_datetime(n_df[nwp_timestamp_col])

    # Subset required columns
    cols_to_pull = [nwp_timestamp_col, "temp_f_fcst"]
    if "simulated_avail_utc" in n_df.columns:
        cols_to_pull.append("simulated_avail_utc")
    if "valid_time_utc" in n_df.columns:
        cols_to_pull.append("valid_time_utc")
    if "nominal_utc" in n_df.columns:
        cols_to_pull.append("nominal_utc")

    n_subset = n_df[cols_to_pull].copy()

    # Join onto demand DataFrame
    merged = demand_df.merge(
        n_subset,
        left_on=target_timestamp_col,
        right_on=nwp_timestamp_col,
        how="left",
    )
    if nwp_timestamp_col != target_timestamp_col and nwp_timestamp_col in merged.columns:
        merged = merged.drop(columns=[nwp_timestamp_col])

    # Causality enforcement
    if enforce_causality and "simulated_avail_utc" in merged.columns and "valid_time_utc" in merged.columns:
        valid_utc = pd.to_datetime(merged["valid_time_utc"])
        avail_utc = pd.to_datetime(merged["simulated_avail_utc"])
        cutoff_utc = valid_utc - pd.Timedelta(hours=forecast_origin_lead_hours)

        violations = merged[avail_utc > cutoff_utc]
        if not violations.empty:
            raise ValueError(
                f"STRICT CAUSALITY VIOLATION DETECTED: {len(violations)} NWP forecast records "
                f"have simulated_avail_utc > forecast_cutoff_utc! "
                f"First violation at target {violations.iloc[0][target_timestamp_col]}: "
                f"avail={violations.iloc[0]['simulated_avail_utc']} > cutoff={cutoff_utc.iloc[0]}"
            )

    # Causally impute missing forecast values
    utc_col = "nominal_utc" if "nominal_utc" in merged.columns else None
    imputed_df, final_state = imputer.transform(
        merged,
        timestamp_col=target_timestamp_col,
        temp_col="temp_f_fcst",
        utc_col=utc_col,
        initial_state=initial_state,
    )

    # Compute piecewise degree proxies for forecast
    cooling_fcst, heating_fcst = compute_degree_proxies(imputed_df["temp_f_fcst"], base_temp=65.0)
    imputed_df["cooling_degree_proxy_fcst"] = cooling_fcst
    imputed_df["heating_degree_proxy_fcst"] = heating_fcst

    # Verify 100% finite values
    for col in ["temp_f_fcst", "cooling_degree_proxy_fcst", "heating_degree_proxy_fcst"]:
        if imputed_df[col].isna().any():
            nan_cnt = int(imputed_df[col].isna().sum())
            raise RuntimeError(f"NaNs detected in operational forecast feature column '{col}': {nan_cnt}")
    return imputed_df, final_state


# ==============================================================================
# EXTENDED WEATHER & SOLAR IRRADIANCE (GHI) INTERFACE
# ==============================================================================

class SolarAndExtendedWeatherInterface:
    """Modular interface for solar irradiance (GHI) and extended meteorological covariates.

    Provenance & Availability Specification:
    - NOAA ISD Historical Station Observations (KPHL, KEWR, KILG):
      Contains TMP (temperature), DEW (dew point), WND (wind speed/direction), SLP (pressure).
      Does NOT archive solar radiation or Global Horizontal Irradiance (GHI).
    - NOAA HRRR Operational NWP Model:
      Contains Downward Shortwave Radiation Flux (DSWRF / GHI at surface) in GRIB2 on AWS S3,
      but records begin in mid-2014. Continuous 2002-2018 historical GHI operational forecasts
      do not exist.
    - Policy:
      In accordance with engineering constraints, we do NOT manufacture synthetic GHI datasets.
      This interface validates GHI when genuinely present (e.g. live operational API feeds)
      and provides graceful fallback when GHI is absent.
    """

    SUPPORTED_SOLAR_COLUMNS: tuple[str, ...] = ("ghi_fcst", "solar_radiation_fcst", "dswrf_fcst")

    @classmethod
    def detect_solar_column(cls, df: pd.DataFrame) -> str | None:
        """Detect presence of genuine solar irradiance forecast columns."""
        for col in cls.SUPPORTED_SOLAR_COLUMNS:
            if col in df.columns:
                return col
        return None

    @classmethod
    def is_solar_available(cls, df: pd.DataFrame) -> bool:
        """Check if reliable solar irradiance forecast is present."""
        col = cls.detect_solar_column(df)
        return col is not None and not df[col].isna().all()

    @classmethod
    def add_solar_features(
        cls,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Extract solar features if genuine solar forecasts are present.

        If solar data is unavailable, returns df unchanged without fabricating synthetic data.
        """
        solar_col = cls.detect_solar_column(df)
        if solar_col is None:
            # Solar data unavailable in historical archive; preserve modular pipeline
            return df.copy()

        out_df = df.copy()
        raw_ghi = out_df[solar_col].to_numpy(dtype=float)

        # Solar features: daylight indicator and normalized GHI (clear-sky ceiling ~ 1000 W/m2)
        out_df["is_daylight_fcst"] = (raw_ghi > 10.0).astype(int)
        out_df["ghi_normalized_fcst"] = np.clip(raw_ghi / 1000.0, 0.0, 1.5)
        return out_df

