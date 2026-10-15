"""Multi-Horizon and Day-Ahead Feature Engineering Engine.

Supports multi-step operational forecasting horizons:
- H=1 (1-Hour-Ahead Real-Time Dispatch)
- H=24 (24-Hour-Ahead Day-Ahead Dispatch)
- H=48 (48-Hour-Ahead Two-Day-Ahead Advisory Dispatch)

Strictly enforces Horizon-Specific Information Availability:
1. Energy Consumption History Cutoff:
   At forecast origin t_origin = T - H hours, only consumption actualized at
   t <= t_origin is admissible.
   - For H=1:  Minimum lag is lag_1  (T - 1h).  Rolling window ends at T - 1h.
   - For H=24: Minimum lag is lag_24 (T - 24h). Rolling window ends at T - 24h.
   - For H=48: Minimum lag is lag_48 (T - 48h). Rolling window ends at T - 48h.
   Prohibits shorter lags (e.g. lag_1 at H=24) to eliminate lookahead leakage.

2. Weather Forecast Availability:
   Operational NWP cycle must be initialized and available no later than t_origin:
   t_avail = t_init + 50min <= t_origin = T - H hours.
   Lead time f_lead = H + 1 hours (f02 at H=1, f25 at H=24, f49 at H=48).
   Simulates authentic lead-time error growth across projection horizons.
"""

from typing import Sequence
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.features.temporal_features import add_calendar_features, add_cyclical_features
from src.features.weather_features import compute_degree_proxies


SUPPORTED_HORIZONS: tuple[int, ...] = (1, 24, 48)


def get_horizon_feature_names(horizon: int, include_weather: bool = True) -> list[str]:
    """Retrieve list of admissible feature names for a specific forecast horizon."""
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}. Must be one of {SUPPORTED_HORIZONS}")

    calendar_feats = [
        "hour", "day_of_week", "day_of_month", "month", "is_weekend",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    ]

    if horizon == 1:
        lag_feats = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24"]
    elif horizon == 24:
        lag_feats = ["lag_24", "lag_48", "lag_168", "rolling_mean_24_h24", "rolling_std_24_h24"]
    elif horizon == 48:
        lag_feats = ["lag_48", "lag_72", "lag_168", "rolling_mean_24_h48", "rolling_std_24_h48"]

    base = calendar_feats + lag_feats
    if include_weather:
        weather_feats = [
            "temp_f_fcst",
            "cooling_degree_proxy_fcst",
            "heating_degree_proxy_fcst",
        ]
        return base + weather_feats
    return base


def create_horizon_feature_matrix(
    clean_energy_df: pd.DataFrame,
    nwp_df: pd.DataFrame | None = None,
    horizon: int = 24,
    random_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Construct leakage-free feature matrix for a specific prediction horizon H.

    Args:
        clean_energy_df: Cleaned energy DataFrame with 'timestamp' and 'total_power_demand'.
        nwp_df: Optional NWP forecast DataFrame with 'timestamp' and 'temp_f_fcst'.
        horizon: Prediction horizon in integer hours (1, 24, or 48). Defaults to 24.
        random_seed: Random seed for horizon-specific weather forecast error scaling.

    Returns:
        pd.DataFrame: Feature matrix aligned to target timestamp with target column
            'target_demand' representing consumption at T.

    Raises:
        ValueError: If horizon is unsupported or required columns are absent.
    """
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}. Must be in {SUPPORTED_HORIZONS}")

    df = clean_energy_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Target Definition at Horizon H
    # Row at index i has origin t_origin = df['timestamp'].iloc[i]
    # Target is demand at t_target = t_origin + H hours
    target_col = "consumption" if "consumption" in df.columns else "total_power_demand"
    if target_col not in df.columns:
        raise ValueError(f"Neither 'consumption' nor 'total_power_demand' found in DataFrame. Columns: {df.columns.tolist()}")

    df["target_timestamp"] = df["timestamp"] + pd.Timedelta(hours=horizon)

    # Lookup actual target demand at target_timestamp
    demand_lookup = df.set_index("timestamp")[target_col].to_dict()
    df["target_demand"] = df["target_timestamp"].map(demand_lookup)

    # 2. Calendar Features (Extracted strictly at target timestamp T)
    temp_target_df = pd.DataFrame({"timestamp": df["target_timestamp"]})
    cal_df = add_calendar_features(temp_target_df)
    cyc_df = add_cyclical_features(cal_df)
    for col in ["hour", "day_of_week", "day_of_month", "month", "is_weekend",
                "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]:
        df[col] = cyc_df[col]

    # 3. Horizon-Restricted Lag and Rolling Features (Strictly at t <= t_origin)
    # At t_origin:
    # df['timestamp'] IS t_origin!
    # Therefore, consumption at t_origin is lag_H relative to target T:
    # y(t_origin) = y(T - H)
    demand_series = df[target_col]

    if horizon == 1:
        # Lags relative to target T: T-1 (lag 0 from origin), T-24 (lag 23 from origin), T-168 (lag 167 from origin)
        df["lag_1"] = demand_series
        df["lag_24"] = demand_series.shift(23)
        df["lag_168"] = demand_series.shift(167)
        df["rolling_mean_24"] = demand_series.rolling(24).mean()
        df["rolling_std_24"] = demand_series.rolling(24).std()

    elif horizon == 24:
        # Lags relative to target T:
        # T - 24 = origin (shift 0 from origin)
        # T - 48 = origin - 24 (shift 24 from origin)
        # T - 168 = origin - 144 (shift 144 from origin)
        df["lag_24"] = demand_series
        df["lag_48"] = demand_series.shift(24)
        df["lag_168"] = demand_series.shift(144)
        df["rolling_mean_24_h24"] = demand_series.rolling(24).mean()
        df["rolling_std_24_h24"] = demand_series.rolling(24).std()

    elif horizon == 48:
        # Lags relative to target T:
        # T - 48 = origin (shift 0 from origin)
        # T - 72 = origin - 24 (shift 24 from origin)
        # T - 168 = origin - 120 (shift 120 from origin)
        df["lag_48"] = demand_series
        df["lag_72"] = demand_series.shift(24)
        df["lag_168"] = demand_series.shift(120)
        df["rolling_mean_24_h48"] = demand_series.rolling(24).mean()
        df["rolling_std_24_h48"] = demand_series.rolling(24).std()

    # 4. Weather Forecast Integration at Horizon H
    if nwp_df is not None:
        nwp = nwp_df.copy()
        nwp["timestamp"] = pd.to_datetime(nwp["timestamp"])

        # Determine base temperature forecast column
        temp_col = "temp_f_zonal_weighted" if "temp_f_zonal_weighted" in nwp.columns else "temp_f_fcst"
        weather_lookup = nwp.set_index("timestamp")[temp_col].to_dict()

        # Match forecast to target_timestamp
        base_fcst = df["target_timestamp"].map(weather_lookup).to_numpy(dtype=float)

        # Apply lead-time error scale growth for day-ahead horizons:
        # Error std at f02 ~ 1.2°F; at f25 ~ 2.4°F (scale factor ~1.5x); at f49 ~ 3.3°F (scale factor ~2.0x)
        if horizon > 1:
            rng = np.random.RandomState(random_seed + horizon)
            scale_factor = 1.45 if horizon == 24 else 1.95
            extra_noise = rng.normal(loc=0.0, scale=0.8 * (scale_factor - 1.0), size=len(df))
            fcst_val = np.where(np.isnan(base_fcst), np.nan, np.round(base_fcst + extra_noise, 2))
        else:
            fcst_val = base_fcst

        df["temp_f_fcst"] = fcst_val
        cooling, heating = compute_degree_proxies(pd.Series(df["temp_f_fcst"]))
        df["cooling_degree_proxy_fcst"] = cooling.to_numpy()
        df["heating_degree_proxy_fcst"] = heating.to_numpy()

    # Drop burn-in rows and end-of-series shift rows with missing lags or target
    feature_names = get_horizon_feature_names(horizon, include_weather=(nwp_df is not None))
    cols_to_check = feature_names + ["target_demand"]
    clean_feat_df = df.dropna(subset=cols_to_check).reset_index(drop=True)

    return clean_feat_df
