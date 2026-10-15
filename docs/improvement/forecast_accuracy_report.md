# Phase 2: Forecast Accuracy, Robustness, and Evaluation Improvements Report
**Project:** AI-Powered Energy Consumption Forecasting System — PJM East  
**Date:** October 2026  
**Status:** COMPLETE & INDEPENDENTLY REVERIFIED  
**Evaluation Script:** `src/evaluation/evaluate_accuracy.py`  
**Test Suite:** 195 passed (all existing 180 + 15 new deterministic regression tests)

---

## 1. Executive Summary & Baseline Reproduction

This report documents the rigorous, deterministic evaluation of the PJM East energy demand forecasting system across prediction horizons $H \in \{1\text{h}, 24\text{h}, 48\text{h}\}$, probabilistic prediction interval calibration, and empirical robustness across operational sub-regimes.

### 1.1 Baseline Metric Reproduction
All reported baseline metrics were deterministically reproduced on the frozen, out-of-sample chronological test holdout partition ($N = 21,773$ aligned test hours, February 7, 2016 to August 3, 2018 UTC):

| Forecast Horizon | Target / Metric | Production NWP Model | Baseline No-Weather | Persistence | Weekly Seasonal Naive | NWP Skill Gain | Statistical Significance ($p$-value) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=1\text{h}$** | **MAE (MW)** | **333.72** | 351.58 | 1,054.91 | 3,436.23 | **+5.08%** | $p < 10^{-15}$ (DM = 15.91) |
| | **RMSE (MW)** | **446.71** | 469.83 | 1,354.76 | 4,737.01 | — | — |
| | **$R^2$** | **0.9952** | 0.9947 | 0.9558 | 0.4594 | — | — |
| | **MAPE (%)** | **1.08%** | 1.13% | 3.43% | 10.70% | — | — |
| | **WAPE (%)** | **1.07%** | 1.13% | 3.39% | 11.05% | — | — |
| **$H=24\text{h}$** | **MAE (MW)** | **1,226.73** | 1,833.64 | 2,207.04 | 3,436.71 | **+33.10%** | $p < 10^{-15}$ (DM = 17.86) |
| | **RMSE (MW)** | **1,629.45** | 2,482.80 | 3,028.07 | 4,737.12 | — | — |
| | **$R^2$** | **0.9360** | 0.8515 | 0.7791 | 0.4594 | — | — |
| | **MAPE (%)** | **3.94%** | 5.80% | 7.02% | 10.70% | — | — |
| | **WAPE (%)** | **3.94%** | 5.90% | 7.10% | 11.05% | — | — |
| **$H=48\text{h}$** | **MAE (MW)** | **1,487.77** | 2,485.98 | 3,232.37 | 3,437.21 | **+40.15%** | $p < 10^{-15}$ (DM = 15.10) |
| | **RMSE (MW)** | **1,949.60** | 3,290.47 | 4,297.64 | 4,737.58 | — | — |
| | **$R^2$** | **0.9084** | 0.7392 | 0.5550 | 0.4593 | — | — |
| | **MAPE (%)** | **4.81%** | 7.88% | 10.36% | 10.71% | — | — |
| | **WAPE (%)** | **4.78%** | 7.99% | 10.39% | 11.05% | — | — |

### 1.2 Conformal Prediction Interval Calibration Reproduction
- **Nominal Target Coverage:** $90.00\%$ ($\alpha = 0.10$).
- **Held-Out Test Sample Count:** $N = 21,762$ test hours.
- **Raw Quantile Regressor:** $86.12\%$ empirical coverage, mean width $4,099.19\text{ MW}$, Winkler score $5,097.43\text{ MW}$.
- **Global Scalar CQR:** $88.09\%$ empirical coverage, mean width $4,212.58\text{ MW}$, Winkler score $5,064.58\text{ MW}$.
- **Adaptive Diurnal CQR (Champion):** **$88.28\%$ empirical coverage**, mean width **$4,180.74\text{ MW}$**, median width $3,477.98\text{ MW}$, Winkler score **$4,930.39\text{ MW}$**.
- **Quantile Monotonicity:** **0 quantile-ordering violations** ($Q_{05} \le Q_{50} \le Q_{95}$ across all test predictions).
- **Nominal Gap:** $-1.72\%$ undercoverage against nominal $90\%$ target. This gap is transparently acknowledged without artificial post-hoc manipulation.

---

## 2. Evaluation Partitions & Data Provenance

### 2.1 Chronological Partitioning Rules
To strictly prevent temporal data leakage and lookahead bias:
1. **Partition Ordering:** Data is split strictly chronologically:
   - **Training:** $70.0\%$ (January 1, 2002 01:00 to September 21, 2013 19:00 UTC; $101,551$ samples)
   - **Validation / Calibration:** $15.0\%$ (September 21, 2013 20:00 to February 7, 2016 10:00 UTC; $21,760$ samples)
   - **Held-Out Test Holdout:** $15.0\%$ (February 7, 2016 11:00 to August 3, 2018 00:00 UTC; $21,773$ samples)
2. **Zero Test Tuning:** All model hyperparameters, tree structures, and conformal calibration adjustment parameters were fit exclusively on Train and Validation splits. The Test holdout was accessed only for final out-of-sample scoring.
3. **Identical Timestamp Alignment:** All baseline models and the production NWP model were evaluated on identical target timestamps ($T$).

### 2.2 Excluded Rows and Missing Observations
- **Raw Hourly Demand Observations:** $145,362$ rows (2002–2018).
- **Lag Burn-In Exclusion:** The initial $168\text{ hours}$ (7 days) are dropped because the weekly lag `lag_168` requires one full preceding week of observations.
- **Horizon Shift Exclusion:** For horizon $H$, the last $H$ rows of the time series have no observed target $y(t + H)$ and are excluded from the test matrix.
- **Weather Observations:** $3$ hourly records in the historical proxy weather dataset had missing temperature entries; these rows were dropped across both NWP and baseline evaluations to maintain identical row counts.

---

## 3. Baseline Definitions & Leakage Audit

### 3.1 Formal Definition of Benchmarks
Every benchmark comparison is defined with mathematical precision:

1. **Persistence Benchmark:**
   $$\hat{y}_{\text{pers}}(T) = y(T - H) = y(t_{\text{origin}})$$
   - At $H=1$: $\hat{y}(T) = y(T - 1\text{h}) = \text{lag}_1$.
   - At $H=24$: $\hat{y}(T) = y(T - 24\text{h}) = \text{lag}_{24}$.
   - At $H=48$: $\hat{y}(T) = y(T - 48\text{h}) = \text{lag}_{48}$.
   *Note: Using $\text{lag}_1$ or $\text{lag}_{24}$ for $H=48$ would constitute future lookahead leakage ($t > T - 48$). The persistence benchmark at $H=48$ strictly uses $\text{lag}_{48}$.*

2. **Daily Seasonal Naive Benchmark:**
   $$\hat{y}_{\text{daily\_sn}}(T) = y(T - 24\text{h}) \quad (\text{for } H \le 24), \quad y(T - 48\text{h}) \quad (\text{for } H = 48)$$

3. **Weekly Seasonal Naive Benchmark:**
   $$\hat{y}_{\text{weekly\_sn}}(T) = y(T - 168\text{h}) = \text{lag}_{168}$$
   Captures standard weekly diurnal cycles (same hour of day, same day of week one week prior).

4. **Baseline No-Weather Tree Model:**
   Gradient-boosted decision tree (`HistGradientBoostingRegressor`) trained strictly on calendar features (`hour`, `day_of_week`, `month`, cyclical encodings) and admissible historical autoregressive lags ($\text{lag}_H, \text{lag}_{H+24}, \text{lag}_{168}$, rolling stats at $t \le T-H$). Zero numerical weather prediction features are included.

5. **Production NWP Model:**
   Identical gradient-boosted architecture augmented with Numerical Weather Prediction temperature forecasts ($\hat{T}_{\text{fcst}}$) and derived cooling/heating degree proxies valid at target time $T$.

### 3.2 Evaluation Metrics
- **Mean Absolute Error (MAE):** $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$ (MW)
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$ (MW)
- **Mean Absolute Percentage Error (MAPE):** $\frac{1}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{y_i} \times 100\%$
- **Weighted Absolute Percentage Error (WAPE):**
  $$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i} \times 100\%$$
  WAPE provides a scale-independent percentage error that weights deviations proportional to grid demand, avoiding distortion from near-zero low-demand periods.
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

---

## 4. Investigation of H=1 NWP Gain & Sign Convention

### 4.1 Sign Convention
In weather forecasting and grid operations, skill gain relative to a reference benchmark is defined as the percentage error reduction:
$$\text{Skill Gain} = \frac{\text{MAE}_{\text{benchmark}} - \text{MAE}_{\text{model}}}{\text{MAE}_{\text{benchmark}}} \times 100\%$$

- A **positive value** indicates that $\text{MAE}_{\text{model}} < \text{MAE}_{\text{benchmark}}$, representing positive forecasting skill (error reduction).
- A **negative value** would indicate deterioration relative to the benchmark.

### 4.2 Verification of H=1 Skill Gain
- $\text{MAE}_{\text{base}} = 351.58\text{ MW}$
- $\text{MAE}_{\text{nwp}} = 333.72\text{ MW}$
$$\text{Gain}_{\text{NWP}} = \frac{351.58 - 333.72}{351.58} \times 100\% = +5.08\%$$

At $H=1$, grid demand is heavily autoregressive ($\text{lag}_1$ correlation $\approx 0.98$). Even so, adding surface weather information provides a statistically significant **$5.08\%$ error reduction** ($p < 10^{-15}$). At day-ahead ($H=24$) and two-day-ahead ($H=48$) horizons, weather dominates thermal demand, driving skill gains of **$+33.10\%$** and **$+40.15\%$** over the no-weather tree baseline.

---

## 5. Statistical Significance Testing (Diebold-Mariano)

To determine whether performance improvements are statistically meaningful, we executed the two-tailed **Diebold-Mariano test** on absolute loss differentials $d_t = |e_{1,t}| - |e_{2,t}|$ with Newey-West spectral variance estimation using lag truncation parameter $h - 1$:

$$\text{DM} = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d}) / N}} \sim \mathcal{N}(0, 1)$$

### Test Results on Held-Out Test Set ($N = 21,773$):
1. **$H=1\text{h}$ — NWP vs No-Weather Baseline:**
   - $\text{DM} = 15.908$, $p = 0.000$ ($p < 10^{-15}$).
   - **Conclusion:** NWP model is significantly superior to the no-weather baseline ($p < 0.001$).
2. **$H=24\text{h}$ — NWP vs No-Weather Baseline:**
   - $\text{DM} = 17.857$, $p = 0.000$ ($p < 10^{-15}$).
   - **Conclusion:** Extreme statistical significance confirming that day-ahead temperature forecasts are essential for load estimation.
3. **$H=48\text{h}$ — NWP vs No-Weather Baseline:**
   - $\text{DM} = 15.100$, $p = 0.000$ ($p < 10^{-15}$).
   - **Conclusion:** Overwhelming rejection of equal predictive ability ($p < 0.001$).

---

## 6. Sliced Error Analysis Across Operational Regimes

### 6.1 Seasonality
Thermal sensitivity creates seasonal error divergence:

| Horizon | Season | Sample Count | NWP MAE (MW) | NWP RMSE (MW) | WAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$H=1\text{h}$** | Winter | 4,861 | 331.49 | 440.22 | 1.02% |
| | Spring | 6,615 | 325.69 | 442.35 | 1.16% |
| | Summer | 5,929 | 334.18 | 449.78 | 0.95% |
| | Fall | 4,368 | 347.77 | 456.17 | 1.21% |
| **$H=24\text{h}$** | Winter | 4,861 | 1,105.56 | 1,411.36 | 3.40% |
| | Spring | 6,615 | 1,023.69 | 1,357.28 | 3.66% |
| | Summer | 5,929 | 1,597.04 | 2,051.58 | 4.54% |
| | Fall | 4,368 | 1,166.43 | 1,585.86 | 4.06% |
| **$H=48\text{h}$** | Winter | 4,861 | 1,235.58 | 1,563.54 | 3.80% |
| | Spring | 6,615 | 1,271.27 | 1,660.15 | 4.54% |
| | Summer | 5,929 | 1,960.95 | 2,489.94 | 5.57% |
| | Fall | 4,368 | 1,454.02 | 1,906.93 | 5.06% |

*Key Finding:* Summer exhibits the highest absolute errors at $H=24$ ($1,597\text{ MW}$) and $H=48$ ($1,961\text{ MW}$) due to nonlinear air-conditioning demand response during peak heatwaves.

### 6.2 Demand Regimes & Peak-Load Stress
Evaluating across demand percentiles on the held-out test set ($N = 21,773$):

| Demand Regime | Threshold | Sample Count | $H=1$ MAE (MW) | $H=24$ MAE (MW) | $H=48$ MAE (MW) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Q1 Low** | $< 25\text{th percentile}$ ($< 25,600\text{ MW}$) | 5,444 | 236.69 | 931.68 | 1,181.59 |
| **Q2 Med-Low** | $25\text{th} - 50\text{th percentile}$ | 5,444 | 333.44 | 1,172.42 | 1,448.17 |
| **Q3 Med-High** | $50\text{th} - 75\text{th percentile}$ | 5,442 | 361.08 | 1,184.18 | 1,459.07 |
| **Q4 High** | $75\text{th} - 95\text{th percentile}$ | 4,354 | 390.70 | 1,578.79 | 1,794.71 |
| **Extreme Peak** | $> 95\text{th percentile}$ ($> 44,000\text{ MW}$) | 1,089 | 455.71 | 1,778.27 | 2,132.59 |

*Operational Implication:* Grid stress hours ($> 95\text{th percentile}$) exhibit higher absolute errors ($1,778\text{ MW}$ at $H=24$), yet WAPE remains tight ($3.73\%$), indicating strong structural stability during critical ramp events.

### 6.3 Day-of-Week Effect
- **Weekday ($N = 15,577$):** $H=1$ MAE = $335.89\text{ MW}$, $H=24$ MAE = $1,238.97\text{ MW}$, $H=48$ MAE = $1,522.39\text{ MW}$.
- **Weekend ($N = 6,196$):** $H=1$ MAE = $328.28\text{ MW}$, $H=24$ MAE = $1,195.98\text{ MW}$, $H=48$ MAE = $1,400.79\text{ MW}$.

---

## 7. Prediction Interval Calibration & Conformal Analysis

### 7.1 Sharpness and Coverage Comparison
Evaluating $90\%$ prediction intervals ($Q_{05}$ to $Q_{95}$) across calibration methods:

| Method | Description | Test Coverage ($90\%$ Target) | Mean Width (MW) | Median Width (MW) | Winkler Score (MW) | Quantile Crossing Violations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Quantiles** | Uncalibrated Pinball Loss Estimators | $86.12\%$ | $4,099.19$ | $3,438.68$ | $5,097.43$ | **0** |
| **Global Scalar CQR** | Single Conformal Adjustment ($\Delta = +113.39\text{ MW}$) | $88.09\%$ | $4,212.58$ | $3,552.07$ | $5,064.58$ | **0** |
| **Adaptive Diurnal CQR** | Hour-Specific Diurnal Scaling Factors | **$88.28\%$** | **$4,180.74$** | **$3,477.98$** | **$4,930.39$** | **0** |

*Winkler Score Interpretation:* The Winkler score ($IS_{0.10}$) measures both sharpness (interval width) and penalties for coverage misses:
$$IS_{\alpha}(l, u, y) = (u - l) + \frac{2}{\alpha}(l - y)\mathbb{I}(y < l) + \frac{2}{\alpha}(y - u)\mathbb{I}(y > u)$$
Adaptive Diurnal CQR achieves the lowest Winkler score ($4,930.39\text{ MW}$), proving it provides the sharpest intervals per unit of empirical coverage.

### 7.2 Diurnal Coverage Distribution
Adaptive diurnal calibration successfully addresses the diurnal variance cycle:
- Overnight hours (00:00–05:00 UTC): $90.3\% - 92.7\%$ coverage.
- Morning ramp hours (06:00–08:00 UTC): $88.4\% - 92.8\%$ coverage.
- Mid-day thermal peak hours (10:00–12:00 UTC): $78.7\% - 81.8\%$ coverage (reflecting high industrial dispatch variance).
- Evening peak hours (19:00–22:00 UTC): $89.9\% - 93.6\%$ coverage.

---

## 8. Weather Experiment Provenance Statement

> [!IMPORTANT]
> **Data Provenance Disclosure:**
> 1. **Historical Training & Evaluation (2002–2018):** All historical weather features in the 2002–2018 dataset use **simulated forecast errors modeled via an AR(1) proxy process derived from genuine surface observations (METAR)**. They are **not** retrospective historical runs from archived NOAA HRRR cycles.
> 2. **Operational Real-Time Integration:** Live operational forecasting uses **genuine, real-time NOAA HRRR GRIB2 forecasts** retrieved from NOAA NOMADS / AWS Open Data and decoded via `ecCodes`.
> 3. **Integrity Rule:** In all scientific and technical communications, simulated historical weather error proxies must never be represented as genuine archived HRRR NWP cycles.

---

## 9. Model Artifact Hash Preservation Audit

All 15 pre-trained production model files, encoders, and calibration artifacts in `models/` were verified before and after Phase 2 execution. Zero weights, calibration parameters, or configuration files were modified.

| Artifact File | Description | SHA-256 Checksum (Pre & Post Evaluation) | Verified Intact |
| :--- | :--- | :--- | :---: |
| `adaptive_calibration_params.json` | Adaptive Diurnal CQR Parameters | `4f8fc5fa95620d54fdb8fe499bdf3cfca5400e811e74bb17dac15aa70e6362ae` | **YES** |
| `conformal_calibration_params.json` | Global Scalar CQR Parameters | `42f5095e8161c8f37fbddc4de52d2d4cc844df35566ce0ecfd9d3d3910dd0bb1` | **YES** |
| `pjm_probabilistic_forecaster_phase9.joblib` | Production Multi-Quantile Model | `51b2bd6837dcc9b62907a7486dcb9c8bbdc1861ba24c162fa99e366587cfadc3` | **YES** |
| `pjm_production_forecaster_phase8.joblib` | Production Multi-Station Forecaster | `f14d5b042afead93cc94e0653ae4e48fb45e800872f7c896bc6fb4d6d372c99d` | **YES** |
| `pjm_tree_forecaster.joblib` | Core Point Forecaster | `e3587d89c7fa1836e8631c4c2aa6d7f7ec4c1ef04fea43f8e0f29eddb8c19b76` | **YES** |
| `pjm_tree_forecaster_phase5a.joblib` | Historical Phase 5a Checkpoint | `806978e5171e1b3dbb4826ac6452fc1ba354a408669dc90731d21157cc660402` | **YES** |
| `pjm_tree_forecaster_phase5a_refit.joblib` | Historical Refit Checkpoint | `a8875bb04d344ced6d7499c6ca3d47aea0cf5053bcba475c781d9f4665798636` | **YES** |
| `pjm_tree_forecaster_phase6.joblib` | Historical Phase 6 Checkpoint | `80360ac7cd094da5feda567bee0d4fae57f4abb56db264e8a6950c857f8a5053` | **YES** |
| `pjm_tree_forecaster_phase7.joblib` | Historical Phase 7 Checkpoint | `a2c832bd15d79854f007faaf6a5d537f232f9ddd35b94c11ba1d04686288a59a` | **YES** |
| `pjm_tree_forecaster_phase7_baseline.joblib` | Historical Phase 7 Baseline | `806978e5171e1b3dbb4826ac6452fc1ba354a408669dc90731d21157cc660402` | **YES** |
| `pjm_tree_forecaster_phase7_observed.joblib` | Historical Phase 7 Observed Weather | `a2c832bd15d79854f007faaf6a5d537f232f9ddd35b94c11ba1d04686288a59a` | **YES** |
| `pjm_tree_forecaster_phase7_operational_nwp.joblib` | Historical Phase 7 NWP Model | `376c31bee5999246187aecc47e32f39f971f14c2dd627d8baf53141a2a012418` | **YES** |
| `production_feature_schema.json` | 18-Feature Production Schema | `d80e6d42a28eef0a6fbdf4cab79986743249c52fcf89dd7ece00166fc8278e39` | **YES** |
| `production_model_config.json` | Production Model Hyperparameters | `855b6574039d528d51e6a63590a9b3880257af38fba627ca35e74293815dda13` | **YES** |
| `weather_imputer_params.json` | Diurnal Weather Imputer Parameters | `213f18685ff2cb6b6b314b311665f6ee711be26b2da33632e18eb5db6c615ac9` | **YES** |

---

## 10. Generated Artifacts Summary

1. `docs/improvement/forecast_accuracy_report.md` (This document)
2. `outputs/improvement/forecast_metrics.json` (Machine-readable comprehensive metric dictionary)
3. `outputs/improvement/forecast_predictions.csv` ($65,319$ rows across H=1, 24, 48 with point predictions, residuals, and prediction intervals)
4. `outputs/improvement/plots/actual_vs_predicted_h1.png` (Publication-grade time-series zoom comparison)
5. `outputs/improvement/plots/residual_distribution_h1.png` (Error density with normal distribution fit)
6. `outputs/improvement/plots/multi_horizon_mae_comparison.png` (Comparative bar chart across all horizons and baselines)
7. `tests/test_forecast_evaluation.py` (15 automated regression test cases)
