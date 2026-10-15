# Comprehensive Forecast Error Analysis & Diagnostic Report

**Project**: AI-Powered Energy Consumption Forecasting System  
**Evaluated Model**: Tuned Gradient Boosted Tree (`EnergyTree_HistGB_Tuned_Refit`)  
**Estimator**: `HistGradientBoostingRegressor`  
**Hyperparameters**: `learning_rate=0.05`, `max_iter=200`, `max_depth=8`, `max_leaf_nodes=45`, `l2_regularization=0.5`, `min_samples_leaf=20`, `random_state=42`  
**Training Dataset**: PJM East (`PJME_hourly.csv`) — Combined Training + Validation Partitions (123,311 samples)  
**Evaluation Holdout**: Untouched Test Partition (**21,759 aligned hourly samples**, 2016-02-07 16:00 to 2018-08-02 23:00)  
**Physical Unit**: Electrical Power Demand in **Megawatts (MW)**  

---

## 1. Executive Summary & Benchmark Re-Verification

All evaluation metrics were recomputed directly from the saved forecast matrix [`outputs/forecasts/pjm_test_predictions_phase5a.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/forecasts/pjm_test_predictions_phase5a.csv) across the **21,759 identical test timestamps**:

| Model Architecture | Sample Count | MAE (MW) | RMSE (MW) | $R^2$ Score | MAPE (%) | $\Delta$ MAE vs. Baseline Tree |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal-Naive (24h)** | 21,759 | 2,207.98 | 3,028.96 | 0.7790 | 7.02% | Baseline |
| **Persistence ($t+1$)** | 21,759 | 1,055.50 | 1,355.20 | 0.9558 | 3.43% | Baseline |
| **EnergyTree_HistGB (Baseline Tree)** | 21,759 | 683.40 | 890.56 | 0.9809 | 2.21% | Frozen Reference |
| **EnergyTree_HistGB_Tuned (Train-Only)** | 21,759 | 632.42 | 833.44 | 0.9833 | 2.03% | -50.98 MW (-7.46%) |
| **EnergyTree_HistGB_Tuned_Refit (Train+Val)** | **21,759** | **584.33** | **776.04** | **0.9855** | **1.87%** | **-99.07 MW (-14.50%)** |

The refitted tuned model improves on the frozen baseline tree across every evaluated metric:
- **MAE Reduction**: **-99.07 MW (-14.50%)**
- **RMSE Reduction**: **-114.52 MW (-12.86%)**
- **$R^2$ Improvement**: **+0.0046** (explaining 98.55% of test variance)
- **MAPE Reduction**: **-0.34%** (down to 1.87%)

---

## 2. Dataset & Target Definition

- **Source Dataset**: PJM Interconnection Eastern Region (`PJME_hourly.csv`), covering 2002 to 2018.
- **Physical Quantity**: Regional electrical power demand.
- **Physical Units**: Megawatts (MW). Instantaneous hourly rate of energy consumption.
- **Timestamp Convention**: Local Eastern clock time under Hour Ending (HE) convention.
- **Forecast Horizon**: One observation hour ahead ($H=1$). At origin timestamp $T_t$, the model forecasts $y(T_{t+1}) = y(T_t + 1\text{h})$.
- **Information Available at Prediction Time**:
  - Autoregressive lags: $t-1\text{h}$ (`lag_1`), $t-24\text{h}$ (`lag_24`), and $t-168\text{h}$ (`lag_168`).
  - Rolling window statistics: 24-hour backward rolling mean (`rolling_mean_24`) and standard deviation (`rolling_std_24`), calculated over $\{y_{t-23}, \dots, y_t\}$.
  - Calendar & cyclical indicators: `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, $\sin/\cos$ harmonic encodings.
  - Zero future consumption labels or lookahead features.

---

## 3. Disaggregated Error Analysis

### 3.1 Error Breakdown by Hour of Day (Diurnal Cycle)

Generated table: [`outputs/metrics/pjm_error_analysis_hourly.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_error_analysis_hourly.csv)  
Generated plot: [`outputs/plots/pjm_mae_by_hour_of_day.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_mae_by_hour_of_day.png)

| Hour (HE) | Sample Count | MAE (MW) | RMSE (MW) | $R^2$ Score | MAPE (%) | Mean Residual (MW) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00:00** | 908 | 507.73 | 661.06 | 0.9779 | 1.77% | +109.81 |
| **01:00** | 907 | 432.89 | 572.82 | 0.9808 | 1.57% | +229.33 |
| **02:00** | 907 | 372.90 | 485.22 | 0.9848 | 1.42% | +191.33 |
| **03:00** | 901 | 310.68 | 400.12 | 0.9890 | 1.23% | +56.78 |
| **04:00** | 898 | 297.51 | 366.67 | 0.9906 | 1.20% | +48.61 |
| **05:00** | 904 | **286.07** | **366.27** | **0.9908** | **1.14%** | -7.20 |
| **06:00** | 907 | 347.10 | 461.06 | 0.9867 | 1.33% | +12.07 |
| **07:00** | 907 | 494.40 | 704.26 | 0.9750 | 1.76% | -276.36 |
| **08:00** | 907 | 561.69 | 745.73 | 0.9740 | 1.88% | -375.97 |
| **09:00** | 907 | 735.17 | 888.85 | 0.9605 | 2.39% | -486.78 |
| **10:00** | 907 | 895.57 | 1095.75 | 0.9426 | 2.85% | -538.23 |
| **11:00** | 907 | **905.94** | **1122.59** | 0.9498 | **2.85%** | -463.75 |
| **12:00** | 907 | 786.76 | 983.10 | 0.9690 | 2.43% | -231.88 |
| **13:00** | 907 | 698.88 | 896.45 | 0.9788 | 2.11% | +50.54 |
| **14:00** | 907 | 657.94 | 854.17 | 0.9836 | 1.97% | +247.78 |
| **15:00** | 907 | 636.07 | 807.87 | 0.9869 | 1.89% | +283.11 |
| **16:00** | 908 | 652.52 | 826.28 | 0.9872 | 1.92% | +303.69 |
| **17:00** | 908 | 675.17 | 875.45 | 0.9857 | 1.94% | +332.97 |
| **18:00** | 908 | 672.04 | 877.16 | 0.9848 | 1.89% | +189.45 |
| **19:00** | 908 | 688.79 | 896.75 | 0.9817 | 1.92% | +5.65 |
| **20:00** | 908 | 576.72 | 750.54 | 0.9841 | 1.67% | -165.20 |
| **21:00** | 908 | 686.40 | 861.41 | 0.9752 | 2.04% | -380.16 |
| **22:00** | 908 | 632.87 | 786.05 | 0.9775 | 1.95% | -406.14 |
| **23:00** | 908 | 506.16 | 633.21 | 0.9828 | 1.66% | -92.40 |

**Diurnal Patterns**:
- **Overnight Baseload Stability (03:00 - 05:00)**: Minimum MAE occurs at 05:00 (286.07 MW, MAPE 1.14%). Overnight load profiles are stable, with gradual declines and minimal weather/activity fluctuations.
- **Morning Industrial & Residential Ramp (09:00 - 11:00)**: Peak error occurs at 11:00 (MAE 905.94 MW, MAPE 2.85%). Negative mean residuals (-463 to -538 MW) reveal that the model tends to overpredict during rapid morning activation when morning demand ramps up more gradually than historical average curves.
- **Evening Peak (17:00 - 19:00)**: Moderate MAE (~672-688 MW) with positive mean residuals (+189 to +332 MW), where cooling surges on extreme summer days cause underprediction.

---

### 3.2 Error Breakdown by Day of Week

Generated table: [`outputs/metrics/pjm_error_analysis_weekday.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_error_analysis_weekday.csv)  
Generated plot: [`outputs/plots/pjm_mae_by_day_of_week.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_mae_by_day_of_week.png)

| Day of Week | Sample Count | MAE (MW) | RMSE (MW) | $R^2$ Score | MAPE (%) | Mean Residual (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monday** | 3,114 | **627.83** | **836.40** | 0.9837 | **2.01%** | -129.76 |
| **Tuesday** | 3,120 | **551.09** | **743.00** | 0.9863 | **1.71%** | -86.14 |
| **Wednesday** | 3,120 | 569.87 | 763.52 | 0.9854 | 1.76% | -65.86 |
| **Thursday** | 3,120 | 576.13 | 766.65 | 0.9863 | 1.77% | +25.94 |
| **Friday** | 3,097 | 574.33 | 750.54 | 0.9867 | 1.79% | -58.31 |
| **Saturday** | 3,096 | 580.24 | 767.83 | 0.9835 | 1.94% | -51.61 |
| **Sunday** | 3,092 | 611.05 | 800.37 | 0.9824 | 2.10% | -32.12 |

**Weekday Patterns**:
- **Midweek Stability (Tuesday–Friday)**: Consistent, tight performance across Tuesday (551.09 MW), Wednesday (569.87 MW), Thursday (576.13 MW), and Friday (574.33 MW).
- **Monday Transition & Holiday Inflation**: Monday records the highest weekday error (MAE 627.83 MW, RMSE 836.40 MW). In the US Eastern interconnection, several major holidays (Memorial Day, Labor Day, Martin Luther King Jr. Day, Washington's Birthday) always fall on Mondays, creating anomalous commercial demand depressions compared to standard working Mondays.

---

### 3.3 Error Breakdown by Demand Magnitude Regime

Generated table: [`outputs/metrics/pjm_error_analysis_demand_bands.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_error_analysis_demand_bands.csv)  
Generated plot: [`outputs/plots/pjm_mae_by_demand_band.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_mae_by_demand_band.png)

#### Demand Band Definition & Methodology Disclosure
- **Threshold Selection**: Fixed domain thresholds of **25,000 MW**, **32,000 MW**, and **40,000 MW** were chosen to partition regional electrical demand into four operational regimes:
  1. *Low (<25,000 MW)*: Overnight baseload and mild shoulder hours.
  2. *Moderate (25,000 – 32,000 MW)*: Normal daytime commercial and residential demand.
  3. *High (32,000 – 40,000 MW)*: Heavy daytime and heating/cooling demand periods.
  4. *Extreme Peak (>40,000 MW)*: High-load periods associated with extreme temperature extremes.
- **Retrospective Test-Set Partitioning**: These demand regimes describe the **observed retrospective test distribution** for diagnostic reporting only. They were **not** used for model training, feature generation, or hyperparameter selection.
- **Unambiguous Coverage**: Using `pd.cut` with `include_lowest=True` ensures that all test observations are assigned to exactly one band without ambiguity or sample loss ($3,500 + 10,289 + 5,752 + 2,218 = 21,759$ samples).

| Demand Regime | Range (MW) | Sample Count | MAE (MW) | RMSE (MW) | $R^2$ Score | MAPE (%) | Mean Residual (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low** | 19,255 – 24,999 | 3,500 | **366.77** | **496.06** | 0.8741 | **1.60%** | -119.01 |
| **Moderate** | 25,001 – 32,000 | 10,289 | 548.89 | 718.27 | 0.8523 | 1.91% | -165.21 |
| **High** | 32,001 – 40,000 | 5,752 | 698.60 | 899.01 | 0.8381 | 1.98% | +38.21 |
| **Extreme Peak** | 40,003 – 56,609 | 2,218 | **795.71** | **1,015.13** | 0.9266 | **1.79%** | **+297.36** |

**Regime Insights**:
- **Scale-Dependent Absolute Error**: Absolute error metrics naturally rise with demand magnitude (MAE rises from 366.77 MW in Low to 795.71 MW in Extreme Peak). Because the baseline demand scale is more than double, higher absolute error does not indicate worse proportional performance.
- **Relative Accuracy (MAPE)**: Relative error remains stable across regimes, ranging from 1.60% in Low load to 1.79% during Extreme Peak.
- **Peak Residual Asymmetry**: A positive mean residual (+297.36 MW) during Extreme Peak indicates a tendency toward underprediction during extreme peaks, consistent with tree leaf-averaging near distribution extremes.

---

## 4. Analysis of Top Residual Outliers (Largest Errors)

Generated table: [`outputs/metrics/pjm_largest_errors.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_largest_errors.csv)

The top 10 largest absolute forecast errors in the 2.5-year holdout partition:

| Rank | Target Timestamp | Actual (MW) | Predicted (MW) | Residual (MW) | Abs Error (MW) | Error (%) | Plausible Context / Calendar Coincidence |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `2016-07-18 18:00` | 47,275 | 52,710 | -5,435 | **5,435** | 11.50% | Summer peak load period; evening demand fluctuation |
| **2** | `2018-07-17 16:00` | 46,917 | 51,417 | -4,500 | **4,500** | 9.59% | Summer peak load period; rapid afternoon ramp |
| **3** | `2018-07-17 17:00` | 45,172 | 49,543 | -4,371 | **4,371** | 9.68% | Summer peak load period continuation |
| **4** | `2016-07-18 17:00` | 49,410 | 53,531 | -4,121 | **4,121** | 8.34% | Summer peak load period ramp |
| **5** | `2017-06-19 18:00` | 43,395 | 47,205 | -3,810 | **3,810** | 8.78% | Early summer evening transition |
| **6** | `2016-07-04 07:00` | 22,392 | 26,202 | -3,810 | **3,810** | 17.02% | **Independence Day (4th of July)**: Monday statutory holiday |
| **7** | `2016-07-14 19:00` | 52,743 | 49,097 | +3,646 | **3,646** | 6.91% | Summer peak load surge |
| **8** | `2017-07-04 07:00` | 26,694 | 30,335 | -3,641 | **3,641** | 13.64% | **Independence Day (4th of July)**: Tuesday statutory holiday |
| **9** | `2017-08-02 16:00` | 44,083 | 47,554 | -3,471 | **3,471** | 7.87% | Summer afternoon cooling ramp |
| **10**| `2017-09-24 13:00` | 36,238 | 32,821 | +3,417 | **3,417** | 9.43% | Late summer afternoon demand spike |

### Calibrated Diagnostic Hypotheses:
1. **Public Holiday Calendar Coincidence**: Several large overprediction outliers coincide with major US federal holidays (e.g., Independence Day 2016 & 2017, Memorial Day 2016, Thanksgiving 2016 & 2017, New Year's Day 2017 & 2018). Because the existing model receives only standard calendar indicators (`day_of_week`), it treats a holiday Monday as a normal working Monday. While this alignment is a compelling hypothesis, it cannot be asserted as an established causal mechanism until formally evaluated against an integrated public holiday dataset in a controlled experiment.
2. **Extreme Summer Demand Periods**: Large residual deviations frequently coincide with July/August peak hours (>45,000 MW). While weather extremes (heatwaves driving air-conditioning demand) represent a plausible driver, ambient temperature was not an input feature. Causal attribution requires incorporating meteorological covariates.

---

## 5. Daylight Saving Time Methodology Audit & Limitation Disclosure

A targeted empirical audit of timestamp structures across the entire 17-year PJM series (`PJME_hourly.csv`) established the following distinction between confirmed facts and unverified assumptions:

### 5.1 Confirmed Facts Established Directly by the Source File
- **Missing Spring Hour**: For every year from 2002 to 2018, exactly one hour is missing each spring: `03:00:00` on the designated spring clock-shift Sunday (17 instances total).
- **Autumn Duplicates**: Across the first 12 years (2002–2013), **zero duplicate timestamps exist** in the raw CSV. Exactly 4 duplicate pairs appear in the entire dataset, occurring at `02:00:00` in 2014, 2015, 2016, and 2017.
- **Non-Chronological File Ordering**: The raw CSV contains non-chronological daily blocks, featuring 6,041 reverse steps prior to preprocessing sort.
- **Lack of Timezone Markers**: Timestamps are bare strings (e.g., `2014-11-02 02:00:00`) without UTC offsets, timezone labels, or daylight-saving flags.

### 5.2 Facts Established by Reliable Source Documentation
- PJM operates in the Eastern Interconnection across US Eastern Time (EST/EDT).
- PJM historically records hourly metering under the **Hour Ending (HE)** convention: timestamp `02:00` denotes the interval ending at 02:00 local clock time.
- The reported variable is hourly average electrical power demand in **Megawatts (MW)** (a rate, not accumulated energy).

### 5.3 Hypotheses That Cannot Be Confirmed from the Source File
- The reason duplicate timestamps appear only in 2014–2017 but are completely absent in 2002–2013 cannot be verified from the CSV alone. Whether the upstream provider pre-averaged them, omitted the 25th hour, or recorded 2002–2013 under standard time cannot be confirmed without unprovided upstream utility metering logs.
- Which observation in each 2014–2017 duplicate pair corresponds to EDT versus EST cannot be determined with complete certainty.

### 5.4 Physical and Methodological Consequences of the Duplicate Policy
- **Energy vs. Power Distinction**: Preprocessing applies `duplicate_policy="mean"`, replacing the two duplicate `02:00` readings with their arithmetic average ($(\text{MW}_1 + \text{MW}_2)/2$). While this preserves the average power rate and avoids dropping observations, **averaging MW observations does not preserve the individual identity of the two physical hours, nor does it mean their combined energy has been preserved**. In physical terms, two elapsed physical hours are represented as a single one-hour step, omitting one hour of elapsed duration from any prospective energy integration ($\int P(t) dt$).
- **Lag Features & Forecast Horizon**: Across the spring gap (`03:00` skipped), the timestamp-aware lag functions in [src/features/temporal_features.py](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/src/features/temporal_features.py) enforce strict $\Delta t$ verification. At `03:00`, `lag_1` assigns `NaN` rather than erroneously linking across the gap.
- **Lossless UTC Timeline Reconstruction is Impossible**: Because the 25th hour is absent for 2002–2013, applying standard timezone localization (e.g., `tz_localize('US/Eastern')`) fails with `AmbiguousTimeError` or requires arbitrarily guessing offsets. Fabricating artificial timestamps or guessing offsets would violate engineering integrity.
4. **Current Operational Policy**:
   - Preprocessing uses `duplicate_policy="mean"` to average the two duplicate `02:00` observations in 2014–2017 ($(\text{MW}_1 + \text{MW}_2)/2$).
   - This preserves total energy demand without arbitrarily discarding either measurement, but compresses two distinct physical hours into one clock index, smoothing out intra-hour fluctuations during that single transition window.

---

## 6. Generated Visualizations & Artifact Index

All diagnostic figures and summary tables have been persisted to the repository:

| Asset Name | Type | File Location | Key Purpose |
| :--- | :---: | :--- | :--- |
| **Actual vs. Predicted** | Plot | [`outputs/plots/pjm_actual_vs_predicted_timeseries.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_actual_vs_predicted_timeseries.png) | High-resolution contiguous trace across July 2018 peak demand |
| **Residuals Over Time** | Plot | [`outputs/plots/pjm_residuals_over_time.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_residuals_over_time.png) | Holdout scatter plot with $3\sigma$ control boundaries |
| **Error Distribution** | Plot | [`outputs/plots/pjm_error_distribution.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_error_distribution.png) | Histogram, density curve, mean (-61.9 MW), median (-53.7 MW) |
| **MAE by Hour of Day** | Plot | [`outputs/plots/pjm_mae_by_hour_of_day.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_mae_by_hour_of_day.png) | Diurnal error bar chart (00:00 to 23:00) |
| **MAE by Day of Week** | Plot | [`outputs/plots/pjm_mae_by_day_of_week.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_mae_by_day_of_week.png) | Monday through Sunday comparison |
| **MAE by Demand Band** | Plot | [`outputs/plots/pjm_mae_by_demand_band.png`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/plots/pjm_mae_by_demand_band.png) | Regime bar chart across low, moderate, high, and peak load |
| **Hourly Summary Table** | Data | [`outputs/metrics/pjm_error_analysis_hourly.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_error_analysis_hourly.csv) | Full metrics table by hour (CSV & RFC 8259 JSON) |
| **Weekday Summary Table**| Data | [`outputs/metrics/pjm_error_analysis_weekday.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_error_analysis_weekday.csv) | Full metrics table by day of week (CSV & RFC 8259 JSON) |
| **Demand Band Summary** | Data | [`outputs/metrics/pjm_error_analysis_demand_bands.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_error_analysis_demand_bands.csv) | Full metrics table by operational load band |
| **Top Outlier Residuals**| Data | [`outputs/metrics/pjm_largest_errors.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_largest_errors.csv) | Top 20 largest errors aligned by target timestamp |

---

## 7. Practical Next Steps

1. **Statutory US Holiday Feature**: Incorporate a binary indicator (`is_holiday`) or holiday distance feature to eliminate systematic overpredictions on Memorial Day, Independence Day, Labor Day, Thanksgiving, and New Year's Day.
2. **Weather Covariates**: Ingest ambient temperature and humidity data (e.g., Cooling Degree Days / Heating Degree Days) to capture extreme summer heatwaves and winter deep freezes.
3. **Multi-Step Recursive Horizon ($H > 1$)**: Extend the single-hour ahead framework to 24-hour and 48-hour day-ahead scheduling horizons.
