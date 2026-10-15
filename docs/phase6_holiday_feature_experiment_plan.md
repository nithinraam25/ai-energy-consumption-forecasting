# Phase 6: Controlled Public-Holiday Feature Experiment Report

**Project**: AI-Powered Energy Consumption Forecasting System  
**Document Status**: COMPLETED — EXPERIMENT EXECUTED & CANDIDATE REJECTED  
**Target Model**: `EnergyTreeForecaster` (`HistGradientBoostingRegressor`)  
**Hyperparameter Configuration**: Fixed Phase 5A optimal parameters (`learning_rate=0.05`, `max_iter=200`, `max_depth=8`, `max_leaf_nodes=45`, `l2_regularization=0.5`, `min_samples_leaf=20`, `random_state=42`)  
**Baseline Reference**: Phase 5A Tuned Model (`Val MAE = 564.93 MW`, `Test MAE = 632.42 MW` train-only)  
**Evaluation Target**: `target_next_hour` (hourly regional electrical power demand in Megawatts - MW)  

---

## 1. Executive Summary & Gate Decision

| Protocol Dimension | Pre-Registered Specification | Measured Outcome | Gate Finding |
| :--- | :--- | :--- | :---: |
| **Modeling Convention** | Nominal parsed-timestamp convention | Consistently applied across baseline & candidate | **COMPLIANT** |
| **Calendar Definition** | US Federal statutory calendar & observed rules | Verified deterministic calendar (2000–2025) | **COMPLIANT** |
| **Validation Benchmark** | Phase 5A reference: **564.93 MW** MAE | Baseline Val MAE: **564.93 MW** (exact match) | **VERIFIED** |
| **Candidate Validation** | Candidate Val MAE | Candidate Val MAE: **565.62 MW** (+0.69 MW / +0.12%) | **MEASURED** |
| **Acceptance Threshold** | Candidate Val MAE $\le 563.93\text{ MW}$ ($\ge 1.0\text{ MW}$ gain) | Val MAE was **565.62 MW** ($> 563.93\text{ MW}$) | **NOT MET** |
| **Test Holdout Isolation**| Zero test-set usage in selection | Candidate evaluated once on holdout for benchmark | **VERIFIED** |
| **Candidate Test MAE** | Diagnostic benchmark across 21,759 samples | Candidate Test MAE: **637.40 MW** (vs Base **632.42 MW**) | **CONFIRMATORY** |
| **Artifact Safety** | Phase 5A models/forecasts preserved | All Phase 5A artifacts preserved without modification | **PRESERVED** |

### Formal Decision: REJECT CANDIDATE & RETAIN PHASE 5A BENCHMARK
The candidate feature set incorporating binary `is_holiday` under the nominal timestamp convention achieved a Validation MAE of **565.62 MW**, failing to satisfy the predefined acceptance criterion ($\le 563.93\text{ MW}$). In accordance with the pre-registered experimental protocol, the candidate is **REJECTED**, and the approved Phase 5A 14-feature model remains the system's champion forecasting architecture.

---

## 2. Epistemic Separation of Dataset Evidence and Modeling Choices

To maintain scientific integrity and auditability, facts derived from the source files, modeling conventions adopted for the experiment, experimental assumptions, and measured empirical results are clearly distinguished:

### 2.1 Source Facts (Empirically Confirmed from `data/raw/PJME_hourly.csv`)
- **Block Organization**: In the raw, un-sorted CSV, data is packaged into 6,057 contiguous daily blocks. Across all blocks, rows begin at `01:00:00` of Date $D$, increment hourly through `23:00:00` of Date $D$, and terminate at `00:00:00` of Date $D+1$ (exactly 0 mismatches across 145,366 rows).
- **Missing Hours**: In every single year from 2002 to 2018, the spring Daylight Saving Time clock shift results in a missing `03:00:00` timestamp, while `02:00:00` is present.
- **Clock Range**: Clock hours range strictly from `00:00:00` to `23:00:00`. No `24:00:00` timestamps exist.
- **Utility Domain Practice**: PJM Interconnection metering systems (eDART / Data Miner) publish operating day demand under Hour Ending (HE 01–24). When converted by standard parsers, `24:00:00` of Date $D$ rolls over into `00:00:00` of Date $D+1$.

### 2.2 Modeling Conventions (Explicit Engineering Choices)
- **Nominal Parsed-Timestamp Convention**: For this experiment, all timestamps are consumed using their nominal parsed clock datetime. A timestamp of `2016-07-04 00:00:00` is treated as belonging to calendar date July 4, 2016.
- **Preservation of Existing Features**: Existing calendar features (`hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, trigonometric harmonics) are preserved without modification. Globally shifting existing features by $-1\text{ hour}$ would alter the baseline representation, creating a different experiment rather than testing the incremental value of holiday features.
- **Nominal Target Date Holiday Mapping**: The holiday feature `is_holiday` is derived directly from the nominal calendar date of the forecast target timestamp $T_{\text{target}}$:
  $$T_{\text{target}} = T_{\text{origin}} + 1\text{ hour}$$
  $$\text{is\_holiday}(T_{\text{origin}}) = \mathbf{1}_{\{T_{\text{target}}.\text{date} \in \text{Observed US Holidays}\}}$$

### 2.3 Experimental Assumptions
- **Public Holiday Calendar Determinism**: US statutory holidays (5 U.S.C. 6103) are established by law decades in advance; calendar dates are deterministic and publicly available at forecast origin $T_{\text{origin}}$.
- **No Target Leakage**: The holiday feature uses exclusively deterministic calendar lookups. Future consumption actuals ($y_{t+1}$) and test partition statistics are completely excluded.
- **Isolation of Architectural Variables**: Hyperparameters were frozen to Phase 5A optimal values (`lr=0.05`, `max_iter=200`, `depth=8`, `leaves=45`, `l2=0.5`) to isolate the effect of feature representation from hyperparameter variability.

### 2.4 Measured Results
- All validation and test metrics were computed across identical eligible sample populations.
- Phase 5A baseline validation score of **564.93 MW** was reproduced to the exact decimal.
- Adding binary `is_holiday` changed validation MAE to **565.62 MW** (-0.69 MW / +0.12% degradation).
- Evaluating on the 21,759 aligned test holdout yielded **637.40 MW** MAE for the candidate vs **632.42 MW** MAE for the baseline.

---

## 3. Holiday Feature Definition & Implementation Details

### 3.1 Public Calendar Authority & Observed Rules
- **Jurisdiction**: United States Federal Statutory Holidays (5 U.S.C. 6103) as administered by the Office of Personnel Management (OPM) and Federal Reserve.
- **Implementation**: Built-in `pandas.tseries.holiday.USFederalHolidayCalendar`, spanning 2000 to 2025 (covering the historical 2002–2018 dataset without external dependencies).
- **Observed Shift Rules**:
  - Holidays falling on **Sunday** are observed on **Monday** (e.g., July 4, 2010 was Sunday $\to$ observed Monday July 5, 2010).
  - Holidays falling on **Saturday** are observed on **Friday** (e.g., July 4, 2009 was Saturday $\to$ observed Friday July 3, 2009).
- **Statutory Scope**:
  1. New Year's Day (January 1)
  2. Martin Luther King Jr. Day (3rd Monday in January)
  3. Washington's Birthday / Presidents' Day (3rd Monday in February)
  4. Memorial Day (Last Monday in May)
  5. Independence Day (July 4)
  6. Labor Day (1st Monday in September)
  7. Columbus Day / Indigenous Peoples' Day (2nd Monday in October)
  8. Veterans Day (November 11)
  9. Thanksgiving Day (4th Thursday in November)
  10. Christmas Day (December 25)  
  *(Juneteenth was enacted in 2021 and does not apply to the historical series).*

### 3.2 Target Timestamp Alignment
- At forecast origin $T_{\text{origin}}$, the target observation is $T_{\text{target}} = T_{\text{origin}} + 1\text{ hour}$.
- For example:
  - At $T_{\text{origin}} = \text{2016-07-03 23:00:00}$: $T_{\text{target}} = \text{2016-07-04 00:00:00} \implies \text{is\_holiday} = 1$.
  - At $T_{\text{origin}} = \text{2016-07-04 22:00:00}$: $T_{\text{target}} = \text{2016-07-04 23:00:00} \implies \text{is\_holiday} = 1$.
  - At $T_{\text{origin}} = \text{2016-07-04 23:00:00}$: $T_{\text{target}} = \text{2016-07-05 00:00:00} \implies \text{is\_holiday} = 0$.
- All 24 target hours of the holiday day receive `is_holiday = 1`.

---

## 4. Controlled Experimental Protocol & Verification

### 4.1 Chronological Partitions
Partitions were maintained without shuffling:
- **Training Set (70%)**: 2002-01-08 01:00:00 to 2013-08-14 07:00:00 (**101,551 samples**)
- **Validation Set (15%)**: 2013-08-14 08:00:00 to 2016-02-07 14:00:00 (**21,760 samples**)
- **Holdout Test Set (15%)**: 2016-02-07 15:00:00 to 2018-08-02 23:00:00 (**21,759 aligned evaluation samples**)

### 4.2 Candidate Feature Sets
- **Baseline Feature Matrix (14 features)**:
  `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `lag_1`, `lag_24`, `lag_168`, `rolling_mean_24`, `rolling_std_24`.
- **Candidate Feature Matrix (15 features)**:
  Baseline 14 features $+$ `is_holiday`.

### 4.3 Validation Results (21,760 Samples)
Saved to: [`outputs/metrics/pjm_holiday_tuning.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_holiday_tuning.csv)

| Configuration | Features | Val MAE (MW) | Val RMSE (MW) | Val $R^2$ | Val MAPE (%) | $\Delta$ Val MAE vs Base | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Phase 5A Reference)** | 14 | **564.93** | 754.10 | 0.9851 | 1.77% | Reference | Approved |
| **Candidate (`is_holiday`)** | 15 | **565.62** | 751.01 | 0.9852 | 1.78% | +0.69 MW (+0.12%) | **REJECTED** |

**Predefined Acceptance Rule Evaluation**:
- Rule: Candidate accepted if $\text{Val MAE} \le 563.93\text{ MW}$ ($\ge 1.0\text{ MW}$ improvement).
- Actual: $565.62\text{ MW} > 563.93\text{ MW}$.
- Outcome: **REJECTED**.

### 4.4 Test Holdout Benchmark Results (21,759 Aligned Samples vs. 21,762 Full Slice)
Saved to: [`outputs/metrics/pjm_model_comparison_phase6.csv`](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/outputs/metrics/pjm_model_comparison_phase6.csv)

#### Common Aligned Benchmark (21,759 Samples)
| Model Architecture | Training Scope | Features | Test MAE (MW) | Test RMSE (MW) | Test $R^2$ | Test MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence ($t+1$)** | Naive Lag ($t-1$) | 1 | 1,055.50 | 1,355.20 | 0.9558 | 3.43% |
| **Seasonal-Naive (24h)** | Seasonal Lag ($t-24$) | 1 | 2,207.98 | 3,028.96 | 0.7790 | 7.02% |
| **EnergyTree_HistGB (Baseline Tree)** | Train-Only (70%) | 14 | **632.42** | **833.44** | **0.9833** | **2.03%** |
| **EnergyTree_Holiday_Candidate** | Train-Only (70%) | 15 | **637.40** | **837.79** | **0.9831** | **2.05%** |
| *EnergyTree_HistGB_Tuned_Refit (Phase 5A)* | *Train + Val (85%)* | *14* | *584.33* | *776.04* | *0.9855* | *1.87%* |

#### Full Test Slice Evaluation (All 21,762 Rows)
Both tree models possess complete predictor feature vectors across all 21,762 rows of the holdout slice without any missing values:
- **Baseline Tree (`EnergyTree_HistGB`, Train-Only)**: MAE **632.39 MW**, RMSE **833.41 MW**.
- **Candidate Tree (`EnergyTree_Holiday_Candidate`, Train-Only)**: MAE **637.37 MW**, RMSE **837.76 MW**.
- **Delta MAE on Full Slice**: **-4.98 MW** (Candidate remains 4.98 MW worse than baseline across all 21,762 rows).

> [!IMPORTANT]
> **Test Row Alignment & Methodological Notes**:
> 1. **Why Exactly 3 Rows are Excluded in Shared Benchmark**:
>    The shared multi-model benchmark evaluates 21,759 samples rather than 21,762 because `SeasonalNaiveForecaster` requires the observation at target timestamp minus 24 hours. For target timestamps `2016-03-14 03:00`, `2017-03-13 03:00`, and `2018-03-12 03:00`, the 24-hour lag fell on the spring Daylight Saving Time skipped hour (`03:00:00` missing), resulting in `NaN` for seasonal-naive. To ensure identical common sample alignment across all baselines, `compare_models` excluded these 3 rows. The tree models themselves can generate predictions for all 21,762 rows.
> 2. **Validation Gate Primacy**:
>    The candidate was **rejected strictly on the validation gate** ($\text{Val MAE} \le 563.93\text{ MW}$). Test metrics were evaluated post-hoc for diagnostic transparency only and were not used in any selection or acceptance decisions.
> 3. **Train-Only Benchmark vs. Production Refit Distinction**:
>    The direct, like-for-like comparator is the **train-only baseline tree (`Test MAE = 632.42 MW`)**, which was fitted on identical 70% training data. The approved production champion (`EnergyTree_HistGB_Tuned_Refit`, `Test MAE = 584.33 MW`) was refitted on 85% data (training + validation) and remains the active champion.
> 4. **Holiday Scope Caveat**:
>    The `is_holiday` feature represents official observed US federal statutory holidays under 5 U.S.C. 6103. It does not capture non-federal, culturally prominent, or commercial celebrations that may influence grid demand (e.g., Super Bowl Sunday, Valentine's Day, St. Patrick's Day, Good Friday/Easter, Halloween, Black Friday, or Christmas Eve).
> 5. **Source Timestamp Ambiguity Caveat**:
>    The source dataset's physical interval semantics (Hour Beginning vs. Hour Ending) remain inconclusive from the CSV alone. The nominal timestamp convention adopted here is an explicit modeling standard to prevent baseline feature shifts, not a resolution of physical ground truth.

---

## 5. Artifact Verification & Cryptographic Baseline

### 5.1 Preservation Audit
All Phase 5A artifacts were inspected to ensure that no Phase 6 execution wrote to or overwrote Phase 5A files:
- `models/pjm_tree_forecaster_phase5a.joblib`
- `models/pjm_tree_forecaster_phase5a_refit.joblib`
- `outputs/metrics/pjm_validation_tuning.csv` and `.json`
- `outputs/metrics/pjm_model_comparison_phase5a.csv` and `.json`
- `outputs/forecasts/pjm_test_predictions_phase5a.csv`

### 5.2 Verification Methodology Disclosure
Prior to the Phase 6 execution, SHA-256 hashes had not been pre-calculated for Phase 5A artifacts. Therefore, an absolute byte-for-byte pre/post hash comparison cannot be claimed. However:
1. Filesystem metadata confirms that all Phase 5A artifact modification timestamps remain at **2026-10-14 18:13:xx** (the Phase 5A completion timestamp), whereas Phase 6 files bear timestamps of **2026-10-14 19:18:xx**.
2. File sizes match the logged Phase 5A outputs exactly.
3. Phase 6 artifacts were strictly directed to dedicated filenames (`*_phase6.*` and `pjm_holiday_tuning.*`).

### 5.3 Baseline SHA-256 Hashes (Recorded Post-Audit)
The current cryptographic hashes are recorded below as an immutable audit baseline:

| File Path | Size (Bytes) | Last Modified | SHA-256 Digest |
| :--- | :---: | :---: | :--- |
| `models/pjm_tree_forecaster_phase5a.joblib` | 1,052,089 | 2026-10-14 18:13:20 | `806978e5171e1b3dbb4826ac6452fc1ba354a408669dc90731d21157cc660402` |
| `models/pjm_tree_forecaster_phase5a_refit.joblib` | 1,052,089 | 2026-10-14 18:13:23 | `a8875bb04d344ced6d7499c6ca3d47aea0cf5053bcba475c781d9f4665798636` |
| `outputs/metrics/pjm_validation_tuning.csv` | 899 | 2026-10-14 18:13:18 | `a61bd26e70c9ff786d3647372f64eeeecc5cb7eebc8e5d311c7cdf333c685757` |
| `outputs/metrics/pjm_model_comparison_phase5a.csv` | 575 | 2026-10-14 18:13:24 | `8370aefebe1536ec2532ac8a964ea2b11b2373ab4489c1c214f4654a8ae21ae3` |
| `outputs/forecasts/pjm_test_predictions_phase5a.csv`| 2,184,813 | 2026-10-14 18:13:24 | `03b165c301c2e8f6f75f1ae428ff710800e7faed6331028cf9647cf80f82499f` |
| `models/pjm_tree_forecaster_phase6.joblib` | 1,052,171 | 2026-10-14 19:18:52 | `80360ac7cd094da5feda567bee0d4fae57f4abb56db264e8a6950c857f8a5053` |
| `outputs/metrics/pjm_holiday_tuning.csv` | 232 | 2026-10-14 19:18:52 | `930b4dde32194f91a0b9db73471cde7ba2de760836548390a68d867ca8837fb3` |
| `outputs/metrics/pjm_model_comparison_phase6.csv` | 467 | 2026-10-14 19:18:53 | `7b9080119dd66849503788fc4e791a1a2b33607c5d64482817a010f8e4a0db81` |
| `outputs/forecasts/pjm_test_predictions_phase6.csv`| 1,782,752 | 2026-10-14 19:18:53 | `cef2d48db1be6b41115396bc32c46344340f776b2e79f213e41d8ed38b92e9f4` |

---

## 6. Root-Cause Analysis & Remaining Limitations

### 6.1 Diagnostic Hypotheses on Degradation Mechanism
1. **Autoregressive Masking by `lag_1`**:
   In an hourly one-step-ahead forecasting regime ($H=1$), `lag_1` carries intense autoregressive inertia. Once demand begins falling on a holiday morning, `lag_1` already captures the reduced load profile for the subsequent 23 hours of the day. A static binary flag provides little incremental split value.
2. **Boundary Discrepancy under Nominal Clock Time**:
   Because PJM load data was originally logged under Hour Ending (HE), timestamp `00:00:00` represents the interval from 23:00 to 24:00 of the preceding day. Tagging `00:00:00` with `is_holiday = 1` forces the tree to predict holiday-level demand during an hour that actually reflects the prior evening's load.
3. **Lack of Interaction with Day-of-Week**:
   Holidays that fall on Thursdays (e.g. Thanksgiving) exhibit high `lag_24` (Wednesday business day load), whereas holidays that fall on Mondays (Memorial Day, Labor Day) exhibit low `lag_24` (Sunday weekend load). A simple un-interacted indicator does not capture this asymmetry.

### 6.2 Remaining Limitations & Timestamp Caveats
- **Persistent Timestamp Semantics Caveat**: While empirical block structures indicate that source PJM tables originated under Hour Ending (HE), the absence of upstream Kaggle extraction code means the physical interval convention cannot be established with 100% certainty. The nominal timestamp convention adopted here is an explicit modeling decision, not a resolution of physical truth.
- **DST Handling Caveat**: The source dataset omits the 25th hour during autumn clock shifts for 2002–2013, making lossless UTC timeline reconstruction fundamentally impossible.
- **Current Operational Recommendation**: Retain the approved **Phase 5A refit tree model** (`Test MAE = 584.33 MW`) as the primary forecaster. Future work should prioritize meteorological covariates and recursive multi-step forecasting horizons ($H \ge 24$).

