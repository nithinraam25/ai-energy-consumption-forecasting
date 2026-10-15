# Phase 7: Controlled Public-Weather Covariate Experiment Plan

**Project**: AI-Powered Energy Consumption Forecasting System  
**Document Status**: **REVISED EXPERIMENT PLAN — AWAITING FORMAL PHASE 7B IMPLEMENTATION AUTHORIZATION**  
**Preceding Milestone**: Phase 7A (Public Weather Data Acquisition & Read-Only Audit) conditionally accepted by Project Lead  
**Implementation Constraint**: **STRICTLY ON HOLD — PHASE 7B IMPLEMENTATION IS NOT AUTHORIZED YET.** Plan and documentation review only. Zero source code added or modified in `src/`, zero weather data merged with demand, zero weather features or imputation implemented, zero models trained or evaluated, and all prior experiment artifacts preserved read-only.  
**Target Model**: `EnergyTreeForecaster` (`HistGradientBoostingRegressor`)  
**Frozen Baseline Reference**: Phase 5A Train-Only Tuned Model (`Val MAE = 564.93 MW`, `Test MAE = 632.42 MW` on 21,759 aligned rows / `632.39 MW` on 21,762 full test rows)  
**Production Champion Reference**: Phase 5A Train+Validation Refitted Model (`Test MAE = 584.33 MW` on 123,311 training samples)  
**Evaluation Target**: `target_next_hour` (hourly regional electrical power demand in Megawatts - MW)  

---

## 1. Motivation & Empirical Diagnostic Background

### 1.1 Diagnostic Findings from Phase 5B Error Analysis
In the Phase 5B comprehensive error analysis ([docs/forecast_error_analysis.md](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/docs/forecast_error_analysis.md)), the single largest cluster of severe forecast residual outliers occurred during summer peak demand periods (>40,000 MW):
- **July 18, 2016 18:00**: Residual **-5,435 MW** (predicted 52,710 MW vs actual 47,275 MW).
- **July 17, 2018 16:00**: Residual **-4,500 MW** (predicted 51,417 MW vs actual 46,917 MW).
- **July 17, 2018 17:00**: Residual **-4,371 MW** (predicted 49,543 MW vs actual 45,172 MW).
- **July 18, 2016 17:00**: Residual **-4,121 MW** (predicted 53,531 MW vs actual 49,410 MW).

During these peak hours, the model consistently over-predicted demand by 4,000 to 5,500 MW because autoregressive lag features (`lag_24`, `lag_168`) were elevated by preceding heatwave conditions while the actual afternoon demand experienced localized load curtailment or cloud cover moderation.

### 1.2 The Physical Driver: Temperature Sensitivity of Grid Load
Electrical power consumption in regional transmission organizations exhibits pronounced non-linear sensitivity to ambient air temperature:
1. **Cooling Load (Summer Air Conditioning)**: Above a base comfort threshold of approximately 65°F (18.3°C), regional electrical demand surges non-linearly due to commercial chillers and residential compressor loads.
2. **Heating Load (Winter Space Heating)**: Below 65°F, electric resistive heating and heat-pump cycling elevate grid demand during cold snaps.
3. **The Current Model's Blind Spot**: The approved Phase 5A model relies exclusively on calendar indices (`hour`, `month`, harmonics) and autoregressive consumption lags (`lag_1`, `lag_24`, `lag_168`). While `month` captures broad climatological seasonality, tree models cannot anticipate intra-seasonal heatwaves, polar vortices, or abrupt frontal passages without meteorological covariates.

---

## 2. Epistemic Classification: Realized-Weather Oracle Proxy vs. Deployable NWP Forecast

### 2.1 The Fundamental Epistemic Distinction
In operational power grid forecasting, confusing **historical realized weather** with **weather forecasts available at prediction issuance time** is a fundamental methodological error. To maintain absolute scientific clarity, two distinct experiment paradigms must be delineated:

- **Paradigm 1: Retrospective Realized-Weather Oracle Proxy (The Current Phase 7 Plan)**:  
  Ingesting historical realized station temperature at target hour $T_{\text{target}}$ represents a **retrospective realized-weather proxy or oracle-style experiment**. It answers a diagnostic question: *"How much predictive value does perfect, zero-noise temperature information at the target hour provide to our feature representation?"* It measures the theoretical upper bound of weather value. It does **not** evaluate a deployable operational forecasting model.
- **Paradigm 2: Operationally Causal Forecasting (Future Operational Deployment)**:  
  At forecast issuance time $T_{\text{origin}}$ (e.g., today at 14:00), an operational forecaster predicting demand for target hour $T_{\text{target}}$ (e.g., next hour at 15:00) **cannot observe the actual thermometer reading at $T_{\text{target}}$**—it lies in the future and does not exist. An operational system must ingest a Numerical Weather Prediction (NWP) forecast (e.g., NOAA HRRR, GFS, or NAM temperature forecasts) issued strictly prior to $T_{\text{origin}}$.

### 2.2 Forecast-Origin Information Availability Distinction
To ensure complete methodological transparency, the pipeline formally distinguishes three separate temporal concepts:
1. **Physical Measurement Timestamp ($t_{\text{meas}}$)**: The exact UTC moment the sensor was read (e.g., `13:54:00 UTC`).
2. **Publication / Transmission Timestamp ($t_{\text{pub}}$)**: The moment the observation was encoded, transmitted over the GTS/telecommunication network, and published to NOAA archives (typically 1 to 5 minutes after measurement).
3. **Pipeline Ingestion Availability ($t_{\text{avail}}$)**: The moment the record would have been ingested, parsed, validated, and made accessible to an operational forecasting model.

> [!WARNING]
> **Non-Interchangeability & Unverifiable Historical-Availability Assumption**:  
> $t_{\text{meas}}$, $t_{\text{pub}}$, and $t_{\text{avail}}$ are **not interchangeable**. In the absence of archival telecommunication network timestamps in historical NOAA ISD files, this retrospective experiment must make the **explicit, unverifiable historical-availability assumption that observations are available at their physical measurement timestamp ($t_{\text{avail}} \approx t_{\text{meas}}$)**.

### 2.3 Strict Information Availability Rules & Operational Cutoffs
To prevent information leakage under this retrospective setup:
1. **Strict Forecast-Origin Cutoff for Causal Inputs**: For any feature constructed at forecast origin $T_{\text{origin}} = T_{\text{target}} - 1\text{h}$ (such as lagged weather, rolling temperature statistics, or historical observations), **only observations satisfying $t_{\text{meas}} \le T_{\text{origin}}$ are eligible**. Any observation reading $t_{\text{meas}} > T_{\text{origin}}$ is strictly prohibited from entering origin-time inputs.
2. **Why Nearest-Observation Selection Around Nominal Hour $T$ May Select $t > T$**:  
   Under the continuous selection window $W(T) = [T - 30\text{ minutes}, \, T + 30\text{ minutes})$, observations occurring in the interval $[T, T + 30\text{ minutes})$ (e.g., at $14:05$ or $14:15$) have physical timestamps $t > T$.  
   *Critical Oracle Implication*: In the retrospective realized-weather setup, selecting weather for target nominal hour $T_{\text{target}}$ means an observation at $T_{\text{target}} - 6\text{ minutes}$ (e.g., `14:54 UTC` for $T_{\text{target}} = 15:00$) occurs **54 minutes after** the forecast origin $T_{\text{origin}} = 14:00$. This temporal dislocation is precisely why the experiment is classified as an **oracle proxy**, not a deployable forecast.
3. **Strictly Causal Forward-Fill**: Forward-fill may only use an eligible observation available at or before the forecast origin. **A future observation ($t > T_{\text{origin}}$) can never fill an earlier target feature.**
4. **Training-Only Imputation Parameters**: Imputation statistics are fitted exclusively on valid observations within the training partition and frozen before validation or test processing.
5. **Partition State Carryover**: Imputation state carried forward across chronological partitions is strictly causal (carrying the last known observation forward); it never propagates future observations backwards into earlier rows.
6. **Operational Milestone Separation (Phase 7B vs. Phase 8)**: Phase 7 establishes whether the model architecture can extract meaningful signal from localized thermal covariates under idealized oracle conditions. If successful, subsequent work will evaluate deployable NWP forecasts issued strictly prior to $T_{\text{origin}}$.

### 2.4 Asymmetric Evidentiary Value of the Proxy Experiment
The conclusions that can be drawn from a realized-weather proxy experiment are strictly asymmetric:

1. **If the Candidate Succeeds ($\text{Val MAE} \le 563.93\text{ MW}$)**:  
   A positive result confirms that local thermal signals contain measurable predictive power under the tested feature encoding. This positive outcome **motivates a subsequent operational experiment** using archived operational NWP forecasts that were genuinely available at forecast issuance time $T_{\text{origin}}$. The realized-weather performance serves as an idealized upper-bound benchmark against which operational NWP forecasts (with typical 1.0°–2.0°F forecast errors) can be evaluated.
2. **If the Candidate Fails ($\text{Val MAE} > 563.93\text{ MW}$)**:  
   A negative result **only rejects the specific tested proxy setup**—for example, indicating that a single airport station is inadequate for a 50,000 sq mile territory, that the 65°F linear proxy feature formulation is suboptimal, or that nominal timestamp alignment introduces thermal lag distortion. **Failure with realized weather does NOT prove that operational weather forecasts cannot help.** Operational forecast products provide spatial grids and multi-hour trajectories that differ fundamentally from a single-point retrospective observation.

---

## 3. Weather Source Schema, Selection Rules & Alignment Protocol

### 3.1 NOAA ISD Source Schema & Pre-Implementation Verification Requirements
Prior to implementing the data loader or ingesting records, the downloaded file schema must be inspected and verified against authoritative NOAA NCEI ISD documentation (`isd-format-document.pdf`):

| Schema Field | Technical Definition | Authoritative NOAA Representation | Pre-Implementation Verification Requirement |
| :--- | :--- | :--- | :--- |
| **`STATION`** | USAF-WBAN Station Identifier | `72408013739` (`KPHL`) | Verify station identifier matches Philadelphia International Airport exactly. |
| **`DATE`** | Observation Timestamp in UTC | ISO format `YYYY-MM-DDTHH:MM:SS` | Verify UTC timezone format and absence of unstated local offsets. |
| **`REPORT_TYPE`** | Meteorological report classification | Routine METAR = `FM-15` | Inspect actual string values. Confirm routine hourly reports are coded `FM-15`. Non-routine (`FM-16`/`SPECI`, `SY-MT`, `SOD`, `SOM`) are ineligible. |
| **`TMP`** | Dry-bulb air temperature and quality | Composite `+0150,1` or split `TMP` / `TMP_QC` | Verify whether temperature and QC code are composite or split. Confirm scaled units in tenths of °C. |
| **Missing Sentinel** | Temperature missing flag | `+9999` (scaled missing value) | Verify missing temperature sentinel matches `+9999`. |
| **Quality Flag** | Operational quality control code | `'1'` (passed all checks), `'5'` (passed manual review) | Verify quality flag definitions. Mask suspect (`'2'`,`'3'`), erroneous (`'6'`,`'7'`), and missing (`'9'`) to `NaN`. |

### 3.2 Temperature Conversion Contract
If `TMP != '+9999'` and quality flag is valid (`'1'` or `'5'`):
$$\text{temp\_c} = \frac{\text{TMP}}{10.0}, \quad \text{temp\_f} = \text{temp\_c} \times \frac{9}{5} + 32.0$$
If `TMP == '+9999'` or quality flag is suspect/erroneous, set $\text{temp\_f} = \text{NaN}$.

### 3.3 Explicit Continuous Hourly Selection Window & Deterministic Tie-Breaking
Rather than applying arbitrary heuristic minute-54 rounding or discontinuous integer approximations, the pipeline aggregates irregular METAR records to a regular hourly grid using an explicit continuous half-open interval:

1. **Continuous Window Definition**: For each target nominal UTC hour $T_{\text{target\_utc}}$:
   $$W(T) = \left[ T - 30\text{ minutes}, \, T + 30\text{ minutes} \right)$$
   Equivalently, every observation timestamp $t$ maps to nominal hour $T = \text{floor}(t + 30\text{ minutes})$.
2. **Eligibility Criteria**:
   - Must be a routine hourly METAR report (`REPORT_TYPE == 'FM-15'`).
   - Must possess a verified valid quality flag (`TMP_QC` in `{'1', '5'}`).
3. **Nearest Eligible Observation Selection**:
   If multiple eligible reports fall within $W(T)$, select the report minimizing absolute physical time distance to the nominal hour:
   $$t_{\text{selected}} = \arg\min_{t \in W(T)} |t - T_{\text{target\_utc}}|$$
   > [!IMPORTANT]
   > The selected observation is explicitly the **nearest eligible observation within the window**, NOT automatically a validated hourly average or integrated quantity.
4. **Deterministic Tie-Breaking Hierarchy**:
   If multiple candidate observations exist within $W(T)$:
   - *1st Criterion*: Smallest absolute physical time difference $|t - T|$.
   - *2nd Criterion (Equidistant Tie)*: If two observations are equidistant from $T$ (e.g., $T-15\text{m}$ and $T+15\text{m}$), select the **earlier timestamp** ($t < T$) to strictly preserve forecast causality.
   - *3rd Criterion (Quality Priority)*: Select the higher quality code rank (`'1'` preferred over `'5'`).
5. **Window Emptiness Behavior**:
   If no eligible observation exists within $W(T)$, the nominal hour is marked as `NaN` (missing) on the continuous hourly grid and passed to the causal imputation pipeline.
6. **Reproducible Boundary Verification**:
   This continuous window definition, tie-breaking, and boundary behavior are verified by executable unit tests in [tests/test_weather_window.py](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/tests/test_weather_window.py), explicitly validating timestamps at `:29:59`, `:30:00`, `:30:01`, `:54:00`, and `:00:00`.

### 3.4 Distinct Timestamp Representations & The Non-Equivalence of Nominal Matching
To avoid conflating disparate time definitions, the pipeline explicitly distinguishes four distinct timestamp concepts:
1. **NOAA UTC Measurement Timestamp ($t_{\text{meas}}$)**: Exact physical observation time in UTC recorded with minute-level precision (e.g., `2016-07-18T17:54:00Z`).
2. **Nominal UTC Target Hour ($T_{\text{target\_utc}}$)**: The standardized top-of-hour reference node on the continuous hourly grid (e.g., `2016-07-18 18:00:00 UTC`).
3. **PJME Naive Local Timestamp**: An unadorned string in `data/raw/PJME_hourly.csv` formatted as `YYYY-MM-DD HH:MM:SS` (e.g., `2016-07-18 18:00:00`). It contains **no timezone tag, no UTC offset, and no physical interval specification**.
4. **Projected Local Eastern Time**: UTC timestamps converted to `America/New_York` prevailing local time (accounting for Eastern Daylight Time UTC-4 vs. Eastern Standard Time UTC-5).

#### Authoritative Demand Metadata Limitations
- `PJME_hourly.csv` contains only two columns (`Datetime` and `PJME_MW`). There is no data dictionary or interval flag.
- In electric utility operations, hourly load is frequently recorded as **Hour Ending (HE 01:00 to 24:00)**: timestamp `18:00` represents integrated energy across `17:00:00` to `17:59:59`. If PJM demand follows HE semantics, a weather observation at `17:54` occurred near the **end** of the demand interval.
- If PJM demand follows **Hour Beginning (HB 00:00 to 23:00)**: timestamp `18:00` represents consumption across `18:00:00` to `18:59:59`. A weather observation at `17:54` occurred **before** the demand interval began.
- If PJM demand is an instantaneous top-of-hour sample at `18:00:00`, the observation at `17:54` occurred 6 minutes prior.

> [!WARNING]
> **Nominal Match Does Not Prove Physical Alignment**:  
> Matching weather at nominal $T$ to demand labeled nominal $T$ is an **explicit modeling convention**, NOT proof of physical simultaneity. Because the true physical interval convention (HE vs. HB vs. instantaneous) cannot be resolved from the demand dataset, this experiment is explicitly designated as a **sensitivity analysis under an explicit nominal alignment assumption**. Timestamps will **not** be shifted (+1h or -1h) without verified external utility provenance.

### 3.5 Predefined Validation Sensitivity Analysis Protocol
To rigorously quantify the impact of interval uncertainty without leaking test information, three candidate alignment hypotheses are predeclared for evaluation **strictly on the validation partition**:

| Hypothesis | Alignment Rule | Operational Interpretation | Evaluation Protocol |
| :--- | :--- | :--- | :--- |
| **Hypothesis A (Primary Candidate)** | Weather at nominal $T$ $\to$ Demand at nominal $T$ | Contemporaneous nominal match (consistent with approved calendar features) | **Primary acceptance decision gate** |
| **Hypothesis B (Sensitivity 1)** | Weather at nominal $T-1$ $\to$ Demand at nominal $T$ | Pure Hour-Ending hypothesis (weather during preceding hour drives recorded load) | Validation diagnostic only |
| **Hypothesis C (Sensitivity 2)** | Two-sample arithmetic mean: $\frac{\text{temp}(T-1) + \text{temp}(T)}{2}$ | Discrete two-point sample average | Validation diagnostic only |

*Methodological Rules*:
1. **Hypothesis C Clarification**: The two-sample mean of nominal $T-1$ and $T$ is **not a true continuous interval-average temperature**; it is strictly an arithmetic average of two discrete hourly point samples.
2. **Decision Primacy**: Hypothesis A remains the **sole predeclared acceptance criterion**. Hypotheses B and C will be reported separately as validation diagnostics.
3. **No Alignment Hunting**: Selecting an alignment based on test-set performance is **strictly prohibited**.
4. **No Physical Certainty Claim**: None of Hypotheses A, B, or C is claimed to be physically confirmed; they represent structured sensitivity probes over bounded interval uncertainty.

### 3.6 Timezone-Aware Processing & Daylight Saving Time (DST) Synchronization Protocol

To eliminate silent timezone misalignment and schema errors, the transformation from raw NOAA observations to the legacy PJME naive timestamp grid is governed by an explicit 5-step protocol:

#### 3.6.1 Step-by-Step Transformation Pipeline
1. **Raw UTC Parsing & Filtering**:
   - Parse observation timestamps strictly with UTC awareness: `t_raw = pd.to_datetime(df['DATE'], utc=True)`.
   - Filter records to valid routine hourly METAR observations (`REPORT_TYPE == 'FM-15'`) and quality codes (`QUALITY_CODE.isin(['1', '5'])`).
2. **Continuous Half-Open Hourly Window Assignment**:
   - Each valid observation at UTC timestamp $t_{\text{meas}}$ is assigned to nominal UTC hour $T_{\text{UTC}} = \lfloor t_{\text{meas}} + 30\text{ min} \rfloor$, corresponding to continuous window $W(T_{\text{UTC}}) = [T_{\text{UTC}} - 30\text{m}, T_{\text{UTC}} + 30\text{m})$.
   - If multiple routine observations fall within $W(T_{\text{UTC}})$, select the observation nearest the top of the hour: $t^* = \arg\min |t - T_{\text{UTC}}|$, breaking ties deterministically toward the earlier timestamp ($t < T_{\text{UTC}}$).
3. **Civil Timezone Localization**:
   - Convert the nominal UTC hourly sequence to Eastern Civil Time (`America/New_York`):
     $$T_{\text{local}} = T_{\text{UTC}}.\text{dt.tz_convert}('America/New_York')$$
   - This localization accurately alternates between Eastern Daylight Time (EDT, UTC-4) and Eastern Standard Time (EST, UTC-5).
4. **DST Anomaly Handling**:
   - **Spring-Forward Transition (March: 23-Hour Civil Day)**:
     - Civil clocks advance from `01:59:59 EST` directly to `03:00:00 EDT`.
     - In physical UTC, time progresses continuously without interruption. When the unbroken hourly UTC sequence is projected into `America/New_York`, the clock hour `02:00:00` naturally does not exist in local civil time.
     - The Kaggle `PJME_hourly.csv` series similarly omits this nonexistent local hour.
     - Stripping the timezone label (`tz_localize(None)`) produces a naive sequence matching the Kaggle demand index with 0 missing-hour mismatches.
   - **Autumn Fall-Back Transition (November: 25-Hour Civil Day)**:
     - Civil clocks fall back from `01:59:59 EDT` to `01:00:00 EST`. Consequently, the local clock hour `01:00:00` occurs twice: first in EDT (UTC-4) at `05:00:00 UTC`, and then 60 physical minutes later in EST (UTC-5) at `06:00:00 UTC`.
     - **Physical Preservation in Intermediate Tables**: The intermediate weather extraction pipeline preserves both distinct physical observations with their unambiguous UTC timestamps and DST indicator flags (`is_dst=True` vs. `is_dst=False`). **Neither raw nor intermediate weather tables permit silent averaging or dropping of physical observations.**
     - **Demand Preprocessing Join Alignment**: The upstream energy pipeline (`src/data/preprocessor.py:preprocess_energy_data(duplicate_policy='mean')`) collapsed the duplicate naive `01:00:00` energy rows by arithmetic mean. To enable a 1-to-1 join onto the preprocessed demand matrix without corrupting sample index alignment, the baseline weather feature pipeline collapses the two physical weather observations into a single row:
       $$\text{temp\_f}_{\text{collapsed}} = \frac{\text{temp\_f}(05:00\,\text{UTC}) + \text{temp\_f}(06:00\,\text{UTC})}{2}$$
     - **Explicit Non-Physicality Warning**:
       > [!WARNING]
       > **Synthetic Schema Artifact, Not a Physical Measurement**:  
       > The averaged scalar $\text{temp\_f}_{\text{collapsed}}$ is **strictly a synthetic schema-alignment artifact** required to match legacy preprocessed demand rows. It does **not** represent a physical hourly temperature measurement. In physical reality, ambient air temperature evolved over two distinct hours separated by 60 minutes.
5. **Bounded Sensitivity Analysis (No Metric-Driven Alignment Hunting)**:
   - Because collapsing two distinct physical hours into one naive local timestamp cannot be scientifically defended as a unique physical representation, the experiment defines a bounded sensitivity protocol:
     - **Primary Specification (Plan Baseline)**: Collapsed arithmetic mean ($\frac{\text{temp}(05:00\,\text{UTC}) + \text{temp}(06:00\,\text{UTC})}{2}$) to match demand preprocessing convention.
     - **Diagnostic Variant S_DST1 (EDT-Only)**: Map only the earlier physical observation (`05:00 UTC`, EDT) to naive `01:00:00`.
     - **Diagnostic Variant S_DST2 (EST-Only)**: Map only the later physical observation (`06:00 UTC`, EST) to naive `01:00:00`.
   - **Methodological Evaluation Rule**:
     Diagnostic variants S_DST1 and S_DST2 will be audited strictly as validation diagnostics to quantify sensitivity across the 16 repeated autumn hours (2002–2017). **It is strictly prohibited to select between EDT, EST, or mean collapsing on the basis of which produces better validation or test MAE.** The primary pre-registered decision remains locked to the primary specification.

---

## 4. Geographic Scope & Single-Station Regional Proxy Representation

### 4.1 Geographic Footprint of PJM East
The PJM East (`PJME`) regional transmission zone covers over 50,000 square miles across four states:
- Eastern Pennsylvania (PECO, PPL, UGI service territories).
- Central and Southern New Jersey (PSE&G, JCP&L, Atlantic City Electric).
- Delaware (Delmarva Power).
- Northern and Eastern Maryland.

### 4.2 Candidate Weather Station Designation
- **Selected Station**: Philadelphia International Airport (`KPHL`, WBAN: 13739, USAF: 724080).
  - *Coordinates*: 39.8719° N, 75.2411° W; Elevation: 9 meters.
  - *Continuous Record*: 1941–Present (unbroken hourly ASOS/METAR coverage through the 2002–2018 study period).
- **Single-Station Regional Proxy Classification**:
  - `KPHL` is designated strictly as a **single-station regional proxy** for the PJM East transmission footprint.
  - **No Population-Weighting Claim**: The plan explicitly avoids describing `KPHL` as a "population-weighted representation." While Philadelphia is the largest metropolitan center in PJM East, representing a multi-state region with a single point station does not account for inland valley temperature inversions, Appalachian elevation gradients, or Atlantic coastal sea-breezes.
- **Initial Experiment Simplification: No Cross-Station Gap Filling**:
  - The initial Phase 7 experiment will **not introduce multi-station blending or cross-station gap filling** (e.g., from `KEWR` or `KILG`).
  - *Rationale*: Introducing secondary stations before station coverage, sensor calibration comparability, microclimate divergence, and deterministic priority rules are rigorously established introduces spatial confounding. Cross-station spatial interpolation is deferred to a future multi-station enhancement phase.

---

## 5. Candidate Feature Definitions & Mathematical Specifications

### 5.1 Proposed Minimal Feature Set
To prevent dimensional explosion, collinearity, and uncalibrated non-linear interactions, the initial weather candidate introduces exactly three features:

1. **`temp_f` (Surface Dry-Bulb Air Temperature in °F)**:
   - Measured ambient dry-bulb temperature converted from Celsius:
     $$\text{temp\_f} = \text{temp\_c} \times \frac{9}{5} + 32.0$$
2. **`cooling_degree_proxy_f` (Hourly Cooling Degree Difference from 65°F)**:
   - Piecewise-linear threshold proxy capturing air-conditioning demand above base comfort:
     $$\text{cooling\_degree_proxy\_f}(T) = \max(0.0, \, \text{temp\_f}(T) - 65.0)$$
3. **`heating_degree_proxy_f` (Hourly Heating Degree Difference from 65°F)**:
   - Piecewise-linear threshold proxy capturing space-heating demand below base comfort:
     $$\text{heating\_degree_proxy\_f}(T) = \max(0.0, \, 65.0 - \text{temp\_f}(T))$$

### 5.2 Methodological Distinction: Hourly Degree Proxies vs. Time-Integrated Degree-Hours
- **Mathematical Clarification**:
  In climatological and building energy modeling, true *Cooling Degree Hours* (CDH) and *Heating Degree Hours* (HDH) represent time-integrated thermal exposure over a defined interval:
  $$\text{CDH} = \int_{t_0}^{t_1} \max(0, \, T(t) - T_{\text{base}}) \, dt$$
  Because our model receives instantaneous or top-of-hour discrete temperature observations, $\max(0, \, \text{temp\_f} - 65.0)$ and $\max(0, \, 65.0 - \text{temp\_f})$ are **hourly degree-based proxies**, not physically integrated degree-hours.
- **Feature Naming Contract**:
  The candidate feature names are formally designated as `cooling_degree_proxy_f` and `heating_degree_proxy_f` (or aliased to `cooling_degree_hours` with an explicit code docstring explaining this distinction).
- **Base Temperature Setting**:
  The reference threshold is fixed at **65.0°F** (18.33°C), matching the universal North American electric utility and ASHRAE base standard. Hyperparameter search over the degree base (e.g., 60°F vs. 68°F) is strictly prohibited during the initial candidate evaluation.

---

## 6. Fully Specified Causal Missing-Data & Imputation Policy

### 6.1 Weather Coverage Denominators & Distinct Accounting

To prevent conflating meteorological data dropouts with demand-side pipeline filtering, the plan establishes three distinct coverage denominators:

1. **Full-Period Nominal Calendar Grid (149,016 hours)**:  
   The complete 17-calendar-year hourly sequence from `2002-01-01 00:00:00 UTC` through `2018-12-31 23:00:00 UTC` ($17 \times 365 + 4\text{ leap days} = 6,209\text{ days} \times 24\text{ hours} = 149,016\text{ hours}$). Across this entire horizon, exactly **148,757 hours (99.826%)** possess an eligible routine METAR observation, with **259 hours (0.174%)** missing.
2. **Modeling Partition Calendar Spans (145,223 total calendar hours)**:  
   The continuous hourly calendar spans corresponding to the three modeling partitions:
   - *Training Span* (`2002-01-08 01:00` to `2013-08-14 07:00`): **101,671 calendar hours** (101,472 matched, 199 missing = 99.804%).
   - *Validation Span* (`2013-08-14 08:00` to `2016-02-07 14:00`): **21,775 calendar hours** (21,758 matched, 17 missing = 99.922%).
   - *Test Span* (`2016-02-07 15:00` to `2018-08-02 23:00`): **21,777 calendar hours** (21,745 matched, 32 missing = 99.853%).
3. **Exact Modeled Forecasting Samples (145,073 total samples)**:  
   The exact evaluation population retained after demand cleaning, lag-feature construction, and missing-target drops:
   - *Training Partition*: **101,551 samples** (101,352 matched weather, 199 missing = 99.804%).
   - *Validation Partition*: **21,760 samples** (21,743 matched weather, 17 missing = 99.922%).
   - *Test Full Slice*: **21,762 samples** (21,730 matched weather, 32 missing = 99.853%).
   - *Test Aligned Slice (Phase 6)*: **21,759 samples** (21,727 matched weather, 32 missing = 99.853%).
   - *Total Modeled Set*: **145,073 samples** (144,825 matched weather, 248 missing = 99.829%).

#### Reconciling Modeled Rows with Calendar Span
The 289-row difference between the cleaned PJME series (145,362 rows) and the modeled feature matrix (145,073 rows) arises entirely from demand-side feature requirements:
- **168 Rows Dropped for Burn-In**: `lag_168` requires 168 hours of historical demand (`2002-01-01 01:00` to `2002-01-08 00:00`).
- **1 Row Dropped at Series End**: `target_next_hour` (`shift(-1)`) has no forward target for the final row (`2018-08-03 00:00`).
- **120 Rows Dropped across Internal Gaps**: Where autoregressive lags cross missing hours in the Kaggle series under `drop_na=True`.
- **Weather-Missing Rows Kept Strictly Distinct**: Weather data missingness accounts for **0 row drops** in the feature matrix. All 248 missing weather observations within the modeling era are imputed causally, preserving the exact 145,073 evaluation rows.

### 6.2 Timestamp-Based Gap Length Measurement
Gap length is measured strictly in **physical elapsed time** on the continuous UTC reference grid, never by row count alone:
$$\Delta t = t_{\text{missing}} - t_{\text{last\_valid\_utc}}$$
where $t_{\text{last\_valid\_utc}}$ is the timestamp of the most recent preceding valid, quality-checked observation.
- **Short Gap**: $\Delta t \le 2\text{ hours}$ (i.e., exactly 1 or 2 consecutive missing hours).
- **Extended Gap**: $\Delta t > 2\text{ hours}$ (or when no prior valid observation exists).

#### Empirical Gap Distribution at KPHL (2002–2018)
As established in the Phase 7A audit, missingness at KPHL consists of 248 contiguous episodes totaling exactly 259 missing hours:
- **1-Hour Gaps**: 239 episodes (239 missing hours, forward-fillable).
- **2-Hour Gaps**: 7 episodes (14 missing hours, forward-fillable).
- **3-Hour Gaps**: 2 episodes (6 missing hours, requiring diurnal fallback):
  1. `2009-03-02 00:00:00` to `2009-03-02 02:00:00 UTC` (3 hours).
  2. `2010-11-07 06:00:00` to `2010-11-07 08:00:00 UTC` (3 hours).
- **Maximum Missing Gap in 17 Years**: **3 hours**. Zero multi-day sensor dropouts exist.

### 6.3 Deterministic Causal Imputation Rules (Zero Future Lookahead)
To eliminate information leakage and respect temporal ordering:

1. **Strict Prohibition of Bidirectional / Future Lookahead Interpolation**:
   > [!CAUTION]
   > Linear interpolation across time ($\frac{\text{temp}_{t-1} + \text{temp}_{t+1}}{2}$) and backward-filling (`bfill`) use future observations ($\text{temp}_{t+1}$) to fill past gaps. This constitutes future lookahead leakage and is **strictly prohibited**.
2. **Fixed Two-Hour Forward-Fill Limit**:
   - For short gaps ($\Delta t \le 2\text{ hours}$), impute using the last known valid temperature observation:
     $$\text{temp\_f}(t) = \text{temp\_f}(t_{\text{last\_valid\_utc}})$$
   - *Fixed Policy*: The two-hour forward-fill limit is **fixed** for this experiment. Any modification requires a separate, pre-approved experiment plan.
3. **Training-Only Fallback Hierarchy (Fitted Once on Training Partition)**:
   When missingness exceeds 2 hours ($\Delta t > 2\text{ hours}$), or when no prior observation exists, missing temperature is imputed using a strict 3-tier hierarchy fitted **exclusively on the training partition** ($N = 101,551$ rows) and frozen as `models/weather_imputer_params.json`:
   - **Tier 1 (Diurnal-Seasonal Cell Mean)**:
     $$\mu_{\text{train}}(\text{month}(t), \, \text{hour}(t)) = \frac{1}{|S_{\text{train}}(m, h)|} \sum_{i \in S_{\text{train}}(m, h)} \text{temp\_f}_i$$
     computed from the 288-cell lookup table ($12 \text{ months} \times 24 \text{ hours}$).
   - **Tier 2 (Monthly Marginal Mean)**: If Tier 1 is NaN (zero valid training samples in that cell), fall back to the monthly training mean across all hours:
     $$\mu_{\text{train}}^{\text{month}}(m) = \text{mean}(\text{temp\_f} \mid \text{month}=m)$$
   - **Tier 3 (Global Training Mean)**: If Tier 2 is NaN, fall back to the global scalar mean across the entire training partition:
     $$\mu_{\text{train}}^{\text{global}} = \text{mean}(\text{temp\_f}_{\text{train}})$$
   - **Fatal Error on Empty Training Statistics**: If no valid global training mean can be calculated (e.g., 100% training observations missing, count == 0), the pipeline must **stop with a fatal `RuntimeError`** rather than silently inventing a default scalar value.
   - Validation and test observations are **never** used to compute or update imputation statistics.

### 6.4 Partition-Boundary Carryover Rules
- **Start of Training Partition (2002-01-08 01:00:00)**: If the initial training timestamp is missing or quality-rejected, no prior valid observation exists ($\Delta t = \infty$).
  - *Rule*: Impute immediately using Tier 1 training diurnal fallback $\mu_{\text{train}}(\text{month}=1, \text{hour}=1)$.
- **Start of Validation and Test Partitions**:
  - The imputer carries forward the state ($t_{\text{last\_valid\_utc}}$, $\text{temp\_f}_{\text{last}}$) from the end of the immediately preceding chronological partition (end of train for validation; end of validation for test).
  - If the elapsed time between the prior partition's last valid observation and the new partition start satisfies $\Delta t \le 2\text{ hours}$, forward-fill applies. If $\Delta t > 2\text{ hours}$, it falls back to $\mu_{\text{train}}(m, h)$.
  - **Zero Backward Leakage Guarantee**: State carryover is strictly unidirectional (past to future). Carrying state across partition boundaries never allows future observations to fill earlier rows.

### 6.5 Common-Timestamp Scoring Protocol
- **No Selective Row Dropping**: The weather-augmented candidate model must **never silently drop rows** with missing weather values while retaining them for the baseline.
- **Complete Test Matrix Guarantee**: Through the combination of causal forward-fill ($\le 2$h) and the 3-tier training fallback ($> 2$h), the candidate feature pipeline is guaranteed to produce finite values for 100% of rows across all partitions.
- **Dual-Benchmark Accounting**:
  If any unforeseen edge case requires excluding un-imputable rows, evaluation must report:
  1. **Common-Timestamp Metric**: Scored strictly on the exact intersection of rows where all models have valid predictions.
  2. **Full-Partition Diagnostic Metric**: Documenting full-partition scores alongside the count and timestamps of any excluded rows.

---

## 7. Experimental Design, Partitions & Acceptance Criteria

### 7.1 Chronological Split Preservation
All data partitions remain strictly identical to the frozen Phase 5A and Phase 6 experimental boundaries:

| Partition | Start Timestamp | End Timestamp | Sample Count | Role in Experiment |
| :--- | :--- | :--- | :---: | :--- |
| **Training (70%)** | 2002-01-08 01:00:00 | 2013-08-14 07:00:00 | **101,551** | Model fitting and diurnal imputation baseline |
| **Validation (15%)** | 2013-08-14 08:00:00 | 2016-02-07 14:00:00 | **21,760** | **Sole arbiter of model acceptance / rejection** |
| **Test Holdout (15%)** | 2016-02-07 15:00:00 | 2018-08-02 23:00:00 | **21,759** aligned / **21,762** full | Locked holdout; evaluated only post-decision |

### 7.2 Baseline vs. Candidate Feature Matrix Specification
- **Baseline Feature Matrix (`W0_Baseline`)**:
  - **14 Approved Features**:
    - *Temporal / Calendar (9)*: `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`.
    - *Autoregressive / Lagged (5)*: `lag_1`, `lag_24`, `lag_168`, `rolling_mean_24`, `rolling_std_24`.
  - *Model Hyperparameters*: Fixed Phase 5A optimal parameters (`learning_rate=0.05`, `max_iter=200`, `max_depth=8`, `max_leaf_nodes=45`, `l2_regularization=0.5`, `min_samples_leaf=20`, `random_state=42`).
- **Candidate Feature Matrix (`W1_Weather`)**:
  - **17 Features**: The identical 14 baseline features $+$ `temp_f`, `cooling_degree_proxy_f`, `heating_degree_proxy_f`.
  - *Model Hyperparameters*: Identical fixed Phase 5A hyperparameters (zero hyperparameter re-tuning in the primary feature comparison).

### 7.3 Methodological Discipline: Benchmark Distinction
To prevent misleading comparisons, project documentation must maintain strict separation between the two Phase 5A reference points:
1. **Phase 5A Train-Only Benchmark**:
   - Trained on: Training partition only (101,551 samples).
   - Validation MAE: **564.93 MW** (reference baseline for candidate comparison).
   - Test MAE: **632.42 MW** (21,759 aligned rows) / **632.39 MW** (21,762 full test rows).
2. **Phase 5A Refitted Champion**:
   - Trained on: Training $+$ Validation partitions combined (**123,311 samples**).
   - Test MAE: **584.33 MW**, Test RMSE: 776.04 MW, Test $R^2$: 0.9855.
3. **Rule of Comparison**:
   The Phase 7 candidate is trained strictly on the 101,551-sample training partition. Therefore, its performance must be compared directly against the **train-only baseline (Val MAE 564.93 MW, Test MAE 632.42 MW)**. Comparing a candidate trained on 101,551 samples directly against the 123,311-sample refitted champion is methodologically invalid.

### 7.4 Predefined Acceptance / Rejection Decision Rules & Fair Comparison Protocol
The decision protocol is locked prior to experiment execution:

1. **Common-Timestamp Scoring Guarantee**:
   - The candidate weather model (`W1_Weather`) and the baseline model (`W0_Baseline`) must be evaluated on the **exact identical timestamps**.
   - If any unforeseen edge case requires excluding un-imputable rows, the evaluation report must:
     1. Report both models scored on the **common intersection of rows**.
     2. Report the original frozen baseline metrics separately on the full partition, explicitly documenting any difference in evaluation populations.
2. **Primary Selection Metric**: Validation Mean Absolute Error (Val MAE in MW).
3. **Candidate Acceptance Threshold**:
   A candidate weather feature configuration is accepted if and only if:
   $$\text{Val MAE}_{\text{candidate}} \le 563.93\text{ MW} \quad (\ge 1.00\text{ MW improvement over baseline 564.93 MW})$$
   subject to the auxiliary non-degradation constraints:
   - Validation RMSE $\le 754.10\text{ MW}$ (Baseline Val RMSE).
   - Validation $R^2 \ge 0.9851$ (Baseline Val $R^2$).
4. **Validation Primacy & Holdout Isolation**:
   - Validation performance alone controls the acceptance decision.
   - The test partition (**21,759 aligned samples / 21,762 full test samples**) remains completely unqueried until the decision is formally locked.
   - **Strict Prohibition on Test-Set Optimization**: Test performance must **never be used to select alignment hypotheses, choose imputation policies, filter features, or tune hyperparameters**.
   - If the candidate fails the validation threshold ($\text{Val MAE} > 563.93\text{ MW}$), it will be declared a **rejected candidate**, exactly as occurred in Phase 6. Test set metrics will be evaluated purely for post-mortem diagnostics.
5. **Benchmark Integrity & Distinction**:
   - Primary candidate comparison is strictly against the **101,551-sample train-only tuned benchmark** (Val MAE: 564.93 MW, Test MAE: 632.42 MW aligned / 632.39 MW full).
   - The **123,311-sample refitted champion** (Test MAE: 584.33 MW) is a separate production artifact trained on train+validation data; comparing a train-only candidate against it is methodologically invalid and prohibited.
6. **Context of the 1.0 MW Validation Improvement Threshold**:
   > [!NOTE]
   > **Experimental Gate vs. Statistical & Operational Deployability**:  
   > Passing the 1.00 MW validation improvement threshold ($\text{Val MAE} \le 563.93\text{ MW}$) is a predefined experimental heuristic gate designed to prevent incorporating noisy or neutral features. **Passing this gate is not evidence of formal statistical significance, nor is it evidence of operational deployability.** In a regional transmission system with 30,000 to 50,000 MW of load, a 1 MW shift represents ~0.003% of peak demand. Practical operational value requires that the feature specifically attenuate the severe summer peak over-prediction outliers (>4,000 MW) identified in Phase 5B. Operational deployment requires further testing against live Numerical Weather Prediction (NWP) forecasts rather than retrospective realized observations.

### 7.5 Exact Sample Identity & Fail-Fast Integrity Protocol

To guarantee that candidate model comparisons are methodologically unassailable and immune to subtle sample drift:

#### 7.5.1 Partition Size and Temporal Boundaries
All experimental evaluations must strictly adhere to the frozen Phase 5A / Phase 6 sample counts:
- **Training Set**: Exactly **101,551 rows** (`2002-01-08 01:00:00` to `2013-08-14 07:00:00`).
- **Validation Set**: Exactly **21,760 rows** (`2013-08-14 08:00:00` to `2016-02-07 14:00:00`).
- **Test Set (Full)**: Exactly **21,762 rows** (`2016-02-07 15:00:00` to `2018-08-02 23:00:00`).
- **Test Set (Aligned Common Slice)**: Exactly **21,759 rows** (matching Phase 6 evaluation slice).

#### 7.5.2 Ordered Vector Comparison Requirement
Verifying sample counts alone is insufficient. The feature engineering pipeline and evaluation scripts must execute full ordered vector assertions between the candidate feature matrix and the frozen Phase 5A baseline matrix:

```python
# 1. Exact row count parity
assert len(df_candidate) == len(df_baseline), (
    f"Row count mismatch: candidate={len(df_candidate)}, baseline={len(df_baseline)}"
)

# 2. Exact ordered timestamp vector match
assert (df_candidate['target_timestamp'].to_numpy() == df_baseline['target_timestamp'].to_numpy()).all(), (
    "Target timestamp vector mismatch between candidate and baseline!"
)

# 3. Monotonic increasing temporal ordering
assert df_candidate['target_timestamp'].is_monotonic_increasing, (
    "Candidate target_timestamp sequence is not monotonically increasing!"
)

# 4. Zero duplicate timestamps
assert df_candidate['target_timestamp'].duplicated().sum() == 0, (
    f"Duplicate timestamps detected in candidate matrix: {df_candidate['target_timestamp'].duplicated().sum()}"
)

# 5. Zero missing values across all features
assert df_candidate[candidate_feature_names].isna().sum().sum() == 0, (
    f"NaNs detected in candidate feature matrix: {df_candidate[candidate_feature_names].isna().sum().to_dict()}"
)

# 6. Exact target value parity
assert np.allclose(df_candidate['target_next_hour'].to_numpy(), df_baseline['target_next_hour'].to_numpy(), rtol=1e-5, atol=1e-5), (
    "Target demand values ('target_next_hour') do not match baseline!"
)
```

#### 7.5.3 Strict Prohibition Against Row-Dropping for Missing Weather
- Missing weather data accounts for **0 dropped rows** in any modeling partition.
- Dropping rows containing missing weather observations merely to artificially improve reported error metrics or avoid imputation is **strictly prohibited**.
- The candidate feature pipeline must produce valid, finite feature values for 100% of the 101,551 training rows, 21,760 validation rows, and 21,762 test rows via the approved causal imputation hierarchy.
- **Reporting Separation**: Reports must strictly separate meteorological missingness (259 calendar hours / 248 modeling hours) from demand preprocessing exclusions (289 rows: 168 burn-in, 1 end-row shift, 120 internal Kaggle gap crossing drops).

## 8. Artifact Preservation & Separate Storage Paths

To ensure strict reproducibility and prevent accidental corruption or overwriting of prior milestone results, Phase 7 outputs will be written strictly to dedicated paths:

| Output Category | Phase 5A Frozen Path (Read-Only) | Phase 6 Negative Result Path (Read-Only) | Phase 7 Proposed Dedicated Path |
| :--- | :--- | :--- | :--- |
| **Model Checkpoint** | `models/pjm_tree_forecaster_phase5a.joblib` | `models/pjm_tree_forecaster_phase6.joblib` | `models/pjm_tree_forecaster_phase7.joblib` |
| **Imputer Statistics** | N/A | N/A | `models/weather_imputer_params.json` |
| **Validation Tuning Metrics** | `outputs/metrics/pjm_validation_tuning.csv` | `outputs/metrics/pjm_holiday_tuning.csv` | `outputs/metrics/pjm_weather_tuning.csv` |
| **Holdout Benchmark Table** | `outputs/metrics/pjm_model_comparison_phase5a.csv` | `outputs/metrics/pjm_model_comparison_phase6.csv` | `outputs/metrics/pjm_model_comparison_phase7.csv` |
| **Aligned Forecast Matrix** | `outputs/forecasts/pjm_test_predictions_phase5a.csv`| `outputs/forecasts/pjm_test_predictions_phase6.csv`| `outputs/forecasts/pjm_test_predictions_phase7.csv`|

---

## 9. Required Pre-Implementation Unit Tests

Unit tests are structured into two tiers: (A) Phase 7A Verified Foundation Tests, and (B) Phase 7B Pre-Implementation Feature Tests.

### 9.1 Phase 7A Verified Foundation Tests (Implemented & Passing)
Verified in [tests/test_weather_window.py](file:///c:/Users/nithi/OneDrive/Desktop/AI%20Powered%20Energy%20Consumption%20Forcasting/AI-Powered%20Energy%20Consumption%20Forecasting/tests/test_weather_window.py) (all 4 tests passing cleanly within the 94-test project suite):
1. **`test_hourly_window_exact_boundaries`**: Verifies that continuous half-open interval $W(T) = [T - 30\text{m}, T + 30\text{m})$ partitions time without overlap or gaps at `:29:59`, `:30:00`, `:30:01`, `:54:00`, and `:00:00`.
2. **`test_select_nearest_observation_in_window`**: Verifies nearest-observation selection ($t^* = \arg\min |t - T|$) when multiple reports exist in a window.
3. **`test_select_nearest_observation_tie_breaking`**: Verifies deterministic tie-breaking preferring earlier timestamps ($t < T$) for equidistant observations.
4. **`test_temperature_parsing_and_quality_masking`**: Verifies numeric scaling (divide by 10), unit conversion (°F), sentinel masking (`+9999`), and quality filtering (validating `'1'`/`'5'` and masking `'A'`, `'6'`, `'9'`, `'P'`, `'2'`).

### 9.2 Phase 7B Pre-Implementation Unit Tests (To Be Implemented Before Modeling)
The following unit tests must be developed in `tests/test_weather_features.py` before any model training or candidate evaluation is executed:
1. **`test_weather_dst_transition_preservation`**: Verify that autumn duplicate hours preserve both distinct UTC observations in intermediate tables before deterministic collapsing, and that spring missing hours align seamlessly.
2. **`test_causal_imputation_strictly_past_only`**: Verify that short gaps ($\le 2$ hours) forward-fill previous observations, that **no future observations ($t > T_{\text{origin}}$) are accessed**, and that bidirectional interpolation is absent.
3. **`test_training_fallback_isolation_and_hierarchy`**: Verify that the 288-cell lookup table is fitted exclusively on training rows, that test/validation data does not alter parameters, and that Tiers 1–3 fallback operates deterministically with fatal error on empty training statistics.
4. **`test_partition_boundary_imputation_state`**: Verify state carryover at partition boundaries (start of validation/test) without backward lookahead.
5. **`test_degree_proxy_mathematical_formulation`**: Verify exact numerical values across boundary conditions:
   - At 65.0°F: $\text{cooling\_proxy} = 0.0$, $\text{heating\_proxy} = 0.0$.
   - At 90.0°F: $\text{cooling\_proxy} = 25.0$, $\text{heating\_proxy} = 0.0$.
   - At 30.0°F: $\text{cooling\_proxy} = 0.0$, $\text{heating\_proxy} = 35.0$.
6. **`test_feature_matrix_sample_parity`**: Verify that merging weather features preserves exact row counts across train (101,551), val (21,760), and test (21,759 aligned / 21,762 full) partitions with zero row drops and zero NaNs.

---

## 10. Concise Audit of Remaining Unresolved Scientific & Methodological Assumptions

Before Phase 7B implementation can be authorized, the following unresolved scientific and data-definition questions are formally preserved as explicit limitations rather than inventing certainty:

| # | Scientific / Methodological Assumption | Current Working Specification | Status | Epistemic Limitation & Residual Risk |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **Demand Interval Physical Semantics** | Hypothesis A: Nominal Contemporaneous Match ($T \to T$) | **Unresolved** | `PJME_hourly.csv` lacks interval metadata (HE vs. HB vs. instantaneous). Primary run uses nominal alignment; Hypotheses B and C are validation diagnostics. True physical alignment remains unprovable without utility provenance. |
| **2** | **Single-Station Regional Representation** | Philadelphia Airport (`KPHL`) as sole load-pocket proxy | **Design Simplification** | PJM East covers >50,000 sq miles across 4 states. Point observations at KPHL do not capture Appalachian cold pockets, New Jersey coastal breezes, or urban heat gradients. Multi-station weighting deferred. |
| **3** | **Historical Telemetry Availability** | Observations assumed available at physical measurement time ($t_{\text{avail}} \approx t_{\text{meas}}$) | **Unverifiable** | Historical telecommunication network broadcast latency is unrecorded in NOAA ISD archive files. In real-time operations, METAR delivery takes several minutes. |
| **4** | **Oracle Proxy vs. Operational NWP** | Retrospective realized-weather target-hour proxy | **Deliberate Design** | Target-hour realized temperature is an oracle diagnostic establishing thermal sensitivity. It does not evaluate a deployable real-time forecasting system. |
| **5** | **Autumn DST Duplicate Hour Averaging** | Two distinct UTC hours collapsed to arithmetic mean | **Synthetic Compromise** | Averaging `05:00 UTC` and `06:00 UTC` observations into a single local "01:00" row is a synthetic tabular convenience required for join compatibility with the legacy preprocessed demand schema, not a physical measurement. |

---

## 11. Implementation Acceptance Criteria for Future Phase 7B Authorization

Phase 7B implementation is **NOT AUTHORIZED** until the Project Lead explicitly verifies that the following gating criteria are satisfied:

- [ ] **1. Formal Scope Authorization**: Explicit written directive authorizing Phase 7B implementation code in `src/`.
- [ ] **2. Test-First Implementation**: All 6 Phase 7B unit tests in `tests/test_weather_features.py` must be implemented and passing before any forecasting pipeline execution or model training.
- [ ] **3. Strict Artifact Preservation**: Phase 5A benchmark artifacts (`pjm_tree_forecaster_phase5a.joblib`, metrics, forecasts) and Phase 6 negative-result artifacts must remain completely untouched in their dedicated directories.
- [ ] **4. Exact Sample Parity Guarantee**: Merged feature matrices must strictly match existing sample counts:
  - Training: exactly **101,551 rows**, 0 NaNs.
  - Validation: exactly **21,760 rows**, 0 NaNs.
  - Test Full: exactly **21,762 rows**, 0 NaNs.
  - Test Aligned: exactly **21,759 rows**, 0 NaNs.
- [ ] **5. Imputer Training Isolation**: Diurnal fallback lookup table ($12 \times 24$) must be fitted exclusively on the 101,551 training samples and persisted to `models/weather_imputer_params.json` before validation or test evaluation.
- [ ] **6. Validation Decision Primacy**: Candidate model acceptance is determined exclusively by $\text{Val MAE} \le 563.93\text{ MW}$ on the validation partition. Test holdout remains locked until validation decision is formally approved.
- [ ] **7. Cryptographic Manifest Integrity**: All 17 raw weather CSVs in `data/raw/weather/` must continue to match the SHA-256 hashes in `acquisition_manifest.txt`.
