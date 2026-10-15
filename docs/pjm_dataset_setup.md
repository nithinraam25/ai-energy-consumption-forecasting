# PJM Hourly Energy Consumption Dataset Setup & Experiment Protocol

This guide describes how to acquire, configure, and execute experiments on the benchmark **PJM Interconnection Hourly Electricity Consumption** dataset.

---

## 1. Dataset Overview

* **Source**: PJM Interconnection (Regional Transmission Organization serving Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and DC).
* **Kaggle Distribution**: [Kaggle Hourly Energy Consumption Dataset (by Rob Mulla)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
* **Target Regional Series**: `PJME_hourly.csv` (PJM East region)
* **Physical Measurement**: Hourly electrical power demand.
* **Original Physical Units**: **Megawatts (MW)**.  
  *(Crucial: Units represent instantaneous power rate in MW, not cumulative electrical energy in MWh).*

---

## 2. Acquisition & File Placement Instructions

Since external datasets are not automatically downloaded by the environment, follow these steps to place the dataset:

1. Navigate to: [https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
2. Click **Download** on Kaggle (or select `PJME_hourly.csv` directly).
   > [!IMPORTANT]
   > Do **not** use your browser's "Save Page As..." function, which saves the Kaggle webpage HTML (`Hourly Energy Consumption.html`). Download the actual ZIP or CSV file.
3. Extract `PJME_hourly.csv`.
4. Place `PJME_hourly.csv` into either of the following supported locations:
   - Primary default path: `data/raw/energy.csv`
   - Regional path: `data/raw/PJME_hourly.csv`

---

## 3. Schema, Physical Units & Temporal Integrity

| Original Column | Normalized Column | Target Type | Units | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `Datetime` | `timestamp` | `datetime64[ns]` | ISO 8601 | Local prevailing Eastern Time without UTC offsets |
| `PJME_MW` | `consumption` | `float64` | **MW** | Instantaneous active power demand |

> [!WARNING]
> **Physical Units Contract**: Measurements represent continuous electrical power demand in **Megawatts (MW)**. Under no circumstances should MW values be relabeled or converted to Megawatt-hours (MWh) without explicit continuous temporal integration $\int P(t) dt$.

### Daylight Saving Time (DST) & Ambiguous Timestamps

Because the source dataset records timestamps in local Eastern Time without explicit UTC offset or timezone identifiers:
1. **Autumn Clock Shift ("Fall Back")**:
   - On the first Sunday of November, clocks shift backward from 02:00 EDT to 01:00 EST.
   - The 01:00 hour occurs **twice** in real elapsed time. In the raw CSV without timezone markers, both entries appear as `YYYY-11-XX 01:00:00`.
   - **Both rows are genuine, distinct physical measurements of grid demand**, with differing MW values.
   - **Resolution Policy**:
     - `duplicate_policy='mean'` (**Default / Recommended**): Averages the two physical demand observations for that duplicated clock hour into a single representative hourly rate ($(\text{MW}_1 + \text{MW}_2)/2$). No demand observation is discarded.
     - `duplicate_policy='keep_first'`: Retains the first chronological observation (EDT reading) and drops the second (EST reading).
     - `duplicate_policy='error'`: Raises a `ValueError` immediately upon detecting any duplicate timestamps.
2. **Spring Clock Shift ("Spring Forward")**:
   - On the second Sunday of March, clocks shift forward from 01:59:59 to 03:00:00, skipping the 02:00 hour.
   - This creates a genuine 1-hour interval gap between 01:00 and 03:00.
   - **Gap-Handling Policy**: Missing hours are **never** filled with zero and are **never** arbitrarily interpolated. Gaps are audited by `validate_energy_data`.
3. **Timestamp-Aware Feature & Target Alignment**:
   - `add_lag_features` verifies exact time deltas ($\Delta t = 24\text{ hours}$ for `lag_24`). If the observation 24 hours prior is missing across a gap, `lag_24` is safely assigned `NaN` rather than pulling an adjacent, non-matching hour.
   - `add_target` verifies exact 1-hour delta ($\Delta t = 1\text{ hour}$). If the immediate next hour is missing across a gap, `target_next_hour` is assigned `NaN` to prevent non-consecutive hours from polluting target labels.
   - `SeasonalNaiveForecaster` performs exact timestamp lookup ($T_{\text{target}} - 24\text{ hours}$).
   - `PersistenceForecaster` performs exact timestamp lookup ($T_{\text{target}} - 1\text{ hour}$).

---

## 4. Reproducible Execution Command

Once `PJME_hourly.csv` or `energy.csv` is placed in `data/raw/`, run:

```bash
python -m src.experiment
```
or
```bash
python main.py
```

The script will:
1. Detect and inspect the raw file schema.
2. Generate an audit report via `validate_energy_data`.
3. Clean and sanitize the series via `preprocess_energy_data(..., duplicate_policy='mean')`.
4. Derive leakage-safe features (`hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `lag_1`, `lag_24`, `lag_168`, `rolling_mean_24`, `rolling_std_24`).
5. Partition into strict chronological slices (70% Train, 15% Validation, 15% Test).
6. Fit `EnergyTreeForecaster` strictly on the training partition.
7. Execute multi-model benchmarking against Persistence and Seasonal-Naive on identical test timestamps.
8. Persist results to `outputs/metrics/pjm_model_comparison.csv` and `outputs/forecasts/pjm_test_predictions.csv`.
