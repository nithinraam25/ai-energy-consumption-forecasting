# Dataset Specifications & Requirements

## 1. Domain Context

Energy consumption forecasting models power demand for residential, commercial, or grid-scale distribution networks. High-precision forecasting directly reduces operational spinning reserve costs, optimizes peak shaving, and facilitates renewable source integration.

---

## 2. Target Schema & Variables

A complete forecasting dataset incorporates both primary consumption signals and correlated exogenous variables:

| Column Name | Type | Unit | Description | Role |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | Datetime (ISO 8601) | - | Monotonically increasing time index | Index |
| `energy_consumption` | Float64 | kWh / kW | Active electrical power / consumed energy | **Target Variable ($y$)** |
| `temperature` | Float64 | °C | Ambient outdoor temperature | Exogenous Feature |
| `humidity` | Float64 | % | Relative outdoor humidity | Exogenous Feature |
| `wind_speed` | Float64 | m/s | Wind velocity (influences heat loss/HVAC) | Exogenous Feature |
| `solar_radiation` | Float64 | W/m² | Solar irradiance (ambient heat gain / PV offset)| Exogenous Feature |
| `holiday_flag` | Binary (0/1) | - | Indicator if date is a statutory/public holiday | Calendar Feature |
| `occupancy_rate` / `is_business_day` | Float64 / Binary | - | Commercial activity or proxy factor | Domain Feature |

### 2.1 Implemented Schema in Phase 2
In Phase 2, the data loader and preprocessor standardize univariate raw inputs into two primary columns:
- **`timestamp`** (datetime64[ns]): Parsed from aliases `timestamp`, `datetime`, `date`, `time`.
- **`consumption`** (float64): Parsed from aliases `consumption`, `consumption_mw`, `pjme_mw`, `power`.

*Note*: No real public dataset has been bundled into the repository yet; unit tests utilize controlled synthetic fixtures. Exogenous weather and holiday features will be connected in subsequent feature engineering phases.

---

## 3. Benchmark Datasets Supported

The architecture is designed to support leading open energy benchmarks:
1. **PJM Hourly Energy Consumption Data**: Regional transmission organization (RTO) in the United States, recording hourly electricity consumption (MW) across multiple utilities (AEP, COMED, DEOK, DOM, DUQ, EKPC, FE, NI, PJME, PJMW, etc.).
2. **UCI Individual Household Electric Power Consumption**: Minute-by-minute measurements of electric power consumption in one household over ~4 years.
3. **Smart Meter Energy Data (London Datastore / Pecan Street)**: Granular smart-meter readings with multi-tenant profiles.

---

## 4. Quality Contracts & Preconditions

Before passing data into the feature pipeline, the dataset must satisfy:
- **No Temporal Inversions**: $t_i > t_{i-1} \quad \forall i$.
- **Strict Bounding**: $0 \le \text{energy\_consumption} \le y_{\max}$ (physical feasibility check).
- **Missing Value Threshold**: Dataset with $> 10\%$ contiguous missing intervals in the validation window will trigger a data-readiness exception.
