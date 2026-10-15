# Modules Catalog & Technical Specifications

This catalog records all codebase modules, their functional responsibilities, architectural interfaces, and validation criteria.

---

## 1. Project Directory Layout (Target Architecture)

```text
AI-Powered Energy Consumption Forecasting/
├── docs/                               # Dedicated technical documentation folder
│   ├── README.md                       # Master documentation index
│   ├── architecture.md                 # System architecture & component design
│   ├── data_pipeline_and_flow.md       # Data lifecycle & flow specifications
│   ├── dataset_specifications.md       # Dataset schema & quality criteria
│   ├── model_learning_and_methodology.md # ML/DL formulation & optimization
│   ├── modules_catalog.md              # Detailed module specs & APIs
│   └── execution_and_validation_log.md # Implementation logs & verification tests
├── data/                               # Dataset storage (raw, processed, external)
│   ├── raw/
│   └── processed/
├── src/                                # Core source code
│   ├── __init__.py
│   ├── config.py                       # Global configuration & hyperparameters
│   ├── data/                           # Data loading, validation & preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── validator.py
│   ├── features/                       # Feature engineering & temporal transforms
│   │   ├── __init__.py
│   │   ├── temporal_features.py
│   │   └── sequence_builder.py
│   ├── models/                         # Model implementations & abstractions
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── baseline.py
│   │   ├── tree_models.py
│   │   └── neural_models.py
│   ├── evaluation/                     # Metrics, backtesting & evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── backtester.py
│   └── utils/                          # Logging, visualization & helpers
│       ├── __init__.py
│       ├── logger.py
│       └── visualizer.py
├── tests/                              # Unit & integration tests
├── models/                             # Saved weights, scalers, and artifacts
└── requirements.txt                    # Project dependencies
```

---

## 2. Module Specifications

### Module 1: `src/config.py`
- **Purpose**: Centralized typed configuration management (Pydantic / Dataclasses).
- **Contains**: File paths, train/val/test split dates, lag parameters, model hyperparameters, random seeds.

### Module 2: `src/data/` (Loader, Preprocessor, Validator)
- **`loader.py`**: Fetches raw CSV/Parquet time-series datasets, parses timestamps.
- **`validator.py`**: Asserts monotonic datetime ordering, detects null rates and out-of-bound values.
- **`preprocessor.py`**: Missing value interpolation, resampling (hourly/daily), outlier winsorization/removal, scaling (StandardScaler/MinMaxScaler).

### Module 3: `src/features/` (Temporal Features, Sequence Builder)
- **`temporal_features.py`**: Generates cyclical time encodings (sin/cos of hour, day-of-week, month), holiday indicators, lag features ($t-k$), and rolling statistics.
- **`sequence_builder.py`**: Constructs sliding window tensors $(X_{lookback}, Y_{horizon})$ for deep recurrent or transformer architectures.

### Module 4: `src/models/` (Base Model, Baselines, Tree Models, Model Selection)
- **`base_model.py`**: Abstract base class (`fit()`, `predict()`, `save()`, `load()`).
- **`baseline.py`**: Non-parametric forecasting baselines (`PersistenceForecaster`, `SeasonalNaiveForecaster`).
- **`tree_models.py`**: `EnergyTreeForecaster` wrapping `HistGradientBoostingRegressor` with anti-leakage guards.
- **`model_selection.py`**: Validation-based hyperparameter tuning (`tune_tree_model_on_validation`) ensuring strict chronological isolation.

### Module 5: `src/evaluation/` (Metrics & Backtesting)
- **`metrics.py`**: Standardized calculation of MAE, RMSE, MAPE, $R^2$, and peak demand error.
- **`backtester.py`**: TimeSeriesSplit cross-validation without lookahead bias.
