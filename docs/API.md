# Energy Trading & Portfolio Optimization System
# API Documentation

Comprehensive API reference for all modules, classes, and functions.

---

## Table of Contents

- [Introduction](#introduction)
- [Data Module (src.data)](#data-module-srcdata)
- [Models Module (src.models)](#models-module-srcmodels)
- [Strategies Module (src.strategies)](#strategies-module-srcstrategies)
- [Backtesting Module (src.backtesting)](#backtesting-module-srcbacktesting)
- [Optimization Module (src.optimization)](#optimization-module-srcoptimization)
- [Dashboard Module (src.dashboard)](#dashboard-module-srcdashboard)
- [Configuration](#configuration)
- [Common Patterns](#common-patterns)
- [Quick Reference](#quick-reference)

---

## Introduction

This API documentation covers all public classes, methods, and functions in the Energy Trading & Portfolio Optimization System. Each section includes:

- **Class signatures** with type hints
- **Method descriptions** with parameters and return types
- **Usage examples** demonstrating common workflows
- **See also** sections linking related functionality

### Import Conventions

```python
# Data management
from src.data.data_manager import DataManager
from src.data.data_fetcher import EIAFetcher, CAISOFetcher
from src.data.synthetic_generator import SyntheticPriceGenerator
from src.data.renewable_generator import WindGenerator, SolarGenerator

# Models
from src.models.price_forecasting import PriceForecastingPipeline
from src.models.renewable_forecasting import RenewableForecastingPipeline
from src.models.feature_engineering import FeatureEngineer

# Strategies
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.spread_trading import SpreadTradingStrategy
from src.strategies.renewable_arbitrage import RenewableArbitrageStrategy

# Backtesting
from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import PerformanceMetrics
from src.backtesting.reporting import BacktestReport

# Optimization
from src.optimization.optimizer import PortfolioOptimizer
from src.optimization.risk_analytics import RiskAnalytics

# Configuration
from src.config.load_config import get_config
```

### Configuration Management

All classes accept an optional `config` parameter. If not provided, configuration is loaded from `config/config.yaml`:

```python
from src.config.load_config import get_config

# Load configuration
config = get_config()

# Pass to any class
manager = DataManager(config=config)
```

---

## Data Module (src.data)

### DataManager

**Purpose**: Manage data persistence using Parquet format with efficient compression and partitioning.

**Class Signature**:
```python
class DataManager:
    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        processed_data_path: Optional[str] = None,
        compression: Optional[str] = None,
        partition_by_date: Optional[bool] = None,
        config: Optional[Dict] = None
    )
```

**Parameters**:
- `raw_data_path`: Path to raw data directory. If None, reads from config.
- `processed_data_path`: Path to processed data directory. If None, reads from config.
- `compression`: Compression codec ("snappy", "gzip", "brotli", "none"). If None, reads from config.
- `partition_by_date`: Whether to partition data by year/month. If None, reads from config.
- `config`: Configuration dictionary. If None, loads from config.yaml.

**Methods**:

#### `save_raw_data()`
```python
def save_raw_data(
    self,
    data: pd.DataFrame,
    source: str,
    dataset: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    validate: bool = True
) -> Optional[Path]
```

Save raw data from external APIs.

**Parameters**:
- `data`: DataFrame to save
- `source`: Data source ('eia', 'caiso', 'synthetic')
- `dataset`: Dataset name (e.g., 'retail_sales', 'lmp_dam')
- `start_date`: Start date (YYYY-MM-DD format). Optional, used for filename.
- `end_date`: End date (YYYY-MM-DD format). Optional, used for filename.
- `validate`: Whether to validate data before saving. Default True.

**Returns**: Path to saved file or directory (if partitioned), or None if data is empty

**Example**:
```python
manager = DataManager()
manager.save_raw_data(
    data=df_prices,
    source='caiso',
    dataset='lmp_np15_dam',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

#### `save_processed_data()`
```python
def save_processed_data(
    self,
    data: pd.DataFrame,
    source: str,
    dataset: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    validate: bool = True,
    format: str = 'parquet'
) -> Optional[Path]
```

Save cleaned and processed data.

**Parameters**:
- `data`: DataFrame to save
- `source`: Data source
- `dataset`: Dataset name
- `start_date`: Optional start date for filename
- `end_date`: Optional end date for filename
- `validate`: Whether to validate data before saving. Default True.
- `format`: Output format ('parquet' or 'csv'). Default 'parquet'.

**Returns**: Path to saved file or directory (if partitioned), or None if data is empty

**Example**:
```python
manager.save_processed_data(
    data=cleaned_df,
    source='synthetic',
    dataset='prices',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

#### `load_data()`
```python
def load_data(
    self,
    source: str,
    dataset: str,
    data_type: str = "raw",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_filter: bool = True
) -> pd.DataFrame
```

Load data from storage with optional date filtering.

**Parameters**:
- `source`: Data source identifier (e.g., "eia", "caiso", "synthetic")
- `dataset`: Dataset name (e.g., "electricity", "lmp", "prices")
- `data_type`: Data type - `'raw'` or `'processed'` (default: `'raw'`)
- `start_date`: Optional filter start date in YYYY-MM-DD format
- `end_date`: Optional filter end date in YYYY-MM-DD format
- `date_filter`: Whether to apply date filtering when `start_date`/`end_date` are provided. When `False`, returns all matching rows regardless of dates. When `True` (default), filters data on DatetimeIndex if dates are specified.

**Returns**: DataFrame with DatetimeIndex

**Example**:
```python
df = manager.load_data(
    source='synthetic',
    dataset='prices',
    data_type='processed',
    start_date='2023-01-01',
    end_date='2023-06-30'
)
```

#### `get_available_datasets()`
```python
def get_available_datasets(
    self,
    data_type: str = 'raw'
) -> List[Dict[str, Any]]
```

List all available datasets.

**Parameters**:
- `data_type`: 'raw' or 'processed'

**Returns**: List of dicts with dataset metadata (name, rows, size_mb, last_modified)

**Example**:
```python
datasets = manager.get_available_datasets(data_type='processed')
for ds in datasets:
    print(f"{ds['name']}: {ds['rows']:,} rows, {ds['size_mb']:.2f} MB")
```

**See Also**: [Notebook 01: Data Exploration](../notebooks/01_data_exploration.ipynb)

---

### EIAFetcher

**Purpose**: Fetch electricity market data from EIA API v2.

**Class Signature**:
```python
class EIAFetcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    )
```

**Parameters**:
- `api_key`: EIA API key. If None, reads from config or EIA_API_KEY environment variable.
- `timeout`: Request timeout in seconds. If None, reads from config (default 120).
- `max_retries`: Maximum retry attempts for failed requests. If None, reads from config (default 3).
- `retry_backoff`: Exponential backoff multiplier for retries. If None, reads from config (default 2.0).
- `config`: Configuration dictionary. If None, loads from config.yaml.

**Note**: EIA API has rate limits. The fetcher automatically implements retry logic with exponential backoff for 429, 500, 502, 503, and 504 status codes.

**Methods**:

#### `fetch_electricity_data()`
```python
def fetch_electricity_data(
    self,
    endpoint: str = "retail-sales",
    frequency: str = "monthly",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    state: Optional[str] = None,
    sector: Optional[str] = None,
    facets: Optional[Dict[str, Any]] = None,
    data_columns: Optional[List[str]] = None
) -> pd.DataFrame
```

Fetch electricity data from EIA API with automatic pagination.

**Parameters**:
- `endpoint`: API endpoint (e.g., 'retail-sales', 'rto', 'operating-generator-capacity'). Default 'retail-sales'.
- `frequency`: Data frequency ('hourly', 'daily', 'monthly', 'annual'). Default 'monthly'.
- `start_date`: Start date (YYYY-MM-DD format). Optional.
- `end_date`: End date (YYYY-MM-DD format). Optional.
- `state`: State code filter (e.g., 'CA', 'TX'). Can be a single string or list of states. Optional.
- `sector`: Sector filter (e.g., 'RES', 'COM', 'IND', 'TRA'). Can be a single string or list of sectors. Optional.
- `facets`: Additional facet filters as dictionary (e.g., `{'plantid': ['12345', '67890']}`). Optional.
- `data_columns`: Specific data columns to retrieve (e.g., `['value', 'price', 'quantity']`). Optional.

**Returns**: DataFrame with timestamp index and requested data columns

**Note**: The API automatically handles pagination (max 5000 rows per request) and includes a 0.5s courtesy delay between requests to respect rate limits.

**Example**:
```python
eia = EIAFetcher()
df = eia.fetch_electricity_data(
    endpoint='retail-sales',
    frequency='monthly',
    start_date='2023-01-01',
    end_date='2023-12-31',
    state='CA'
)
```

**See Also**: [EIA API Documentation](https://www.eia.gov/opendata/documentation.php)

---

### CAISOFetcher

**Purpose**: Fetch market data from CAISO OASIS (California Independent System Operator).

**Class Signature**:
```python
class CAISOFetcher:
    def __init__(
        self,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    )
```

**Parameters**:
- `timeout`: Request timeout in seconds. If None, reads from config (default 120).
- `max_retries`: Maximum retry attempts for failed requests. If None, reads from config (default 3).
- `retry_backoff`: Exponential backoff multiplier for retries. If None, reads from config (default 2.0).
- `config`: Configuration dictionary. If None, loads from config.yaml.

**Note**: CAISO OASIS returns data as ZIP files containing CSVs. The fetcher automatically extracts and parses the data.

**Methods**:

#### `fetch_lmp_data()`
```python
def fetch_lmp_data(
    self,
    market_run_id: str = "DAM",
    nodes: Optional[List[str]] = None,
    start_date: str = None,
    end_date: str = None,
    query_name: str = "PRC_LMP"
) -> pd.DataFrame
```

Fetch Locational Marginal Prices from CAISO.

**Parameters**:
- `market_run_id`: Market run type. Default 'DAM'.
  - `'DAM'`: Day-ahead market (hourly prices)
  - `'RTM'`: Real-time market (5-minute prices)
- `nodes`: List of pricing node IDs (e.g., `['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND']`). Optional, fetches all nodes if None.
- `start_date`: Start datetime in format `YYYYMMDDThh:mm-0000` (e.g., '20230101T00:00-0000'). Required.
- `end_date`: End datetime in same format. Required.
- `query_name`: Query name. Default 'PRC_LMP' for day-ahead, use 'PRC_INTVL_LMP' for real-time 5-min data.

**Returns**: DataFrame with LMP data including:
  - Timestamp columns (INTERVALSTARTTIME_GMT, INTERVALENDTIME_GMT)
  - Node identification
  - Price components: LMP (total), MCC (marginal cost of congestion), MLC (marginal loss component), MCE (marginal cost of energy)

**Important**: The `market_run_id` and `query_name` must be consistent:
  - Day-ahead market: `market_run_id='DAM'`, `query_name='PRC_LMP'`
  - Real-time market: `market_run_id='RTM'`, `query_name='PRC_INTVL_LMP'`

**Date Format Requirements**:
  - Exact format: `YYYYMMDDThh:mm-0000` (20 characters total)
  - Must end with `-0000` (UTC timezone)
  - Example: `20230101T00:00-0000` for January 1, 2023 at midnight UTC

**Example**:
```python
caiso = CAISOFetcher()
df = caiso.fetch_lmp_data(
    market_run_id='DAM',
    nodes=['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND'],
    start_date='20230101T00:00-0000',
    end_date='20230131T23:59-0000'
)
```

#### `fetch_demand_forecast()`
```python
def fetch_demand_forecast(
    self,
    start_date: str = None,
    end_date: str = None,
    query_name: str = "SLD_FCST"
) -> pd.DataFrame
```

Fetch system load (demand) forecast from CAISO.

**Parameters**:
- `start_date`: Start datetime in format `YYYYMMDDThh:mm-0000` (e.g., '20230101T00:00-0000'). Required.
- `end_date`: End datetime in same format. Required.
- `query_name`: Query name. Default 'SLD_FCST' for system load forecast.

**Returns**: DataFrame with demand forecast including:
  - Timestamp columns (INTERVALSTARTTIME_GMT, INTERVALENDTIME_GMT)
  - MW or LOAD columns with forecasted demand in megawatts

**Date Format Requirements**: Same as `fetch_lmp_data()` - exact format `YYYYMMDDThh:mm-0000` with UTC timezone `-0000`

**Example**:
```python
df_demand = caiso.fetch_demand_forecast(
    start_date='20230101T00:00-0000',
    end_date='20230107T23:59-0000'
)
```

**See Also**: [CAISO OASIS Documentation](https://www.caiso.com/oasis)

---

### SyntheticPriceGenerator

**Purpose**: Generate synthetic electricity price data using Ornstein-Uhlenbeck mean-reverting process.

**Class Signature**:
```python
class SyntheticPriceGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Methods**:

#### `generate_price_series()`
```python
def generate_price_series(
    self,
    start_date: str,
    end_date: str,
    frequency: str = 'H',
    initial_price: float = 50.0,
    n_scenarios: int = 1
) -> pd.DataFrame
```

Generate synthetic price time series.

**Parameters**:
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `frequency`: Time frequency ('H' hourly, 'D' daily)
- `initial_price`: Starting price ($/MWh)
- `n_scenarios`: Number of scenarios to generate

**Returns**: DataFrame with columns ['price', 'scenario_id'] (if multiple scenarios)

**Example**:
```python
gen = SyntheticPriceGenerator()
prices = gen.generate_price_series(
    start_date='2023-01-01',
    end_date='2023-12-31',
    frequency='H',
    initial_price=50.0,
    n_scenarios=1
)
```

#### `generate_stress_scenarios()`
```python
def generate_stress_scenarios(
    self,
    base_series: pd.DataFrame,
    scenario_type: str,
    intensity: float = 1.0
) -> pd.DataFrame
```

Generate stress test scenarios.

**Parameters**:
- `base_series`: Base price series
- `scenario_type`: 'high_volatility', 'sustained_high_prices', 'price_spike', 'negative_prices'
- `intensity`: Scenario intensity multiplier (1.0 = default)

**Returns**: Stressed price series

**Example**:
```python
stressed = gen.generate_stress_scenarios(
    base_series=prices,
    scenario_type='high_volatility',
    intensity=1.5
)
```

#### `calibrate_from_data()`
```python
def calibrate_from_data(
    self,
    historical_prices: pd.Series,
    dt: float
) -> Dict[str, float]
```

Calibrate OU parameters from historical data.

**Parameters**:
- `historical_prices`: Historical price series
- `dt`: Time step in years (1/252 for daily, 1/8760 for hourly)

**Returns**: Dict with calibrated parameters (theta, mu, sigma)

**Example**:
```python
params = gen.calibrate_from_data(
    historical_prices=historical_df['price'],
    dt=1/8760  # Hourly data
)
print(f"Mean reversion speed: {params['theta']:.4f}")
print(f"Long-term mean: {params['mu']:.2f}")
print(f"Volatility: {params['sigma']:.4f}")
```

**See Also**: [ALGORITHMS.md - Ornstein-Uhlenbeck Process](ALGORITHMS.md#ornstein-uhlenbeck-process)

---

### WindGenerator

**Purpose**: Generate synthetic wind power generation profiles.

**Class Signature**:
```python
class WindGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Methods**:

#### `generate_wind_profile()`
```python
def generate_wind_profile(
    self,
    start_date: str,
    end_date: str,
    frequency: str = 'H',
    n_scenarios: int = 1
) -> pd.DataFrame
```

Generate wind generation time series.

**Parameters**:
- `start_date`: Start date
- `end_date`: End date
- `frequency`: Time frequency ('H' hourly)
- `n_scenarios`: Number of scenarios

**Returns**: DataFrame with columns ['generation_mw', 'capacity_factor', 'wind_speed_mps']

**Example**:
```python
wind_gen = WindGenerator()
wind = wind_gen.generate_wind_profile(
    start_date='2023-01-01',
    end_date='2023-12-31',
    frequency='H'
)

print(f"Mean capacity factor: {wind['capacity_factor'].mean():.2%}")
print(f"Max generation: {wind['generation_mw'].max():.1f} MW")
```

#### `generate_low_wind_scenario()`
```python
def generate_low_wind_scenario(
    self,
    base_profile: pd.DataFrame,
    start_time: str,
    duration_hours: int,
    reduction_factor: float = 0.5
) -> pd.DataFrame
```

Generate stress test scenario with low wind period.

**Parameters**:
- `base_profile`: Base wind generation profile
- `start_time`: Start of low wind period (YYYY-MM-DD HH:MM:SS)
- `duration_hours`: Duration in hours
- `reduction_factor`: Generation reduction factor (0.5 = 50% reduction)

**Returns**: Stressed generation profile

**Example**:
```python
low_wind = wind_gen.generate_low_wind_scenario(
    base_profile=wind,
    start_time='2023-06-15 00:00:00',
    duration_hours=72,  # 3 days
    reduction_factor=0.3  # 70% reduction
)
```

#### `calculate_capacity_factor()`
```python
def calculate_capacity_factor(
    self,
    generation_profile: pd.DataFrame
) -> float
```

Calculate overall capacity factor.

**Parameters**:
- `generation_profile`: Wind generation DataFrame

**Returns**: Capacity factor (0 to 1)

**Example**:
```python
cf = wind_gen.calculate_capacity_factor(wind)
print(f"Capacity Factor: {cf:.2%}")  # e.g., "Capacity Factor: 35.2%"
```

**See Also**: [ALGORITHMS.md - Wind Power Curve](ALGORITHMS.md#wind-power-curve)

---

### SolarGenerator

**Purpose**: Generate synthetic solar power generation profiles.

**Class Signature**:
```python
class SolarGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Methods**:

#### `generate_solar_profile()`
```python
def generate_solar_profile(
    self,
    start_date: str,
    end_date: str,
    frequency: str = 'H',
    n_scenarios: int = 1
) -> pd.DataFrame
```

Generate solar generation time series.

**Parameters**:
- `start_date`: Start date
- `end_date`: End date
- `frequency`: Time frequency ('H' hourly)
- `n_scenarios`: Number of scenarios

**Returns**: DataFrame with columns ['generation_mw', 'capacity_factor', 'irradiance_w_m2']

**Example**:
```python
solar_gen = SolarGenerator()
solar = solar_gen.generate_solar_profile(
    start_date='2023-01-01',
    end_date='2023-12-31',
    frequency='H'
)

print(f"Mean capacity factor: {solar['capacity_factor'].mean():.2%}")
print(f"Peak irradiance: {solar['irradiance_w_m2'].max():.1f} W/m²")
```

#### `generate_cloudy_scenario()`
```python
def generate_cloudy_scenario(
    self,
    base_profile: pd.DataFrame,
    start_time: str,
    duration_hours: int,
    cloud_cover_level: float = 0.7
) -> pd.DataFrame
```

Generate stress test scenario with cloudy period.

**Parameters**:
- `base_profile`: Base solar generation profile
- `start_time`: Start of cloudy period
- `duration_hours`: Duration in hours
- `cloud_cover_level`: Cloud cover level (0-1, where 1 = completely overcast)

**Returns**: Stressed generation profile

**Example**:
```python
cloudy = solar_gen.generate_cloudy_scenario(
    base_profile=solar,
    start_time='2023-06-15 10:00:00',
    duration_hours=8,  # Daylight hours
    cloud_cover_level=0.8  # 80% cloud cover
)
```

**See Also**: [ALGORITHMS.md - Solar Irradiance Model](ALGORITHMS.md#solar-irradiance-model)

---

## Models Module (src.models)

### PriceForecastingPipeline

**Purpose**: Orchestrate price forecasting with multiple models (ARIMA, XGBoost, LSTM) and ensemble creation.

**Class Signature**:
```python
class PriceForecastingPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Methods**:

#### `prepare_data()`
```python
def prepare_data(
    self,
    data: pd.DataFrame,
    target_col: str = 'price'
) -> 'PriceForecastingPipeline'
```

Prepare data with train/validation/test split (80/10/10).

**Parameters**:
- `data`: DataFrame with price data (DatetimeIndex)
- `target_col`: Column name for target variable

**Returns**: self (for method chaining)

**Example**:
```python
pipeline = PriceForecastingPipeline()
pipeline.prepare_data(prices, target_col='price')
```

#### `train_models()`
```python
def train_models(
    self,
    model_types: Optional[List[str]] = None
) -> 'PriceForecastingPipeline'
```

Train specified forecasting models.

**Parameters**:
- `model_types`: List of models to train (['arima', 'xgboost', 'lstm'], or None for all)

**Returns**: self (for method chaining)

**Example**:
```python
# Train all models
pipeline.train_models()

# Train specific models
pipeline.train_models(model_types=['arima', 'xgboost'])
```

#### `evaluate_models()`
```python
def evaluate_models(self) -> Dict[str, Dict[str, float]]
```

Evaluate all trained models on validation set.

**Returns**: Dict mapping model name to metrics dict (rmse, mae, mape, r2)

**Example**:
```python
evaluation = pipeline.evaluate_models()

for model, metrics in evaluation.items():
    print(f"{model}:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  MAPE: {metrics['mape']:.2%}")
    print(f"  R²: {metrics['r2']:.3f}")
```

#### `create_ensemble()`
```python
def create_ensemble(
    self,
    weights: Optional[Dict[str, float]] = None
) -> 'PriceForecastingPipeline'
```

Create ensemble forecast with weighted averaging.

**Parameters**:
- `weights`: Optional custom weights (if None, use inverse RMSE weighting)

**Returns**: self (for method chaining)

**Example**:
```python
# Automatic inverse RMSE weighting
pipeline.create_ensemble()

# Custom weights
pipeline.create_ensemble(weights={
    'arima': 0.2,
    'xgboost': 0.3,
    'lstm': 0.5
})
```

#### `predict()`
```python
def predict(
    self,
    data: pd.DataFrame,
    model_name: str = 'ensemble',
    steps: Optional[int] = None
) -> pd.DataFrame
```

Generate forecasts.

**Parameters**:
- `data`: Input data for forecasting
- `model_name`: Model to use ('arima', 'xgboost', 'lstm', 'ensemble')
- `steps`: Number of steps ahead to forecast (if None, single-step)

**Returns**: DataFrame with forecasts

**Example**:
```python
# Single-step forecast for entire dataset
forecasts = pipeline.predict(prices, model_name='ensemble')

# Multi-step ahead (24-hour forecast)
forecast_24h = pipeline.predict(
    prices.tail(168),  # Use last week as input
    model_name='ensemble',
    steps=24
)
```

#### `save_models()`
```python
def save_models(self) -> Dict[str, Path]
```

Save all trained models to disk.

**Returns**: Dict mapping model name to saved file path

**Example**:
```python
saved_paths = pipeline.save_models()
print(f"ARIMA model saved to: {saved_paths['arima']}")
print(f"XGBoost model saved to: {saved_paths['xgboost']}")
print(f"LSTM model saved to: {saved_paths['lstm']}")
```

**See Also**: [Notebook 02: Price Forecasting](../notebooks/02_price_forecasting.ipynb), [ALGORITHMS.md - Forecasting Models](ALGORITHMS.md#forecasting-models)

---

### FeatureEngineer

**Purpose**: Create features for machine learning models.

**Class Signature**:
```python
class FeatureEngineer:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Methods**:

#### `create_features()`
```python
def create_features(
    self,
    data: pd.DataFrame,
    target_col: str = 'price',
    include_target: bool = True
) -> pd.DataFrame
```

Create all features (lags, rolling stats, time features).

**Parameters**:
- `data`: Input DataFrame
- `target_col`: Target column name
- `include_target`: Whether to include target column in output

**Returns**: DataFrame with all engineered features

**Example**:
```python
engineer = FeatureEngineer()
features_df = engineer.create_features(
    data=prices,
    target_col='price',
    include_target=True
)

print(f"Original columns: {list(prices.columns)}")
print(f"Feature columns: {list(features_df.columns)}")
# Output: lag_1, lag_2, ..., rolling_mean_24, ..., hour, day_of_week, ...
```

#### `create_lag_features()`
```python
def create_lag_features(
    self,
    data: pd.DataFrame,
    target_col: str,
    lags: List[int]
) -> pd.DataFrame
```

Create lagged features.

**Parameters**:
- `data`: Input DataFrame
- `target_col`: Column to lag
- `lags`: List of lag periods (e.g., [1, 2, 3, 24, 168])

**Returns**: DataFrame with lag columns

**Example**:
```python
lags_df = engineer.create_lag_features(
    data=prices,
    target_col='price',
    lags=[1, 2, 3, 24, 168]  # 1-3h, 24h (1 day), 168h (1 week)
)
```

#### `create_rolling_features()`
```python
def create_rolling_features(
    self,
    data: pd.DataFrame,
    target_col: str,
    windows: List[int]
) -> pd.DataFrame
```

Create rolling statistics features.

**Parameters**:
- `data`: Input DataFrame
- `target_col`: Column for rolling stats
- `windows`: List of window sizes (e.g., [24, 168, 720])

**Returns**: DataFrame with rolling_mean, rolling_std, rolling_min, rolling_max columns

**Example**:
```python
rolling_df = engineer.create_rolling_features(
    data=prices,
    target_col='price',
    windows=[24, 168]  # 1 day, 1 week
)
# Creates: rolling_mean_24, rolling_std_24, rolling_min_24, rolling_max_24, etc.
```

#### `create_time_features()`
```python
def create_time_features(
    self,
    data: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame
```

Create time-based features from DatetimeIndex.

**Parameters**:
- `data`: Input DataFrame with DatetimeIndex
- `features`: List of time features ('hour', 'day', 'day_of_week', 'month', 'is_weekend')

**Returns**: DataFrame with time feature columns

**Example**:
```python
time_df = engineer.create_time_features(
    data=prices,
    features=['hour', 'day_of_week', 'month', 'is_weekend']
)
```

#### `create_sequences_for_lstm()`
```python
def create_sequences_for_lstm(
    self,
    features: pd.DataFrame,
    target: pd.Series,
    lookback: int,
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]
```

Create 3D sequences for LSTM training.

**Parameters**:
- `features`: Feature DataFrame
- `target`: Target series
- `lookback`: Number of past timesteps to use (e.g., 168 for 1 week)
- `forecast_horizon`: Number of future steps to predict (e.g., 1 for single-step)

**Returns**: Tuple of (X, y) arrays with shape (samples, lookback, features), (samples, forecast_horizon)

**Example**:
```python
X, y = engineer.create_sequences_for_lstm(
    features=features_df,
    target=prices['price'],
    lookback=168,  # Use past week
    forecast_horizon=1  # Predict next hour
)

print(f"X shape: {X.shape}")  # (samples, 168, num_features)
print(f"y shape: {y.shape}")  # (samples, 1)
```

**See Also**: [ALGORITHMS.md - Feature Engineering](ALGORITHMS.md#feature-engineering)

---

### RenewableForecastingPipeline

**Purpose**: Forecast wind and solar generation with curtailment analysis.

**Class Signature**:
```python
class RenewableForecastingPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
```

**Methods**:

#### `prepare_data()`
```python
def prepare_data(
    self,
    wind_data: pd.DataFrame,
    solar_data: pd.DataFrame
) -> 'RenewableForecastingPipeline'
```

Prepare renewable generation data.

**Parameters**:
- `wind_data`: Wind generation DataFrame
- `solar_data`: Solar generation DataFrame

**Returns**: self

**Example**:
```python
rf_pipeline = RenewableForecastingPipeline()
rf_pipeline.prepare_data(wind, solar)
```

#### `train_wind_models()`
```python
def train_wind_models(
    self,
    model_types: List[str]
) -> 'RenewableForecastingPipeline'
```

Train wind forecasting models.

**Parameters**:
- `model_types`: List of models (['arima', 'xgboost', 'lstm'])

**Returns**: self

**Example**:
```python
rf_pipeline.train_wind_models(['arima', 'xgboost'])
```

#### `train_solar_models()`
```python
def train_solar_models(
    self,
    model_types: List[str]
) -> 'RenewableForecastingPipeline'
```

Train solar forecasting models.

**Parameters**:
- `model_types`: List of models

**Returns**: self

**Example**:
```python
rf_pipeline.train_solar_models(['arima', 'xgboost'])
```

#### `forecast()`
```python
def forecast(
    self,
    wind_data: pd.DataFrame,
    solar_data: pd.DataFrame,
    model_name: str = 'ensemble',
    steps: Optional[int] = None,
    include_curtailment: bool = False,
    price_forecast: Optional[np.ndarray] = None
) -> Dict[str, pd.DataFrame]
```

Generate renewable generation forecasts.

**Parameters**:
- `wind_data`: Historical wind data
- `solar_data`: Historical solar data
- `model_name`: Model to use (default: 'ensemble')
- `steps`: Forecast horizon in hours (default: None, uses config default)
- `include_curtailment`: Whether to analyze curtailment (default: False)
- `price_forecast`: Optional price forecast array for curtailment analysis

**Returns**: Dict with 'wind_forecast', 'solar_forecast', 'combined_forecast', optionally 'curtailment'

**Example**:
```python
forecasts = rf_pipeline.forecast(
    wind_data=wind,
    solar_data=solar,
    model_name='ensemble',
    steps=24,
    include_curtailment=True,
    price_forecast=price_forecasts
)

print(f"24-hour wind forecast: {forecasts['wind_forecast']['generation_mw'].values}")
print(f"Curtailment events: {len(forecasts['curtailment'])}")
```

#### `generate_scenarios()`
```python
def generate_scenarios(
    self,
    base_wind_data: pd.DataFrame,
    base_solar_data: pd.DataFrame,
    n_scenarios: int = 1000,
    include_stress_tests: bool = False
) -> Dict
```

Generate Monte Carlo scenarios for renewable generation.

**Parameters**:
- `base_wind_data`: Base wind generation
- `base_solar_data`: Base solar generation
- `n_scenarios`: Number of scenarios
- `include_stress_tests`: Whether to include stress scenarios (low wind, cloudy)

**Returns**: Dict with scenario arrays and statistics

**Example**:
```python
scenarios = rf_pipeline.generate_scenarios(
    base_wind_data=wind,
    base_solar_data=solar,
    n_scenarios=1000,
    include_stress_tests=True
)

# Analyze scenario distribution
print(f"Mean generation: {scenarios['mean_generation']:.2f} MW")
print(f"95th percentile: {scenarios['p95_generation']:.2f} MW")
print(f"5th percentile: {scenarios['p5_generation']:.2f} MW")
```

**See Also**: [Notebook 05: Renewable Energy Analysis](../notebooks/05_renewable_energy_analysis.ipynb)

---

## Strategies Module (src.strategies)

### BaseStrategy (Abstract)

**Purpose**: Abstract base class for all trading strategies.

**Class Signature**:
```python
class BaseStrategy(ABC):
    def __init__(
        self,
        name: str,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    )
```

**Abstract Methods**:

#### `generate_signals()`
```python
@abstractmethod
def generate_signals(
    self,
    data: pd.DataFrame,
    **kwargs
) -> pd.DataFrame
```

Generate trading signals. Must be implemented by subclasses.

**Parameters**:
- `data`: Price or feature DataFrame
- `**kwargs`: Strategy-specific parameters

**Returns**: DataFrame with 'signal' column (-1 for sell, 0 for neutral, 1 for buy)

**Methods**:

#### `calculate_position_size()`
```python
def calculate_position_size(
    self,
    signal_strength: float,
    account_value: float,
    current_price: float,
    method: str = 'fixed_fractional'
) -> float
```

Calculate position size based on risk management rules.

**Parameters**:
- `signal_strength`: Signal strength (-1 to 1)
- `account_value`: Current account value
- `current_price`: Current price
- `method`: Position sizing method ('fixed_fractional', 'kelly', 'volatility_adjusted')

**Returns**: Position size (number of units)

**Example**:
```python
position = strategy.calculate_position_size(
    signal_strength=0.8,
    account_value=1000000,
    current_price=50.0,
    method='fixed_fractional'  # Use 2% risk per trade
)
```

#### `apply_risk_limits()`
```python
def apply_risk_limits(
    self,
    signals: pd.DataFrame,
    current_positions: Dict,
    account_value: float,
    current_prices: Dict
) -> pd.DataFrame
```

Apply risk management constraints.

**Parameters**:
- `signals`: Generated signals
- `current_positions`: Current holdings
- `account_value`: Account value
- `current_prices`: Current market prices

**Returns**: Risk-adjusted signals

**Example**:
```python
adjusted_signals = strategy.apply_risk_limits(
    signals=raw_signals,
    current_positions={'asset1': 100, 'asset2': -50},
    account_value=1000000,
    current_prices={'asset1': 50.0, 'asset2': 75.0}
)
```

**See Also**: [Implementing Custom Strategies](#implementing-custom-strategies)

---

### MeanReversionStrategy

**Purpose**: Bollinger Bands mean reversion strategy.

**Class Signature**:
```python
class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    )
```

**Parameters**:
- `config`: Optional global configuration dictionary. If None, loads from `config/config.yaml`
- `strategy_config`: Optional strategy-specific configuration. If None, loads from `config['strategies']['mean_reversion']`

**Strategy Configuration Parameters** (set in `config/config.yaml` or `strategy_config` dict):
- `window`: Moving average window (default: 20)
- `num_std`: Number of standard deviations for Bollinger Bands (default: 2.0)
- `entry_threshold`: How close to band to trigger entry signal, 0-1 scale (default: 0.9)
- `exit_threshold`: When to exit position, 0-1 scale (default: 0.5)
- `min_volatility`: Minimum volatility filter (default: 0.01)
- `max_volatility`: Maximum volatility filter (default: 0.10)

**Methods**:

#### `generate_signals()`
```python
def generate_signals(
    self,
    data: pd.DataFrame,
    asset: str = 'default',
    **kwargs
) -> pd.DataFrame
```

Generate mean reversion signals based on Bollinger Bands.

**Parameters**:
- `data`: DataFrame with 'price' or 'close' column and DatetimeIndex
- `asset`: Asset identifier for position tracking (default: 'default')
- `**kwargs`: Additional parameters (currently unused)

**Returns**: DataFrame with columns:
- `timestamp`: Signal timestamp (datetime)
- `asset`: Asset identifier (str)
- `signal`: Trading signal - `1` (buy), `-1` (sell), `0` (hold/neutral)
- `strength`: Signal confidence score (0-1 float)
- `reason`: Human-readable explanation of the signal (str)

**Example**:
```python
from src.strategies.mean_reversion import MeanReversionStrategy

# Initialize with config-driven parameters
strategy = MeanReversionStrategy()

# Or override specific parameters
strategy_config = {
    'window': 24,  # 24-hour MA
    'num_std': 2.0,  # 2 standard deviations
    'entry_threshold': 0.9,  # Enter at 90% of band
    'exit_threshold': 0.5   # Exit at 50% of band
}
strategy = MeanReversionStrategy(strategy_config=strategy_config)

# Generate signals
signals = strategy.generate_signals(prices, asset='CAISO_NP15')

# Analyze signals
buy_signals = signals[signals['signal'] == 1]
sell_signals = signals[signals['signal'] == -1]

print(f"Buy opportunities: {len(buy_signals)}")
print(f"Sell opportunities: {len(sell_signals)}")
print(f"Average buy strength: {buy_signals['strength'].mean():.2f}")

# Example output
#   timestamp           asset       signal  strength  reason
# 0 2023-01-01 14:00  CAISO_NP15    1      0.92      Price at lower band (oversold)
# 1 2023-01-02 08:00  CAISO_NP15   -1      0.88      Price at upper band (overbought)
```

**See Also**: [ALGORITHMS.md - Mean Reversion](ALGORITHMS.md#mean-reversion-bollinger-bands)

---

### MomentumStrategy

**Purpose**: Moving average crossover momentum strategy.

**Class Signature**:
```python
class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    )
```

**Parameters**:
- `config`: Optional global configuration dictionary
- `strategy_config`: Optional strategy-specific configuration with:
  - `fast_window`: Fast MA window (default 10)
  - `slow_window`: Slow MA window (default 30)
  - `signal_threshold`: Minimum difference to trigger signal (default 0.02 = 2%)
  - `trend_confirmation`: Whether to require price/MA alignment (default True)

**Methods**:

#### `generate_signals()`
```python
def generate_signals(
    self,
    data: pd.DataFrame,
    asset: str = 'default',
    **kwargs
) -> pd.DataFrame
```

Generate signals on MA crossover.

**Returns**: DataFrame with columns ['signal', 'fast_ma', 'slow_ma', 'ma_diff']

**Example**:
```python
strategy = MomentumStrategy(
    fast_period=10,
    slow_period=30,
    signal_threshold=0.02
)

signals = strategy.generate_signals(prices)

# Crossover points
crossovers = signals[signals['signal'] != 0]
print(f"Crossover events: {len(crossovers)}")
```

**See Also**: [ALGORITHMS.md - Momentum](ALGORITHMS.md#momentum-ma-crossover)

---

### SpreadTradingStrategy

**Purpose**: Statistical arbitrage on price spreads.

**Class Signature**:
```python
class SpreadTradingStrategy(BaseStrategy):
    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    )
```

**Parameters**:
- `config`: Optional global configuration dictionary
- `strategy_config`: Optional strategy-specific configuration with:
  - `lookback_window`: Historical window for spread statistics (default 60)
  - `entry_z_score`: Z-score threshold for entry (default 2.0)
  - `exit_z_score`: Z-score threshold for exit (default 0.5)
  - `spread_type`: Type of spread 'difference' or 'ratio' (default 'difference')
  - `use_dynamic_hedge_ratio`: Use rolling OLS hedge ratio (default True)

**Methods**:

#### `generate_signals()`
```python
def generate_signals(
    self,
    data: pd.DataFrame,
    asset: str = 'spread',
    **kwargs
) -> pd.DataFrame
```

Generate signals based on spread z-score.

**Returns**: DataFrame with columns ['signal', 'spread', 'z_score']

**Example**:
```python
# Data should have multiple price columns for spread calculation
# e.g., peak vs off-peak prices
strategy = SpreadTradingStrategy(
    lookback_period=168,
    entry_quantile=0.8,
    exit_quantile=0.5
)

signals = strategy.generate_signals(prices_with_spread)

# Analyze z-scores
high_z = signals[signals['z_score'] > 2.0]
print(f"Extreme spread events: {len(high_z)}")
```

**See Also**: [ALGORITHMS.md - Spread Trading](ALGORITHMS.md#spread-trading-statistical-arbitrage)

---

### RenewableArbitrageStrategy

**Purpose**: Exploit generation-price correlation for renewable assets.

**Class Signature**:
```python
class RenewableArbitrageStrategy(BaseStrategy):
    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None,
        price_forecaster=None,
        renewable_forecaster=None
    )
```

**Parameters**:
- `config`: Optional global configuration dictionary
- `strategy_config`: Optional strategy-specific configuration with:
  - `forecast_horizon`: Hours ahead to forecast (default 24)
  - `generation_threshold_high`: Capacity factor threshold for high generation (default 0.7)
  - `generation_threshold_low`: Capacity factor threshold for low generation (default 0.3)
  - `price_sensitivity`: Weight of price forecast vs generation forecast (default 0.5)
  - `correlation_factor`: Expected correlation between generation and prices (default -0.6)
- `price_forecaster`: Optional PriceForecastingPipeline instance
- `renewable_forecaster`: Optional RenewableForecastingPipeline instance

**Methods**:

#### `set_forecasters()`
```python
def set_forecasters(
    self,
    price_forecaster: PriceForecastingPipeline,
    renewable_forecaster: RenewableForecastingPipeline
)
```

Set forecasting pipelines.

**Parameters**:
- `price_forecaster`: Trained price forecaster
- `renewable_forecaster`: Trained renewable forecaster

**Example**:
```python
strategy = RenewableArbitrageStrategy()

# Set forecasters
strategy.set_forecasters(
    price_forecaster=price_pipeline,
    renewable_forecaster=renewable_pipeline
)
```

#### `generate_signals()`
```python
def generate_signals(
    self,
    data: pd.DataFrame,
    wind_data: Optional[pd.DataFrame] = None,
    solar_data: Optional[pd.DataFrame] = None,
    asset: str = 'default',
    **kwargs
) -> pd.DataFrame
```

Generate signals based on generation-price correlation.

**Parameters**:
- `data`: Price data with DatetimeIndex
- `wind_data`: Optional wind generation data
- `solar_data`: Optional solar generation data
- `asset`: Asset identifier for position tracking (default: 'default')

**Returns**: DataFrame with columns ['signal', 'expected_generation', 'expected_price', 'price_impact']

**Example**:
```python
signals = strategy.generate_signals(
    data=prices,
    wind_data=wind,
    solar_data=solar
)

# High generation periods (expect lower prices)
high_gen_signals = signals[signals['signal'] == -1]
print(f"High generation trade opportunities: {len(high_gen_signals)}")
```

**See Also**: [ALGORITHMS.md - Renewable Arbitrage](ALGORITHMS.md#renewable-arbitrage)

---

## Backtesting Module (src.backtesting)

### BacktestEngine

**Purpose**: Event-driven backtesting with realistic order execution.

**Class Signature**:
```python
class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1000000,
        transaction_cost: float = 0.001,
        slippage_model: str = 'fixed',
        config: Optional[Dict] = None
    )
```

**Parameters**:
- `initial_capital`: Starting capital (default $1M)
- `transaction_cost`: Transaction cost as fraction (default 0.001 = 0.1%)
- `slippage_model`: Slippage model ('fixed', 'proportional', 'market_impact')

**Methods**:

#### `run()`
```python
def run(
    self,
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_trades: bool = True,
    trade_format: Optional[Union[str, List[str]]] = None
) -> BacktestResult
```

Run backtest simulation.

**Parameters**:
- `price_data`: Price DataFrame
- `signals`: Signal DataFrame from strategy
- `start_date`: Optional start date filter
- `end_date`: Optional end date filter
- `save_trades`: Whether to save trade history
- `trade_format`: Trade log output format(s). Accepts:
  - `'parquet'`: Save trades as Parquet file
  - `'csv'`: Save trades as CSV file
  - `['parquet', 'csv']`: Save in both formats
  - `None`: Use default from config `transaction_log_format` (default: `'parquet'`)

**Returns**: BacktestResult dataclass with equity_curve, trades, portfolio_history, metrics

**Example**:
```python
engine = BacktestEngine(
    initial_capital=1000000,
    transaction_cost=0.001
)

result = engine.run(
    price_data=prices,
    signals=strategy_signals,
    save_trades=True,
    trade_format='parquet'  # Save trades in Parquet format
)

print(f"Final Portfolio Value: ${result.equity_curve.iloc[-1]:,.2f}")
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
print(f"Number of Trades: {result.metrics['num_trades']}")

# Example: Save trades in both formats for compatibility
result_multi = engine.run(
    price_data=prices,
    signals=strategy_signals,
    save_trades=True,
    trade_format=['parquet', 'csv']  # Saves both .parquet and .csv
)
```

#### `reset()`
```python
def reset(self)
```

Reset engine state for new backtest.

**Example**:
```python
# Run multiple backtests
for strategy in strategies:
    engine.reset()
    result = engine.run(prices, strategy.generate_signals(prices))
    results[strategy.name] = result
```

**See Also**: [Notebook 03: Strategy Backtesting](../notebooks/03_strategy_backtesting.ipynb)

---

### PerformanceMetrics

**Purpose**: Calculate comprehensive performance metrics.

**Class Signature**:
```python
class PerformanceMetrics:
    def __init__(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        initial_capital: float = 1000000,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    )
```

**Parameters**:
- `equity_curve`: Portfolio value time series
- `trades`: Trade history DataFrame
- `returns`: Optional return series (calculated if None)
- `initial_capital`: Starting capital
- `risk_free_rate`: Risk-free rate for Sharpe calculation
- `periods_per_year`: Trading periods per year (252 for daily, 252*24 for hourly)

**Methods**:

#### `calculate_all()`
```python
def calculate_all(self) -> Dict[str, float]
```

Calculate all performance metrics.

**Returns**: Dict with metrics:
- total_return
- annualized_return
- volatility
- sharpe_ratio
- sortino_ratio
- calmar_ratio
- max_drawdown
- win_rate
- profit_factor
- avg_win
- avg_loss

**Example**:
```python
metrics = PerformanceMetrics(
    equity_curve=result.equity_curve,
    trades=result.trades,
    initial_capital=1000000,
    risk_free_rate=0.03  # 3% risk-free rate
)

all_metrics = metrics.calculate_all()

for metric, value in all_metrics.items():
    print(f"{metric}: {value:.4f}")
```

#### `get_summary()`
```python
def get_summary(self) -> Dict[str, str]
```

Get formatted summary of key metrics.

**Returns**: Dict with formatted strings

**Example**:
```python
summary = metrics.get_summary()

print("Performance Summary:")
print(f"  Return: {summary['total_return']}")
print(f"  Sharpe: {summary['sharpe_ratio']}")
print(f"  Max DD: {summary['max_drawdown']}")
```

#### `to_dataframe()`
```python
def to_dataframe(self) -> pd.DataFrame
```

Convert metrics to DataFrame for easy comparison.

**Returns**: DataFrame with metrics

**Example**:
```python
df = metrics.to_dataframe()
df.to_csv('performance_metrics.csv')
```

**See Also**: [ALGORITHMS.md - Performance Metrics](ALGORITHMS.md#performance-metrics)

---

### BacktestReport

**Purpose**: Generate visualizations and reports.

**Class Signature**:
```python
class BacktestReport:
    def __init__(
        self,
        result: BacktestResult,
        metrics: Optional[PerformanceMetrics] = None
    )
```

**Methods**:

#### `generate_full_report()`
```python
def generate_full_report(self) -> Dict[str, go.Figure]
```

Generate all visualizations.

**Returns**: Dict mapping plot name to Plotly Figure

**Example**:
```python
report = BacktestReport(result)
figures = report.generate_full_report()

# Display or save figures
for name, fig in figures.items():
    fig.show()
    fig.write_html(f"{name}.html")
```

#### `plot_equity_curve()`
```python
def plot_equity_curve(
    self,
    show_drawdown: bool = True,
    show_trades: bool = True
) -> go.Figure
```

Plot equity curve with optional drawdown and trade markers.

**Parameters**:
- `show_drawdown`: Show drawdown subplot
- `show_trades`: Show trade entry/exit markers

**Returns**: Plotly Figure

**Example**:
```python
fig = report.plot_equity_curve(
    show_drawdown=True,
    show_trades=True
)
fig.show()
```

#### `plot_trade_distribution()`
```python
def plot_trade_distribution(self) -> go.Figure
```

Plot histogram of trade P&L.

**Returns**: Plotly Figure

**Example**:
```python
fig = report.plot_trade_distribution()
fig.show()
```

#### `save_report()`
```python
def save_report(
    self,
    output_path: Union[Path, str],
    include_all: bool = True
) -> Path
```

Save interactive HTML report with all charts.

**Parameters**:
- `output_path`: Output file path
- `include_all`: Include all visualizations

**Returns**: Path to saved file

**Example**:
```python
report_path = report.save_report(
    output_path='backtest_report.html',
    include_all=True
)
print(f"Report saved to: {report_path}")
```

---

## Optimization Module (src.optimization)

### PortfolioOptimizer

**Purpose**: Orchestrate portfolio optimization with multiple methods.

**Class Signature**:
```python
class PortfolioOptimizer:
    def __init__(
        self,
        returns: Union[pd.DataFrame, BacktestResult],
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    )
```

**Parameters**:
- `returns`: Return DataFrame or BacktestResult
- `config`: Configuration dict
- `risk_analytics`: Optional RiskAnalytics instance

**Methods**:

#### `optimize()`
```python
def optimize(
    self,
    method: str = 'mean_variance',
    **kwargs
) -> OptimizationResult
```

Run optimization with specified method.

**Parameters**:
- `method`: Optimization method ('mean_variance', 'risk_parity', 'black_litterman', 'min_cvar')
- `**kwargs`: Method-specific parameters

**Returns**: OptimizationResult with weights, expected_return, expected_risk, sharpe_ratio, metadata

**Example**:
```python
optimizer = PortfolioOptimizer(returns=strategy_returns)

# Mean-variance optimization
mv_result = optimizer.optimize(
    method='mean_variance',
    objective='max_sharpe'
)

print(f"Optimal Weights:")
for asset, weight in mv_result.weights.items():
    print(f"  {asset}: {weight:.2%}")

print(f"\nExpected Return: {mv_result.expected_return:.2%}")
print(f"Expected Risk: {mv_result.expected_risk:.2%}")
print(f"Sharpe Ratio: {mv_result.sharpe_ratio:.2f}")
```

#### `optimize_with_renewable_objectives()`
```python
def optimize_with_renewable_objectives(
    self,
    base_method: str,
    objectives: List[str]
) -> OptimizationResult
```

Optimize with renewable-specific objectives.

**Parameters**:
- `base_method`: Base optimization method
- `objectives`: List of objectives (['max_utilization', 'min_curtailment', 'max_recs', 'min_variance'])

**Returns**: OptimizationResult

**Example**:
```python
result = optimizer.optimize_with_renewable_objectives(
    base_method='mean_variance',
    objectives=['max_utilization', 'min_curtailment', 'max_recs']
)
```

#### `compare_methods()`
```python
def compare_methods(
    self,
    methods: List[str]
) -> pd.DataFrame
```

Compare multiple optimization methods.

**Parameters**:
- `methods`: List of methods to compare

**Returns**: Comparison DataFrame

**Example**:
```python
comparison = optimizer.compare_methods([
    'mean_variance',
    'risk_parity',
    'black_litterman',
    'min_cvar'
])

print(comparison)
# Shows expected_return, expected_risk, sharpe_ratio for each method
```

#### `save_results()`
```python
def save_results(
    self,
    result: OptimizationResult,
    filepath: Union[Path, str]
) -> Dict[str, Path]
```

Save optimization results.

**Parameters**:
- `result`: OptimizationResult to save
- `filepath`: Output file path

**Returns**: Dict mapping output type to file path

**Example**:
```python
saved_paths = optimizer.save_results(
    result=mv_result,
    filepath='optimization_results.json'
)
```

**See Also**: [Notebook 04: Portfolio Optimization](../notebooks/04_portfolio_optimization.ipynb)

---

### MeanVarianceOptimizer

**Purpose**: Markowitz mean-variance optimization.

**Class Signature**:
```python
class MeanVarianceOptimizer:
    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    )
```

**Methods**:

#### `optimize()`
```python
def optimize(
    self,
    objective: str = 'max_sharpe',
    risk_aversion: Optional[float] = None
) -> OptimizationResult
```

Optimize portfolio weights.

**Parameters**:
- `objective`: Objective function ('max_sharpe', 'min_risk', 'max_return', 'efficient_frontier')
- `risk_aversion`: Risk aversion parameter (for utility maximization)

**Returns**: OptimizationResult

**Example**:
```python
mv_optimizer = MeanVarianceOptimizer(returns=returns_df)

# Maximum Sharpe ratio
max_sharpe_result = mv_optimizer.optimize(objective='max_sharpe')

# Minimum risk
min_risk_result = mv_optimizer.optimize(objective='min_risk')

# Custom risk aversion
utility_result = mv_optimizer.optimize(
    objective='utility',
    risk_aversion=2.5
)
```

#### `compute_efficient_frontier()`
```python
def compute_efficient_frontier(
    self,
    n_points: int = 50,
    return_range: Optional[Tuple[float, float]] = None
) -> pd.DataFrame
```

Compute efficient frontier.

**Parameters**:
- `n_points`: Number of points on frontier
- `return_range`: Optional (min_return, max_return) range

**Returns**: DataFrame with columns ['return', 'risk', 'sharpe']

**Example**:
```python
frontier = mv_optimizer.compute_efficient_frontier(
    n_points=50
)

# Plot frontier
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=frontier['risk'],
    y=frontier['return'],
    mode='lines+markers',
    name='Efficient Frontier'
))
fig.update_layout(
    title='Efficient Frontier',
    xaxis_title='Risk (Volatility)',
    yaxis_title='Expected Return'
)
fig.show()
```

**See Also**: [ALGORITHMS.md - Mean-Variance Optimization](ALGORITHMS.md#mean-variance-optimization-markowitz)

---

### BlackLittermanOptimizer

**Purpose**: Black-Litterman model with investor views.

**Class Signature**:
```python
class BlackLittermanOptimizer:
    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    )
```

**Methods**:

#### `add_view()`
```python
def add_view(
    self,
    asset: str,
    expected_return: float,
    confidence: float
)
```

Add investor view.

**Parameters**:
- `asset`: Asset name
- `expected_return`: Expected return for asset
- `confidence`: Confidence level (0-1, where 1 = 100% confident)

**Example**:
```python
bl_optimizer = BlackLittermanOptimizer(returns=returns_df)

# Add views
bl_optimizer.add_view(
    asset='Mean_Reversion',
    expected_return=0.08,  # 8% expected return
    confidence=0.7  # 70% confident
)

bl_optimizer.add_view(
    asset='Renewable_Arbitrage',
    expected_return=0.12,  # 12% expected return
    confidence=0.8  # 80% confident
)
```

#### `add_renewable_view()`
```python
def add_renewable_view(
    self,
    asset: str,
    generation_forecast: float,
    price_forecast: float,
    correlation: float = -0.6
)
```

Add view based on renewable generation forecast.

**Parameters**:
- `asset`: Renewable asset name
- `generation_forecast`: Expected capacity factor
- `price_forecast`: Expected average price
- `correlation`: Generation-price correlation (default -0.6)

**Example**:
```python
# Add renewable-based view
bl_optimizer.add_renewable_view(
    asset='Wind',
    generation_forecast=0.40,  # 40% CF expected
    price_forecast=45.0,  # $45/MWh expected
    correlation=-0.6
)
```

#### `optimize()`
```python
def optimize(
    self,
    views: Dict[str, float],
    view_confidences: Dict[str, float]
) -> OptimizationResult
```

Optimize with views.

**Parameters**:
- `views`: Dict mapping asset to expected return
- `view_confidences`: Dict mapping asset to confidence level

**Returns**: OptimizationResult

**Example**:
```python
result = bl_optimizer.optimize(
    views={
        'Mean_Reversion': 0.08,
        'Renewable_Arbitrage': 0.12
    },
    view_confidences={
        'Mean_Reversion': 0.7,
        'Renewable_Arbitrage': 0.8
    }
)
```

**See Also**: [ALGORITHMS.md - Black-Litterman Model](ALGORITHMS.md#black-litterman-model)

---

### RiskAnalytics

**Purpose**: Calculate portfolio risk metrics and perform stress testing.

**Class Signature**:
```python
class RiskAnalytics:
    def __init__(
        self,
        result: BacktestResult,
        config: Optional[Dict] = None,
        benchmark_returns: Optional[pd.Series] = None
    )
```

**Parameters**:
- `result`: BacktestResult from backtest
- `config`: Configuration dict
- `benchmark_returns`: Optional benchmark return series

**Methods**:

#### `calculate_portfolio_var()`
```python
def calculate_portfolio_var(
    self,
    confidence_level: float = 0.95,
    method: str = 'historical',
    horizon: int = 1
) -> Dict[str, float]
```

Calculate Value at Risk.

**Parameters**:
- `confidence_level`: Confidence level (0.95 for 95%)
- `method`: Calculation method ('historical', 'parametric', 'monte_carlo')
- `horizon`: Time horizon (days)

**Returns**: Dict with VaR, confidence_level, method

**Example**:
```python
risk_analytics = RiskAnalytics(backtest_result)

# 95% VaR
var_95 = risk_analytics.calculate_portfolio_var(
    confidence_level=0.95,
    method='historical'
)
print(f"95% 1-day VaR: ${var_95['var']:,.2f}")

# 99% VaR
var_99 = risk_analytics.calculate_portfolio_var(
    confidence_level=0.99,
    method='parametric'
)
print(f"99% 1-day VaR: ${var_99['var']:,.2f}")
```

#### `calculate_portfolio_cvar()`
```python
def calculate_portfolio_cvar(
    self,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> Dict[str, float]
```

Calculate Conditional Value at Risk (Expected Shortfall).

**Parameters**:
- `confidence_level`: Confidence level
- `method`: Calculation method

**Returns**: Dict with CVaR, VaR, confidence_level

**Example**:
```python
cvar_95 = risk_analytics.calculate_portfolio_cvar(
    confidence_level=0.95,
    method='historical'
)
print(f"95% CVaR: ${cvar_95['cvar']:,.2f}")
print(f"Expected loss in worst 5% scenarios: ${cvar_95['cvar']:,.2f}")
```

#### `calculate_risk_decomposition()`
```python
def calculate_risk_decomposition(
    self,
    confidence_level: float = 0.95
) -> pd.DataFrame
```

Decompose risk by asset (marginal and component VaR).

**Parameters**:
- `confidence_level`: Confidence level for VaR calculation

**Returns**: DataFrame with columns ['asset', 'weight', 'marginal_var', 'component_var', 'contribution_pct']

**Example**:
```python
decomposition = risk_analytics.calculate_risk_decomposition(
    confidence_level=0.95
)

print("Risk Decomposition:")
print(decomposition.to_string(index=False))

# Identify largest risk contributors
top_risk = decomposition.nlargest(3, 'contribution_pct')
print(f"\nTop 3 Risk Contributors:")
print(top_risk[['asset', 'contribution_pct']])
```

#### `run_scenario_analysis()`
```python
def run_scenario_analysis(
    self,
    scenario_names: List[str]
) -> pd.DataFrame
```

Run stress tests.

**Parameters**:
- `scenario_names`: List of scenarios ('high_volatility', 'price_spike', 'low_wind', 'cloudy_period')

**Returns**: DataFrame with scenario results

**Example**:
```python
scenarios = risk_analytics.run_scenario_analysis([
    'high_volatility',
    'price_spike',
    'low_wind',
    'cloudy_period'
])

print("Stress Test Results:")
print(scenarios[['scenario', 'portfolio_loss', 'var_breach']])
```

#### `calculate_all_metrics()`
```python
def calculate_all_metrics(self) -> Dict[str, Any]
```

Calculate all risk metrics.

**Returns**: Dict with comprehensive risk metrics

**Example**:
```python
all_metrics = risk_analytics.calculate_all_metrics()

print("Risk Metrics:")
print(f"  95% VaR: ${all_metrics['var_95']:,.2f}")
print(f"  95% CVaR: ${all_metrics['cvar_95']:,.2f}")
print(f"  Portfolio volatility: {all_metrics['volatility']:.2%}")
print(f"  Information ratio: {all_metrics['information_ratio']:.2f}")
```

**See Also**: [ALGORITHMS.md - Risk Analytics](ALGORITHMS.md#risk-analytics)

---

## Dashboard Module (src.dashboard)

### Dashboard Utilities

Shared utility functions for the Streamlit dashboard.

#### `load_backtest_data()`
```python
def load_backtest_data(
    strategy_name: str,
    date_range: Tuple[datetime, datetime],
    use_cache: bool = True
) -> BacktestResult
```

Load or run backtest with caching.

**Parameters**:
- `strategy_name`: Strategy name
- `date_range`: (start, end) datetime tuple
- `use_cache`: Use cached results if available

**Returns**: BacktestResult

**Example**:
```python
import streamlit as st

@st.cache_data
def load_data(strategy, start, end):
    return load_backtest_data(strategy, (start, end), use_cache=True)

result = load_data('Mean Reversion', datetime(2023,1,1), datetime(2023,12,31))
```

#### `load_optimization_results()`
```python
def load_optimization_results(
    method: str,
    constraints: Dict,
    use_cache: bool = True
) -> OptimizationResult
```

Load or compute optimization with caching.

**Parameters**:
- `method`: Optimization method
- `constraints`: Constraint dict
- `use_cache`: Use cached results

**Returns**: OptimizationResult

**Example**:
```python
@st.cache_data
def load_opt(method, constraints):
    return load_optimization_results(method, constraints, use_cache=True)

result = load_opt('mean_variance', {'min_weight': 0.0, 'max_weight': 0.5})
```

#### `create_date_range_selector()`
```python
def create_date_range_selector() -> Tuple[datetime, datetime]
```

Create Streamlit date range widget.

**Returns**: (start_date, end_date) tuple

**Example**:
```python
import streamlit as st

st.sidebar.header("Date Range")
start_date, end_date = create_date_range_selector()
```

#### `create_strategy_selector()`
```python
def create_strategy_selector() -> str
```

Create Streamlit strategy dropdown.

**Returns**: Selected strategy name

**Example**:
```python
strategy = create_strategy_selector()
# Returns: 'Mean Reversion', 'Momentum', etc.
```

#### `create_parameter_sliders()`
```python
def create_parameter_sliders(
    strategy_name: str
) -> Dict[str, Any]
```

Create strategy-specific parameter sliders.

**Parameters**:
- `strategy_name`: Strategy name

**Returns**: Dict of parameter values

**Example**:
```python
params = create_parameter_sliders('Mean Reversion')
# Returns: {'lookback_period': 24, 'entry_threshold': 2.0, 'exit_threshold': 0.5}
```

#### `apply_dashboard_theme()`
```python
def apply_dashboard_theme(
    fig: go.Figure
) -> go.Figure
```

Apply consistent Plotly theme.

**Parameters**:
- `fig`: Plotly Figure

**Returns**: Themed Figure

**Example**:
```python
fig = go.Figure(data=...)
fig = apply_dashboard_theme(fig)
st.plotly_chart(fig, use_container_width=True)
```

---

## Configuration

### get_config()

**Purpose**: Load configuration from config.yaml.

**Function Signature**:
```python
def get_config() -> Dict[str, Any]
```

**Returns**: Configuration dictionary

**Example**:
```python
from src.config.load_config import get_config

config = get_config()

# Access configuration sections
data_config = config['data']
models_config = config['models']
strategies_config = config['strategies']
optimization_config = config['optimization']
risk_config = config['risk']
dashboard_config = config['dashboard']
```

### Configuration Structure

```yaml
data:
  eia_api_key: ${EIA_API_KEY}
  caiso_base_url: "https://oasis.caiso.com/oasisapi/SingleZip"
  storage_format: "parquet"
  compression: "snappy"

models:
  arima:
    order: [2, 1, 2]
    seasonal_order: [1, 1, 1, 24]
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  lstm:
    units: [50, 50]
    dropout: 0.2
    epochs: 100

strategies:
  mean_reversion:
    lookback_period: 20
    entry_threshold: 2.0
    exit_threshold: 0.5
  momentum:
    fast_period: 12
    slow_period: 48
    signal_threshold: 0.02

optimization:
  methods: ['mean_variance', 'risk_parity', 'black_litterman', 'min_cvar']
  constraints:
    min_weight: 0.0
    max_weight: 1.0
  risk_free_rate: 0.03

risk:
  var:
    confidence_levels: [0.95, 0.99]
    methods: ['historical', 'parametric']
  stress_scenarios:
    - 'high_volatility'
    - 'price_spike'
    - 'low_wind'
    - 'cloudy_period'

dashboard:
  default_date_range: '1Y'
  refresh_interval: 60  # seconds
  chart_theme: 'plotly_white'
```

---

## Common Patterns

### Error Handling

All methods raise descriptive exceptions:

```python
try:
    data = manager.load_data(...)
except FileNotFoundError:
    print("Data file not found. Generate data first.")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

### Logging

Use `logging.getLogger(__name__)` for consistent logging:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Starting backtest...")
logger.warning("High volatility detected")
logger.error("Optimization failed")
```

### Type Hints

All methods have type hints:

```python
def optimize(
    self,
    method: str,
    risk_aversion: Optional[float] = None
) -> OptimizationResult:
    pass
```

### Docstrings

Google-style docstrings with examples:

```python
def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals from price data.

    Args:
        data: DataFrame with price column and DatetimeIndex

    Returns:
        DataFrame with 'signal' column (-1, 0, 1)

    Raises:
        ValueError: If data is empty or missing required columns

    Example:
        >>> strategy = MeanReversionStrategy()
        >>> signals = strategy.generate_signals(prices)
        >>> buy_signals = signals[signals['signal'] == 1]
    """
    pass
```

### Configuration Pattern

Optional config parameter with fallback to config.yaml:

```python
def __init__(self, config: Optional[Dict] = None):
    if config is None:
        from src.config.load_config import get_config
        config = get_config()

    self.config = config
```

### Caching Pattern

Use `@st.cache_data` for expensive operations in dashboard:

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_expensive_data():
    # Expensive operation
    return data
```

---

## Quick Reference

### Data Pipeline
```python
DataManager → EIAFetcher/CAISOFetcher → save_raw_data → load_data
```

### Forecasting
```python
FeatureEngineer → PriceForecastingPipeline → train_models → predict
```

### Trading
```python
Strategy → generate_signals → BacktestEngine → run → BacktestResult
```

### Optimization
```python
PortfolioOptimizer → optimize → OptimizationResult
```

### Risk
```python
RiskAnalytics → calculate_all_metrics → RiskReport → generate_full_report
```

### Dashboard
```python
streamlit run src/dashboard/app.py → Navigate pages → Interactive analysis
```

---

**For more information, see:**
- [ALGORITHMS.md](ALGORITHMS.md) - Mathematical foundations
- [Notebooks](../notebooks/) - Interactive tutorials
- [README.md](../README.md) - Project overview
