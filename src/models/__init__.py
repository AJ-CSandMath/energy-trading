"""
Machine learning models module for energy price forecasting and renewable generation.

This module provides:
- FeatureEngineer: Transform time-series data into ML-ready features
- ARIMAForecaster: Classical time-series forecasting
- XGBoostForecaster: Gradient boosting for price prediction
- LSTMForecaster: Deep learning for sequential patterns
- PriceForecastingPipeline: End-to-end forecasting orchestration
- RenewableFeatureEngineer: Renewable-specific feature engineering
- QuantileXGBoostForecaster: Quantile regression for probabilistic forecasts
- WindForecaster: Wind generation forecasting with curtailment
- SolarForecaster: Solar generation forecasting with curtailment
- MonteCarloSimulator: Correlated scenario generation
- RenewableForecastingPipeline: End-to-end renewable forecasting orchestration
"""

from src.models.feature_engineering import FeatureEngineer
from src.models.price_forecasting import (
    ARIMAForecaster,
    XGBoostForecaster,
    LSTMForecaster,
    PriceForecastingPipeline
)
from src.models.renewable_forecasting import (
    RenewableFeatureEngineer,
    QuantileXGBoostForecaster,
    WindForecaster,
    SolarForecaster,
    MonteCarloSimulator,
    RenewableForecastingPipeline,
    calculate_generation_metrics,
    apply_curtailment
)

__all__ = [
    # Price forecasting
    'FeatureEngineer',
    'ARIMAForecaster',
    'XGBoostForecaster',
    'LSTMForecaster',
    'PriceForecastingPipeline',
    # Renewable forecasting
    'RenewableFeatureEngineer',
    'QuantileXGBoostForecaster',
    'WindForecaster',
    'SolarForecaster',
    'MonteCarloSimulator',
    'RenewableForecastingPipeline',
    'calculate_generation_metrics',
    'apply_curtailment'
]
