"""
Shared Utilities for Streamlit Dashboard

This module provides shared utilities for the Streamlit dashboard including data loaders,
session state management, interactive controls, and chart styling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Internal imports
from src.backtesting.engine import BacktestEngine
from src.backtesting.reporting import BacktestReport
from src.optimization.optimizer import PortfolioOptimizer, OptimizationResult
from src.optimization.risk_analytics import RiskAnalytics
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.spread_trading import SpreadTradingStrategy
from src.strategies.renewable_arbitrage import RenewableArbitrageStrategy
from src.data.data_manager import DataManager
from src.data.synthetic_generator import SyntheticPriceGenerator
from src.data.renewable_generator import WindGenerator, SolarGenerator
from src.models.price_forecasting import PriceForecastingPipeline
from src.models.renewable_forecasting import RenewableForecastingPipeline
from src.config.load_config import get_config

# Get logger
logger = logging.getLogger(__name__)

# Constants (reuse from reporting.py)
COLORS = {
    'equity': '#1f77b4',
    'drawdown': '#d62728',
    'win': '#2ca02c',
    'loss': '#d62728',
    'neutral': '#7f7f7f',
    'var': '#d62728',
    'cvar': '#ff7f0e'
}

LAYOUT_DEFAULTS = {
    'height': 400,
    'template': 'plotly_white',
    'hovermode': 'x unified',
    'showlegend': True
}


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(ttl=3600)
def load_backtest_data(
    strategy_name: str,
    date_range: Tuple[datetime, datetime],
    initial_capital: float = 1000000,
    transaction_cost: float = 0.001,
    strategy_params: Optional[Dict] = None,
    data_source: str = 'synthetic',
    use_cache: bool = True
) -> Any:
    """
    Load or run backtest.

    Args:
        strategy_name: Name of strategy to backtest
        date_range: Tuple of (start_date, end_date)
        initial_capital: Initial capital for backtesting
        transaction_cost: Transaction cost as percentage (e.g., 0.001 = 0.1%)
        strategy_params: Dict of strategy-specific parameters
        data_source: Data source to use ('synthetic', 'eia', 'caiso')
        use_cache: Whether to use cached results

    Returns:
        BacktestResult
    """
    try:
        config = get_config()

        if strategy_params is None:
            strategy_params = {}

        # Create config override with user-specified values
        config_override = config.copy()
        if 'backtesting' not in config_override:
            config_override['backtesting'] = {}
        config_override['backtesting']['initial_capital'] = initial_capital
        if 'transaction_costs' not in config_override['backtesting']:
            config_override['backtesting']['transaction_costs'] = {}
        config_override['backtesting']['transaction_costs']['percentage_cost'] = transaction_cost

        # Load price data using provided data source
        price_data = load_price_data(data_source, date_range)

        if price_data.empty:
            raise ValueError("No price data available")

        # Initialize strategy
        strategy_map = {
            'Mean Reversion': MeanReversionStrategy,
            'Momentum': MomentumStrategy,
            'Spread Trading': SpreadTradingStrategy,
            'Renewable Arbitrage': RenewableArbitrageStrategy
        }

        strategy_class = strategy_map.get(strategy_name)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = strategy_class(config=config_override)

        # Apply strategy parameters by setting attributes
        for param_name, param_value in strategy_params.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, param_value)

        # Generate signals from strategy
        signals = strategy.generate_signals(price_data)

        # Run backtest with proper API
        engine = BacktestEngine(strategies=strategy, config=config_override)
        result = engine.run(
            price_data=price_data,
            signals=signals,
            start_date=date_range[0].strftime('%Y-%m-%d'),
            end_date=date_range[1].strftime('%Y-%m-%d')
        )

        return result

    except Exception as e:
        logger.error(f"Error loading backtest data: {str(e)}")
        handle_error(e, "Loading backtest data")
        return None


@st.cache_data(ttl=3600)
def load_optimization_results(
    method: str,
    constraints: Dict,
    use_cache: bool = True
) -> Optional[OptimizationResult]:
    """
    Load or compute optimization.

    Args:
        method: Optimization method name
        constraints: Dict of constraint parameters
        use_cache: Whether to use cached results

    Returns:
        OptimizationResult or None
    """
    try:
        config = get_config()

        # Load returns data (simplified - would normally load from backtest)
        generator = SyntheticPriceGenerator(config=config)
        price_data = generator.generate_price_series(
            start_date='2023-01-01',
            end_date='2024-12-31',
            frequency='D',
            n_scenarios=1
        )
        returns = price_data['price'].pct_change().dropna()
        returns = pd.DataFrame({'returns': returns})

        # Run optimization
        optimizer = PortfolioOptimizer(returns=returns, config=config)
        result = optimizer.optimize(method=method, **constraints)

        return result

    except Exception as e:
        logger.error(f"Error loading optimization results: {str(e)}")
        handle_error(e, "Loading optimization results")
        return None


@st.cache_data(ttl=3600)
def load_price_data(
    source: str = 'synthetic',
    date_range: Optional[Tuple[datetime, datetime]] = None
) -> pd.DataFrame:
    """
    Load price data (real or synthetic).

    Args:
        source: Data source ('synthetic', 'real', or specific source like 'eia', 'caiso')
        date_range: Optional date range tuple

    Returns:
        DataFrame with price data
    """
    try:
        config = get_config()

        if date_range is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            date_range = (start_date, end_date)

        if source == 'synthetic':
            generator = SyntheticPriceGenerator(config=config)
            data = generator.generate_price_series(
                start_date=date_range[0].strftime('%Y-%m-%d'),
                end_date=date_range[1].strftime('%Y-%m-%d'),
                frequency='H',
                n_scenarios=1
            )
        else:
            # Handle 'real' by using preferred source from config
            if source == 'real':
                preferred_source = config.get('dashboard', {}).get('data_sources', {}).get('real_data', {}).get('preferred_source', 'caiso')
                actual_source = preferred_source
            else:
                actual_source = source

            # Use DataManager to load real data
            try:
                data_manager = DataManager(config=config)
                # Try to load real data (assuming 'electricity' or 'lmp' dataset)
                # Try multiple dataset names in order of preference
                for dataset_name in ['electricity', 'lmp', 'prices']:
                    try:
                        data = data_manager.load_data(
                            source=actual_source,
                            dataset=dataset_name,
                            data_type='processed',  # Try processed first
                            start_date=date_range[0].strftime('%Y-%m-%d'),
                            end_date=date_range[1].strftime('%Y-%m-%d'),
                            date_filter=True
                        )
                        if not data.empty:
                            logger.info(f"Successfully loaded real data from {actual_source}/{dataset_name}")
                            return data
                    except FileNotFoundError:
                        continue

                # If no processed data found, try raw data
                for dataset_name in ['electricity', 'lmp', 'prices']:
                    try:
                        data = data_manager.load_data(
                            source=actual_source,
                            dataset=dataset_name,
                            data_type='raw',
                            start_date=date_range[0].strftime('%Y-%m-%d'),
                            end_date=date_range[1].strftime('%Y-%m-%d'),
                            date_filter=True
                        )
                        if not data.empty:
                            logger.info(f"Successfully loaded real data from {actual_source}/{dataset_name} (raw)")
                            return data
                    except FileNotFoundError:
                        continue

                # If we get here, no real data was found
                raise FileNotFoundError(f"No real data found for source '{actual_source}'")

            except Exception as e:
                logger.warning(f"Real data source '{actual_source}' not available ({str(e)}), falling back to synthetic")
                generator = SyntheticPriceGenerator(config=config)
                data = generator.generate_price_series(
                    start_date=date_range[0].strftime('%Y-%m-%d'),
                    end_date=date_range[1].strftime('%Y-%m-%d'),
                    frequency='H',
                    n_scenarios=1
                )

        return data

    except Exception as e:
        logger.error(f"Error loading price data: {str(e)}")
        handle_error(e, "Loading price data")
        return pd.DataFrame()


@st.cache_resource(ttl=86400)  # Cache models for 24 hours
def _load_or_train_forecasting_model(
    forecast_type: str,
    data_source: str,
    model_types: Tuple[str, ...]
) -> Any:
    """
    Load or train forecasting models with caching.

    Args:
        forecast_type: Type of forecast ('price', 'wind', 'solar')
        data_source: Data source for training
        model_types: Tuple of model types to train

    Returns:
        Trained pipeline
    """
    config = get_config()

    if forecast_type == 'price':
        pipeline = PriceForecastingPipeline(config=config)

        # Load training data
        if data_source == 'synthetic':
            generator = SyntheticPriceGenerator(config=config)
            historical_data = generator.generate_price_series(
                start_date='2023-01-01',
                end_date='2024-12-31',
                frequency='H',
                n_scenarios=1
            )
        else:
            # Load real data
            date_range = (datetime(2023, 1, 1), datetime(2024, 12, 31))
            historical_data = load_price_data(data_source, date_range)

        # Train pipeline
        logger.info(f"Training {forecast_type} forecasting models: {model_types}")
        pipeline.prepare_data(historical_data, target_col=historical_data.columns[0])
        pipeline.train_models(model_types=list(model_types))

        # Evaluate models to populate metrics
        pipeline.evaluate_models()

        # Create ensemble using evaluation metrics
        pipeline.create_ensemble()

        return pipeline

    elif forecast_type in ['wind', 'solar']:
        pipeline = RenewableForecastingPipeline(config=config, resource_type=forecast_type)

        # Load training data for renewables
        if forecast_type == 'wind':
            generator = WindGenerator(config=config)
            historical_data = generator.generate_wind_profile(
                start_date='2023-01-01',
                end_date='2024-12-31',
                n_scenarios=1,
                capacity_mw=100.0
            )
        else:
            generator = SolarGenerator(config=config)
            historical_data = generator.generate_solar_profile(
                start_date='2023-01-01',
                end_date='2024-12-31',
                n_scenarios=1,
                capacity_mw=50.0
            )

        # Train pipeline
        logger.info(f"Training {forecast_type} forecasting models")
        pipeline.prepare_data(historical_data)
        pipeline.train_models()

        return pipeline

    else:
        raise ValueError(f"Unknown forecast type: {forecast_type}")


@st.cache_data(ttl=3600)
def load_forecast_data(
    forecast_type: str = 'price',
    horizon: int = 24,
    models: Optional[List[str]] = None,
    confidence_level: float = 0.95,
    data_source: str = 'synthetic',
    scenario: str = 'base'
) -> Dict[str, Any]:
    """
    Load or generate forecasts with real ML models.

    Args:
        forecast_type: Type of forecast ('price', 'wind', 'solar')
        horizon: Forecast horizon in hours
        models: List of models to use (default: ['ensemble'])
        confidence_level: Confidence level for intervals (default: 0.95)
        data_source: Data source for training/forecasting
        scenario: Scenario type ('base', 'optimistic', 'pessimistic', 'custom')

    Returns:
        Dictionary containing:
        - 'forecast': DataFrame with forecasts and confidence intervals
        - 'metrics': Dict of validation metrics by model
        - 'residuals': Dict of residuals by model
        - 'historical': DataFrame with historical data
    """
    try:
        config = get_config()

        if models is None:
            models = ['ensemble']

        # Load or train models
        model_types = tuple(['arima', 'xgboost', 'lstm'] if 'ensemble' in models or len(models) > 1 else models)
        pipeline = _load_or_train_forecasting_model(forecast_type, data_source, model_types)

        # Load recent historical data for context
        if data_source == 'synthetic':
            if forecast_type == 'price':
                generator = SyntheticPriceGenerator(config=config)
                historical_data = generator.generate_price_series(
                    start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    frequency='H',
                    n_scenarios=1
                )
            elif forecast_type == 'wind':
                generator = WindGenerator(config=config)
                historical_data = generator.generate_wind_profile(
                    start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    n_scenarios=1,
                    capacity_mw=100.0
                )
            else:
                generator = SolarGenerator(config=config)
                historical_data = generator.generate_solar_profile(
                    start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    n_scenarios=1,
                    capacity_mw=50.0
                )
        else:
            date_range = (datetime.now() - timedelta(days=90), datetime.now())
            historical_data = load_price_data(data_source, date_range)

        # Get validation metrics from pipeline
        metrics_dict = {}
        residuals_dict = {}

        if hasattr(pipeline, 'metrics') and pipeline.metrics:
            for model_name, model_metrics in pipeline.metrics.items():
                metrics_dict[model_name] = model_metrics

        # Generate forecasts for each model
        forecast_results = {}
        for model_name in models:
            model_key = model_name.lower()

            # Generate forecast
            forecast_df = pipeline.predict(historical_data, model_name=model_key, steps=horizon)

            # Apply scenario adjustments
            if scenario == 'optimistic':
                forecast_df['forecast'] *= 1.1
                forecast_df['upper_ci'] *= 1.15
                forecast_df['lower_ci'] *= 1.05
            elif scenario == 'pessimistic':
                forecast_df['forecast'] *= 0.9
                forecast_df['upper_ci'] *= 0.95
                forecast_df['lower_ci'] *= 0.85
            elif scenario == 'custom':
                # Custom scenarios can be implemented later
                pass

            forecast_results[model_name] = forecast_df

        # Use primary model for main forecast (ensemble if available)
        primary_model = 'ensemble' if 'ensemble' in models else models[0]
        main_forecast = forecast_results[primary_model].copy()

        # Rename columns to match expected format
        main_forecast = main_forecast.rename(columns={
            'timestamp': 'timestamp',
            'forecast': 'forecast',
            'lower_ci': 'lower_bound',
            'upper_ci': 'upper_bound'
        })

        # Compute real residuals for each model if test data available
        feature_importance_dict = {}

        if hasattr(pipeline, 'y_test') and pipeline.y_test is not None:
            for model_name in models:
                model_key = model_name.lower()

                # Skip ensemble for residuals computation
                if model_key == 'ensemble':
                    continue

                if model_key in pipeline.models:
                    try:
                        if model_key == 'arima':
                            # ARIMA: predict on test set
                            if hasattr(pipeline, 'X_test') and len(pipeline.y_test) > 0:
                                arima_model = pipeline.models['arima']
                                steps = len(pipeline.y_test)
                                pred_result = arima_model.predict(steps=steps, return_conf_int=False)
                                y_pred = pred_result['forecast']
                                residuals = pipeline.y_test.values - y_pred
                                residuals_dict[model_name] = residuals

                        elif model_key == 'xgboost':
                            # XGBoost: predict on X_test
                            if hasattr(pipeline, 'X_test') and len(pipeline.X_test) > 0:
                                xgb_model = pipeline.models['xgboost']
                                pred_result = xgb_model.predict(pipeline.X_test, return_conf_int=False)
                                y_pred = pred_result['forecast']
                                residuals = pipeline.y_test.values - y_pred
                                residuals_dict[model_name] = residuals

                                # Extract feature importances for XGBoost
                                if hasattr(xgb_model, 'model') and hasattr(xgb_model.model, 'feature_importances_'):
                                    importances = xgb_model.model.feature_importances_
                                    feature_names = pipeline.feature_engineer.get_feature_names()

                                    if len(importances) == len(feature_names):
                                        # Sort by importance
                                        importance_pairs = sorted(
                                            zip(feature_names, importances),
                                            key=lambda x: x[1],
                                            reverse=True
                                        )
                                        top_n = min(15, len(importance_pairs))  # Top 15 features
                                        top_features = [pair[0] for pair in importance_pairs[:top_n]]
                                        top_importances = [pair[1] for pair in importance_pairs[:top_n]]

                                        feature_importance_dict['xgboost'] = {
                                            'features': top_features,
                                            'importances': top_importances
                                        }

                        elif model_key == 'lstm':
                            # LSTM: predict on X_test_seq
                            if hasattr(pipeline, 'X_test_seq') and hasattr(pipeline, 'y_test_seq'):
                                if len(pipeline.X_test_seq) > 0 and len(pipeline.y_test_seq) > 0:
                                    lstm_model = pipeline.models['lstm']
                                    pred_result = lstm_model.predict(pipeline.X_test_seq, return_conf_int=False)
                                    y_pred = pred_result['forecast']
                                    residuals = pipeline.y_test_seq - y_pred
                                    residuals_dict[model_name] = residuals

                    except Exception as e:
                        logger.warning(f"Could not compute residuals for {model_name}: {str(e)}")
                        # Skip this model's residuals
                        continue

        result = {
            'forecast': main_forecast,
            'metrics': metrics_dict,
            'residuals': residuals_dict,
            'historical': historical_data,
            'all_forecasts': forecast_results,
            'feature_importance': feature_importance_dict,
            'confidence_level': confidence_level
        }

        return result

    except Exception as e:
        logger.error(f"Error loading forecast data: {str(e)}")
        handle_error(e, "Loading forecast data")

        # Return dummy data for graceful degradation
        forecast_dates = pd.date_range(
            start=datetime.now(),
            periods=horizon,
            freq='H'
        )
        dummy_forecast = pd.DataFrame({
            'forecast': np.random.randn(horizon).cumsum() + 50,
            'lower_bound': np.random.randn(horizon).cumsum() + 45,
            'upper_bound': np.random.randn(horizon).cumsum() + 55
        }, index=forecast_dates)

        return {
            'forecast': dummy_forecast,
            'metrics': {},
            'residuals': {},
            'historical': pd.DataFrame(),
            'all_forecasts': {},
            'feature_importance': {},
            'confidence_level': confidence_level
        }


# =============================================================================
# Session State Management
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.selected_strategy = 'Mean Reversion'
        st.session_state.date_range = (
            datetime.now() - timedelta(days=365),
            datetime.now()
        )
        st.session_state.optimization_method = 'mean_variance'
        st.session_state.risk_level = 0.95
        st.session_state.data_source = 'synthetic'
        st.session_state.last_update = datetime.now()


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Get value from session state with default.

    Args:
        key: Session state key
        default: Default value if key not found

    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any):
    """
    Set value in session state.

    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


# =============================================================================
# Interactive Controls
# =============================================================================

def create_date_range_selector(container=None) -> Tuple[datetime, datetime]:
    """
    Create date range selector widget.

    Args:
        container: Streamlit container to render in (default: st)

    Returns:
        Tuple of (start_date, end_date)
    """
    if container is None:
        container = st

    config = get_config()
    default_range = config.get('dashboard', {}).get('date_ranges', {}).get('default_range', '1Y')

    # Calculate default dates
    end_date = datetime.now()
    if default_range == '1Y':
        start_date = end_date - timedelta(days=365)
    elif default_range == '6M':
        start_date = end_date - timedelta(days=180)
    elif default_range == '3M':
        start_date = end_date - timedelta(days=90)
    elif default_range == '1M':
        start_date = end_date - timedelta(days=30)
    else:
        start_date = end_date - timedelta(days=365)

    col1, col2 = container.columns(2)
    with col1:
        start = container.date_input("Start Date", value=start_date.date())
    with col2:
        end = container.date_input("End Date", value=end_date.date())

    return (datetime.combine(start, datetime.min.time()),
            datetime.combine(end, datetime.max.time()))


def create_strategy_selector() -> str:
    """
    Create strategy selection dropdown.

    Returns:
        Strategy name
    """
    strategies = [
        'Mean Reversion',
        'Momentum',
        'Spread Trading',
        'Renewable Arbitrage'
    ]

    return st.selectbox("Select Strategy", strategies)


def create_parameter_sliders(strategy_name: str) -> Dict:
    """
    Create parameter adjustment sliders.

    Args:
        strategy_name: Name of strategy

    Returns:
        Dict with parameter values
    """
    config = get_config()
    controls_config = config.get('dashboard', {}).get('controls', {})

    params = {}

    if strategy_name == 'Mean Reversion':
        mr_config = controls_config.get('mean_reversion', {})
        params['window'] = st.slider(
            "Window",
            min_value=mr_config.get('window', [10, 50, 20])[0],
            max_value=mr_config.get('window', [10, 50, 20])[1],
            value=mr_config.get('window', [10, 50, 20])[2]
        )
        params['num_std'] = st.slider(
            "Number of Std Dev",
            min_value=mr_config.get('num_std', [1.0, 3.0, 2.0])[0],
            max_value=mr_config.get('num_std', [1.0, 3.0, 2.0])[1],
            value=mr_config.get('num_std', [1.0, 3.0, 2.0])[2]
        )

    elif strategy_name == 'Momentum':
        mom_config = controls_config.get('momentum', {})
        params['fast_window'] = st.slider(
            "Fast Window",
            min_value=mom_config.get('fast_window', [5, 30, 10])[0],
            max_value=mom_config.get('fast_window', [5, 30, 10])[1],
            value=mom_config.get('fast_window', [5, 30, 10])[2]
        )
        params['slow_window'] = st.slider(
            "Slow Window",
            min_value=mom_config.get('slow_window', [20, 100, 30])[0],
            max_value=mom_config.get('slow_window', [20, 100, 30])[1],
            value=mom_config.get('slow_window', [20, 100, 30])[2]
        )

    elif strategy_name == 'Spread Trading':
        spread_config = controls_config.get('spread_trading', {})
        params['lookback_window'] = st.slider(
            "Lookback Window",
            min_value=spread_config.get('lookback_window', [30, 120, 60])[0],
            max_value=spread_config.get('lookback_window', [30, 120, 60])[1],
            value=spread_config.get('lookback_window', [30, 120, 60])[2]
        )
        params['entry_z_score'] = st.slider(
            "Entry Z-Score",
            min_value=spread_config.get('entry_z_score', [1.0, 3.0, 2.0])[0],
            max_value=spread_config.get('entry_z_score', [1.0, 3.0, 2.0])[1],
            value=spread_config.get('entry_z_score', [1.0, 3.0, 2.0])[2],
            step=0.1
        )
        params['exit_z_score'] = st.slider(
            "Exit Z-Score",
            min_value=spread_config.get('exit_z_score', [0.0, 1.0, 0.5])[0],
            max_value=spread_config.get('exit_z_score', [0.0, 1.0, 0.5])[1],
            value=spread_config.get('exit_z_score', [0.0, 1.0, 0.5])[2],
            step=0.1
        )

    elif strategy_name == 'Renewable Arbitrage':
        renew_config = controls_config.get('renewable_arbitrage', {})
        params['forecast_horizon'] = st.slider(
            "Forecast Horizon (hours)",
            min_value=renew_config.get('forecast_horizon', [6, 72, 24])[0],
            max_value=renew_config.get('forecast_horizon', [6, 72, 24])[1],
            value=renew_config.get('forecast_horizon', [6, 72, 24])[2]
        )
        params['generation_threshold_high'] = st.slider(
            "Generation Threshold High",
            min_value=renew_config.get('generation_threshold_high', [0.5, 0.9, 0.7])[0],
            max_value=renew_config.get('generation_threshold_high', [0.5, 0.9, 0.7])[1],
            value=renew_config.get('generation_threshold_high', [0.5, 0.9, 0.7])[2],
            step=0.05
        )
        params['generation_threshold_low'] = st.slider(
            "Generation Threshold Low",
            min_value=renew_config.get('generation_threshold_low', [0.1, 0.5, 0.3])[0],
            max_value=renew_config.get('generation_threshold_low', [0.1, 0.5, 0.3])[1],
            value=renew_config.get('generation_threshold_low', [0.1, 0.5, 0.3])[2],
            step=0.05
        )

    return params


def create_optimization_controls() -> Dict:
    """
    Create optimization constraint inputs.

    Returns:
        Dict with constraint values
    """
    config = get_config()
    opt_config = config.get('dashboard', {}).get('controls', {}).get('optimization', {})

    constraints = {}

    col1, col2 = st.columns(2)
    with col1:
        constraints['min_weight'] = st.slider(
            "Min Weight",
            min_value=opt_config.get('min_weight', [0.0, 0.2, 0.0])[0],
            max_value=opt_config.get('min_weight', [0.0, 0.2, 0.0])[1],
            value=opt_config.get('min_weight', [0.0, 0.2, 0.0])[2]
        )
    with col2:
        constraints['max_weight'] = st.slider(
            "Max Weight",
            min_value=opt_config.get('max_weight', [0.1, 1.0, 0.3])[0],
            max_value=opt_config.get('max_weight', [0.1, 1.0, 0.3])[1],
            value=opt_config.get('max_weight', [0.1, 1.0, 0.3])[2]
        )

    constraints['risk_aversion'] = st.slider(
        "Risk Aversion",
        min_value=opt_config.get('risk_aversion', [0.5, 5.0, 2.0])[0],
        max_value=opt_config.get('risk_aversion', [0.5, 5.0, 2.0])[1],
        value=opt_config.get('risk_aversion', [0.5, 5.0, 2.0])[2]
    )

    constraints['allow_short'] = st.checkbox(
        "Allow Short Positions",
        value=opt_config.get('allow_short', False)
    )

    return constraints


def create_scenario_selector() -> List[str]:
    """
    Create stress test scenario dropdown.

    Returns:
        List of selected scenarios
    """
    config = get_config()
    scenarios = config.get('risk', {}).get('stress_scenarios', {}).keys()

    return st.multiselect(
        "Select Scenarios",
        options=list(scenarios),
        default=[]
    )


# =============================================================================
# Chart Styling
# =============================================================================

def apply_dashboard_theme(fig: go.Figure) -> go.Figure:
    """
    Apply consistent theme to Plotly figures.

    Args:
        fig: Plotly figure

    Returns:
        Styled figure
    """
    config = get_config()

    # Read from session_state first, fallback to config
    theme = st.session_state.get('chart_theme') or config.get('dashboard', {}).get('charts', {}).get('theme', 'plotly_white')
    height = st.session_state.get('chart_height') or config.get('dashboard', {}).get('charts', {}).get('default_height', 400)

    fig.update_layout(
        template=theme,
        height=height,
        hovermode=LAYOUT_DEFAULTS['hovermode'],
        showlegend=LAYOUT_DEFAULTS['showlegend']
    )

    return fig


def format_currency(value: float) -> str:
    """
    Format as currency.

    Args:
        value: Numeric value

    Returns:
        Formatted currency string
    """
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format as percentage.

    Args:
        value: Numeric value (0.1234 = 12.34%)

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def create_metric_card(
    label: str,
    value: float,
    delta: Optional[float] = None,
    format_type: str = 'number'
):
    """
    Create Streamlit metric display.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        format_type: Format type ('currency', 'percentage', 'number')
    """
    if format_type == 'currency':
        formatted_value = format_currency(value)
        formatted_delta = format_currency(delta) if delta is not None else None
    elif format_type == 'percentage':
        formatted_value = format_percentage(value)
        formatted_delta = format_percentage(delta) if delta is not None else None
    else:
        formatted_value = f"{value:,.2f}"
        formatted_delta = f"{delta:,.2f}" if delta is not None else None

    st.metric(label=label, value=formatted_value, delta=formatted_delta)


# =============================================================================
# Data Refresh
# =============================================================================

def refresh_data():
    """Refresh all cached data."""
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.last_update = datetime.now()
    st.success("Data refreshed successfully!")


def auto_refresh_check() -> bool:
    """
    Check if auto-refresh needed.

    Returns:
        True if refresh needed
    """
    config = get_config()

    # Read from session_state first, fall back to config
    enable_auto_refresh = st.session_state.get('enable_auto_refresh')
    if enable_auto_refresh is None:
        enable_auto_refresh = config.get('dashboard', {}).get('enable_auto_refresh', False)

    if not enable_auto_refresh:
        return False

    # Get update frequency from session_state or config
    update_frequency = st.session_state.get('update_frequency')
    if update_frequency is None:
        update_frequency = config.get('dashboard', {}).get('update_frequency', 300)

    # Get last_update timestamp
    last_update = st.session_state.get('last_update')
    if last_update is None:
        last_update = datetime.now() - timedelta(seconds=update_frequency + 1)

    return (datetime.now() - last_update).total_seconds() > update_frequency


# =============================================================================
# Error Handling
# =============================================================================

def handle_error(error: Exception, context: str):
    """
    Display user-friendly error messages.

    Args:
        error: Exception object
        context: Context description
    """
    error_msg = f"Error in {context}: {str(error)}"
    st.error(error_msg)
    logger.error(error_msg, exc_info=True)


def validate_inputs(date_range: Tuple[datetime, datetime]) -> Tuple[bool, str]:
    """
    Validate user inputs.

    Args:
        date_range: Date range tuple

    Returns:
        Tuple of (is_valid, error_message)
    """
    if date_range[0] >= date_range[1]:
        return False, "Start date must be before end date"

    if (date_range[1] - date_range[0]).days < 1:
        return False, "Date range must be at least 1 day"

    return True, ""


# =============================================================================
# Helper Functions
# =============================================================================

def get_available_strategies() -> List[str]:
    """
    List available strategies.

    Returns:
        List of strategy names
    """
    return [
        'Mean Reversion',
        'Momentum',
        'Spread Trading',
        'Renewable Arbitrage'
    ]


def get_available_assets() -> List[str]:
    """
    List available assets from data.

    Returns:
        List of asset names
    """
    return ['Energy', 'Wind', 'Solar', 'Natural Gas']


def calculate_comparison_metrics(results: List) -> pd.DataFrame:
    """
    Calculate metrics for comparison.

    Args:
        results: List of BacktestResult or OptimizationResult objects

    Returns:
        DataFrame with comparison table
    """
    metrics = []

    for i, result in enumerate(results):
        if hasattr(result, 'metrics'):  # BacktestResult
            # metrics is a dict, not an object with attributes
            result_metrics = result.metrics if isinstance(result.metrics, dict) else {}
            metrics.append({
                'Name': f'Strategy {i+1}',
                'Total Return': result_metrics.get('total_return', 0.0),
                'Sharpe Ratio': result_metrics.get('sharpe_ratio', 0.0),
                'Max Drawdown': result_metrics.get('max_drawdown', 0.0),
                'Win Rate': result_metrics.get('win_rate', 0.0)
            })
        else:  # OptimizationResult
            metrics.append({
                'Name': f'Portfolio {i+1}',
                'Expected Return': result.expected_return,
                'Expected Risk': result.expected_risk,
                'Sharpe Ratio': result.sharpe_ratio
            })

    return pd.DataFrame(metrics)
