"""
Pytest configuration file with shared fixtures for all tests.

Provides common test data, mock objects, and helper functions used across
unit and integration tests.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestResult
from src.config.load_config import get_config
from src.data.data_manager import DataManager
from src.data.renewable_generator import SolarGenerator, WindGenerator
from src.data.synthetic_generator import SyntheticPriceGenerator
from src.optimization.optimizer import OptimizationResult

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_config() -> Dict:
    """
    Provide test configuration dictionary.

    Returns minimal config with test-specific values, temporary paths,
    and small model parameters for fast tests.
    """
    return {
        "data": {
            "raw_data_path": "test_data/raw",
            "processed_data_path": "test_data/processed",
            "partition_by_date": False,
            "compression": "snappy",
        },
        "models": {
            "price_forecasting": {
                "arima": {"order": (1, 1, 1)},
                "xgboost": {"n_estimators": 10, "max_depth": 3},
                "lstm": {"units": 10, "layers": 1, "epochs": 2, "batch_size": 16},
            },
            "features": {
                "lags": [1, 2, 3],
                "rolling_windows": [12, 24],
                "time_features": ["hour", "day_of_week", "month", "is_weekend"],
            },
            "forecasting": {
                "lookback_hours": 24,
                "forecast_horizon": 1,
            },
        },
        "strategies": {
            "mean_reversion": {
                "window": 20,
                "num_std": 2.0,
                "entry_threshold": 0.8,
                "exit_threshold": 0.2,
            },
            "risk_management": {
                "max_position_size": 0.2,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
            },
        },
        "backtesting": {
            "initial_capital": 100000.0,
            "transaction_costs": {
                "fixed_cost": 2.0,
                "percentage_cost": 0.001,
            },
        },
        "optimization": {
            "risk_free_rate": 0.02,
            "constraints": {
                "min_weight": 0.0,
                "max_weight": 0.4,
            },
        },
    }


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """
    Provide synthetic price DataFrame for testing.

    Generates 100 hourly price observations using SyntheticPriceGenerator
    with fixed random seed for reproducibility.
    """
    np.random.seed(42)
    generator = SyntheticPriceGenerator()
    start_date = "2023-01-01"
    end_date = "2023-01-05"  # ~100 hours
    prices = generator.generate_price_series(
        start_date=start_date, end_date=end_date, frequency="H", n_scenarios=1
    )
    return prices


@pytest.fixture
def sample_wind_data() -> pd.DataFrame:
    """
    Provide synthetic wind generation DataFrame.

    Generates 100 hourly wind observations using WindGenerator
    with fixed random seed.
    """
    np.random.seed(42)
    generator = WindGenerator()
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    wind_data = generator.generate_wind_profile(
        start_date=start_date, end_date=end_date, n_scenarios=1, capacity_mw=100.0
    )
    return wind_data


@pytest.fixture
def sample_solar_data() -> pd.DataFrame:
    """
    Provide synthetic solar generation DataFrame.

    Generates 100 hourly solar observations using SolarGenerator
    with fixed random seed.
    """
    np.random.seed(42)
    generator = SolarGenerator()
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    solar_data = generator.generate_solar_profile(
        start_date=start_date, end_date=end_date, n_scenarios=1, capacity_mw=100.0
    )
    return solar_data


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """
    Provide temporary data directory for tests.

    Uses pytest's tmp_path fixture to create isolated directories
    for raw and processed data. Automatically cleaned up after test.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def mock_data_manager(temp_data_dir: Path) -> DataManager:
    """
    Provide DataManager instance with temporary paths.

    Returns DataManager configured to use temp_data_dir for all operations.
    """
    return DataManager(
        raw_data_path=str(temp_data_dir / "raw"),
        processed_data_path=str(temp_data_dir / "processed"),
    )


# =============================================================================
# Backtest & Optimization Result Fixtures
# =============================================================================


@pytest.fixture
def sample_backtest_result() -> BacktestResult:
    """
    Provide minimal BacktestResult for testing.

    Creates BacktestResult with synthetic equity curve, trades,
    and portfolio history for use in optimization and metrics tests.
    """
    # Create synthetic equity curve
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    equity = 100000 * np.cumprod(1 + np.random.randn(100) * 0.02)
    equity_curve = pd.DataFrame({"equity": equity}, index=dates)

    # Create sample trades
    trades = pd.DataFrame(
        {
            "timestamp": dates[:10],
            "asset": ["ASSET_A"] * 10,
            "action": ["buy", "sell"] * 5,
            "quantity": [100] * 10,
            "price": [50 + np.random.randn() for _ in range(10)],
            "realized_pnl": np.random.randn(10) * 1000,
        }
    )

    # Create portfolio history
    portfolio_history = pd.DataFrame(
        {
            "timestamp": dates,
            "cash": 100000 - np.cumsum(np.random.randn(100) * 1000),
            "equity": equity,
            "val_ASSET_A": np.random.rand(100) * 50000,
            "pos_ASSET_A": np.random.randint(-100, 100, 100),
        }
    ).set_index("timestamp")

    # Create metrics
    metrics = {
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.15,
        "total_return": 0.25,
        "win_rate": 0.60,
    }

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        portfolio_history=portfolio_history,
        metrics=metrics,
    )


@pytest.fixture
def sample_optimization_result() -> OptimizationResult:
    """
    Provide minimal OptimizationResult for testing.

    Creates OptimizationResult with sample weights and metrics.
    """
    assets = ["ASSET_A", "ASSET_B", "ASSET_C"]
    weights = pd.Series([0.4, 0.35, 0.25], index=assets)

    return OptimizationResult(
        weights=weights,
        expected_return=0.12,
        expected_risk=0.18,
        sharpe_ratio=0.67,
        method="mean_variance",
        constraints_satisfied=True,
        optimization_status="optimal",
    )


# =============================================================================
# Helper Functions
# =============================================================================


def assert_dataframe_equal(
    df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = False, rtol: float = 1e-5
):
    """
    Custom assertion for DataFrame comparison with tolerance.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        check_dtype: Whether to check dtypes match
        rtol: Relative tolerance for numeric comparisons
    """
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, rtol=rtol)


def assert_series_equal(
    s1: pd.Series, s2: pd.Series, check_dtype: bool = False, rtol: float = 1e-5
):
    """
    Custom assertion for Series comparison with tolerance.

    Args:
        s1: First Series
        s2: Second Series
        check_dtype: Whether to check dtypes match
        rtol: Relative tolerance for numeric comparisons
    """
    pd.testing.assert_series_equal(s1, s2, check_dtype=check_dtype, rtol=rtol)


def create_mock_api_response(data: Dict, format: str = "json") -> Dict:
    """
    Helper to create mock API responses.

    Args:
        data: Data to include in response
        format: Response format ('json' or 'csv')

    Returns:
        Mock response dictionary
    """
    if format == "json":
        return {
            "response": {
                "data": data,
                "total": len(data) if isinstance(data, list) else 1,
            }
        }
    else:
        return data
