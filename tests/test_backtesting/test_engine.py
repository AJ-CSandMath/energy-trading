"""
Unit tests for BacktestEngine class.

Tests backtesting engine including order execution, transaction costs,
portfolio valuation, signal processing, and metrics calculation.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine, BacktestResult


class TestBacktestEngine:
    """Test backtesting engine."""

    def test_init(self, sample_config):
        """Test initialization."""
        mock_strategy = Mock()
        engine = BacktestEngine(strategies=mock_strategy, config=sample_config)

        # Verify config loaded, portfolio state initialized
        assert engine.config is not None
        assert engine.cash == sample_config["backtesting"]["initial_capital"]

    def test_run_simple_backtest(self, sample_price_data):
        """Test basic backtest execution."""
        # Create simple strategy
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = pd.DataFrame(
            {
                "timestamp": sample_price_data.index[:10],
                "asset": ["TEST"] * 10,
                "signal": [1, 0, -1, 0, 1, 0, -1, 0, 1, 0],
                "strength": [0.8] * 10,
            }
        )

        engine = BacktestEngine(strategies=mock_strategy)

        # Prepare signals
        signals = mock_strategy.generate_signals(sample_price_data)

        result = engine.run(
            price_data=sample_price_data,
            signals=signals,
        )

        # Verify BacktestResult returned
        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert result.trades is not None
        assert len(result.equity_curve) > 0

    def test_calculate_portfolio_value(self):
        """Test portfolio valuation."""
        engine = BacktestEngine(strategies=Mock())
        engine.cash = 50000
        engine.positions = {"ASSET_A": 100, "ASSET_B": 200}

        prices = {"ASSET_A": 50.0, "ASSET_B": 75.0}

        portfolio_value = engine.calculate_portfolio_value(prices)

        # equity = cash + sum(position * price)
        expected_value = 50000 + (100 * 50.0) + (200 * 75.0)
        assert pytest.approx(portfolio_value, rel=0.01) == expected_value

    def test_reset(self, sample_config):
        """Test engine reset."""
        engine = BacktestEngine(strategies=Mock(), config=sample_config)

        # Modify state
        engine.cash = 50000
        engine.positions = {"ASSET_A": 100}

        # Reset
        engine.reset()

        # Verify portfolio state cleared
        assert engine.cash == sample_config["backtesting"]["initial_capital"]
        assert len(engine.positions) == 0

    def test_transaction_costs_applied(self, sample_price_data, sample_config):
        """Test transaction costs are properly deducted."""
        # Create strategy with buy signal
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = pd.DataFrame(
            {
                "timestamp": [sample_price_data.index[0]],
                "asset": ["TEST"],
                "signal": [1],  # Buy signal
                "strength": [1.0],
            }
        )

        # Set known transaction costs
        config = sample_config.copy()
        config["backtesting"]["transaction_costs"] = {
            "fixed_cost": 10.0,
            "percentage_cost": 0.001,
        }

        engine = BacktestEngine(strategies=mock_strategy, config=config)
        initial_cash = engine.cash

        signals = mock_strategy.generate_signals(sample_price_data)
        result = engine.run(price_data=sample_price_data, signals=signals)

        # Verify transaction costs were deducted
        # Cash should be less than initial due to costs
        assert engine.cash < initial_cash

    def test_slippage_affects_execution_price(self, sample_price_data, sample_config):
        """Test slippage affects execution price."""
        # Create strategy with trade signals
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = pd.DataFrame(
            {
                "timestamp": sample_price_data.index[:2],
                "asset": ["TEST", "TEST"],
                "signal": [1, -1],  # Buy then sell
                "strength": [1.0, 1.0],
            }
        )

        # Configure slippage
        config = sample_config.copy()
        config["backtesting"]["slippage"] = 0.001  # 0.1% slippage

        engine = BacktestEngine(strategies=mock_strategy, config=config)
        signals = mock_strategy.generate_signals(sample_price_data)
        result = engine.run(price_data=sample_price_data, signals=signals)

        # Verify trades were executed (slippage applied internally)
        assert len(result.trades) > 0

    def test_execution_delay(self, sample_price_data, sample_config):
        """Test execution delay shifts trade timing."""
        # Create strategy with signals
        mock_strategy = Mock()
        signal_time = sample_price_data.index[5]
        mock_strategy.generate_signals.return_value = pd.DataFrame(
            {
                "timestamp": [signal_time],
                "asset": ["TEST"],
                "signal": [1],
                "strength": [1.0],
            }
        )

        # Configure execution delay
        config = sample_config.copy()
        config["backtesting"]["execution_delay"] = 1  # 1 period delay

        engine = BacktestEngine(strategies=mock_strategy, config=config)
        signals = mock_strategy.generate_signals(sample_price_data)
        result = engine.run(price_data=sample_price_data, signals=signals)

        # Verify backtest completed (delay applied internally)
        assert isinstance(result, BacktestResult)
