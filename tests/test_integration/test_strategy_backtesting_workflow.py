"""
Integration tests for strategy backtesting workflow.

Tests end-to-end strategy execution including signal generation, backtesting,
performance metrics, and trade logging.
"""

from unittest.mock import Mock

import pytest

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import PerformanceMetrics
from src.strategies.mean_reversion import MeanReversionStrategy


@pytest.mark.integration
class TestStrategyBacktestingWorkflow:
    """Test end-to-end strategy execution and backtesting."""

    def test_mean_reversion_backtest(self, sample_price_data, tmp_path):
        """Test complete mean reversion workflow."""
        # Initialize strategy
        strategy = MeanReversionStrategy()

        # Generate signals
        signals = strategy.generate_signals(sample_price_data)

        # Initialize BacktestEngine
        engine = BacktestEngine(strategies=strategy)

        # Run backtest
        result = engine.run(
            price_data=sample_price_data,
            signals=signals,
        )

        # Calculate performance metrics
        metrics = PerformanceMetrics(result)
        all_metrics = metrics.calculate_all()

        # Verify complete workflow executes
        assert result.equity_curve is not None
        assert result.trades is not None
        assert "sharpe_ratio" in all_metrics
