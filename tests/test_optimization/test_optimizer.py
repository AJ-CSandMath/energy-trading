"""
Unit tests for portfolio optimization classes.

Tests mean-variance, risk parity, Black-Litterman, minimum CVaR optimization,
and the main PortfolioOptimizer orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
)


class TestPortfolioOptimizer:
    """Test main orchestrator."""

    def test_init_with_dataframe(self):
        """Test initialization with returns DataFrame."""
        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "ASSET_A": np.random.randn(100) * 0.02,
                "ASSET_B": np.random.randn(100) * 0.03,
                "ASSET_C": np.random.randn(100) * 0.025,
            }
        )

        optimizer = PortfolioOptimizer(returns=returns)
        assert optimizer.returns is not None
        assert len(optimizer.returns.columns) == 3

    def test_init_with_backtest_result(self, sample_backtest_result):
        """Test initialization with BacktestResult."""
        optimizer = PortfolioOptimizer(returns=sample_backtest_result)

        # Verify returns extracted
        assert optimizer.returns is not None

    def test_optimize_mean_variance(self):
        """Test mean-variance method."""
        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "ASSET_A": np.random.randn(100) * 0.02,
                "ASSET_B": np.random.randn(100) * 0.03,
            }
        )

        optimizer = PortfolioOptimizer(returns=returns)
        result = optimizer.optimize(method="mean_variance", objective="max_sharpe")

        # Verify OptimizationResult returned
        assert isinstance(result, OptimizationResult)
        assert result.weights is not None

        # Verify weights sum to 1.0
        weight_sum = result.weights.sum()
        np.testing.assert_almost_equal(weight_sum, 1.0, decimal=5)

        # Check weights respect constraints
        assert (result.weights >= 0).all()  # Long-only
        assert (result.weights <= 1.0).all()
