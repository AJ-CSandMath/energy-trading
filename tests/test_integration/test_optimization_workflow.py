"""
Integration tests for portfolio optimization workflow.

Tests end-to-end optimization including optimization from backtest results,
risk analytics integration, and method comparison.
"""

import pytest

from src.optimization.optimizer import PortfolioOptimizer
from src.optimization.risk_analytics import RiskAnalytics


@pytest.mark.integration
class TestOptimizationWorkflow:
    """Test end-to-end optimization."""

    def test_portfolio_optimization_from_backtest(self, sample_backtest_result, tmp_path):
        """Test optimization from backtest results."""
        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns=sample_backtest_result)

        # Run mean-variance optimization
        result = optimizer.optimize(method="mean_variance", objective="max_sharpe")

        # Verify OptimizationResult returned
        assert result.weights is not None
        assert abs(result.weights.sum() - 1.0) < 0.01  # Weights sum to 1

    def test_risk_analytics_integration(self, sample_backtest_result):
        """Test risk analytics integration."""
        # Initialize RiskAnalytics
        analytics = RiskAnalytics(sample_backtest_result)

        # Calculate all risk metrics
        var_dict = analytics.calculate_portfolio_var()
        cvar_dict = analytics.calculate_portfolio_cvar()

        # Verify metrics calculated
        assert var_dict["var_pct"] > 0
        assert cvar_dict["cvar_pct"] >= var_dict["var_pct"]
