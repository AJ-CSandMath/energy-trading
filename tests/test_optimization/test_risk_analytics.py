"""
Unit tests for RiskAnalytics class and risk calculation functions.

Tests VaR, CVaR, correlation matrices, risk decomposition, and stress testing.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.risk_analytics import (
    RiskAnalytics,
    calculate_correlation_matrix,
    calculate_cvar,
    calculate_var,
)


class TestRiskFunctions:
    """Test individual risk functions."""

    def test_calculate_var_historical(self):
        """Test historical VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.02)

        var_95 = calculate_var(returns, confidence_level=0.95, method='historical')

        # VaR = -percentile(returns, 5)
        expected_var = -np.percentile(returns, 5)
        assert pytest.approx(var_95, rel=0.01) == expected_var

    def test_calculate_cvar_historical(self):
        """Test historical CVaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.02)

        cvar_95 = calculate_cvar(returns, confidence_level=0.95, method='historical')
        var_95 = calculate_var(returns, confidence_level=0.95, method='historical')

        # CVaR >= VaR (always true)
        assert cvar_95 >= var_95

    def test_calculate_correlation_matrix(self):
        """Test correlation matrix."""
        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "ASSET_A": np.random.randn(100),
                "ASSET_B": np.random.randn(100),
                "ASSET_C": np.random.randn(100),
            }
        )

        corr_matrix = calculate_correlation_matrix(returns)

        # Verify symmetric matrix with 1.0 on diagonal
        assert corr_matrix.shape == (3, 3)
        np.testing.assert_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0])
        assert np.allclose(corr_matrix, corr_matrix.T)


class TestRiskAnalytics:
    """Test RiskAnalytics class."""

    def test_init(self, sample_backtest_result):
        """Test initialization."""
        analytics = RiskAnalytics(sample_backtest_result)

        # Verify returns extracted
        assert analytics.returns is not None
        assert len(analytics.returns) > 0

    def test_calculate_portfolio_var(self, sample_backtest_result):
        """Test portfolio VaR."""
        analytics = RiskAnalytics(sample_backtest_result)

        var_dict = analytics.calculate_portfolio_var(confidence_level=0.95)

        # Verify both percentage and dollar terms returned
        assert "var_pct" in var_dict
        assert "var_dollar" in var_dict
        assert var_dict["var_pct"] > 0

    def test_calculate_portfolio_cvar(self, sample_backtest_result):
        """Test portfolio CVaR."""
        analytics = RiskAnalytics(sample_backtest_result)

        cvar_dict = analytics.calculate_portfolio_cvar(confidence_level=0.95)
        var_dict = analytics.calculate_portfolio_var(confidence_level=0.95)

        # Verify CVaR >= VaR
        assert cvar_dict["cvar_pct"] >= var_dict["var_pct"]

    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_var_different_confidence_levels(self, sample_backtest_result, confidence_level):
        """Test VaR at different confidence levels."""
        analytics = RiskAnalytics(sample_backtest_result)
        var_dict = analytics.calculate_portfolio_var(confidence_level=confidence_level)

        assert var_dict["var_pct"] > 0
        assert var_dict["var_dollar"] > 0
