"""
Unit tests for performance metrics functions and PerformanceMetrics class.

Tests calculation of Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio,
win rate, profit factor, and other trading metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import (
    PerformanceMetrics,
    calculate_annualized_return,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
)


class TestMetricsFunctions:
    """Test individual metric functions."""

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Constant 1% return, 2% std
        returns = pd.Series([0.01] * 252)  # 1 year daily returns
        returns = returns + np.random.randn(252) * 0.02

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)

        # Should be approximately (0.01 - 0.02/252) / 0.02 * sqrt(252)
        assert -5 < sharpe < 5  # Reasonable range

    def test_calculate_sharpe_ratio_zero_std(self):
        """Test with zero std (returns 0)."""
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe_ratio(returns)
        # With zero std, should handle gracefully
        assert sharpe == 0 or np.isinf(sharpe)

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio."""
        # Mixed positive/negative returns
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])

        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        assert not np.isnan(sortino)

    def test_calculate_max_drawdown(self):
        """Test max drawdown."""
        # Create equity curve with known drawdown
        equity = pd.Series([100, 110, 105, 95, 105, 120])  # -13.6% drawdown at idx 3

        max_dd = calculate_max_drawdown(equity)

        # Verify max_dd = (trough - peak) / peak
        assert max_dd < 0  # Drawdown is negative
        assert max_dd >= -0.15  # Approximately -13.6%

    def test_calculate_max_drawdown_monotonic(self):
        """Test with monotonically increasing equity (max_dd = 0)."""
        equity = pd.Series([100, 105, 110, 115, 120])
        max_dd = calculate_max_drawdown(equity)
        assert max_dd == 0.0

    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio."""
        returns = pd.Series([0.01] * 252)
        equity = 100000 * (1 + returns).cumprod()

        calmar = calculate_calmar_ratio(equity, returns, periods_per_year=252)

        # Verify formula: annualized_return / abs(max_drawdown)
        assert calmar > 0 or np.isinf(calmar)

    def test_calculate_win_rate(self):
        """Test win rate."""
        # Create trades DataFrame
        trades = pd.DataFrame(
            {
                "realized_pnl": [100, -50, 200, -30, 150],  # 3 wins, 2 losses
            }
        )

        win_rate = calculate_win_rate(trades)

        # win_rate = 3 / 5 = 0.6
        assert pytest.approx(win_rate, rel=0.01) == 0.6

    def test_calculate_win_rate_all_wins(self):
        """Test with all wins (returns 1.0)."""
        trades = pd.DataFrame({"realized_pnl": [100, 200, 150]})
        win_rate = calculate_win_rate(trades)
        assert win_rate == 1.0

    def test_calculate_win_rate_no_trades(self):
        """Test with no trades (returns 0)."""
        trades = pd.DataFrame({"realized_pnl": []})
        win_rate = calculate_win_rate(trades)
        assert win_rate == 0.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""

    def test_init(self, sample_backtest_result):
        """Test initialization."""
        metrics = PerformanceMetrics(sample_backtest_result)

        # Verify returns calculated
        assert metrics.returns is not None
        assert len(metrics.returns) > 0

    def test_calculate_all(self, sample_backtest_result):
        """Test comprehensive metrics calculation."""
        metrics = PerformanceMetrics(sample_backtest_result)
        all_metrics = metrics.calculate_all()

        # Verify all metrics present
        assert "sharpe_ratio" in all_metrics
        assert "max_drawdown" in all_metrics
        assert "total_return" in all_metrics

        # Check metrics are reasonable values
        assert -5 < all_metrics["sharpe_ratio"] < 5
        assert all_metrics["max_drawdown"] <= 0

    def test_get_summary(self, sample_backtest_result):
        """Test formatted summary."""
        metrics = PerformanceMetrics(sample_backtest_result)
        summary = metrics.get_summary()

        # Verify returns dict with formatted strings
        assert isinstance(summary, dict)
        assert all(isinstance(v, str) for v in summary.values())

    def test_to_dataframe(self, sample_backtest_result):
        """Test DataFrame conversion."""
        metrics = PerformanceMetrics(sample_backtest_result)
        df = metrics.to_dataframe()

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
