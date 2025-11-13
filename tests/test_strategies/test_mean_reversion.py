"""
Unit tests for MeanReversionStrategy class.

Tests mean reversion strategy signal generation, Bollinger Bands calculation,
band position metrics, and volatility filtering.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.mean_reversion import MeanReversionStrategy


class TestMeanReversionStrategy:
    """Test mean reversion strategy."""

    def test_init(self, sample_config):
        """Test initialization."""
        strategy = MeanReversionStrategy(config=sample_config)

        # Verify parameters loaded
        assert hasattr(strategy, "window")
        assert hasattr(strategy, "num_std")
        assert hasattr(strategy, "entry_threshold")
        assert hasattr(strategy, "exit_threshold")

    def test_generate_signals_buy(self):
        """Test buy signal generation."""
        np.random.seed(42)
        strategy = MeanReversionStrategy()

        # Create price data with price near lower Bollinger Band
        prices = np.array([50] * 20 + [45, 44, 43, 42, 41])  # Drop below lower band
        df = pd.DataFrame(
            {"price": prices},
            index=pd.date_range("2023-01-01", periods=len(prices), freq="H"),
        )

        signals = strategy.generate_signals(df)

        # Verify buy signal generated
        buy_signals = signals[signals["signal"] == 1]
        assert len(buy_signals) > 0

        # Check signal strength calculated
        assert "strength" in signals.columns
        assert (signals["strength"] >= 0).all()
        assert (signals["strength"] <= 1).all()

    def test_generate_signals_sell(self):
        """Test sell signal generation."""
        np.random.seed(42)
        strategy = MeanReversionStrategy()

        # Create price data with price near upper Bollinger Band
        prices = np.array([50] * 20 + [55, 56, 57, 58, 59])  # Rise above upper band
        df = pd.DataFrame(
            {"price": prices},
            index=pd.date_range("2023-01-01", periods=len(prices), freq="H"),
        )

        signals = strategy.generate_signals(df)

        # Verify sell signal generated
        sell_signals = signals[signals["signal"] == -1]
        assert len(sell_signals) > 0

    def test_calculate_bands(self):
        """Test Bollinger Bands calculation."""
        strategy = MeanReversionStrategy(window=20, num_std=2.0)

        prices = pd.Series([50 + np.sin(i / 10) * 5 for i in range(100)])
        middle, upper, lower = strategy.calculate_bands(prices)

        # Verify middle band = rolling mean
        rolling_mean = prices.rolling(window=20).mean()
        pd.testing.assert_series_equal(middle, rolling_mean, check_names=False)

        # Verify upper = middle + num_std * std
        rolling_std = prices.rolling(window=20).std()
        expected_upper = rolling_mean + 2.0 * rolling_std
        pd.testing.assert_series_equal(upper, expected_upper, check_names=False)

    def test_get_required_columns(self):
        """Test required columns."""
        strategy = MeanReversionStrategy()
        required = strategy.get_required_columns()
        assert "price" in required

    @pytest.mark.parametrize("window", [10, 20, 50])
    def test_different_window_sizes(self, window):
        """Test with different window sizes."""
        strategy = MeanReversionStrategy(window=window)
        prices = pd.Series([50 + np.random.randn() for _ in range(100)])
        middle, upper, lower = strategy.calculate_bands(prices)

        # Verify bands calculated with correct window
        assert len(middle) == len(prices)

    @pytest.mark.parametrize("num_std", [1.5, 2.0, 2.5])
    def test_different_num_std(self, num_std):
        """Test with different num_std."""
        strategy = MeanReversionStrategy(num_std=num_std)
        prices = pd.Series([50] * 30 + [45] * 10)  # Price drop
        df = pd.DataFrame(
            {"price": prices},
            index=pd.date_range("2023-01-01", periods=len(prices), freq="H"),
        )

        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.DataFrame)
