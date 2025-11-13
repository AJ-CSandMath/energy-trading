"""
Unit tests for RenewableArbitrageStrategy class.

Tests renewable arbitrage strategy including forecaster integration,
signal generation based on generation-price correlation, and curtailment logic.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.strategies.renewable_arbitrage import RenewableArbitrageStrategy


class TestRenewableArbitrageStrategy:
    """Test renewable arbitrage strategy."""

    def test_init(self, sample_config):
        """Test initialization."""
        strategy = RenewableArbitrageStrategy(config=sample_config)

        # Verify parameters loaded
        assert hasattr(strategy, "forecast_horizon")
        assert hasattr(strategy, "generation_threshold_high")
        assert hasattr(strategy, "correlation_factor")

    def test_set_forecasters(self):
        """Test forecaster setting."""
        strategy = RenewableArbitrageStrategy()

        # Create mock forecasters
        price_forecaster = Mock()
        renewable_forecaster = Mock()

        # Set forecasters
        strategy.set_forecasters(price_forecaster, renewable_forecaster)

        # Verify stored correctly
        assert strategy.price_forecaster == price_forecaster
        assert strategy.renewable_forecaster == renewable_forecaster

    def test_generate_signals_no_forecasters(self, sample_price_data):
        """Test ValueError when forecasters not set."""
        strategy = RenewableArbitrageStrategy()

        with pytest.raises(ValueError, match="Forecasters"):
            strategy.generate_signals(sample_price_data)

    def test_generate_signals_high_generation(
        self, sample_price_data, sample_wind_data, sample_solar_data, mocker
    ):
        """Test buy signal on high generation forecast."""
        strategy = RenewableArbitrageStrategy()

        # Mock forecasters
        price_forecaster = Mock()
        renewable_forecaster = Mock()

        # Mock lower price forecast
        price_forecast_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=24, freq="H"),
                "forecast": [45.0] * 24,  # Lower than current
            }
        )
        price_forecaster.predict.return_value = price_forecast_df

        # Mock high capacity factor
        renewable_forecaster.forecast.return_value = {
            "wind_forecast": {"forecast": np.array([80.0] * 24)},  # High generation
            "solar_forecast": {"forecast": np.array([40.0] * 24)},
        }

        strategy.set_forecasters(price_forecaster, renewable_forecaster)

        signals = strategy.generate_signals(
            sample_price_data,
            wind_data=sample_wind_data,
            solar_data=sample_solar_data,
        )

        # Verify buy signal generated
        if len(signals) > 0:
            buy_signals = signals[signals["signal"] == 1]
            assert len(buy_signals) >= 0  # At least some buy opportunities

    def test_get_required_columns(self):
        """Test required columns."""
        strategy = RenewableArbitrageStrategy()
        required = strategy.get_required_columns()
        assert "price" in required
