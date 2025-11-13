"""
Unit tests for BaseStrategy abstract class.

Tests base strategy functionality including position sizing,
risk management, position tracking, and state persistence.
"""

import pandas as pd
import pytest

from src.strategies.base_strategy import BaseStrategy


# Create concrete test strategy for testing
class TestConcreteStrategy(BaseStrategy):
    """Concrete strategy for testing BaseStrategy."""

    def generate_signals(self, data, **kwargs):
        """Minimal implementation."""
        return pd.DataFrame(
            {
                "timestamp": data.index[:5],
                "asset": ["TEST"] * 5,
                "signal": [1, -1, 0, 1, -1],
                "strength": [0.8, 0.6, 0.0, 0.9, 0.7],
            }
        )

    def get_required_columns(self):
        """Minimal implementation."""
        return ["price"]


class TestBaseStrategy:
    """Test base strategy functionality."""

    def test_init(self, sample_config):
        """Test initialization."""
        strategy = TestConcreteStrategy(name="test", config=sample_config)

        # Verify name, config set
        assert strategy.name == "test"
        assert strategy.config is not None

        # Check risk parameters loaded
        assert hasattr(strategy, "max_position_size")
        assert hasattr(strategy, "stop_loss_pct")

        # Verify position tracking initialized
        assert isinstance(strategy.current_positions, dict)
        assert isinstance(strategy.entry_prices, dict)

    def test_calculate_position_size_fixed_fractional(self):
        """Test fixed fractional sizing."""
        strategy = TestConcreteStrategy(name="test")
        strategy.max_position_size = 0.2

        position_size = strategy.calculate_position_size(
            signal_strength=1.0,
            current_price=50.0,
            account_value=100000.0,
            sizing_method="fixed_fractional",
        )

        # position_size = (account_value * max_position_size * strength) / price
        expected_size = (100000 * 0.2 * 1.0) / 50.0
        assert pytest.approx(position_size, rel=0.01) == expected_size

    def test_calculate_position_size_different_strengths(self):
        """Test with different signal strengths."""
        strategy = TestConcreteStrategy(name="test")
        strategy.max_position_size = 0.2

        size_full = strategy.calculate_position_size(
            signal_strength=1.0,
            current_price=50.0,
            account_value=100000.0,
        )

        size_half = strategy.calculate_position_size(
            signal_strength=0.5,
            current_price=50.0,
            account_value=100000.0,
        )

        # Half strength should give half position size
        assert pytest.approx(size_half, rel=0.01) == size_full * 0.5

    def test_update_positions(self):
        """Test position tracking."""
        strategy = TestConcreteStrategy(name="test")

        # Update position with buy
        strategy.update_positions(asset="ASSET_A", quantity=100, price=50.0)

        # Verify current_positions updated
        assert strategy.current_positions["ASSET_A"] == 100

        # Verify entry_prices updated
        assert strategy.entry_prices["ASSET_A"] == 50.0

        # Add to position
        strategy.update_positions(asset="ASSET_A", quantity=50, price=52.0)

        # Verify weighted average entry price
        expected_entry = (100 * 50.0 + 50 * 52.0) / 150
        assert pytest.approx(strategy.entry_prices["ASSET_A"], rel=0.01) == expected_entry

    def test_get_strategy_state(self):
        """Test state serialization."""
        strategy = TestConcreteStrategy(name="test")
        strategy.update_positions(asset="ASSET_A", quantity=100, price=50.0)
        strategy.update_positions(asset="ASSET_B", quantity=200, price=75.0)

        state = strategy.get_strategy_state()

        # Verify all data included
        assert "current_positions" in state
        assert "entry_prices" in state
        assert state["current_positions"]["ASSET_A"] == 100
        assert state["entry_prices"]["ASSET_A"] == 50.0

    def test_load_strategy_state(self):
        """Test state deserialization."""
        strategy = TestConcreteStrategy(name="test")

        state = {
            "current_positions": {"ASSET_A": 100, "ASSET_B": 200},
            "entry_prices": {"ASSET_A": 50.0, "ASSET_B": 75.0},
        }

        strategy.load_strategy_state(state)

        # Verify positions restored
        assert strategy.current_positions["ASSET_A"] == 100
        assert strategy.entry_prices["ASSET_A"] == 50.0

    def test_validate_data_valid(self, sample_price_data):
        """Test validation passes."""
        strategy = TestConcreteStrategy(name="test")
        validated = strategy.validate_data(sample_price_data)
        assert len(validated) == len(sample_price_data)

    def test_validate_data_no_datetime_index(self):
        """Test ValueError for invalid index."""
        strategy = TestConcreteStrategy(name="test")
        df = pd.DataFrame({"price": [50, 51, 52]})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            strategy.validate_data(df)

    def test_validate_data_missing_columns(self, sample_price_data):
        """Test ValueError for missing required columns."""
        strategy = TestConcreteStrategy(name="test")
        df = sample_price_data.drop(columns=["price"])

        with pytest.raises(ValueError, match="column"):
            strategy.validate_data(df)
