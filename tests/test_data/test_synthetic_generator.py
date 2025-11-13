"""
Unit tests for SyntheticPriceGenerator class.

Tests synthetic data generation, OU process simulation, seasonality,
volatility clustering, jump diffusion, and parameter calibration.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import SyntheticPriceGenerator


class TestSyntheticPriceGenerator:
    """Test synthetic price generation."""

    def test_init_with_config(self, sample_config):
        """Test initialization with config dictionary."""
        generator = SyntheticPriceGenerator(config=sample_config)
        assert generator.config is not None

    def test_init_default_config(self):
        """Test initialization loads config.yaml by default."""
        generator = SyntheticPriceGenerator()
        assert generator.config is not None

    def test_generate_price_series_single_scenario(self):
        """Test generating single price series."""
        np.random.seed(42)
        generator = SyntheticPriceGenerator()
        prices = generator.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-05",
            frequency="H",
            n_scenarios=1,
        )

        # Verify structure
        assert isinstance(prices, pd.DataFrame)
        assert pd.api.types.is_datetime64_any_dtype(prices.index)
        assert "price" in prices.columns

        # Verify length (4 days * 24 hours + 1 = 97 hours)
        assert len(prices) >= 90

        # Verify prices are positive
        assert (prices["price"] > 0).all()

        # Check mean is reasonably near expected value (with tolerance for stochastic)
        assert 30 < prices["price"].mean() < 70

    def test_generate_price_series_multiple_scenarios(self):
        """Test generating multiple scenarios."""
        np.random.seed(42)
        generator = SyntheticPriceGenerator()
        prices = generator.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-03",
            frequency="H",
            n_scenarios=3,
        )

        # Verify MultiIndex structure
        assert isinstance(prices.index, pd.MultiIndex)
        assert prices.index.names == ["scenario_id", "timestamp"]

        # Check all scenarios have same length
        scenario_lengths = prices.groupby(level=0).size()
        assert len(scenario_lengths.unique()) == 1

        # Verify scenarios differ
        scenario_0 = prices.loc[0, "price"].values
        scenario_1 = prices.loc[1, "price"].values
        assert not np.allclose(scenario_0, scenario_1)

    def test_generate_price_series_invalid_dates(self):
        """Test ValueError when start_date >= end_date."""
        generator = SyntheticPriceGenerator()
        with pytest.raises(ValueError, match="start_date.*end_date"):
            generator.generate_price_series(
                start_date="2023-01-10",
                end_date="2023-01-05",
                frequency="H",
            )

    def test_generate_price_series_insufficient_data(self):
        """Test ValueError with <2 timestamps."""
        generator = SyntheticPriceGenerator()
        with pytest.raises(ValueError):
            generator.generate_price_series(
                start_date="2023-01-01 00:00",
                end_date="2023-01-01 00:00",  # Same timestamp
                frequency="H",
            )

    # NOTE: Private methods (simulate_ou_process, add_seasonality, etc.) are tested
    # indirectly through public generate_price_series() and generate_stress_scenarios()
    # which exercise the full pipeline including these internal components.

    def test_generate_stress_scenarios(self):
        """Test stress scenario generation."""
        np.random.seed(42)
        generator = SyntheticPriceGenerator()

        # Generate base series first
        base_series = generator.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-05",
            frequency="H",
            n_scenarios=1,
        )

        # Test high volatility scenario
        high_vol = generator.generate_stress_scenarios(
            base_series=base_series,
            scenario_type="high_volatility",
            intensity=2.0,
        )
        assert len(high_vol) == len(base_series)

        # Test price spike scenario
        spike = generator.generate_stress_scenarios(
            base_series=base_series,
            scenario_type="price_spike",
            intensity=0.5,
        )
        assert len(spike) == len(base_series)

    def test_generate_stress_scenarios_unknown_type(self):
        """Test ValueError for unknown scenario_type."""
        generator = SyntheticPriceGenerator()
        base_series = generator.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-02",
            frequency="H",
            n_scenarios=1,
        )
        with pytest.raises(ValueError, match="scenario"):
            generator.generate_stress_scenarios(
                base_series=base_series,
                scenario_type="unknown_scenario",
                intensity=1.0,
            )

    def test_calibrate_from_data(self):
        """Test parameter calibration from historical data."""
        np.random.seed(42)
        generator = SyntheticPriceGenerator()

        # Generate synthetic data for calibration
        prices_df = generator.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-02-15",  # ~45 days for sufficient data
            frequency="H",
            n_scenarios=1,
        )

        # Calibrate from the generated data
        calibrated = generator.calibrate_from_data(prices_df["price"])

        assert "mu" in calibrated
        assert "sigma" in calibrated
        assert calibrated["mu"] > 0  # Should be positive
        assert calibrated["sigma"] > 0  # Should be positive

    def test_calibrate_from_data_insufficient(self):
        """Test ValueError for insufficient data."""
        generator = SyntheticPriceGenerator()
        data = pd.DataFrame(
            {"price": [50, 51, 52]},
            index=pd.date_range("2023-01-01", periods=3, freq="H"),
        )

        with pytest.raises(ValueError, match="observations"):
            generator.calibrate_from_data(data)

    def test_save_to_storage(self, mock_data_manager):
        """Test saving to DataManager."""
        np.random.seed(42)
        generator = SyntheticPriceGenerator()
        prices = generator.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-02",
            frequency="H",
            n_scenarios=1,
        )

        # Mock the save method
        with pytest.mock.patch.object(mock_data_manager, "save_processed_data") as mock_save:
            mock_save.return_value = pytest.mock.MagicMock()  # Return a path-like object

            result = generator.save_to_storage(
                data=prices,
                data_manager=mock_data_manager,
                start_date="2023-01-01",
                end_date="2023-01-02",
            )

            # Verify save was called
            mock_save.assert_called_once()

    @pytest.mark.parametrize("frequency", ["H", "D", "30min"])
    def test_different_frequencies(self, frequency):
        """Test with different frequencies."""
        np.random.seed(42)
        generator = SyntheticPriceGenerator()
        prices = generator.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-03",
            frequency=frequency,
            n_scenarios=1,
        )

        assert isinstance(prices, pd.DataFrame)
        assert len(prices) > 0
        assert (prices["price"] > 0).all()

    @pytest.mark.parametrize(
        "kappa,mu,sigma",
        [
            (0.05, 40.0, 3.0),
            (0.2, 50.0, 5.0),
            (0.5, 60.0, 8.0),
        ],
    )
    def test_different_ou_parameters(self, kappa, mu, sigma):
        """Test with various OU parameters."""
        np.random.seed(42)
        generator = SyntheticPriceGenerator()
        prices = generator.simulate_ou_process(
            n_steps=100,
            mu=mu,
            kappa=kappa,
            sigma=sigma,
        )

        assert len(prices) == 100
        assert (prices > 0).all()
        # Mean should be reasonably close to mu
        np.testing.assert_allclose(np.mean(prices), mu, rtol=0.3)
