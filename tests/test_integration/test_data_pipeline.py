"""
Integration tests for end-to-end data pipeline.

Tests complete data workflow including generation, storage, loading,
validation, and cleanup.
"""

import pandas as pd
import pytest

from src.data.data_manager import DataManager
from src.data.renewable_generator import SolarGenerator, WindGenerator
from src.data.synthetic_generator import SyntheticPriceGenerator


@pytest.mark.integration
class TestDataPipeline:
    """Test complete data workflow."""

    def test_synthetic_data_generation_and_storage(self, tmp_path):
        """Test full synthetic data pipeline."""
        # Generate synthetic price data
        price_gen = SyntheticPriceGenerator()
        prices = price_gen.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-03",
            frequency="H",
            n_scenarios=1,
        )

        # Generate wind data
        wind_gen = WindGenerator()
        wind_data = wind_gen.generate_wind_profile(
            start_date="2023-01-01",
            end_date="2023-01-03",
            n_scenarios=1,
            capacity_mw=100.0,
        )

        # Generate solar data
        solar_gen = SolarGenerator()
        solar_data = solar_gen.generate_solar_profile(
            start_date="2023-01-01",
            end_date="2023-01-03",
            n_scenarios=1,
            capacity_mw=100.0,
        )

        # Save all data using DataManager
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        price_path = manager.save_raw_data(prices, source="synthetic", dataset="prices")
        wind_path = manager.save_raw_data(wind_data, source="synthetic", dataset="wind")
        solar_path = manager.save_raw_data(solar_data, source="synthetic", dataset="solar")

        # Load data back and verify integrity
        loaded_prices = manager.load_data(source="synthetic", dataset="prices", data_type="raw")
        loaded_wind = manager.load_data(source="synthetic", dataset="wind", data_type="raw")
        loaded_solar = manager.load_data(source="synthetic", dataset="solar", data_type="raw")

        # Verify data integrity
        assert len(loaded_prices) == len(prices)
        assert len(loaded_wind) == len(wind_data)
        assert len(loaded_solar) == len(solar_data)

    def test_data_validation_and_cleanup(self, tmp_path):
        """Test data quality workflow."""
        manager = DataManager(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed"),
        )

        # Generate data with intentional issues
        price_gen = SyntheticPriceGenerator()
        prices = price_gen.generate_price_series(
            start_date="2023-01-01",
            end_date="2023-01-02",
            frequency="H",
        )

        # Create duplicates
        duplicated_prices = pd.concat([prices, prices.head(5)])

        # Validate and cleanup
        cleaned_prices = manager.remove_duplicates(duplicated_prices)

        # Verify issues resolved
        assert len(cleaned_prices) == len(prices)
        assert not cleaned_prices.index.duplicated().any()
