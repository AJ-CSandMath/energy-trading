"""
Unit tests for WindGenerator and SolarGenerator classes.

Tests wind and solar generation profiles, power curves, ramp rates,
capacity factors, and intermittency metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.renewable_generator import SolarGenerator, WindGenerator


class TestWindGenerator:
    """Test wind generation."""

    def test_init(self, sample_config):
        """Test initialization with config."""
        generator = WindGenerator(config=sample_config)
        assert generator.config is not None

    def test_generate_wind_profile(self):
        """Test wind profile generation."""
        np.random.seed(42)
        generator = WindGenerator()
        wind_data = generator.generate_wind_profile(
            start_date="2023-01-01",
            end_date="2023-01-03",
            n_scenarios=1,
            capacity_mw=100.0,
        )

        # Verify DataFrame structure
        assert isinstance(wind_data, pd.DataFrame)
        assert "generation_mw" in wind_data.columns
        assert "capacity_factor" in wind_data.columns
        assert "wind_speed_mps" in wind_data.columns

        # Check capacity factor in reasonable range
        cf_mean = wind_data["capacity_factor"].mean()
        assert 0.1 < cf_mean < 0.6

        # Verify generation respects capacity
        assert (wind_data["generation_mw"] >= 0).all()
        assert (wind_data["generation_mw"] <= 100.0 * 1.01).all()  # Small tolerance

    def test_generate_wind_profile_multiple_scenarios(self):
        """Test multiple scenarios."""
        np.random.seed(42)
        generator = WindGenerator()
        wind_data = generator.generate_wind_profile(
            start_date="2023-01-01",
            end_date="2023-01-02",
            n_scenarios=3,
            capacity_mw=100.0,
        )

        # Verify MultiIndex structure
        assert isinstance(wind_data.index, pd.MultiIndex)
        assert wind_data.index.names == ["scenario_id", "timestamp"]

    def test_generate_wind_speeds(self):
        """Test wind speed generation."""
        np.random.seed(42)
        generator = WindGenerator()
        wind_speeds = generator.generate_wind_speeds(
            n_steps=168,  # 1 week
            weibull_k=2.0,
            weibull_lambda=7.0,
        )

        # Verify Weibull distribution characteristics
        assert len(wind_speeds) == 168
        assert (wind_speeds >= 0).all()
        assert 4 < np.mean(wind_speeds) < 10  # Typical mean for Weibull(2, 7)

        # Check temporal autocorrelation (AR(1) with phi=0.85)
        autocorr = np.corrcoef(wind_speeds[:-1], wind_speeds[1:])[0, 1]
        assert autocorr > 0.5  # Strong positive autocorrelation

    def test_apply_power_curve(self):
        """Test power curve application."""
        generator = WindGenerator()
        capacity_mw = 100.0

        # Test cut-in speed (v < 3 m/s)
        power_low = generator.apply_power_curve(np.array([1.0, 2.0]), capacity_mw)
        assert (power_low == 0).all()

        # Test rated speed (v >= 12 m/s)
        power_high = generator.apply_power_curve(np.array([12.0, 15.0]), capacity_mw)
        np.testing.assert_allclose(power_high, capacity_mw, rtol=0.01)

        # Test cut-out speed (v >= 25 m/s)
        power_cutout = generator.apply_power_curve(np.array([25.0, 30.0]), capacity_mw)
        assert (power_cutout == 0).all()

        # Test cubic relationship in between
        power_mid = generator.apply_power_curve(np.array([6.0]), capacity_mw)
        assert 0 < power_mid[0] < capacity_mw

    def test_apply_ramp_rate_limits(self):
        """Test ramp rate constraints."""
        generator = WindGenerator()
        capacity_mw = 100.0

        # Create series with excessive ramps
        generation = np.array([10.0, 90.0, 10.0, 90.0])  # 80 MW jumps

        # Apply limits
        limited = generator.apply_ramp_rate_limits(
            generation,
            capacity_mw=capacity_mw,
            max_ramp_rate_pct=0.2,  # 20% per period = 20 MW
        )

        # Verify ramps are clamped
        ramps = np.abs(np.diff(limited))
        assert (ramps <= 20.0 * 1.01).all()  # Allow small tolerance

    def test_generate_low_wind_scenario(self):
        """Test stress scenario."""
        np.random.seed(42)
        generator = WindGenerator()

        normal_profile = generator.generate_wind_profile(
            start_date="2023-01-01",
            end_date="2023-01-02",
            n_scenarios=1,
            capacity_mw=100.0,
        )

        low_wind_profile = generator.generate_low_wind_scenario(
            base_generation=normal_profile["generation_mw"],
            reduction_factor=0.5,
            duration_hours=12,
        )

        # Verify generation reduced
        assert low_wind_profile.mean() < normal_profile["generation_mw"].mean()

    def test_calculate_capacity_factor(self):
        """Test CF calculation."""
        generator = WindGenerator()
        generation = pd.Series([30, 40, 50, 60])  # MW
        capacity_mw = 100.0

        cf = generator.calculate_capacity_factor(generation, capacity_mw)

        # Verify CF = total_energy / (capacity * hours)
        expected_cf = generation.mean() / capacity_mw
        np.testing.assert_almost_equal(cf, expected_cf)

    def test_calculate_intermittency_metrics(self):
        """Test variability metrics."""
        np.random.seed(42)
        generator = WindGenerator()
        generation = pd.Series(np.random.rand(168) * 100)

        metrics = generator.calculate_intermittency_metrics(generation)

        # Verify metrics calculated
        assert "cv" in metrics  # Coefficient of variation
        assert "ramp_rate_mean" in metrics
        assert "ramp_rate_std" in metrics
        assert "autocorr_1h" in metrics

        assert metrics["cv"] > 0
        assert -1 <= metrics["autocorr_1h"] <= 1

    @pytest.mark.parametrize("capacity_mw", [50, 100, 200])
    def test_wind_different_capacities(self, capacity_mw):
        """Test with different capacities."""
        np.random.seed(42)
        generator = WindGenerator()
        wind_data = generator.generate_wind_profile(
            start_date="2023-01-01",
            end_date="2023-01-02",
            n_scenarios=1,
            capacity_mw=capacity_mw,
        )

        # Verify generation scales with capacity
        assert (wind_data["generation_mw"] <= capacity_mw * 1.01).all()


class TestSolarGenerator:
    """Test solar generation."""

    def test_init(self, sample_config):
        """Test initialization."""
        generator = SolarGenerator(config=sample_config)
        assert generator.config is not None

    def test_generate_solar_profile(self):
        """Test solar profile generation."""
        np.random.seed(42)
        generator = SolarGenerator()
        solar_data = generator.generate_solar_profile(
            start_date="2023-06-01",  # Summer for higher generation
            end_date="2023-06-03",
            n_scenarios=1,
            capacity_mw=100.0,
        )

        # Verify DataFrame structure
        assert isinstance(solar_data, pd.DataFrame)
        assert "generation_mw" in solar_data.columns
        assert "capacity_factor" in solar_data.columns
        assert "irradiance_w_m2" in solar_data.columns

        # Check zero generation at night
        night_mask = (solar_data.index.hour < 6) | (solar_data.index.hour > 20)
        assert (solar_data.loc[night_mask, "generation_mw"] == 0).all()

        # Verify peak generation during day
        day_mask = (solar_data.index.hour >= 10) & (solar_data.index.hour <= 14)
        day_generation = solar_data.loc[day_mask, "generation_mw"]
        assert day_generation.max() > 50.0  # Significant generation at peak

        # Check capacity factor in range
        cf_mean = solar_data["capacity_factor"].mean()
        assert 0.1 < cf_mean < 0.4

    def test_calculate_solar_position(self):
        """Test solar position calculation."""
        generator = SolarGenerator(latitude=35.0)  # Mid-latitude
        timestamps = pd.date_range("2023-06-21", periods=24, freq="H")  # Summer solstice

        elevation_angles = generator.calculate_solar_position(timestamps)

        # Verify elevation angles reasonable
        assert len(elevation_angles) == 24
        assert (elevation_angles >= 0).all()
        assert (elevation_angles <= 90).all()

        # Check zero elevation at night
        night_hours = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]
        assert (elevation_angles[night_hours] == 0).all()

        # Verify peak elevation at solar noon
        noon_idx = 12
        assert elevation_angles[noon_idx] == elevation_angles.max()

    def test_generate_irradiance(self):
        """Test irradiance generation."""
        np.random.seed(42)
        generator = SolarGenerator()
        timestamps = pd.date_range("2023-06-01", periods=168, freq="H")
        elevation_angles = generator.calculate_solar_position(timestamps)

        irradiance = generator.generate_irradiance(
            elevation_angles=elevation_angles,
            cloud_cover_mean=0.3,
        )

        # Verify clear-sky irradiance formula
        assert len(irradiance) == 168
        assert (irradiance >= 0).all()

        # Check cloud cover effects
        day_mask = elevation_angles > 0
        assert irradiance[day_mask].mean() < 1000  # Reduced by clouds

    def test_apply_pv_conversion(self):
        """Test PV conversion."""
        generator = SolarGenerator()
        irradiance = np.array([0, 500, 1000])
        capacity_mw = 100.0

        power = generator.apply_pv_conversion(
            irradiance=irradiance,
            capacity_mw=capacity_mw,
            panel_efficiency=0.2,
            inverter_efficiency=0.96,
        )

        # Verify DC power calculation
        assert len(power) == 3
        assert power[0] == 0  # No irradiance
        assert power[1] < power[2]  # Higher irradiance = more power
        assert (power <= capacity_mw).all()

    def test_generate_cloudy_scenario(self):
        """Test cloudy stress scenario."""
        np.random.seed(42)
        generator = SolarGenerator()

        normal_profile = generator.generate_solar_profile(
            start_date="2023-06-01",
            end_date="2023-06-02",
            n_scenarios=1,
            capacity_mw=100.0,
        )

        cloudy_profile = generator.generate_cloudy_scenario(
            base_generation=normal_profile["generation_mw"],
            cloud_cover_level=0.8,
            duration_hours=8,
        )

        # Verify generation reduced during cloudy period
        assert cloudy_profile.mean() < normal_profile["generation_mw"].mean()

    def test_calculate_capacity_factor(self):
        """Test CF calculation."""
        generator = SolarGenerator()
        generation = pd.Series([0, 30, 50, 30, 0])  # Typical daily pattern
        capacity_mw = 100.0

        cf = generator.calculate_capacity_factor(generation, capacity_mw)

        expected_cf = generation.mean() / capacity_mw
        np.testing.assert_almost_equal(cf, expected_cf)

    def test_calculate_intermittency_metrics(self):
        """Test variability metrics."""
        np.random.seed(42)
        generator = SolarGenerator()
        generation = pd.Series(np.random.rand(168) * 100)

        metrics = generator.calculate_intermittency_metrics(generation)

        assert "cv" in metrics
        assert "ramp_rate_mean" in metrics
        assert metrics["cv"] > 0

    @pytest.mark.parametrize("latitude", [25, 35, 45])
    def test_solar_different_latitudes(self, latitude):
        """Test with different latitudes."""
        np.random.seed(42)
        generator = SolarGenerator(latitude=latitude)
        solar_data = generator.generate_solar_profile(
            start_date="2023-06-01",
            end_date="2023-06-02",
            n_scenarios=1,
            capacity_mw=100.0,
        )

        # Verify generation varies with latitude
        assert len(solar_data) > 0
        assert (solar_data["generation_mw"] >= 0).all()
