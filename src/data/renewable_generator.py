"""
Renewable energy generation profile synthesis module.

This module provides WindGenerator and SolarGenerator classes for generating
realistic synthetic renewable energy generation profiles with:
- Weather-driven patterns (Weibull for wind, Beta for solar)
- Temporal autocorrelation and intermittency
- Ramp rate constraints
- Capacity factor modeling
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Get logger (no basicConfig - central config handles logging)
logger = logging.getLogger(__name__)


class WindGenerator:
    """
    Generator for synthetic wind power generation profiles.

    Simulates realistic wind generation using Weibull wind speed distribution,
    turbine power curves, temporal correlation, and operational constraints.

    Features:
    - Weibull-distributed wind speeds with AR(1) temporal correlation
    - Realistic turbine power curve (cut-in, rated, cut-out)
    - Diurnal and seasonal patterns
    - Intermittency (outages, maintenance)
    - Ramp rate constraints

    Example:
        >>> generator = WindGenerator()
        >>> profile = generator.generate_wind_profile(
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31',
        ...     frequency='H'
        ... )
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize wind generation profile generator.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml.
        """
        # Load config if not provided
        if config is None:
            from src.config.load_config import get_config
            config = get_config()

        # Get wind generation parameters
        wind_config = config.get("synthetic", {}).get("wind", {})

        # Turbine specifications
        self.capacity_mw = wind_config.get("capacity_mw", 100.0)
        self.hub_height = wind_config.get("hub_height", 80)
        self.turbine_type = wind_config.get("turbine_type", "generic_2mw")

        # Capacity factor statistics
        self.capacity_factor_mean = wind_config.get("capacity_factor_mean", 0.35)
        self.capacity_factor_std = wind_config.get("capacity_factor_std", 0.05)

        # Power curve parameters
        self.cut_in_speed = wind_config.get("cut_in_speed", 3.0)
        self.rated_speed = wind_config.get("rated_speed", 12.0)
        self.cut_out_speed = wind_config.get("cut_out_speed", 25.0)

        # Variability parameters
        self.max_ramp_rate = wind_config.get("max_ramp_rate", 0.15)
        self.autocorrelation = wind_config.get("autocorrelation", 0.85)

        # Weibull distribution for wind speeds
        self.weibull_shape = wind_config.get("weibull_shape", 2.0)
        self.weibull_scale = wind_config.get("weibull_scale", 7.5)

        # Intermittency
        self.outage_rate = wind_config.get("outage_rate", 0.01)
        self.maintenance_fraction = wind_config.get("maintenance_fraction", 0.015)

        # Random seed
        self.random_seed = wind_config.get("random_seed", 43)
        self.rng = np.random.default_rng(self.random_seed)

        logger.info(
            f"WindGenerator initialized: capacity={self.capacity_mw}MW, "
            f"CF_mean={self.capacity_factor_mean:.2f}, cut_in={self.cut_in_speed}m/s"
        )

    def generate_wind_profile(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'H',
        n_scenarios: int = 1
    ) -> pd.DataFrame:
        """
        Generate wind generation time series.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Time frequency ('H' for hourly, 'D' for daily)
            n_scenarios: Number of scenarios to generate

        Returns:
            DataFrame with columns ['generation_mw', 'capacity_factor', 'wind_speed_mps', 'scenario_id']

        Raises:
            ValueError: If dates are invalid
        """
        # Validate dates
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        if start_dt >= end_dt:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")

        logger.info(
            f"Generating wind profile: start={start_date}, end={end_date}, "
            f"freq={frequency}, scenarios={n_scenarios}"
        )

        # Create timestamp index with inclusive end
        # For hourly data over a full year, we want exactly 8760 hours (non-leap) or 8784 (leap)
        if frequency == 'H':
            # Ensure end_date includes the last hour of the day
            end_dt_inclusive = end_dt + pd.Timedelta(hours=23)
            timestamps = pd.date_range(start=start_dt, end=end_dt_inclusive, freq=frequency)
        else:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq=frequency, inclusive='both')

        n_steps = len(timestamps)

        # Validate expected length for full-year hourly data
        if frequency == 'H':
            expected_hours = int((end_dt - start_dt).total_seconds() / 3600) + 24  # +24 for final day
            if abs(n_steps - expected_hours) > 1:  # Allow 1-hour tolerance
                logger.warning(
                    f"Hourly timestamp count {n_steps} differs from expected {expected_hours}. "
                    f"Date range: {start_date} to {end_date}"
                )

        # Calculate time step in hours
        if n_steps < 2:
            raise ValueError("Need at least 2 timestamps to infer time step")

        dt_hours = (timestamps[1] - timestamps[0]).total_seconds() / 3600.0

        # Generate scenarios
        all_scenarios = []

        for scenario_idx in range(n_scenarios):
            # Step 1: Generate wind speeds
            wind_speeds = self._generate_wind_speeds(timestamps, n_steps)

            # Step 2: Apply power curve
            power_output = self._apply_power_curve(
                wind_speeds=wind_speeds,
                capacity_mw=self.capacity_mw,
                cut_in=self.cut_in_speed,
                rated=self.rated_speed,
                cut_out=self.cut_out_speed
            )

            # Step 3: Add intermittency
            power_output = self._add_intermittency(power_output, n_steps, dt_hours)

            # Step 4: Apply ramp rate limits
            power_output = self._apply_ramp_rate_limits(
                power_series=power_output,
                max_ramp_rate_per_min=self.max_ramp_rate,
                dt_hours=dt_hours
            )

            # Calculate capacity factor
            capacity_factor = power_output / self.capacity_mw

            # Validate capacity factor is within expected range
            mean_cf = capacity_factor.mean()
            if mean_cf < 0.05:
                logger.warning(
                    f"Wind capacity factor ({mean_cf:.2%}) is unrealistically low. "
                    f"Expected range: 15-50%. Check Weibull parameters, power curve, and intermittency settings."
                )
            elif mean_cf > 0.70:
                logger.warning(
                    f"Wind capacity factor ({mean_cf:.2%}) is unrealistically high. "
                    f"Expected range: 15-50%."
                )

            all_scenarios.append({
                'generation_mw': power_output,
                'capacity_factor': capacity_factor,
                'wind_speed_mps': wind_speeds,
                'scenario_id': scenario_idx
            })

        # Create DataFrame
        if n_scenarios == 1:
            df = pd.DataFrame({
                'generation_mw': all_scenarios[0]['generation_mw'],
                'capacity_factor': all_scenarios[0]['capacity_factor'],
                'wind_speed_mps': all_scenarios[0]['wind_speed_mps']
            }, index=timestamps)
        else:
            # Multiple scenarios - use MultiIndex to avoid duplicate timestamps
            dfs = []
            for scenario in all_scenarios:
                scenario_df = pd.DataFrame({
                    'generation_mw': scenario['generation_mw'],
                    'capacity_factor': scenario['capacity_factor'],
                    'wind_speed_mps': scenario['wind_speed_mps'],
                    'scenario_id': scenario['scenario_id']
                }, index=timestamps)
                scenario_df = scenario_df.set_index('scenario_id', append=True)
                scenario_df = scenario_df.reorder_levels(['scenario_id', scenario_df.index.names[0]])
                dfs.append(scenario_df)
            df = pd.concat(dfs)

        logger.info(
            f"Generated wind profile: {len(df)} observations, "
            f"mean CF={df['capacity_factor'].mean():.2%}"
        )

        return df

    def _generate_wind_speeds(
        self,
        timestamps: pd.DatetimeIndex,
        n_steps: int
    ) -> np.ndarray:
        """
        Generate synthetic wind speed time series using Weibull distribution with temporal correlation.

        Args:
            timestamps: DatetimeIndex
            n_steps: Number of time steps

        Returns:
            Array of wind speeds in m/s
        """
        # Generate base Weibull-distributed samples
        base_speeds = self.rng.weibull(self.weibull_shape, n_steps) * self.weibull_scale

        # Add temporal autocorrelation using AR(1) process
        # v_t = φ*v_{t-1} + sqrt(1-φ²)*ε_t
        phi = self.autocorrelation
        correlated_speeds = np.zeros(n_steps)
        correlated_speeds[0] = base_speeds[0]

        for t in range(1, n_steps):
            innovation = self.rng.normal(0, 1)
            # AR(1) with stochastic component and base pattern preservation
            correlated_speeds[t] = (
                phi * correlated_speeds[t-1] +
                np.sqrt(1 - phi**2) * (base_speeds[t] - base_speeds.mean()) +
                base_speeds.mean() +
                0.1 * innovation  # Add small stochastic component for proper AR(1) dynamics
            )

        # Ensure non-negative
        correlated_speeds = np.maximum(correlated_speeds, 0)

        # Add diurnal pattern (wind typically stronger at night/morning)
        hours = timestamps.hour
        diurnal_factor = 1.0 + 0.2 * np.cos(2 * np.pi * (hours - 3) / 24)
        correlated_speeds *= diurnal_factor

        # Add seasonal variation (stronger in winter/spring)
        month = timestamps.month
        seasonal_factor = 1.0 + 0.15 * np.cos(2 * np.pi * (month - 1) / 12)
        correlated_speeds *= seasonal_factor

        return correlated_speeds

    def _apply_power_curve(
        self,
        wind_speeds: np.ndarray,
        capacity_mw: float,
        cut_in: float,
        rated: float,
        cut_out: float
    ) -> np.ndarray:
        """
        Convert wind speeds to power output using turbine power curve.

        Piecewise function:
        - v < cut_in: P = 0
        - cut_in ≤ v < rated: P = capacity * ((v - cut_in)/(rated - cut_in))³
        - rated ≤ v < cut_out: P = capacity
        - v ≥ cut_out: P = 0

        Args:
            wind_speeds: Array of wind speeds in m/s
            capacity_mw: Turbine capacity in MW
            cut_in: Cut-in wind speed
            rated: Rated wind speed
            cut_out: Cut-out wind speed

        Returns:
            Power output in MW
        """
        power = np.zeros_like(wind_speeds)

        # Below cut-in: no generation
        mask_below_cut_in = wind_speeds < cut_in

        # Between cut-in and rated: cubic power curve
        mask_ramp = (wind_speeds >= cut_in) & (wind_speeds < rated)
        power[mask_ramp] = capacity_mw * (
            ((wind_speeds[mask_ramp] - cut_in) / (rated - cut_in)) ** 3
        )

        # Between rated and cut-out: full power
        mask_rated = (wind_speeds >= rated) & (wind_speeds < cut_out)
        power[mask_rated] = capacity_mw

        # Above cut-out: shutdown for safety
        mask_cut_out = wind_speeds >= cut_out

        return power

    def _add_intermittency(
        self,
        power_output: np.ndarray,
        n_steps: int,
        dt_hours: float
    ) -> np.ndarray:
        """
        Add realistic intermittency and variability.

        Includes:
        - Random outages (Poisson process)
        - Scheduled maintenance periods
        - Performance degradation (blade icing, soiling)

        Args:
            power_output: Base power output array
            n_steps: Number of time steps
            dt_hours: Time step duration in hours

        Returns:
            Adjusted power output
        """
        adjusted_power = power_output.copy()

        # Random outages using Poisson process
        # Outage rate is per day, convert to per time step
        outage_prob = self.outage_rate * dt_hours / 24.0
        outages = self.rng.random(n_steps) < outage_prob
        adjusted_power[outages] = 0

        # Scheduled maintenance (continuous periods)
        total_maintenance_hours = int(n_steps * self.maintenance_fraction)
        if total_maintenance_hours > 0:
            maintenance_start = self.rng.integers(0, n_steps - total_maintenance_hours)
            adjusted_power[maintenance_start:maintenance_start + total_maintenance_hours] = 0

        # Performance degradation (random factor 0.95-1.0)
        degradation_factor = self.rng.uniform(0.95, 1.0, n_steps)
        adjusted_power *= degradation_factor

        return adjusted_power

    def _apply_ramp_rate_limits(
        self,
        power_series: np.ndarray,
        max_ramp_rate_per_min: float,
        dt_hours: float
    ) -> np.ndarray:
        """
        Enforce realistic ramp rate constraints.

        Args:
            power_series: Power series in MW
            max_ramp_rate_per_min: Maximum ramp rate as fraction of capacity per minute
            dt_hours: Time step duration in hours

        Returns:
            Ramp-limited power series
        """
        limited_power = power_series.copy()

        # Convert per-minute rate to per-time-step limit
        max_ramp = max_ramp_rate_per_min * 60 * dt_hours * self.capacity_mw

        for t in range(1, len(power_series)):
            delta = limited_power[t] - limited_power[t-1]

            if abs(delta) > max_ramp:
                # Limit the change
                if delta > 0:
                    limited_power[t] = limited_power[t-1] + max_ramp
                else:
                    limited_power[t] = limited_power[t-1] - max_ramp

        return limited_power

    def generate_low_wind_scenario(
        self,
        base_profile: pd.DataFrame,
        start_time: str,
        duration_hours: int,
        reduction_factor: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate stress test scenario with extended low wind period.

        Args:
            base_profile: Base wind generation profile
            start_time: Start time for low wind period (YYYY-MM-DD HH:MM:SS)
            duration_hours: Duration of low wind period in hours
            reduction_factor: Factor to reduce generation (0.5 = 50% reduction)

        Returns:
            Stressed generation profile
        """
        stressed = base_profile.copy()

        # Find the index range for the low wind period
        start_dt = pd.to_datetime(start_time)
        end_dt = start_dt + pd.Timedelta(hours=duration_hours)

        mask = (stressed.index >= start_dt) & (stressed.index < end_dt)

        # Reduce generation during this period
        stressed.loc[mask, 'generation_mw'] *= reduction_factor
        stressed.loc[mask, 'capacity_factor'] *= reduction_factor
        stressed.loc[mask, 'wind_speed_mps'] *= np.sqrt(reduction_factor)

        logger.info(
            f"Generated low wind scenario: {duration_hours}h starting {start_time}, "
            f"reduction={reduction_factor:.1%}"
        )

        return stressed

    def calculate_capacity_factor(
        self,
        generation_profile: pd.DataFrame
    ) -> float:
        """
        Calculate capacity factor from generation profile.

        Args:
            generation_profile: DataFrame with 'generation_mw' column

        Returns:
            Capacity factor (0 to 1)
        """
        total_energy = generation_profile['generation_mw'].sum()
        hours = len(generation_profile)
        capacity_factor = total_energy / (self.capacity_mw * hours)

        return capacity_factor

    def calculate_intermittency_metrics(
        self,
        generation_profile: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate variability metrics for generation profile.

        Returns:
            Dictionary with metrics:
            - coefficient_of_variation
            - mean_ramp_rate
            - max_ramp_rate
            - ramp_rate_95th_percentile
            - autocorr_1h, autocorr_6h, autocorr_24h
        """
        gen = generation_profile['generation_mw'].values

        # Coefficient of variation
        cv = gen.std() / gen.mean() if gen.mean() > 0 else 0

        # Ramp rates
        ramps = np.abs(np.diff(gen))
        mean_ramp = ramps.mean()
        max_ramp = ramps.max()
        ramp_95th = np.percentile(ramps, 95)

        # Autocorrelations
        gen_series = pd.Series(gen)
        autocorr_1h = gen_series.autocorr(lag=1)
        autocorr_6h = gen_series.autocorr(lag=6)
        autocorr_24h = gen_series.autocorr(lag=24) if len(gen) >= 24 else np.nan

        return {
            'coefficient_of_variation': cv,
            'mean_ramp_rate': mean_ramp,
            'max_ramp_rate': max_ramp,
            'ramp_rate_95th_percentile': ramp_95th,
            'autocorr_1h': autocorr_1h,
            'autocorr_6h': autocorr_6h,
            'autocorr_24h': autocorr_24h
        }

    def save_to_storage(
        self,
        data: pd.DataFrame,
        data_manager,
        start_date: str,
        end_date: str
    ) -> Optional[Path]:
        """
        Save generated wind profile to DataManager.

        Args:
            data: DataFrame with wind generation data
            data_manager: DataManager instance
            start_date: Start date string
            end_date: End date string

        Returns:
            Path to saved file/directory, or None if save failed
        """
        try:
            path = data_manager.save_processed_data(
                data=data,
                source='synthetic',
                dataset='wind',
                start_date=start_date,
                end_date=end_date
            )

            logger.info(f"Saved wind profile to {path}")
            return path

        except Exception as e:
            logger.error(f"Failed to save wind profile: {str(e)}")
            return None


class SolarGenerator:
    """
    Generator for synthetic solar power generation profiles.

    Simulates realistic solar generation using solar position calculations,
    clear-sky irradiance models, cloud cover, and PV system characteristics.

    Features:
    - Simplified solar position algorithm
    - Clear-sky and cloudy irradiance modeling
    - Beta-distributed cloud cover with temporal correlation
    - Temperature derating
    - Inverter efficiency
    - Soiling and degradation

    Example:
        >>> generator = SolarGenerator()
        >>> profile = generator.generate_solar_profile(
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31',
        ...     frequency='H'
        ... )
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize solar generation profile generator.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml.
        """
        # Load config if not provided
        if config is None:
            from src.config.load_config import get_config
            config = get_config()

        # Get solar generation parameters
        solar_config = config.get("synthetic", {}).get("solar", {})

        # System configuration
        self.capacity_mw = solar_config.get("capacity_mw", 100.0)
        self.tilt_angle = solar_config.get("tilt_angle", 30)
        self.azimuth = solar_config.get("azimuth", 180)
        self.latitude = solar_config.get("latitude", 35.0)
        self.longitude = solar_config.get("longitude", -120.0)

        # Capacity factor statistics
        self.capacity_factor_mean = solar_config.get("capacity_factor_mean", 0.25)
        self.capacity_factor_std = solar_config.get("capacity_factor_std", 0.03)

        # PV system parameters
        self.module_efficiency = solar_config.get("module_efficiency", 0.18)
        self.inverter_efficiency = solar_config.get("inverter_efficiency", 0.97)
        self.temperature_coeff = solar_config.get("temperature_coeff", -0.004)

        # Cloud cover modeling
        self.cloud_alpha = solar_config.get("cloud_alpha", 2.0)
        self.cloud_beta = solar_config.get("cloud_beta", 5.0)
        self.cloud_persistence = solar_config.get("cloud_persistence", 0.90)

        # Degradation
        self.annual_degradation = solar_config.get("annual_degradation", 0.005)

        # Random seed
        self.random_seed = solar_config.get("random_seed", 44)
        self.rng = np.random.default_rng(self.random_seed)

        logger.info(
            f"SolarGenerator initialized: capacity={self.capacity_mw}MW, "
            f"CF_mean={self.capacity_factor_mean:.2f}, lat={self.latitude}, lon={self.longitude}"
        )

    def generate_solar_profile(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'H',
        n_scenarios: int = 1
    ) -> pd.DataFrame:
        """
        Generate solar generation time series.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Time frequency ('H' for hourly)
            n_scenarios: Number of scenarios to generate

        Returns:
            DataFrame with columns ['generation_mw', 'capacity_factor', 'irradiance_w_m2', 'scenario_id']

        Raises:
            ValueError: If dates are invalid
        """
        # Validate dates
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        if start_dt >= end_dt:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")

        logger.info(
            f"Generating solar profile: start={start_date}, end={end_date}, "
            f"freq={frequency}, scenarios={n_scenarios}"
        )

        # Create timestamp index with inclusive end
        # For hourly data over a full year, we want exactly 8760 hours (non-leap) or 8784 (leap)
        if frequency == 'H':
            # Ensure end_date includes the last hour of the day
            end_dt_inclusive = end_dt + pd.Timedelta(hours=23)
            timestamps = pd.date_range(start=start_dt, end=end_dt_inclusive, freq=frequency)
        else:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq=frequency, inclusive='both')

        n_steps = len(timestamps)

        # Validate expected length for full-year hourly data
        if frequency == 'H':
            expected_hours = int((end_dt - start_dt).total_seconds() / 3600) + 24  # +24 for final day
            if abs(n_steps - expected_hours) > 1:  # Allow 1-hour tolerance
                logger.warning(
                    f"Hourly timestamp count {n_steps} differs from expected {expected_hours}. "
                    f"Date range: {start_date} to {end_date}"
                )

        # Generate scenarios
        all_scenarios = []

        for scenario_idx in range(n_scenarios):
            # Step 1: Calculate solar position
            elevation_angles = self._calculate_solar_position(timestamps)

            # Step 2: Generate irradiance
            irradiance = self._generate_irradiance(
                timestamps=timestamps,
                elevation_angles=elevation_angles,
                n_steps=n_steps
            )

            # Step 3: Apply PV conversion
            power_output = self._apply_pv_conversion(
                irradiance=irradiance,
                capacity_mw=self.capacity_mw,
                module_efficiency=self.module_efficiency,
                inverter_efficiency=self.inverter_efficiency
            )

            # Step 4: Add weather variability
            power_output = self._add_weather_variability(
                power_output=power_output,
                timestamps=timestamps
            )

            # Calculate capacity factor
            capacity_factor = power_output / self.capacity_mw

            all_scenarios.append({
                'generation_mw': power_output,
                'capacity_factor': capacity_factor,
                'irradiance_w_m2': irradiance,
                'scenario_id': scenario_idx
            })

        # Create DataFrame
        if n_scenarios == 1:
            df = pd.DataFrame({
                'generation_mw': all_scenarios[0]['generation_mw'],
                'capacity_factor': all_scenarios[0]['capacity_factor'],
                'irradiance_w_m2': all_scenarios[0]['irradiance_w_m2']
            }, index=timestamps)
        else:
            # Multiple scenarios - use MultiIndex to avoid duplicate timestamps
            dfs = []
            for scenario in all_scenarios:
                scenario_df = pd.DataFrame({
                    'generation_mw': scenario['generation_mw'],
                    'capacity_factor': scenario['capacity_factor'],
                    'irradiance_w_m2': scenario['irradiance_w_m2'],
                    'scenario_id': scenario['scenario_id']
                }, index=timestamps)
                scenario_df = scenario_df.set_index('scenario_id', append=True)
                scenario_df = scenario_df.reorder_levels(['scenario_id', scenario_df.index.names[0]])
                dfs.append(scenario_df)
            df = pd.concat(dfs)

        logger.info(
            f"Generated solar profile: {len(df)} observations, "
            f"mean CF={df['capacity_factor'].mean():.2%}"
        )

        return df

    def _calculate_solar_position(
        self,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Calculate sun elevation angles using simplified solar position algorithm.

        Args:
            timestamps: DatetimeIndex

        Returns:
            Array of elevation angles in degrees
        """
        # Convert to day of year and hour
        day_of_year = timestamps.dayofyear
        hour = timestamps.hour + timestamps.minute / 60.0

        # Solar declination (simplified)
        declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year + 284)))

        # Hour angle
        solar_noon = 12.0  # Simplified - assumes local solar noon at 12:00
        hour_angle = 15.0 * (hour - solar_noon)

        # Elevation angle
        lat_rad = np.radians(self.latitude)
        dec_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)

        sin_elevation = (
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
        )

        elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))

        # Set negative elevations to zero (sun below horizon)
        elevation = np.maximum(elevation, 0)

        return elevation

    def _generate_irradiance(
        self,
        timestamps: pd.DatetimeIndex,
        elevation_angles: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Generate solar irradiance time series with cloud cover.

        Args:
            timestamps: DatetimeIndex
            elevation_angles: Sun elevation angles
            n_steps: Number of time steps

        Returns:
            Irradiance array in W/m²
        """
        # Clear-sky irradiance (simplified model)
        # I_clear = I_0 * sin(elevation)
        I_0 = 1000.0  # W/m² at sea level
        clear_sky_irradiance = I_0 * np.sin(np.radians(elevation_angles))
        clear_sky_irradiance = np.maximum(clear_sky_irradiance, 0)

        # Generate cloud cover using Beta distribution with temporal correlation
        # Beta(2,5) gives right-skewed distribution (mostly clear)
        base_cloud_cover = self.rng.beta(self.cloud_alpha, self.cloud_beta, n_steps)

        # Add temporal correlation using AR(1)
        phi = self.cloud_persistence
        cloud_cover = np.zeros(n_steps)
        cloud_cover[0] = base_cloud_cover[0]

        for t in range(1, n_steps):
            innovation = self.rng.normal(0, 0.1)
            cloud_cover[t] = phi * cloud_cover[t-1] + (1 - phi) * base_cloud_cover[t] + innovation
            cloud_cover[t] = np.clip(cloud_cover[t], 0, 1)

        # Apply cloud cover to reduce irradiance
        irradiance = clear_sky_irradiance * (1 - cloud_cover * 0.8)  # Clouds reduce by up to 80%

        return irradiance

    def _apply_pv_conversion(
        self,
        irradiance: np.ndarray,
        capacity_mw: float,
        module_efficiency: float,
        inverter_efficiency: float
    ) -> np.ndarray:
        """
        Convert irradiance to AC power output.

        Args:
            irradiance: Irradiance in W/m²
            capacity_mw: System capacity in MW
            module_efficiency: Module efficiency (0-1)
            inverter_efficiency: Inverter efficiency (0-1)

        Returns:
            AC power output in MW
        """
        # DC power: P_dc = capacity * (irradiance / 1000) * efficiency
        power_dc = capacity_mw * (irradiance / 1000.0) * module_efficiency

        # Temperature derating (simplified - assume 5°C above 25°C on average)
        temp_derating = 1.0 + self.temperature_coeff * 5.0
        power_dc *= temp_derating

        # AC power: P_ac = P_dc * inverter_efficiency
        power_ac = power_dc * inverter_efficiency

        # Ensure non-negative and within capacity
        power_ac = np.clip(power_ac, 0, capacity_mw)

        return power_ac

    def _add_weather_variability(
        self,
        power_output: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Add realistic weather-driven variability.

        Includes:
        - Passing clouds (brief dips)
        - Seasonal patterns
        - Soiling/degradation

        Args:
            power_output: Base power output
            timestamps: DatetimeIndex

        Returns:
            Adjusted power output
        """
        # Ensure power_output is a mutable NumPy array before mutation
        adjusted_power = np.asarray(power_output, dtype=float).copy()

        # Passing clouds - random brief dips (only during daylight hours)
        daylight_mask = adjusted_power > 0.01
        n_daylight = daylight_mask.sum()

        if n_daylight > 0:
            # 2% of daylight hours have passing clouds
            n_cloud_events = int(n_daylight * 0.02)
            if n_cloud_events > 0:
                cloud_indices = self.rng.choice(np.where(daylight_mask)[0], size=n_cloud_events, replace=False)
                # Ensure cloud_indices is an integer ndarray before indexing
                cloud_indices = np.asarray(cloud_indices, dtype=int)
                cloud_reduction = self.rng.uniform(0.3, 0.7, n_cloud_events)
                adjusted_power[cloud_indices] *= cloud_reduction

        # Soiling/degradation - gradual reduction over time
        # 0.5-1% per year
        days_elapsed = (timestamps - timestamps[0]).days
        degradation_factor = 1.0 - (self.annual_degradation * days_elapsed / 365.0)
        degradation_factor = np.maximum(degradation_factor, 0.90)  # Max 10% degradation
        adjusted_power *= degradation_factor

        return adjusted_power

    def generate_cloudy_scenario(
        self,
        base_profile: pd.DataFrame,
        start_time: str,
        duration_hours: int,
        cloud_cover_level: float = 0.7
    ) -> pd.DataFrame:
        """
        Generate stress test scenario with extended cloudy period.

        Args:
            base_profile: Base solar generation profile
            start_time: Start time for cloudy period
            duration_hours: Duration in hours
            cloud_cover_level: Cloud cover level (0-1, where 1 = completely overcast)

        Returns:
            Stressed generation profile
        """
        stressed = base_profile.copy()

        # Find the index range
        start_dt = pd.to_datetime(start_time)
        end_dt = start_dt + pd.Timedelta(hours=duration_hours)

        mask = (stressed.index >= start_dt) & (stressed.index < end_dt)

        # Reduce generation based on cloud cover
        reduction_factor = 1.0 - cloud_cover_level * 0.8
        stressed.loc[mask, 'generation_mw'] *= reduction_factor
        stressed.loc[mask, 'capacity_factor'] *= reduction_factor
        stressed.loc[mask, 'irradiance_w_m2'] *= reduction_factor

        logger.info(
            f"Generated cloudy scenario: {duration_hours}h starting {start_time}, "
            f"cloud_cover={cloud_cover_level:.1%}"
        )

        return stressed

    def calculate_capacity_factor(
        self,
        generation_profile: pd.DataFrame
    ) -> float:
        """
        Calculate capacity factor from generation profile.

        Args:
            generation_profile: DataFrame with 'generation_mw' column

        Returns:
            Capacity factor (0 to 1)
        """
        total_energy = generation_profile['generation_mw'].sum()
        hours = len(generation_profile)
        capacity_factor = total_energy / (self.capacity_mw * hours)

        return capacity_factor

    def calculate_intermittency_metrics(
        self,
        generation_profile: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate variability metrics for generation profile.

        Returns:
            Dictionary with metrics
        """
        gen = generation_profile['generation_mw'].values

        # Coefficient of variation
        cv = gen.std() / gen.mean() if gen.mean() > 0 else 0

        # Ramp rates
        ramps = np.abs(np.diff(gen))
        mean_ramp = ramps.mean()
        max_ramp = ramps.max()
        ramp_95th = np.percentile(ramps, 95)

        # Autocorrelations
        gen_series = pd.Series(gen)
        autocorr_1h = gen_series.autocorr(lag=1)
        autocorr_6h = gen_series.autocorr(lag=6)
        autocorr_24h = gen_series.autocorr(lag=24) if len(gen) >= 24 else np.nan

        return {
            'coefficient_of_variation': cv,
            'mean_ramp_rate': mean_ramp,
            'max_ramp_rate': max_ramp,
            'ramp_rate_95th_percentile': ramp_95th,
            'autocorr_1h': autocorr_1h,
            'autocorr_6h': autocorr_6h,
            'autocorr_24h': autocorr_24h
        }

    def save_to_storage(
        self,
        data: pd.DataFrame,
        data_manager,
        start_date: str,
        end_date: str
    ) -> Optional[Path]:
        """
        Save generated solar profile to DataManager.

        Args:
            data: DataFrame with solar generation data
            data_manager: DataManager instance
            start_date: Start date string
            end_date: End date string

        Returns:
            Path to saved file/directory, or None if save failed
        """
        try:
            path = data_manager.save_processed_data(
                data=data,
                source='synthetic',
                dataset='solar',
                start_date=start_date,
                end_date=end_date
            )

            logger.info(f"Saved solar profile to {path}")
            return path

        except Exception as e:
            logger.error(f"Failed to save solar profile: {str(e)}")
            return None


if __name__ == "__main__":
    # Setup logging
    from src.config.load_config import setup_logging
    setup_logging()

    print("Renewable Generation Profiles Example")
    print("=" * 50)

    # Wind generation example
    print("\n1. Wind Generation Profile")
    print("-" * 50)

    wind_gen = WindGenerator()
    wind_profile = wind_gen.generate_wind_profile(
        start_date='2024-01-01',
        end_date='2024-01-31',
        frequency='H',
        n_scenarios=1
    )

    print(f"Generated {len(wind_profile)} hourly observations")
    print(f"\nWind generation statistics:")
    print(f"  Mean: {wind_profile['generation_mw'].mean():.2f} MW")
    print(f"  Max:  {wind_profile['generation_mw'].max():.2f} MW")
    print(f"  CF:   {wind_profile['capacity_factor'].mean():.2%}")

    # Calculate metrics
    wind_metrics = wind_gen.calculate_intermittency_metrics(wind_profile)
    print(f"\nIntermittency metrics:")
    for key, value in wind_metrics.items():
        print(f"  {key}: {value:.3f}")

    # Solar generation example
    print("\n2. Solar Generation Profile")
    print("-" * 50)

    solar_gen = SolarGenerator()
    solar_profile = solar_gen.generate_solar_profile(
        start_date='2024-06-01',
        end_date='2024-06-30',
        frequency='H',
        n_scenarios=1
    )

    print(f"Generated {len(solar_profile)} hourly observations")
    print(f"\nSolar generation statistics:")
    print(f"  Mean: {solar_profile['generation_mw'].mean():.2f} MW")
    print(f"  Max:  {solar_profile['generation_mw'].max():.2f} MW")
    print(f"  CF:   {solar_profile['capacity_factor'].mean():.2%}")

    # Calculate metrics
    solar_metrics = solar_gen.calculate_intermittency_metrics(solar_profile)
    print(f"\nIntermittency metrics:")
    for key, value in solar_metrics.items():
        print(f"  {key}: {value:.3f}")

    # Stress scenarios
    print("\n3. Stress Scenarios")
    print("-" * 50)

    # Low wind scenario
    low_wind = wind_gen.generate_low_wind_scenario(
        base_profile=wind_profile,
        start_time='2024-01-15 00:00:00',
        duration_hours=72,
        reduction_factor=0.3
    )

    print(f"Low wind scenario - mean generation: {low_wind['generation_mw'].mean():.2f} MW")

    # Cloudy scenario
    cloudy = solar_gen.generate_cloudy_scenario(
        base_profile=solar_profile,
        start_time='2024-06-15 00:00:00',
        duration_hours=48,
        cloud_cover_level=0.8
    )

    print(f"Cloudy scenario - mean generation: {cloudy['generation_mw'].mean():.2f} MW")
