"""
Synthetic price data generation module for energy trading simulations.

This module provides the SyntheticPriceGenerator class for generating realistic
synthetic energy price time series using stochastic processes including:
- Ornstein-Uhlenbeck mean-reverting process
- Multi-scale seasonality (daily, weekly, yearly)
- Volatility clustering (GARCH-like)
- Jump diffusion for price spikes
"""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Get logger (no basicConfig - central config handles logging)
logger = logging.getLogger(__name__)


class SyntheticPriceGenerator:
    """
    Generator for synthetic energy price time series with realistic market dynamics.

    Uses Ornstein-Uhlenbeck process for mean reversion, Fourier series for seasonality,
    GARCH-like volatility clustering, and compound Poisson process for jumps/spikes.

    Example:
        >>> from src.config.load_config import get_config
        >>> config = get_config()
        >>> gen = SyntheticPriceGenerator(config=config)
        >>> prices = gen.generate_price_series(
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31',
        ...     frequency='H'
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize synthetic price generator.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml.
        """
        # Load config if not provided
        if config is None:
            from src.config.load_config import get_config
            config = get_config()

        # Get synthetic price configuration
        price_config = config.get("synthetic", {}).get("price", {})

        # OU process parameters
        ou_params = price_config.get("ou_params", {})
        self.kappa = ou_params.get("kappa", 1.2)
        self.mu = ou_params.get("mu", 50.0)
        self.sigma = ou_params.get("sigma", 15.0)

        # Seasonality parameters
        seasonality = price_config.get("seasonality", {})
        self.seasonality_params = {
            "daily": seasonality.get("daily", {"a1": 5.0, "b1": -2.0}),
            "weekly": seasonality.get("weekly", {"a2": 3.0, "b2": 1.0}),
            "yearly": seasonality.get("yearly", {"a3": 10.0, "b3": -5.0})
        }

        # Volatility clustering parameters (GARCH)
        garch = price_config.get("volatility_clustering", {})
        self.garch_params = {
            "omega": garch.get("omega", 0.1),
            "alpha": garch.get("alpha", 0.15),
            "beta": garch.get("beta", 0.80)
        }

        # Jump diffusion parameters
        jumps = price_config.get("jumps", {})
        self.lambda_jumps = jumps.get("lambda", 0.02)
        self.jump_mu = jumps.get("jump_mu", 0.0)
        self.jump_sigma = jumps.get("jump_sigma", 20.0)

        # Default settings
        self.default_initial_price = price_config.get("default_initial_price", 50.0)
        self.random_seed = price_config.get("random_seed", 42)

        logger.info(
            f"SyntheticPriceGenerator initialized: kappa={self.kappa}, "
            f"mu={self.mu}, sigma={self.sigma}"
        )

    def generate_price_series(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'H',
        initial_price: Optional[float] = None,
        n_scenarios: int = 1
    ) -> pd.DataFrame:
        """
        Generate synthetic price time series with realistic market dynamics.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Time frequency (any pandas frequency string, e.g., 'H', 'D', '30min', 'W')
            initial_price: Starting price. If None, uses default from config.
            n_scenarios: Number of scenarios to generate (default 1)

        Returns:
            DataFrame with DatetimeIndex and columns ['price', 'scenario_id']
            If n_scenarios=1, only 'price' column is returned.

        Raises:
            ValueError: If start_date >= end_date or invalid parameters
        """
        # Validate dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if start_dt >= end_dt:
            raise ValueError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )

        logger.info(
            f"Generating price series: {start_date} to {end_date}, "
            f"freq={frequency}, scenarios={n_scenarios}"
        )

        # Create datetime index with inclusive end
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

        # Calculate time step in days from timestamp difference
        if n_steps < 2:
            raise ValueError("Need at least 2 timestamps to infer time step")

        dt_days = (timestamps[1] - timestamps[0]).total_seconds() / 86400.0

        # Set initial price
        x0 = initial_price if initial_price is not None else self.default_initial_price

        # Generate scenarios
        all_scenarios = []

        for scenario_id in range(n_scenarios):
            # Set seed for reproducibility
            seed = self.random_seed + scenario_id if self.random_seed else None

            # Generate base OU process
            ou_prices = self._simulate_ou_process(
                n_steps=n_steps,
                dt=dt_days,
                x0=x0,
                kappa=self.kappa,
                mu=self.mu,
                sigma=self.sigma,
                seed=seed
            )

            # Add seasonality
            seasonal_adj = self._add_seasonality(timestamps, self.seasonality_params)
            prices_with_seasonality = ou_prices + seasonal_adj

            # Clamp prices to avoid log of non-positive values
            min_price = 1e-6
            if np.any(prices_with_seasonality <= 0):
                n_clamped = np.sum(prices_with_seasonality <= 0)
                logger.warning(
                    f"Scenario {scenario_id}: Clamped {n_clamped} non-positive prices "
                    f"(min={prices_with_seasonality.min():.6f}) to {min_price} before log"
                )
                prices_with_seasonality = np.maximum(prices_with_seasonality, min_price)

            # Add volatility clustering
            returns = np.diff(np.log(prices_with_seasonality))
            returns = np.concatenate([[0], returns])  # Prepend 0 for first value
            clustered_returns = self._add_volatility_clustering(
                returns, self.garch_params
            )

            # Convert back to prices
            log_prices = np.log(prices_with_seasonality[0]) + np.cumsum(clustered_returns)
            prices_with_vol = np.exp(log_prices)

            # Add jump diffusion
            final_prices = self._add_jump_diffusion(
                prices_with_vol,
                self.lambda_jumps,
                self.jump_mu,
                self.jump_sigma,
                dt_days,
                seed=seed
            )

            # Ensure prices are positive
            final_prices = np.maximum(final_prices, 1.0)

            all_scenarios.append(final_prices)

        # Create DataFrame
        if n_scenarios == 1:
            df = pd.DataFrame({'price': all_scenarios[0]}, index=timestamps)
        else:
            # Use MultiIndex for multiple scenarios to avoid duplicate timestamps
            # Set name for timestamps index
            timestamps.name = 'timestamp'
            dfs = []
            for i in range(n_scenarios):
                scenario_df = pd.DataFrame(
                    {'price': all_scenarios[i], 'scenario_id': i},
                    index=timestamps
                )
                scenario_df = scenario_df.set_index('scenario_id', append=True)
                scenario_df = scenario_df.reorder_levels(['scenario_id', 'timestamp'])
                dfs.append(scenario_df)
            df = pd.concat(dfs)

        logger.info(f"Generated {len(df)} price observations across {n_scenarios} scenario(s)")

        return df

    def simulate_ou_process(
        self,
        n_steps: int,
        mu: float,
        kappa: float,
        sigma: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Public wrapper for simulating Ornstein-Uhlenbeck process.

        Provides sensible defaults for dt and x0 based on configuration.

        Args:
            n_steps: Number of time steps
            mu: Long-term mean
            kappa: Mean reversion speed
            sigma: Volatility
            seed: Random seed for reproducibility

        Returns:
            Array of OU process values
        """
        # Derive dt from frequency: hourly = 1/24 days
        dt = 1.0 / 24.0
        # Start at the mean
        x0 = mu

        return self._simulate_ou_process(
            n_steps=n_steps,
            dt=dt,
            x0=x0,
            kappa=kappa,
            mu=mu,
            sigma=sigma,
            seed=seed
        )

    def _simulate_ou_process(
        self,
        n_steps: int,
        dt: float,
        x0: float,
        kappa: float,
        mu: float,
        sigma: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate Ornstein-Uhlenbeck process using exact discretization.

        Uses the formula:
        X_{t+Δ} = mu + (X_t - mu) * exp(-kappa * Δ) +
                  sigma * sqrt((1 - exp(-2*kappa*Δ))/(2*kappa)) * ε

        Args:
            n_steps: Number of time steps
            dt: Time step size (in days)
            x0: Initial value
            kappa: Mean reversion speed (must be > 0)
            mu: Long-term mean
            sigma: Volatility (must be > 0)
            seed: Random seed for reproducibility

        Returns:
            Array of OU process values

        Raises:
            ValueError: If kappa <= 0 or sigma <= 0
        """
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        rng = np.random.default_rng(seed)

        # Pre-compute constants
        exp_kappa_dt = np.exp(-kappa * dt)
        std_noise = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

        # Simulate process
        x = np.zeros(n_steps)
        x[0] = x0

        for t in range(1, n_steps):
            epsilon = rng.standard_normal()
            x[t] = mu + (x[t-1] - mu) * exp_kappa_dt + std_noise * epsilon

        return x

    def _add_seasonality(
        self,
        timestamps: pd.DatetimeIndex,
        seasonality_params: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """
        Add multi-scale seasonal patterns using Fourier series.

        Implements:
        - Daily cycle: a1*cos(2π*hour/24) + b1*sin(2π*hour/24)
        - Weekly cycle: a2*cos(2π*day_of_week/7) + b2*sin(2π*day_of_week/7)
        - Yearly cycle: a3*cos(2π*day_of_year/365) + b3*sin(2π*day_of_year/365)

        Args:
            timestamps: DatetimeIndex of timestamps
            seasonality_params: Dict with 'daily', 'weekly', 'yearly' coefficient dicts

        Returns:
            Array of seasonal adjustments
        """
        seasonal_adj = np.zeros(len(timestamps))

        # Daily cycle
        daily_params = seasonality_params.get("daily", {})
        if daily_params:
            hours = timestamps.hour + timestamps.minute / 60.0
            a1 = daily_params.get("a1", 0.0)
            b1 = daily_params.get("b1", 0.0)
            seasonal_adj += a1 * np.cos(2 * np.pi * hours / 24)
            seasonal_adj += b1 * np.sin(2 * np.pi * hours / 24)

        # Weekly cycle
        weekly_params = seasonality_params.get("weekly", {})
        if weekly_params:
            day_of_week = timestamps.dayofweek
            a2 = weekly_params.get("a2", 0.0)
            b2 = weekly_params.get("b2", 0.0)
            seasonal_adj += a2 * np.cos(2 * np.pi * day_of_week / 7)
            seasonal_adj += b2 * np.sin(2 * np.pi * day_of_week / 7)

        # Yearly cycle
        yearly_params = seasonality_params.get("yearly", {})
        if yearly_params:
            day_of_year = timestamps.dayofyear
            a3 = yearly_params.get("a3", 0.0)
            b3 = yearly_params.get("b3", 0.0)
            seasonal_adj += a3 * np.cos(2 * np.pi * day_of_year / 365)
            seasonal_adj += b3 * np.sin(2 * np.pi * day_of_year / 365)

        return seasonal_adj

    def _add_volatility_clustering(
        self,
        returns: np.ndarray,
        garch_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Add GARCH(1,1) volatility clustering to returns.

        Implements: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}

        Args:
            returns: Array of returns
            garch_params: Dict with 'omega', 'alpha', 'beta' parameters

        Returns:
            Returns array with clustered volatility

        Raises:
            ValueError: If GARCH parameters are invalid or unstable
        """
        omega = garch_params.get("omega", 0.1)
        alpha = garch_params.get("alpha", 0.15)
        beta = garch_params.get("beta", 0.80)

        # Validate GARCH parameters for stability
        if alpha < 0:
            raise ValueError(f"GARCH alpha parameter must be non-negative, got {alpha}")
        if beta < 0:
            raise ValueError(f"GARCH beta parameter must be non-negative, got {beta}")
        if alpha + beta >= 1:
            raise ValueError(
                f"GARCH parameters must satisfy alpha + beta < 1 for stability, "
                f"got alpha={alpha}, beta={beta}, sum={alpha + beta}"
            )
        if omega <= 0:
            raise ValueError(f"GARCH omega parameter must be positive, got {omega}")

        n = len(returns)
        variance = np.zeros(n)
        variance[0] = omega / (1 - alpha - beta)  # Unconditional variance

        # Guard against invalid variance
        if variance[0] <= 0:
            raise ValueError(f"Initial variance must be positive, got {variance[0]}")

        # Iterate through time steps
        for t in range(1, n):
            variance[t] = omega + alpha * (returns[t-1] ** 2) + beta * variance[t-1]

        # Scale returns by time-varying volatility
        # Guard sqrt operations
        if np.any(variance <= 0):
            logger.warning("Negative or zero variance detected, clipping to small positive value")
            variance = np.maximum(variance, 1e-10)

        clustered_returns = returns * np.sqrt(variance) / np.sqrt(variance[0])

        return clustered_returns

    def _add_jump_diffusion(
        self,
        price_series: np.ndarray,
        lambda_jumps: float,
        jump_mu: float,
        jump_sigma: float,
        dt: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Add rare price spikes using compound Poisson process.

        Jump sizes are interpreted as percentage points and applied as
        exp(jump_size/100). For example, a jump_size of 2.0 results in
        a ~2% price change (exp(0.02) ≈ 1.020).

        Args:
            price_series: Base price series
            lambda_jumps: Jump intensity (jumps per day)
            jump_mu: Mean jump size in percentage points (e.g., 0.0 for no directional bias)
            jump_sigma: Jump size volatility in percentage points (typical range: 1.0-5.0)
            dt: Time step size in days
            seed: Random seed

        Returns:
            Price series with jumps added
        """
        rng = np.random.default_rng(seed)

        n = len(price_series)
        prices_with_jumps = price_series.copy()

        # Generate jump times using Poisson process
        for t in range(n):
            # Number of jumps in this time step
            n_jumps = rng.poisson(lambda_jumps * dt)

            if n_jumps > 0:
                # Generate jump sizes (log-normal)
                jump_sizes = rng.normal(jump_mu, jump_sigma, n_jumps)
                total_jump = np.sum(jump_sizes)

                # Add jumps to price
                prices_with_jumps[t] = prices_with_jumps[t] * np.exp(total_jump / 100.0)

        return prices_with_jumps

    def generate_stress_scenarios(
        self,
        base_series: pd.DataFrame,
        scenario_type: str,
        intensity: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate specific stress test scenarios from base price series.

        Args:
            base_series: Base price DataFrame with 'price' column
            scenario_type: One of 'high_volatility', 'price_spike', 'price_crash', 'low_liquidity'
            intensity: Stress intensity multiplier (default 1.0)

        Returns:
            DataFrame with stressed price series

        Raises:
            ValueError: If scenario_type is unknown
        """
        if 'price' not in base_series.columns:
            raise ValueError("base_series must have 'price' column")

        stressed_series = base_series.copy()
        prices = stressed_series['price'].values

        if scenario_type == 'high_volatility':
            # Increase volatility by 2-3x
            mean_price = np.mean(prices)
            volatility_multiplier = 2.0 + intensity
            prices = mean_price + (prices - mean_price) * volatility_multiplier

        elif scenario_type == 'price_spike':
            # Add large positive jumps
            n_spikes = max(1, int(len(prices) * 0.01 * intensity))  # 1% of observations
            spike_indices = np.random.choice(len(prices), size=n_spikes, replace=False)
            spike_magnitude = 100 * intensity  # $/MWh
            prices[spike_indices] += spike_magnitude

        elif scenario_type == 'price_crash':
            # Add large negative jumps
            n_crashes = max(1, int(len(prices) * 0.01 * intensity))
            crash_indices = np.random.choice(len(prices), size=n_crashes, replace=False)
            crash_magnitude = -50 * intensity  # $/MWh
            prices[crash_indices] = np.maximum(prices[crash_indices] + crash_magnitude, 1.0)

        elif scenario_type == 'low_liquidity':
            # Simulate low liquidity with increased bid-ask spread (higher variance)
            returns = np.diff(np.log(prices))
            spread_multiplier = 1.5 * intensity
            adjusted_returns = returns * spread_multiplier
            prices[1:] = prices[0] * np.exp(np.cumsum(adjusted_returns))

        else:
            raise ValueError(
                f"Unknown scenario_type: {scenario_type}. "
                "Must be one of: 'high_volatility', 'price_spike', 'price_crash', 'low_liquidity'"
            )

        # Ensure positive prices
        prices = np.maximum(prices, 1.0)
        stressed_series['price'] = prices

        logger.info(
            f"Generated {scenario_type} stress scenario with intensity={intensity}"
        )

        return stressed_series

    def calibrate_from_data(
        self,
        historical_prices: pd.Series,
        dt: float = 1.0 / 24.0
    ) -> Dict[str, float]:
        """
        Calibrate OU parameters from historical price data using MLE.

        Implements discrete-time AR(1) regression:
        X_{t+1} = a + b*X_t + ε

        Then extracts:
        - kappa = -ln(b)/dt
        - mu = a/(1-b)
        - sigma = sqrt(2*kappa*var(ε)/(1-b²))

        Args:
            historical_prices: Series or DataFrame column with historical prices
            dt: Time step size (default 1/24 for hourly data)

        Returns:
            Dict with calibrated parameters {'kappa', 'mu', 'sigma'}

        Raises:
            ValueError: If insufficient data or calibration fails
        """
        if len(historical_prices) < 10:
            raise ValueError("Need at least 10 observations for calibration")

        prices = historical_prices.values if isinstance(historical_prices, pd.Series) else historical_prices

        # Remove NaN values
        prices = prices[~np.isnan(prices)]

        # Set up AR(1) regression: X_{t+1} = a + b*X_t + ε
        X = prices[:-1].reshape(-1, 1)
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept
        y = prices[1:]

        # OLS regression
        params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        a, b = params

        # Check stability (b should be < 1 for mean reversion)
        if b >= 1.0:
            logger.warning(f"AR(1) coefficient b={b:.3f} >= 1, no mean reversion detected")
            b = 0.95  # Force mean reversion

        # Extract OU parameters
        kappa = -np.log(b) / dt
        mu = a / (1 - b)

        # Estimate sigma from residuals
        residual_var = np.var(y - X @ params)
        sigma = np.sqrt(2 * kappa * residual_var / (1 - b**2))

        calibrated_params = {
            'kappa': float(kappa),
            'mu': float(mu),
            'sigma': float(sigma)
        }

        logger.info(
            f"Calibrated OU parameters: kappa={kappa:.3f}, mu={mu:.2f}, sigma={sigma:.2f}"
        )

        return calibrated_params

    def save_to_storage(
        self,
        data: pd.DataFrame,
        data_manager,
        start_date: str,
        end_date: str
    ) -> Optional[Path]:
        """
        Save generated price series to DataManager.

        Args:
            data: DataFrame with generated prices
            data_manager: Instance of DataManager
            start_date: Start date string
            end_date: End date string

        Returns:
            Path to saved file/directory, or None if save failed
        """
        try:
            path = data_manager.save_processed_data(
                data=data,
                source='synthetic',
                dataset='prices',
                start_date=start_date,
                end_date=end_date
            )

            logger.info(f"Saved synthetic prices to {path}")
            return path

        except Exception as e:
            logger.error(f"Failed to save synthetic prices: {str(e)}")
            return None


if __name__ == "__main__":
    # Setup logging
    from src.config.load_config import setup_logging
    setup_logging()

    # Example usage
    print("Synthetic Price Generator Example")
    print("=" * 50)

    # Initialize generator
    gen = SyntheticPriceGenerator()

    # Generate price series
    prices = gen.generate_price_series(
        start_date='2024-01-01',
        end_date='2024-01-31',
        frequency='H',
        initial_price=50.0,
        n_scenarios=3
    )

    print(f"\nGenerated {len(prices)} price observations")
    print(f"Price statistics:")
    print(f"  Mean: ${prices['price'].mean():.2f}/MWh")
    print(f"  Std:  ${prices['price'].std():.2f}/MWh")
    print(f"  Min:  ${prices['price'].min():.2f}/MWh")
    print(f"  Max:  ${prices['price'].max():.2f}/MWh")

    print("\nFirst few rows:")
    print(prices.head())

    # Test stress scenarios
    base_prices = prices[prices['scenario_id'] == 0][['price']].copy()
    stressed = gen.generate_stress_scenarios(
        base_series=base_prices,
        scenario_type='price_spike',
        intensity=1.5
    )

    print(f"\nStress scenario (price_spike) statistics:")
    print(f"  Mean: ${stressed['price'].mean():.2f}/MWh")
    print(f"  Max:  ${stressed['price'].max():.2f}/MWh")

    # Test calibration
    print("\nCalibrating from generated data...")
    calibrated = gen.calibrate_from_data(base_prices['price'], dt=1.0/24.0)
    print(f"Calibrated parameters:")
    for param, value in calibrated.items():
        print(f"  {param}: {value:.3f}")
