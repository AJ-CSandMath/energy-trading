"""
Renewable Arbitrage Strategy Module

Implements a renewable-aware arbitrage strategy that exploits the correlation between
renewable energy generation forecasts and electricity prices.

Strategy Logic:
- Forecast renewable generation (wind and solar) using forecasting pipelines
- Forecast electricity prices using price forecasting pipeline
- Exploit negative correlation: high generation → lower prices, low generation → higher prices
- Buy when high generation expected (prices likely to drop)
- Sell when low generation expected (prices likely to rise)

This strategy is particularly relevant for renewable energy portfolio management and
demonstrates understanding of how renewable generation impacts market prices.

Applications:
- Trading around renewable generation forecasts
- Curtailment arbitrage opportunities
- Optimizing generation schedules based on price forecasts

Example:
    >>> from src.models.price_forecasting import PriceForecastingPipeline
    >>> from src.models.renewable_forecasting import RenewableForecastingPipeline
    >>>
    >>> strategy = RenewableArbitrageStrategy()
    >>> strategy.set_forecasters(price_forecaster, renewable_forecaster)
    >>> signals = strategy.generate_signals(price_data, wind_data=wind_df, solar_data=solar_df)
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class RenewableArbitrageStrategy(BaseStrategy):
    """
    Renewable-aware arbitrage strategy exploiting generation-price correlations.

    Integrates with price and renewable forecasting pipelines to generate signals
    based on expected relationship between generation and prices.

    Attributes:
        forecast_horizon: Hours ahead to forecast
        generation_threshold_high: Capacity factor threshold for high generation
        generation_threshold_low: Capacity factor threshold for low generation
        price_sensitivity: Weight of price forecast vs generation forecast
        correlation_factor: Expected correlation between generation and prices
        price_forecaster: Price forecasting pipeline
        renewable_forecaster: Renewable forecasting pipeline
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None,
        price_forecaster=None,
        renewable_forecaster=None
    ):
        """
        Initialize renewable arbitrage strategy.

        Args:
            config: Optional global configuration dictionary
            strategy_config: Optional strategy-specific configuration
            price_forecaster: Optional PriceForecastingPipeline instance
            renewable_forecaster: Optional RenewableForecastingPipeline instance
        """
        # Call parent constructor
        super().__init__(
            name='renewable_arbitrage',
            config=config,
            strategy_config=strategy_config
        )

        # Load strategy-specific parameters
        self.forecast_horizon = self.strategy_config.get('forecast_horizon', 24)
        self.generation_threshold_high = self.strategy_config.get('generation_threshold_high', 0.7)
        self.generation_threshold_low = self.strategy_config.get('generation_threshold_low', 0.3)
        self.price_sensitivity = self.strategy_config.get('price_sensitivity', 0.5)
        self.correlation_factor = self.strategy_config.get('correlation_factor', -0.6)

        # Curtailment parameters
        self.consider_curtailment = self.strategy_config.get('consider_curtailment', True)
        self.curtailment_price_threshold = self.strategy_config.get('curtailment_price_threshold', -5.0)

        # Renewable capacities
        self.wind_capacity_mw = self.strategy_config.get('wind_capacity_mw', 100.0)
        self.solar_capacity_mw = self.strategy_config.get('solar_capacity_mw', 100.0)

        # Forecast confidence filtering
        self.forecast_confidence_threshold = self.strategy_config.get('forecast_confidence_threshold', 0.6)
        self.max_forecast_error = self.strategy_config.get('max_forecast_error', 0.15)

        # Price impact sensitivity for generation-price relationship
        self.price_impact_sensitivity = self.strategy_config.get('price_impact_sensitivity', 0.1)

        # Store forecasters
        self.price_forecaster = price_forecaster
        self.renewable_forecaster = renewable_forecaster

        self.logger.info(
            f"RenewableArbitrageStrategy initialized: horizon={self.forecast_horizon}h, "
            f"gen_threshold_high={self.generation_threshold_high:.1%}, "
            f"gen_threshold_low={self.generation_threshold_low:.1%}, "
            f"correlation_factor={self.correlation_factor:.2f}"
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        wind_data: Optional[pd.DataFrame] = None,
        solar_data: Optional[pd.DataFrame] = None,
        asset: str = 'default',
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate signals based on renewable generation and price forecasts.

        Args:
            data: DataFrame with current price data and DatetimeIndex
            wind_data: Optional DataFrame with wind generation data
            solar_data: Optional DataFrame with solar generation data
            asset: Asset identifier for position tracking (default: 'default')
            **kwargs: Additional parameters

        Returns:
            DataFrame with columns:
                - timestamp: Signal timestamp
                - asset: Asset identifier
                - signal: 1 (buy), -1 (sell), 0 (hold)
                - strength: Signal confidence (0-1)
                - reason: Description of signal

        Raises:
            ValueError: If forecasters not set or data validation fails
        """
        # Validate forecasters are set
        if self.price_forecaster is None or self.renewable_forecaster is None:
            raise ValueError(
                "Forecasters not set. Call set_forecasters() before generating signals."
            )

        # Validate input data
        validated_data = self.validate_data(data)

        # Determine price column
        price_col = 'price' if 'price' in validated_data.columns else 'close'
        current_price = validated_data[price_col].iloc[-1]

        try:
            # Generate price forecast
            self.logger.info(f"Generating price forecast for {self.forecast_horizon} hours")
            price_forecast_df = self.price_forecaster.predict(
                validated_data,
                model_name='ensemble',
                steps=self.forecast_horizon
            )
            # Set index to timestamp for proper signal generation
            price_forecast_df = price_forecast_df.set_index('timestamp', drop=False)
            price_forecast = price_forecast_df['forecast'].values

            # Generate renewable forecasts
            wind_forecast = None
            solar_forecast = None
            total_capacity = 0

            # Create zero-filled placeholders for missing data
            # This avoids calling forecast() with empty DataFrames
            if wind_data is not None or solar_data is not None:
                # Use provided data or create zero-filled placeholder
                if wind_data is None:
                    # Create zero-filled wind data with same length as solar
                    wind_data = pd.DataFrame({
                        'wind_speed': np.zeros(len(solar_data)),
                        'wind_generation': np.zeros(len(solar_data))
                    }, index=solar_data.index if solar_data is not None else pd.DatetimeIndex([]))

                if solar_data is None:
                    # Create zero-filled solar data with same length as wind
                    solar_data = pd.DataFrame({
                        'solar_irradiance': np.zeros(len(wind_data)),
                        'solar_generation': np.zeros(len(wind_data))
                    }, index=wind_data.index if wind_data is not None else pd.DatetimeIndex([]))

                # Single unified forecast call
                self.logger.info("Generating renewable forecasts (wind and solar)")
                renewable_result = self.renewable_forecaster.forecast(
                    wind_data=wind_data,
                    solar_data=solar_data,
                    model_name='ensemble',
                    steps=self.forecast_horizon
                )

                # Extract forecasts from unified result
                wind_forecast = renewable_result['wind_forecast']['forecast'].values
                solar_forecast = renewable_result['solar_forecast']['forecast'].values

                # Set capacities based on original (non-None) data
                if wind_data is not None and not (wind_data == 0).all().all():
                    total_capacity += self.wind_capacity_mw
                if solar_data is not None and not (solar_data == 0).all().all():
                    total_capacity += self.solar_capacity_mw

        except Exception as e:
            self.logger.error(f"Forecast generation failed: {e}")
            # Return empty signals DataFrame
            return pd.DataFrame(columns=[
                'timestamp', 'signal', 'strength', 'reason',
                'capacity_factor', 'price_forecast', 'current_price'
            ])

        # Calculate combined renewable generation forecast
        if wind_forecast is not None and solar_forecast is not None:
            total_gen = wind_forecast + solar_forecast
        elif wind_forecast is not None:
            total_gen = wind_forecast
        elif solar_forecast is not None:
            total_gen = solar_forecast
        else:
            self.logger.warning("No renewable forecasts available")
            return pd.DataFrame(columns=[
                'timestamp', 'signal', 'strength', 'reason',
                'capacity_factor', 'price_forecast', 'current_price'
            ])

        # Calculate capacity factor
        if total_capacity > 0:
            capacity_factor = total_gen / total_capacity
        else:
            capacity_factor = np.zeros_like(total_gen)

        # Calculate statistics for normalization
        cf_mean = np.mean(capacity_factor)
        cf_std = np.std(capacity_factor) if np.std(capacity_factor) > 0 else 1.0

        # Analyze generation-price relationship
        signals = []

        for t in range(len(price_forecast)):
            timestamp = price_forecast_df.index[t]
            cf = capacity_factor[t]
            forecast_price = price_forecast[t]

            # Expected price impact based on generation
            # Negative correlation: high CF → lower expected price
            cf_deviation = (cf - cf_mean) / cf_std
            price_impact = self.correlation_factor * cf_deviation

            # Adjusted price forecast using configured sensitivity
            adjusted_price = forecast_price * (1 + price_impact * self.price_impact_sensitivity)

            signal = 0
            strength = 0.0
            reason = ""

            # Buy signal: high generation → lower prices expected
            if cf > self.generation_threshold_high and adjusted_price < current_price:
                signal = 1
                # Calculate strength based on both generation and price deviation
                gen_strength = (cf - self.generation_threshold_high) / (1 - self.generation_threshold_high)
                price_strength = (current_price - adjusted_price) / current_price
                strength = (
                    self.price_sensitivity * price_strength +
                    (1 - self.price_sensitivity) * gen_strength
                )
                strength = np.clip(strength, 0, 1)
                reason = (
                    f"High generation forecast: CF={cf:.1%}, "
                    f"expected price=${adjusted_price:.2f} (current ${current_price:.2f}), "
                    f"buy signal"
                )

            # Sell signal: low generation → higher prices expected
            elif cf < self.generation_threshold_low and adjusted_price > current_price:
                signal = -1
                # Calculate strength
                gen_strength = (self.generation_threshold_low - cf) / self.generation_threshold_low
                price_strength = (adjusted_price - current_price) / current_price
                strength = (
                    self.price_sensitivity * price_strength +
                    (1 - self.price_sensitivity) * gen_strength
                )
                strength = np.clip(strength, 0, 1)
                reason = (
                    f"Low generation forecast: CF={cf:.1%}, "
                    f"expected price=${adjusted_price:.2f} (current ${current_price:.2f}), "
                    f"sell signal"
                )

            # Check for curtailment opportunities
            if self.consider_curtailment and signal == 0:
                if total_gen[t] > total_capacity * 0.9 and forecast_price < self.curtailment_price_threshold:
                    signal = -1  # Sell (curtail generation)
                    strength = 0.7
                    reason = (
                        f"Curtailment opportunity: high generation ({total_gen[t]:.1f} MW), "
                        f"negative price (${forecast_price:.2f})"
                    )

            # Only add signal if it's not a hold
            if signal != 0:
                signals.append({
                    'timestamp': timestamp,
                    'asset': asset,
                    'signal': signal,
                    'strength': strength,
                    'reason': reason,
                    'capacity_factor': cf,
                    'price_forecast': forecast_price,
                    'adjusted_price': adjusted_price,
                    'current_price': current_price,
                    'wind_forecast': wind_forecast[t] if wind_forecast is not None else np.nan,
                    'solar_forecast': solar_forecast[t] if solar_forecast is not None else np.nan
                })

        # Create signals DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            self.logger.info(
                f"Generated {len(signals_df)} renewable arbitrage signals "
                f"({(signals_df['signal'] == 1).sum()} buy, "
                f"{(signals_df['signal'] == -1).sum()} sell)"
            )
        else:
            # No signals generated
            signals_df = pd.DataFrame(columns=[
                'timestamp', 'asset', 'signal', 'strength', 'reason',
                'capacity_factor', 'price_forecast', 'adjusted_price', 'current_price',
                'wind_forecast', 'solar_forecast'
            ])
            self.logger.info("No renewable arbitrage signals generated")

        return signals_df

    def set_forecasters(self, price_forecaster, renewable_forecaster):
        """
        Set forecasting pipelines.

        Args:
            price_forecaster: PriceForecastingPipeline instance
            renewable_forecaster: RenewableForecastingPipeline instance
        """
        self.price_forecaster = price_forecaster
        self.renewable_forecaster = renewable_forecaster

        self.logger.info("Forecasters set for renewable arbitrage strategy")

    def get_required_columns(self) -> List[str]:
        """
        Return list of required data columns.

        Returns:
            List with 'price' (plus optional 'wind_generation', 'solar_generation')
        """
        return ['price']

    def calculate_generation_price_correlation(
        self,
        prices: pd.Series,
        generation: pd.Series,
        window: int = 168
    ) -> pd.Series:
        """
        Calculate historical correlation between generation and prices.

        Args:
            prices: Price series
            generation: Generation series
            window: Rolling window for correlation calculation (default: 168 = 1 week)

        Returns:
            Rolling correlation series
        """
        correlation = prices.rolling(window=window).corr(generation)

        self.logger.info(
            f"Generation-price correlation: mean={correlation.mean():.2f}, "
            f"std={correlation.std():.2f}"
        )

        return correlation

    def analyze_curtailment_opportunity(
        self,
        generation_forecast: pd.Series,
        price_forecast: pd.Series,
        grid_capacity: float
    ) -> pd.DataFrame:
        """
        Identify curtailment arbitrage opportunities.

        Opportunities exist when generation exceeds grid capacity and prices are
        below threshold, making curtailment economically beneficial.

        Args:
            generation_forecast: Forecasted generation (MW)
            price_forecast: Forecasted prices ($/MWh)
            grid_capacity: Grid injection capacity (MW)

        Returns:
            DataFrame with curtailment opportunities
        """
        opportunities = []

        for idx in range(len(generation_forecast)):
            gen = generation_forecast.iloc[idx]
            price = price_forecast.iloc[idx]

            # Check if curtailment is needed and beneficial
            if gen > grid_capacity and price < self.curtailment_price_threshold:
                curtailment_amount = gen - grid_capacity
                economic_benefit = -price * curtailment_amount  # Negative price means benefit

                opportunities.append({
                    'timestamp': generation_forecast.index[idx],
                    'generation_mw': gen,
                    'grid_capacity_mw': grid_capacity,
                    'curtailment_mw': curtailment_amount,
                    'price_per_mwh': price,
                    'economic_benefit': economic_benefit
                })

        if opportunities:
            opp_df = pd.DataFrame(opportunities)
            self.logger.info(
                f"Identified {len(opp_df)} curtailment opportunities, "
                f"total benefit: ${opp_df['economic_benefit'].sum():.2f}"
            )
        else:
            opp_df = pd.DataFrame(columns=[
                'timestamp', 'generation_mw', 'grid_capacity_mw',
                'curtailment_mw', 'price_per_mwh', 'economic_benefit'
            ])
            self.logger.info("No curtailment opportunities identified")

        return opp_df

    def optimize_generation_schedule(
        self,
        generation_forecast: pd.Series,
        price_forecast: pd.Series,
        capacity: float
    ) -> pd.Series:
        """
        Optimize generation schedule based on price forecasts.

        Simple heuristic: generate more when prices are high, curtail when prices
        are low (if curtailment is allowed).

        Args:
            generation_forecast: Base generation forecast (MW)
            price_forecast: Price forecast ($/MWh)
            capacity: Generation capacity (MW)

        Returns:
            Optimized generation schedule (MW)
        """
        # Normalize prices to 0-1 range
        price_norm = (price_forecast - price_forecast.min()) / (price_forecast.max() - price_forecast.min())

        # Simple heuristic: scale generation by price
        # High price → generate at capacity
        # Low price → consider curtailment
        optimized = generation_forecast.copy()

        for idx in range(len(generation_forecast)):
            gen = generation_forecast.iloc[idx]
            price = price_forecast.iloc[idx]
            price_n = price_norm.iloc[idx]

            # If price is very low and below curtailment threshold, reduce generation
            if price < self.curtailment_price_threshold:
                optimized.iloc[idx] = min(gen * 0.5, capacity)  # Curtail to 50%

            # If price is very high, ensure generation at maximum feasible level
            elif price_n > 0.8:
                optimized.iloc[idx] = min(gen, capacity)

        revenue_base = (generation_forecast * price_forecast).sum()
        revenue_optimized = (optimized * price_forecast).sum()

        self.logger.info(
            f"Generation schedule optimized: "
            f"base revenue=${revenue_base:.2f}, "
            f"optimized revenue=${revenue_optimized:.2f}, "
            f"improvement=${revenue_optimized - revenue_base:.2f}"
        )

        return optimized
