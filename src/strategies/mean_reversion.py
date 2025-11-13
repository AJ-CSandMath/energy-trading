"""
Mean Reversion Strategy Module

Implements a mean reversion trading strategy using Bollinger Bands. Assumes that
energy prices tend to revert to their mean due to supply/demand equilibrium.

Strategy Logic:
- Buy when price drops significantly below the lower Bollinger Band (oversold)
- Sell when price rises significantly above the upper Bollinger Band (overbought)
- Exit when price returns to the middle band (mean)

Bollinger Bands consist of:
- Middle Band: Simple Moving Average (SMA) of price
- Upper Band: SMA + (num_std * standard deviation)
- Lower Band: SMA - (num_std * standard deviation)

The strategy works well in range-bound markets but can underperform in trending markets.

Example:
    >>> strategy = MeanReversionStrategy()
    >>> signals = strategy.generate_signals(price_data)
    >>> # signals contains buy/sell/hold signals with strength and reasoning
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands.

    Identifies overbought/oversold conditions by measuring how far price has
    deviated from its moving average relative to recent volatility.

    Attributes:
        window: Rolling window for mean/std calculation
        num_std: Number of standard deviations for bands
        entry_threshold: How close to band to trigger signal (0-1)
        exit_threshold: When to exit position (0-1)
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    ):
        """
        Initialize mean reversion strategy.

        Args:
            config: Optional global configuration dictionary
            strategy_config: Optional strategy-specific configuration
        """
        # Call parent constructor
        super().__init__(
            name='mean_reversion',
            config=config,
            strategy_config=strategy_config
        )

        # Load strategy-specific parameters from config
        self.window = self.strategy_config.get('window', 20)
        self.num_std = self.strategy_config.get('num_std', 2.0)
        self.entry_threshold = self.strategy_config.get('entry_threshold', 0.9)
        self.exit_threshold = self.strategy_config.get('exit_threshold', 0.5)

        # Optional volatility filters
        self.min_volatility = self.strategy_config.get('min_volatility', 0.01)
        self.max_volatility = self.strategy_config.get('max_volatility', 0.10)

        self.logger.info(
            f"MeanReversionStrategy initialized: window={self.window}, "
            f"num_std={self.num_std}, entry_threshold={self.entry_threshold}, "
            f"exit_threshold={self.exit_threshold}"
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        asset: str = 'default',
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate mean reversion signals based on Bollinger Bands.

        Args:
            data: DataFrame with 'price' or 'close' column and DatetimeIndex
            asset: Asset identifier for position tracking (default: 'default')
            **kwargs: Additional parameters (unused)

        Returns:
            DataFrame with columns:
                - timestamp: Signal timestamp
                - asset: Asset identifier
                - signal: 1 (buy), -1 (sell), 0 (hold)
                - strength: Signal confidence (0-1)
                - reason: Description of signal

        Raises:
            ValueError: If data validation fails
        """
        # Validate input data
        validated_data = self.validate_data(data)

        # Determine price column
        price_col = 'price' if 'price' in validated_data.columns else 'close'
        prices = validated_data[price_col]

        # Calculate Bollinger Bands
        middle, upper, lower = self.calculate_bands(prices, self.window, self.num_std)

        # Calculate band position (0 = lower band, 1 = upper band)
        band_width = upper - lower
        band_position = (prices - lower) / band_width

        # Handle division by zero (when volatility is zero)
        band_position = band_position.fillna(0.5)

        # Calculate volatility for filtering
        volatility = prices.pct_change().rolling(window=self.window).std()

        # Calculate previous band position for crossing detection
        band_position_prev = band_position.shift(1)

        # Initialize signals
        signals = []

        for idx in range(len(validated_data)):
            timestamp = validated_data.index[idx]
            price = prices.iloc[idx]
            band_pos = band_position.iloc[idx]
            band_pos_prev = band_position_prev.iloc[idx] if idx > 0 else np.nan
            vol = volatility.iloc[idx] if idx >= self.window else np.nan

            # Skip if insufficient data
            if idx < self.window or np.isnan(band_pos):
                continue

            # Apply volatility filters
            if not np.isnan(vol):
                if vol < self.min_volatility or vol > self.max_volatility:
                    # Volatility outside acceptable range, no signal
                    continue

            # Generate signals based on band position
            signal = 0
            strength = 0.0
            reason = ""

            # Buy signal: price near lower band (oversold)
            if band_pos < (1 - self.entry_threshold):
                signal = 1
                strength = (1 - self.entry_threshold - band_pos) / (1 - self.entry_threshold)
                strength = np.clip(strength, 0, 1)
                reason = f"Price at {band_pos:.1%} of Bollinger Band (oversold), buy signal"

            # Sell signal: price near upper band (overbought)
            elif band_pos > self.entry_threshold:
                signal = -1
                strength = (band_pos - self.entry_threshold) / (1 - self.entry_threshold)
                strength = np.clip(strength, 0, 1)
                reason = f"Price at {band_pos:.1%} of Bollinger Band (overbought), sell signal"

            # Exit long: price crosses upward through middle band
            elif not np.isnan(band_pos_prev):
                # Check if we have a long position
                current_pos = self.current_positions.get(asset, 0)

                # Long exit: crosses upward through exit_threshold (middle band)
                if (current_pos > 0 and
                    band_pos_prev < self.exit_threshold and
                    band_pos >= self.exit_threshold):
                    signal = -1  # Close long position
                    strength = abs(band_pos - 0.5) * 2
                    reason = f"Price at {band_pos:.1%}, crossed up through middle band, exit long"

                # Short exit: crosses downward through exit_threshold (middle band)
                elif (current_pos < 0 and
                      band_pos_prev > self.exit_threshold and
                      band_pos <= self.exit_threshold):
                    signal = 1  # Close short position
                    strength = abs(band_pos - 0.5) * 2
                    reason = f"Price at {band_pos:.1%}, crossed down through middle band, exit short"

            # Only add signal if it's not a hold
            if signal != 0:
                signals.append({
                    'timestamp': timestamp,
                    'asset': asset,
                    'signal': signal,
                    'strength': strength,
                    'reason': reason,
                    'band_position': band_pos,
                    'price': price,
                    'middle_band': middle.iloc[idx],
                    'upper_band': upper.iloc[idx],
                    'lower_band': lower.iloc[idx]
                })

        # Create signals DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            self.logger.info(
                f"Generated {len(signals_df)} mean reversion signals "
                f"({(signals_df['signal'] == 1).sum()} buy, "
                f"{(signals_df['signal'] == -1).sum()} sell)"
            )
        else:
            # No signals generated, return empty DataFrame with correct structure
            signals_df = pd.DataFrame(columns=[
                'timestamp', 'asset', 'signal', 'strength', 'reason',
                'band_position', 'price', 'middle_band', 'upper_band', 'lower_band'
            ])
            self.logger.info("No mean reversion signals generated")

        return signals_df

    def get_required_columns(self) -> List[str]:
        """
        Return list of required data columns.

        Returns:
            List with 'price' or 'close'
        """
        return ['price']

    def calculate_bands(
        self,
        prices: pd.Series,
        window: int,
        num_std: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Price series
            window: Rolling window for mean/std calculation
            num_std: Number of standard deviations for bands

        Returns:
            Tuple of (middle_band, upper_band, lower_band) as Series
        """
        # Middle band: Simple Moving Average
        middle = prices.rolling(window=window).mean()

        # Standard deviation
        std = prices.rolling(window=window).std()

        # Upper and lower bands
        upper = middle + num_std * std
        lower = middle - num_std * std

        self.logger.debug(
            f"Calculated Bollinger Bands: window={window}, num_std={num_std}"
        )

        return middle, upper, lower
