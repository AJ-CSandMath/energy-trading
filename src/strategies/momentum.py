"""
Momentum Strategy Module

Implements a momentum trading strategy using moving average crossovers. Follows
trends in energy markets by identifying when fast-moving averages cross slow-moving
averages.

Strategy Logic:
- Buy when fast MA crosses above slow MA (uptrend starting)
- Sell when fast MA crosses below slow MA (downtrend starting)
- Hold during the trend until opposite crossover occurs

The strategy assumes that trends persist due to supply/demand imbalances and
weather patterns in energy markets. Works well in trending markets but suffers
from whipsaws in sideways markets.

Example:
    >>> strategy = MomentumStrategy()
    >>> signals = strategy.generate_signals(price_data)
    >>> # signals contains trend-following buy/sell signals
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using moving average crossover.

    Generates buy signals when fast MA crosses above slow MA and sell signals
    when fast MA crosses below slow MA. Optional trend confirmation requires
    price to be above/below both MAs.

    Attributes:
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        signal_threshold: Minimum crossover magnitude to trigger signal
        trend_confirmation: Whether to require price/MA alignment
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    ):
        """
        Initialize momentum strategy.

        Args:
            config: Optional global configuration dictionary
            strategy_config: Optional strategy-specific configuration
        """
        # Call parent constructor
        super().__init__(
            name='momentum',
            config=config,
            strategy_config=strategy_config
        )

        # Load strategy-specific parameters
        self.fast_window = self.strategy_config.get('fast_window', 10)
        self.slow_window = self.strategy_config.get('slow_window', 30)
        self.signal_threshold = self.strategy_config.get('signal_threshold', 0.02)
        self.trend_confirmation = self.strategy_config.get('trend_confirmation', True)

        # Validate parameters
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be less than "
                f"slow_window ({self.slow_window})"
            )

        # Optional filters
        self.min_trend_strength = self.strategy_config.get('min_trend_strength', 0.01)

        self.logger.info(
            f"MomentumStrategy initialized: fast_window={self.fast_window}, "
            f"slow_window={self.slow_window}, signal_threshold={self.signal_threshold:.1%}, "
            f"trend_confirmation={self.trend_confirmation}"
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        asset: str = 'default',
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate momentum signals based on MA crossover.

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

        # Calculate moving averages
        fast_ma = prices.rolling(window=self.fast_window).mean()
        slow_ma = prices.rolling(window=self.slow_window).mean()

        # Calculate MA difference (percentage)
        ma_diff = (fast_ma - slow_ma) / slow_ma

        # Detect crossovers
        ma_diff_prev = ma_diff.shift(1)
        crossover_up = (ma_diff > 0) & (ma_diff_prev <= 0)  # Fast crosses above slow
        crossover_down = (ma_diff < 0) & (ma_diff_prev >= 0)  # Fast crosses below slow

        # Calculate trend direction
        trend = self.detect_trend(prices, self.slow_window)

        # Initialize signals
        signals = []

        for idx in range(len(validated_data)):
            timestamp = validated_data.index[idx]
            price = prices.iloc[idx]

            # Skip if insufficient data
            if idx < self.slow_window:
                continue

            fast_ma_val = fast_ma.iloc[idx]
            slow_ma_val = slow_ma.iloc[idx]
            ma_diff_val = ma_diff.iloc[idx]

            if np.isnan(fast_ma_val) or np.isnan(slow_ma_val):
                continue

            signal = 0
            strength = 0.0
            reason = ""

            # Buy signal: fast MA crosses above slow MA
            if crossover_up.iloc[idx] and abs(ma_diff_val) > self.signal_threshold:
                # Optional trend confirmation
                if self.trend_confirmation:
                    if price > slow_ma_val:
                        signal = 1
                        strength = min(abs(ma_diff_val) / self.signal_threshold, 1.0)
                        reason = (
                            f"Fast MA ({self.fast_window}) crossed above "
                            f"Slow MA ({self.slow_window}), diff={ma_diff_val:.2%}, "
                            f"trend confirmed"
                        )
                else:
                    signal = 1
                    strength = min(abs(ma_diff_val) / self.signal_threshold, 1.0)
                    reason = (
                        f"Fast MA ({self.fast_window}) crossed above "
                        f"Slow MA ({self.slow_window}), diff={ma_diff_val:.2%}"
                    )

            # Sell signal: fast MA crosses below slow MA
            elif crossover_down.iloc[idx] and abs(ma_diff_val) > self.signal_threshold:
                # Optional trend confirmation
                if self.trend_confirmation:
                    if price < slow_ma_val:
                        signal = -1
                        strength = min(abs(ma_diff_val) / self.signal_threshold, 1.0)
                        reason = (
                            f"Fast MA ({self.fast_window}) crossed below "
                            f"Slow MA ({self.slow_window}), diff={ma_diff_val:.2%}, "
                            f"trend confirmed"
                        )
                else:
                    signal = -1
                    strength = min(abs(ma_diff_val) / self.signal_threshold, 1.0)
                    reason = (
                        f"Fast MA ({self.fast_window}) crossed below "
                        f"Slow MA ({self.slow_window}), diff={ma_diff_val:.2%}"
                    )

            # Only add signal if it's not a hold
            if signal != 0:
                signals.append({
                    'timestamp': timestamp,
                    'asset': asset,
                    'signal': signal,
                    'strength': strength,
                    'reason': reason,
                    'price': price,
                    'fast_ma': fast_ma_val,
                    'slow_ma': slow_ma_val,
                    'ma_diff_pct': ma_diff_val,
                    'trend': trend.iloc[idx] if idx < len(trend) else 0
                })

        # Create signals DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            self.logger.info(
                f"Generated {len(signals_df)} momentum signals "
                f"({(signals_df['signal'] == 1).sum()} buy, "
                f"{(signals_df['signal'] == -1).sum()} sell)"
            )
        else:
            # No signals generated
            signals_df = pd.DataFrame(columns=[
                'timestamp', 'asset', 'signal', 'strength', 'reason',
                'price', 'fast_ma', 'slow_ma', 'ma_diff_pct', 'trend'
            ])
            self.logger.info("No momentum signals generated")

        return signals_df

    def get_required_columns(self) -> List[str]:
        """
        Return list of required data columns.

        Returns:
            List with 'price' or 'close'
        """
        return ['price']

    def calculate_moving_averages(
        self,
        prices: pd.Series,
        windows: List[int]
    ) -> Dict[int, pd.Series]:
        """
        Calculate multiple moving averages.

        Args:
            prices: Price series
            windows: List of window sizes

        Returns:
            Dictionary mapping window size to MA series
        """
        mas = {}
        for window in windows:
            mas[window] = prices.rolling(window=window).mean()

        self.logger.debug(f"Calculated moving averages for windows: {windows}")

        return mas

    def detect_trend(
        self,
        prices: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Detect overall trend direction.

        Args:
            prices: Price series
            window: Window for trend calculation

        Returns:
            Series with 1 (uptrend), -1 (downtrend), 0 (sideways)
        """
        # Calculate moving average
        ma = prices.rolling(window=window).mean()

        # Compare price to MA
        price_vs_ma = prices > ma

        # Calculate MA slope
        ma_slope = ma.diff() / ma

        # Determine trend
        trend = pd.Series(0, index=prices.index)
        trend[price_vs_ma & (ma_slope > self.min_trend_strength)] = 1  # Uptrend
        trend[~price_vs_ma & (ma_slope < -self.min_trend_strength)] = -1  # Downtrend

        return trend
