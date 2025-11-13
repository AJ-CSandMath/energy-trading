"""
Spread Trading Strategy Module

Implements a spread trading strategy for inter-market arbitrage. Exploits price
differences between related energy markets or assets (e.g., day-ahead vs real-time
prices, different nodes in the grid).

Strategy Logic:
- Calculate spread between two related assets
- Monitor statistical properties of the spread (mean, std deviation)
- Buy spread when it's unusually low (z-score below threshold)
- Sell spread when it's unusually high (z-score above threshold)
- Exit when spread reverts to mean

The strategy assumes that related energy markets maintain a statistical relationship
(cointegration) and that deviations from this relationship are temporary.

Applications in Energy Markets:
- Day-ahead vs real-time prices at the same node
- Prices at different nodes (congestion arbitrage)
- On-peak vs off-peak prices

Example:
    >>> strategy = SpreadTradingStrategy()
    >>> # data should have 'price_1' and 'price_2' columns
    >>> signals = strategy.generate_signals(price_data)
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.strategies.base_strategy import BaseStrategy


class SpreadTradingStrategy(BaseStrategy):
    """
    Spread trading strategy for inter-market arbitrage.

    Monitors the spread between two related assets and generates signals when
    the spread deviates significantly from its historical mean (measured by z-score).

    Attributes:
        lookback_window: Window for spread statistics calculation
        entry_z_score: Z-score threshold for entry
        exit_z_score: Z-score threshold for exit
        spread_type: Type of spread ('difference' or 'ratio')
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    ):
        """
        Initialize spread trading strategy.

        Args:
            config: Optional global configuration dictionary
            strategy_config: Optional strategy-specific configuration
        """
        # Call parent constructor
        super().__init__(
            name='spread_trading',
            config=config,
            strategy_config=strategy_config
        )

        # Load strategy-specific parameters
        self.lookback_window = self.strategy_config.get('lookback_window', 60)
        self.entry_z_score = self.strategy_config.get('entry_z_score', 2.0)
        self.exit_z_score = self.strategy_config.get('exit_z_score', 0.5)
        self.spread_type = self.strategy_config.get('spread_type', 'difference')

        # Hedge ratio settings
        self.use_dynamic_hedge_ratio = self.strategy_config.get('use_dynamic_hedge_ratio', True)
        self.hedge_ratio_window = self.strategy_config.get('hedge_ratio_window', 120)

        # Risk management
        self.max_spread_deviation = self.strategy_config.get('max_spread_deviation', 4.0)
        self.correlation_threshold = self.strategy_config.get('correlation_threshold', 0.7)

        self.logger.info(
            f"SpreadTradingStrategy initialized: lookback={self.lookback_window}, "
            f"entry_z={self.entry_z_score}, exit_z={self.exit_z_score}, "
            f"spread_type={self.spread_type}"
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        asset: str = 'spread',
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate spread trading signals.

        Args:
            data: DataFrame with 'price_1' and 'price_2' columns and DatetimeIndex
            asset: Asset identifier for position tracking (default: 'spread')
            **kwargs: Additional parameters (unused)

        Returns:
            DataFrame with columns:
                - timestamp: Signal timestamp
                - asset: Asset identifier
                - signal: 1 (buy spread), -1 (sell spread), 0 (hold)
                - strength: Signal confidence (0-1)
                - reason: Description of signal

        Raises:
            ValueError: If data validation fails or required columns missing
        """
        # Validate input data
        validated_data = self.validate_data(data)

        # Explicit validation for required spread columns
        if 'price_1' not in validated_data.columns or 'price_2' not in validated_data.columns:
            raise ValueError(
                "Spread trading requires 'price_1' and 'price_2' columns. "
                f"Found columns: {list(validated_data.columns)}"
            )

        # Extract price series
        price_1 = validated_data['price_1']
        price_2 = validated_data['price_2']

        # Calculate spread
        spread = self.calculate_spread(price_1, price_2, self.spread_type)

        # Calculate spread statistics
        spread_mean = spread.rolling(window=self.lookback_window).mean()
        spread_std = spread.rolling(window=self.lookback_window).std()

        # Calculate z-score
        z_score = (spread - spread_mean) / spread_std
        z_score = z_score.fillna(0)

        # Calculate hedge ratio if using dynamic hedging
        if self.use_dynamic_hedge_ratio:
            hedge_ratio = self.calculate_hedge_ratio(price_1, price_2)
        else:
            hedge_ratio = pd.Series(1.0, index=validated_data.index)

        # Calculate correlation for filtering
        correlation = price_1.rolling(window=self.lookback_window).corr(price_2)

        # Initialize signals
        signals = []

        for idx in range(len(validated_data)):
            timestamp = validated_data.index[idx]

            # Skip if insufficient data
            if idx < self.lookback_window:
                continue

            z = z_score.iloc[idx]
            corr = correlation.iloc[idx]
            hedge = hedge_ratio.iloc[idx] if self.use_dynamic_hedge_ratio else 1.0

            if np.isnan(z) or np.isnan(corr):
                continue

            # Skip if correlation too low
            if abs(corr) < self.correlation_threshold:
                continue

            # Skip if z-score is extreme (potential regime change)
            if abs(z) > self.max_spread_deviation:
                self.logger.warning(
                    f"Extreme z-score detected: {z:.2f} at {timestamp}, skipping"
                )
                continue

            signal = 0
            strength = 0.0
            reason = ""

            # Buy spread signal: z-score < -entry_z_score
            # (spread unusually low, buy asset 1, sell asset 2)
            if z < -self.entry_z_score:
                signal = 1
                strength = min(abs(z) / self.entry_z_score, 1.0)
                reason = (
                    f"Spread z-score={z:.2f} (< -{self.entry_z_score}), "
                    f"spread unusually low, buy spread"
                )

            # Sell spread signal: z-score > entry_z_score
            # (spread unusually high, sell asset 1, buy asset 2)
            elif z > self.entry_z_score:
                signal = -1
                strength = min(abs(z) / self.entry_z_score, 1.0)
                reason = (
                    f"Spread z-score={z:.2f} (> {self.entry_z_score}), "
                    f"spread unusually high, sell spread"
                )

            # Exit long spread: z-score > -exit_z_score
            # (spread returning to mean)
            elif z > -self.exit_z_score and z < 0:
                current_pos = self.current_positions.get(asset, 0)
                if current_pos > 0:
                    signal = -1  # Close long spread
                    strength = abs(z) / self.exit_z_score
                    reason = f"Spread z-score={z:.2f}, reverting to mean, exit long spread"

            # Exit short spread: z-score < exit_z_score
            # (spread returning to mean)
            elif z < self.exit_z_score and z > 0:
                current_pos = self.current_positions.get(asset, 0)
                if current_pos < 0:
                    signal = 1  # Close short spread
                    strength = abs(z) / self.exit_z_score
                    reason = f"Spread z-score={z:.2f}, reverting to mean, exit short spread"

            # Only add signal if it's not a hold
            if signal != 0:
                # Decompose spread signal into individual asset signals
                asset_signals = self.decompose_spread_signal(signal, hedge)

                signals.append({
                    'timestamp': timestamp,
                    'asset': asset,
                    'signal': signal,
                    'strength': strength,
                    'reason': reason,
                    'z_score': z,
                    'spread': spread.iloc[idx],
                    'spread_mean': spread_mean.iloc[idx],
                    'spread_std': spread_std.iloc[idx],
                    'hedge_ratio': hedge,
                    'correlation': corr,
                    'asset_1_signal': asset_signals['asset_1_signal'],
                    'asset_2_signal': asset_signals['asset_2_signal']
                })

        # Create signals DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            self.logger.info(
                f"Generated {len(signals_df)} spread trading signals "
                f"({(signals_df['signal'] == 1).sum()} buy spread, "
                f"{(signals_df['signal'] == -1).sum()} sell spread)"
            )
        else:
            # No signals generated
            signals_df = pd.DataFrame(columns=[
                'timestamp', 'asset', 'signal', 'strength', 'reason',
                'z_score', 'spread', 'spread_mean', 'spread_std',
                'hedge_ratio', 'correlation', 'asset_1_signal', 'asset_2_signal'
            ])
            self.logger.info("No spread trading signals generated")

        return signals_df

    def get_required_columns(self) -> List[str]:
        """
        Return list of required data columns.

        Returns:
            List with 'price_1' and 'price_2'
        """
        return ['price_1', 'price_2']

    def calculate_spread(
        self,
        price_1: pd.Series,
        price_2: pd.Series,
        spread_type: str
    ) -> pd.Series:
        """
        Calculate spread between two price series.

        Args:
            price_1: First price series
            price_2: Second price series
            spread_type: Type of spread ('difference' or 'ratio')

        Returns:
            Spread series
        """
        if spread_type == 'difference':
            spread = price_1 - price_2
        elif spread_type == 'ratio':
            spread = price_1 / price_2
        else:
            raise ValueError(f"Unknown spread_type: {spread_type}")

        return spread

    def calculate_hedge_ratio(
        self,
        price_1: pd.Series,
        price_2: pd.Series
    ) -> pd.Series:
        """
        Calculate optimal hedge ratio for spread using rolling OLS regression.

        The hedge ratio determines how many units of asset 2 to trade per unit
        of asset 1 to create a hedged spread position.

        Args:
            price_1: First price series
            price_2: Second price series

        Returns:
            Hedge ratio series (beta from regressing price_1 on price_2)
        """
        # Use rolling window for dynamic hedge ratio
        window = self.hedge_ratio_window

        # Calculate rolling covariance and variance
        cov = price_1.rolling(window=window).cov(price_2)
        var = price_2.rolling(window=window).var()

        # Hedge ratio = cov(p1, p2) / var(p2)
        hedge_ratio = cov / var

        # Fill NaN values with 1.0 (equal weights)
        hedge_ratio = hedge_ratio.fillna(1.0)

        # Clip to reasonable range
        hedge_ratio = hedge_ratio.clip(0.1, 10.0)

        self.logger.debug(
            f"Calculated hedge ratio: mean={hedge_ratio.mean():.2f}, "
            f"std={hedge_ratio.std():.2f}"
        )

        return hedge_ratio

    def decompose_spread_signal(
        self,
        spread_signal: int,
        hedge_ratio: float
    ) -> Dict[str, float]:
        """
        Convert spread signal to individual asset signals.

        A spread position consists of:
        - Long spread: buy 1 unit of asset 1, sell hedge_ratio units of asset 2
        - Short spread: sell 1 unit of asset 1, buy hedge_ratio units of asset 2

        Args:
            spread_signal: Spread signal (1 = buy spread, -1 = sell spread)
            hedge_ratio: Hedge ratio for the spread

        Returns:
            Dictionary with 'asset_1_signal' and 'asset_2_signal'
        """
        if spread_signal == 1:
            # Buy spread: long asset 1, short asset 2
            asset_1_signal = 1
            asset_2_signal = -hedge_ratio
        elif spread_signal == -1:
            # Sell spread: short asset 1, long asset 2
            asset_1_signal = -1
            asset_2_signal = hedge_ratio
        else:
            asset_1_signal = 0
            asset_2_signal = 0

        return {
            'asset_1_signal': asset_1_signal,
            'asset_2_signal': asset_2_signal
        }
