"""
Base Strategy Module

Provides the abstract base class for all trading strategies. Defines the common
interface for signal generation, position sizing, and risk management.

All concrete strategies must inherit from BaseStrategy and implement:
- generate_signals(): Generate trading signals from market data
- get_required_columns(): Return list of required data columns

The base class provides common functionality:
- Position sizing (fixed fractional and Kelly Criterion)
- Risk management (stop-loss, take-profit, position limits)
- Position tracking and state persistence
- Data validation

Example:
    >>> class MyStrategy(BaseStrategy):
    ...     def __init__(self, config=None, strategy_config=None):
    ...         super().__init__(name='my_strategy', config=config, strategy_config=strategy_config)
    ...
    ...     def generate_signals(self, data, **kwargs):
    ...         # Implement signal generation logic
    ...         signals = pd.DataFrame(...)
    ...         return signals
    ...
    ...     def get_required_columns(self):
    ...         return ['price', 'volume']
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config.load_config import get_config


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Provides common interface and functionality for all strategies including
    signal generation, position sizing, risk management, and state tracking.

    Attributes:
        name: Strategy name
        config: Global configuration dictionary
        strategy_config: Strategy-specific configuration
        max_position_size: Maximum position size as fraction of portfolio
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        current_positions: Dictionary tracking current positions
        entry_prices: Dictionary tracking entry prices for positions
        logger: Logger instance
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict] = None,
        strategy_config: Optional[Dict] = None
    ):
        """
        Initialize base strategy.

        Args:
            name: Strategy name (e.g., 'mean_reversion')
            config: Optional global configuration dictionary
            strategy_config: Optional strategy-specific configuration from strategies.yaml
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Load global configuration
        if config is None:
            config = get_config()
        self.config = config

        # Load strategy-specific configuration
        if strategy_config is None:
            strategy_config = {}
        self.strategy_config = strategy_config

        # Load risk management parameters with proper precedence:
        # 1. strategy_config['risk'] (includes merged global.risk from strategies.yaml)
        # 2. Legacy config.yaml strategies.risk (backward compatibility)
        # 3. Defaults
        risk_from_sc = self.strategy_config.get('risk', {})
        risk_from_legacy = self.config.get('strategies', {}).get('risk', {})
        # Merge with strategy_config taking precedence
        risk_config = {**risk_from_legacy, **risk_from_sc}

        # Set risk attributes with backward compatibility for stop_loss/stop_loss_pct names
        self.max_position_size = risk_config.get('max_position_size', 0.2)
        self.stop_loss_pct = risk_config.get('stop_loss_pct', risk_config.get('stop_loss', 0.05))
        self.take_profit_pct = risk_config.get('take_profit_pct', risk_config.get('take_profit', 0.10))

        # Load position sizing method from strategy config (merged with global)
        # This allows strategies.yaml global.position_sizing_method to be used
        self.position_sizing_method = self.strategy_config.get('position_sizing_method', 'fixed_fractional')

        # Apply backward-compatible top-level overrides from strategy_config
        # (allows ad-hoc per-strategy overrides outside nested 'risk')
        if 'max_position_size' in self.strategy_config:
            self.max_position_size = self.strategy_config['max_position_size']
        if 'stop_loss_pct' in self.strategy_config or 'stop_loss' in self.strategy_config:
            self.stop_loss_pct = self.strategy_config.get('stop_loss_pct', self.strategy_config.get('stop_loss', self.stop_loss_pct))
        if 'take_profit_pct' in self.strategy_config or 'take_profit' in self.strategy_config:
            self.take_profit_pct = self.strategy_config.get('take_profit_pct', self.strategy_config.get('take_profit', self.take_profit_pct))

        # Initialize position tracking
        self.current_positions = {}  # {asset: position_size}
        self.entry_prices = {}  # {asset: entry_price}

        # Strategy metadata
        self.metadata = {
            'name': self.name,
            'description': self.__class__.__doc__,
            'parameters': self.strategy_config
        }

        self.logger.info(
            f"BaseStrategy '{name}' initialized: "
            f"max_position_size={self.max_position_size:.1%}, "
            f"stop_loss_pct={self.stop_loss_pct:.1%}, "
            f"take_profit_pct={self.take_profit_pct:.1%}, "
            f"position_sizing_method={self.position_sizing_method}"
        )

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals from market data.

        Must be implemented by concrete strategies.

        Args:
            data: DataFrame with market data (DatetimeIndex, price/OHLC columns)
            **kwargs: Additional strategy-specific parameters

        Returns:
            DataFrame with columns:
                - timestamp: Signal timestamp
                - signal: 1 (buy), -1 (sell), 0 (hold)
                - strength: Signal confidence (0-1)
                - reason: Description of why signal was generated

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Return list of required data columns for this strategy.

        Must be implemented by concrete strategies.

        Returns:
            List of column names required in input data

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_required_columns()")

    def calculate_position_size(
        self,
        signal_strength: float,
        account_value: float,
        current_price: float,
        method: Optional[str] = None
    ) -> float:
        """
        Calculate position size based on signal strength and risk parameters.

        Args:
            signal_strength: Signal confidence (0-1)
            account_value: Total account value
            current_price: Current asset price
            method: Position sizing method ('fixed_fractional' or 'kelly').
                   If None, uses self.position_sizing_method from config.

        Returns:
            Position size (number of units to trade)
        """
        # Use configured default method if not specified
        if method is None:
            method = self.position_sizing_method

        if method == 'fixed_fractional':
            # Position size = (account_value * max_position_size * signal_strength) / price
            position_value = account_value * self.max_position_size * signal_strength
            position_size = position_value / current_price

            self.logger.debug(
                f"Fixed fractional sizing: account=${account_value:.2f}, "
                f"max_position={self.max_position_size:.1%}, strength={signal_strength:.2f}, "
                f"price=${current_price:.2f} → size={position_size:.2f} units"
            )

        elif method == 'kelly':
            # Kelly Criterion position sizing
            kelly_fraction = self.calculate_kelly_fraction()
            position_value = account_value * kelly_fraction * signal_strength
            position_size = position_value / current_price

            self.logger.debug(
                f"Kelly sizing: kelly_fraction={kelly_fraction:.2%}, "
                f"strength={signal_strength:.2f} → size={position_size:.2f} units"
            )

        else:
            raise ValueError(f"Unknown position sizing method: {method}")

        # Ensure position size respects max_position_size constraint
        max_units = (account_value * self.max_position_size) / current_price
        position_size = min(position_size, max_units)

        return position_size

    def calculate_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        fractional_kelly: Optional[float] = None
    ) -> float:
        """
        Calculate Kelly Criterion fraction for position sizing.

        Args:
            win_rate: Historical win rate (e.g., 0.55 for 55%)
            avg_win_loss_ratio: Average win / average loss (e.g., 1.5)
            fractional_kelly: Fraction of Kelly to use (e.g., 0.5 for half-Kelly)

        Returns:
            Kelly fraction (capped at max_position_size)
        """
        # Load defaults from config if not provided
        # Read from strategy_config first (includes merged global.kelly), then fall back to legacy path
        if win_rate is None:
            kelly_config = self.strategy_config.get('kelly', {})
            if not kelly_config:
                # Fallback to legacy config path for backward compatibility
                kelly_config = self.config.get('strategies', {}).get('risk', {}).get('kelly', {})
            win_rate = kelly_config.get('win_rate', 0.55)

        if avg_win_loss_ratio is None:
            kelly_config = self.strategy_config.get('kelly', {})
            if not kelly_config:
                # Fallback to legacy config path for backward compatibility
                kelly_config = self.config.get('strategies', {}).get('risk', {}).get('kelly', {})
            avg_win_loss_ratio = kelly_config.get('avg_win_loss_ratio', 1.5)

        if fractional_kelly is None:
            kelly_config = self.strategy_config.get('kelly', {})
            if not kelly_config:
                # Fallback to legacy config path for backward compatibility
                kelly_config = self.config.get('strategies', {}).get('risk', {}).get('kelly', {})
            fractional_kelly = kelly_config.get('fractional_kelly', 0.5)

        # Kelly formula: (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        if avg_win_loss_ratio <= 0:
            self.logger.warning("Invalid avg_win_loss_ratio, defaulting to 0")
            return 0.0

        kelly = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio

        # Apply fractional Kelly for safety
        kelly = kelly * fractional_kelly

        # Clip to reasonable range
        kelly = np.clip(kelly, 0, self.max_position_size)

        self.logger.debug(
            f"Kelly calculation: win_rate={win_rate:.1%}, "
            f"win_loss_ratio={avg_win_loss_ratio:.2f}, "
            f"fractional={fractional_kelly:.1%} → kelly={kelly:.2%}"
        )

        return kelly

    def apply_risk_limits(
        self,
        signals: pd.DataFrame,
        current_positions: Optional[Dict] = None,
        account_value: float = 100000.0,
        current_prices: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Apply risk management rules to signals.

        Args:
            signals: DataFrame with trading signals
            current_positions: Dictionary of current positions {asset: size}
            account_value: Total account value
            current_prices: Dictionary of current prices {asset: price}

        Returns:
            Filtered signals DataFrame with risk limits applied
        """
        if current_positions is None:
            current_positions = self.current_positions

        if current_prices is None:
            current_prices = {}

        filtered_signals = signals.copy()
        filtered_signals['risk_adjusted'] = False

        # Check stop-loss and take-profit for existing positions
        exit_signals = []

        for asset, position_size in current_positions.items():
            if position_size == 0:
                continue

            entry_price = self.entry_prices.get(asset, 0)
            current_price = current_prices.get(asset, 0)

            if entry_price == 0 or current_price == 0:
                continue

            # Calculate P&L percentage
            pnl_pct = (current_price - entry_price) / entry_price

            # Check stop-loss (for long positions)
            if position_size > 0 and pnl_pct <= -self.stop_loss_pct:
                self.logger.warning(
                    f"Stop-loss triggered for {asset}: "
                    f"entry=${entry_price:.2f}, current=${current_price:.2f}, "
                    f"loss={pnl_pct:.1%}"
                )
                exit_signals.append({
                    'timestamp': pd.Timestamp.now(),
                    'asset': asset,
                    'signal': -1,  # Sell to exit long
                    'strength': 1.0,
                    'reason': f'Stop-loss triggered: {pnl_pct:.1%} loss',
                    'risk_adjusted': True
                })

            # Check stop-loss (for short positions)
            elif position_size < 0 and pnl_pct >= self.stop_loss_pct:
                self.logger.warning(
                    f"Stop-loss triggered for {asset} short: "
                    f"entry=${entry_price:.2f}, current=${current_price:.2f}, "
                    f"loss={pnl_pct:.1%}"
                )
                exit_signals.append({
                    'timestamp': pd.Timestamp.now(),
                    'asset': asset,
                    'signal': 1,  # Buy to exit short
                    'strength': 1.0,
                    'reason': f'Stop-loss triggered: {pnl_pct:.1%} loss',
                    'risk_adjusted': True
                })

            # Check take-profit (for long positions)
            elif position_size > 0 and pnl_pct >= self.take_profit_pct:
                self.logger.info(
                    f"Take-profit triggered for {asset}: "
                    f"entry=${entry_price:.2f}, current=${current_price:.2f}, "
                    f"profit={pnl_pct:.1%}"
                )
                exit_signals.append({
                    'timestamp': pd.Timestamp.now(),
                    'asset': asset,
                    'signal': -1,  # Sell to take profit
                    'strength': 1.0,
                    'reason': f'Take-profit triggered: {pnl_pct:.1%} profit',
                    'risk_adjusted': True
                })

            # Check take-profit (for short positions)
            elif position_size < 0 and pnl_pct <= -self.take_profit_pct:
                self.logger.info(
                    f"Take-profit triggered for {asset} short: "
                    f"entry=${entry_price:.2f}, current=${current_price:.2f}, "
                    f"profit={-pnl_pct:.1%}"
                )
                exit_signals.append({
                    'timestamp': pd.Timestamp.now(),
                    'asset': asset,
                    'signal': 1,  # Buy to take profit
                    'strength': 1.0,
                    'reason': f'Take-profit triggered: {-pnl_pct:.1%} profit',
                    'risk_adjusted': True
                })

        # Add exit signals to filtered signals
        if exit_signals:
            exit_df = pd.DataFrame(exit_signals)
            filtered_signals = pd.concat([filtered_signals, exit_df], ignore_index=True)

        # Check max position size constraint for new signals
        for idx, row in filtered_signals.iterrows():
            if row.get('risk_adjusted', False):
                continue  # Already a risk-adjusted signal

            asset = row.get('asset', 'default')
            signal = row['signal']

            if signal == 0:
                continue  # Hold signal, no position change

            # Check if new position would exceed max_position_size
            # Calculate current position value
            current_pos = current_positions.get(asset, 0)
            current_price = current_prices.get(asset, 0)

            if current_price > 0:
                current_pos_value = abs(current_pos * current_price)
                max_pos_value = self.max_position_size * account_value

                # Reject signals that would increase already-large positions
                if current_pos_value >= max_pos_value:
                    if (current_pos > 0 and signal > 0) or (current_pos < 0 and signal < 0):
                        self.logger.warning(
                            f"Signal rejected for {asset}: current position value ${current_pos_value:.2f} "
                            f">= max ${max_pos_value:.2f}"
                        )
                        filtered_signals.at[idx, 'signal'] = 0
                        filtered_signals.at[idx, 'risk_adjusted'] = True
                        filtered_signals.at[idx, 'reason'] += ' (REJECTED: max position)'\


        n_modified = filtered_signals['risk_adjusted'].sum()
        if n_modified > 0:
            self.logger.info(f"Risk limits modified {n_modified} signals")

        return filtered_signals

    def update_positions(
        self,
        asset: str,
        position_change: float,
        execution_price: float
    ):
        """
        Update internal position tracking after trade execution.

        Args:
            asset: Asset identifier
            position_change: Change in position size (positive for buy, negative for sell)
            execution_price: Price at which trade was executed
        """
        # Update position
        current_pos = self.current_positions.get(asset, 0)
        new_pos = current_pos + position_change
        self.current_positions[asset] = new_pos

        # Update entry price
        if new_pos == 0:
            # Position closed, remove entry price
            self.entry_prices.pop(asset, None)
        elif abs(new_pos) > abs(current_pos):
            # Position increased, update entry price (weighted average)
            if asset in self.entry_prices:
                old_entry = self.entry_prices[asset]
                weighted_entry = (
                    (abs(current_pos) * old_entry + abs(position_change) * execution_price) /
                    abs(new_pos)
                )
                self.entry_prices[asset] = weighted_entry
            else:
                self.entry_prices[asset] = execution_price

        self.logger.info(
            f"Position updated: {asset} "
            f"{current_pos:+.2f} → {new_pos:+.2f} "
            f"(change={position_change:+.2f} @ ${execution_price:.2f})"
        )

    def get_strategy_state(self) -> Dict:
        """
        Return current strategy state for persistence.

        Returns:
            Dictionary with current_positions, entry_prices, parameters, metadata
        """
        return {
            'name': self.name,
            'current_positions': self.current_positions.copy(),
            'entry_prices': self.entry_prices.copy(),
            'parameters': self.strategy_config.copy(),
            'metadata': self.metadata.copy()
        }

    def load_strategy_state(self, state: Dict):
        """
        Load previously saved strategy state.

        Args:
            state: Dictionary with strategy state (from get_strategy_state())
        """
        self.current_positions = state.get('current_positions', {}).copy()
        self.entry_prices = state.get('entry_prices', {}).copy()

        # Update parameters if provided
        if 'parameters' in state:
            self.strategy_config.update(state['parameters'])

        self.logger.info(
            f"Strategy state loaded: {len(self.current_positions)} positions"
        )

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input data format and quality.

        Args:
            data: DataFrame to validate

        Returns:
            Validated (and potentially cleaned) DataFrame

        Raises:
            ValueError: If data fails validation
        """
        # Check DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        # Check required columns, with special handling for 'price' vs 'close'
        required_cols = self.get_required_columns()
        missing_cols = []

        for col in required_cols:
            # Allow 'price' or 'close' interchangeably
            if col == 'price':
                if 'price' not in data.columns and 'close' not in data.columns:
                    missing_cols.append(col)
            elif col not in data.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for excessive NaN values
        for col in required_cols:
            # Handle price/close alternatives
            check_col = col
            if col == 'price' and 'price' not in data.columns and 'close' in data.columns:
                check_col = 'close'

            if check_col in data.columns:
                nan_pct = data[check_col].isna().sum() / len(data)
                if nan_pct > 0.5:
                    raise ValueError(
                        f"Column '{check_col}' has {nan_pct:.1%} NaN values (>50% threshold)"
                    )

        # Forward-fill small gaps (max 3 periods)
        validated_data = data.copy()
        for col in required_cols:
            # Handle price/close alternatives
            check_col = col
            if col == 'price' and 'price' not in data.columns and 'close' in data.columns:
                check_col = 'close'

            if check_col in validated_data.columns and validated_data[check_col].isna().any():
                validated_data[check_col] = validated_data[check_col].fillna(method='ffill', limit=3)
                remaining_nans = validated_data[check_col].isna().sum()
                if remaining_nans > 0:
                    self.logger.warning(
                        f"Column '{check_col}' still has {remaining_nans} NaN values after forward-fill"
                    )

        # Check minimum data length
        if len(validated_data) < 10:
            raise ValueError(f"Insufficient data: {len(validated_data)} rows (minimum 10)")

        self.logger.debug(
            f"Data validated: {len(validated_data)} rows, "
            f"{len(validated_data.columns)} columns"
        )

        return validated_data
