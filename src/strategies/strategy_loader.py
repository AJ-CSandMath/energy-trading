"""
Strategy Loader Module

Utilities for loading and managing multiple trading strategies from configuration.

Provides:
- load_strategy_config(): Load strategies.yaml configuration
- create_strategy(): Factory function to instantiate strategies by name
- StrategyManager: Orchestrate multiple strategies, combine signals, apply portfolio-level risk

Example:
    >>> from src.strategies.strategy_loader import StrategyManager
    >>>
    >>> # Initialize manager
    >>> manager = StrategyManager()
    >>> manager.load_strategies(['mean_reversion', 'momentum', 'renewable_arbitrage'])
    >>>
    >>> # Generate combined signals
    >>> data = {
    >>>     'prices': price_df,
    >>>     'wind': wind_df,
    >>>     'solar': solar_df
    >>> }
    >>> signals = manager.generate_combined_signals(data)
    >>>
    >>> # Get specific strategy
    >>> mean_rev = manager.get_strategy('mean_reversion')
    >>> mean_rev_signals = mean_rev.generate_signals(price_df)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import copy
import yaml
import numpy as np
import pandas as pd

from src.config.load_config import get_config
from src.strategies.base_strategy import BaseStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.spread_trading import SpreadTradingStrategy
from src.strategies.renewable_arbitrage import RenewableArbitrageStrategy


def deep_merge_config(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two configuration dictionaries.

    Values in override take precedence over base. Nested dictionaries are
    recursively merged rather than replaced.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge_config(result[key], value)
        else:
            # Override value
            result[key] = copy.deepcopy(value)

    return result


def load_strategy_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load strategies.yaml configuration.

    Args:
        config_path: Optional path to strategies.yaml (default: config/strategies.yaml)

    Returns:
        Dictionary with strategy configuration

    Raises:
        FileNotFoundError: If config file not found and no defaults available
    """
    logger = logging.getLogger(__name__)

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'strategies.yaml'

    try:
        with open(config_path, 'r') as f:
            strategy_config = yaml.safe_load(f)

        logger.info(f"Loaded strategy configuration from {config_path}")
        return strategy_config

    except FileNotFoundError:
        logger.warning(f"Strategy config file not found: {config_path}, using defaults")
        # Return minimal defaults
        return {
            'global': {
                'position_sizing_method': 'fixed_fractional',
                'risk': {
                    'max_position_size': 0.2,
                    'stop_loss': 0.05,
                    'take_profit': 0.10
                }
            }
        }


def create_strategy(strategy_name: str, config: Optional[Dict] = None) -> BaseStrategy:
    """
    Factory function to create strategy instance by name.

    Args:
        strategy_name: Name of strategy ('mean_reversion', 'momentum', etc.)
        config: Optional configuration dictionary

    Returns:
        Instantiated strategy object

    Raises:
        ValueError: If strategy_name is unknown
    """
    logger = logging.getLogger(__name__)

    # Load strategy config
    strategy_config_all = load_strategy_config()

    # Get global config from strategies.yaml
    global_config = strategy_config_all.get('global', {})

    # Get strategy-specific config
    strategy_config_raw = strategy_config_all.get(strategy_name, {})

    # Deep merge global config with strategy-specific config
    # Strategy-specific settings override global defaults
    strategy_config = deep_merge_config(global_config, strategy_config_raw)

    # Map strategy names to classes
    strategy_classes = {
        'mean_reversion': MeanReversionStrategy,
        'momentum': MomentumStrategy,
        'spread_trading': SpreadTradingStrategy,
        'renewable_arbitrage': RenewableArbitrageStrategy
    }

    if strategy_name not in strategy_classes:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available strategies: {list(strategy_classes.keys())}"
        )

    # Instantiate strategy
    strategy_class = strategy_classes[strategy_name]
    strategy = strategy_class(config=config, strategy_config=strategy_config)

    logger.info(f"Created strategy: {strategy_name}")

    return strategy


class StrategyManager:
    """
    Manage multiple trading strategies and combine their signals.

    Handles strategy lifecycle, signal combination, portfolio-level risk management,
    and state persistence.

    Attributes:
        config: Global configuration dictionary
        strategy_config: Strategy-specific configuration from strategies.yaml
        strategies: Dictionary of loaded strategies {name: strategy_instance}
        logger: Logger instance
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        strategy_config_path: Optional[Path] = None
    ):
        """
        Initialize strategy manager.

        Args:
            config: Optional global configuration dictionary
            strategy_config_path: Optional path to strategies.yaml
        """
        self.logger = logging.getLogger(__name__)

        # Load global config
        if config is None:
            config = get_config()
        self.config = config

        # Load strategy config
        self.strategy_config = load_strategy_config(strategy_config_path)

        # Storage for loaded strategies
        self.strategies = {}

        # Extract portfolio config
        self.portfolio_config = self.strategy_config.get('portfolio', {})
        self.allow_multiple_strategies = self.portfolio_config.get('allow_multiple_strategies', True)
        self.strategy_weights = self.portfolio_config.get('strategy_weights', {})
        self.conflict_resolution = self.portfolio_config.get('conflict_resolution', 'weighted_average')
        self.max_total_exposure = self.portfolio_config.get('max_total_exposure', 0.8)
        self.max_strategies_per_asset = self.portfolio_config.get('max_strategies_per_asset', 2)

        self.logger.info("StrategyManager initialized")

    def load_strategies(
        self,
        strategy_names: Optional[List[str]] = None
    ) -> 'StrategyManager':
        """
        Load strategies from configuration.

        Args:
            strategy_names: Optional list of strategy names to load
                          (default: from config['strategies']['default_strategies'])

        Returns:
            Self for method chaining
        """
        if strategy_names is None:
            # Load default strategies from config
            strategy_names = self.config.get('strategies', {}).get('default_strategies', [])

        for name in strategy_names:
            try:
                # Check if enabled in config
                strategy_cfg = self.strategy_config.get(name, {})
                if not strategy_cfg.get('enabled', True):
                    self.logger.info(f"Strategy '{name}' is disabled, skipping")
                    continue

                # Create strategy instance
                strategy = create_strategy(name, self.config)
                self.strategies[name] = strategy

                self.logger.info(f"Loaded strategy: {name}")

            except Exception as e:
                self.logger.error(f"Failed to load strategy '{name}': {e}")

        self.logger.info(f"Loaded {len(self.strategies)} strategies: {list(self.strategies.keys())}")

        return self

    def generate_combined_signals(
        self,
        data: Dict[str, pd.DataFrame],
        account_value: float = 100000.0,
        current_prices: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate signals from all strategies and combine them.

        Args:
            data: Dictionary with data for each strategy
                  e.g., {'prices': price_df, 'wind': wind_df, 'solar': solar_df}
            account_value: Total account value for position sizing
            current_prices: Optional dict of current prices {asset: price}

        Returns:
            Combined signals DataFrame
        """
        all_signals = []

        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                self.logger.info(f"Generating signals from {strategy_name}")

                # Determine which data to pass based on strategy type
                if strategy_name == 'renewable_arbitrage':
                    signals = strategy.generate_signals(
                        data=data.get('prices', pd.DataFrame()),
                        wind_data=data.get('wind'),
                        solar_data=data.get('solar')
                    )
                elif strategy_name == 'spread_trading':
                    # Spread trading needs price_1 and price_2
                    # Only use 'spread' data, no fallback to avoid invalid data
                    spread_data = data.get('spread')
                    if spread_data is None or spread_data.empty:
                        self.logger.info(
                            f"{strategy_name} requires 'spread' data with price_1 and price_2 columns, skipping"
                        )
                        continue

                    # Validate required columns are present
                    if not {'price_1', 'price_2'}.issubset(set(spread_data.columns)):
                        self.logger.warning(
                            f"{strategy_name} requires 'price_1' and 'price_2' columns; skipping"
                        )
                        continue

                    signals = strategy.generate_signals(spread_data)
                else:
                    # Mean reversion and momentum use price data
                    signals = strategy.generate_signals(data.get('prices', pd.DataFrame()))

                if not signals.empty:
                    # Add strategy name to signals
                    signals['strategy'] = strategy_name

                    # Add to collection
                    all_signals.append(signals)

                    self.logger.info(
                        f"{strategy_name} generated {len(signals)} signals"
                    )

            except Exception as e:
                self.logger.error(f"Signal generation failed for {strategy_name}: {e}")

        # Combine signals
        if not all_signals:
            self.logger.warning("No signals generated from any strategy")
            return pd.DataFrame(columns=['timestamp', 'signal', 'strength', 'reason', 'strategy', 'asset'])

        combined_signals = pd.concat(all_signals, ignore_index=True)

        # Apply conflict resolution if multiple strategies signal same asset
        if self.allow_multiple_strategies:
            combined_signals = self._resolve_conflicts(combined_signals)

        # Apply portfolio-level risk limits
        combined_signals = self.apply_portfolio_risk_limits(
            combined_signals,
            account_value=account_value,
            current_prices=current_prices
        )

        self.logger.info(
            f"Combined signals: {len(combined_signals)} total "
            f"({(combined_signals['signal'] == 1).sum()} buy, "
            f"{(combined_signals['signal'] == -1).sum()} sell)"
        )

        return combined_signals

    def _resolve_conflicts(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve conflicts when multiple strategies signal same asset/timestamp.

        Args:
            signals: DataFrame with signals from multiple strategies

        Returns:
            Resolved signals DataFrame
        """
        if 'asset' not in signals.columns:
            # No asset column, can't detect conflicts
            return signals

        # Group by timestamp and asset
        grouped = signals.groupby(['timestamp', 'asset'])

        resolved_signals = []

        for (timestamp, asset), group in grouped:
            if len(group) == 1:
                # No conflict, keep as is
                resolved_signals.append(group.iloc[0].to_dict())
                continue

            # Multiple strategies for same asset/timestamp
            if self.conflict_resolution == 'weighted_average':
                # Weighted average of signals
                total_weight = 0
                weighted_signal = 0
                weighted_strength = 0

                for _, row in group.iterrows():
                    strategy_name = row['strategy']
                    weight = self.strategy_weights.get(strategy_name, 1.0 / len(self.strategies))
                    weighted_signal += row['signal'] * weight
                    weighted_strength += row['strength'] * weight
                    total_weight += weight

                # Normalize
                final_signal = int(np.sign(weighted_signal))
                final_strength = weighted_strength / total_weight if total_weight > 0 else 0

                resolved_signals.append({
                    'timestamp': timestamp,
                    'asset': asset,
                    'signal': final_signal,
                    'strength': final_strength,
                    'reason': f"Combined from {len(group)} strategies",
                    'strategy': 'combined'
                })

            elif self.conflict_resolution == 'strongest_signal':
                # Take signal with highest strength
                strongest = group.loc[group['strength'].idxmax()]
                resolved_signals.append(strongest.to_dict())

            elif self.conflict_resolution == 'no_trade':
                # If signals conflict, don't trade
                signals_set = set(group['signal'])
                if len(signals_set) > 1 and 0 not in signals_set:
                    # Conflicting signals (buy and sell), ignore
                    self.logger.debug(
                        f"Conflicting signals for {asset} at {timestamp}, ignoring"
                    )
                else:
                    # All agree, take first one
                    resolved_signals.append(group.iloc[0].to_dict())

        return pd.DataFrame(resolved_signals)

    def apply_portfolio_risk_limits(
        self,
        signals: pd.DataFrame,
        account_value: float = 100000.0,
        current_prices: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Apply portfolio-level risk constraints.

        Args:
            signals: Combined signals DataFrame
            account_value: Total account value
            current_prices: Optional dict of current prices

        Returns:
            Filtered signals DataFrame
        """
        if current_prices is None:
            current_prices = {}

        filtered_signals = signals.copy()

        # Check total exposure
        total_position_value = 0
        for strategy in self.strategies.values():
            for asset, pos in strategy.current_positions.items():
                price = current_prices.get(asset, 0)
                total_position_value += abs(pos * price)

        exposure_ratio = total_position_value / account_value

        if exposure_ratio >= self.max_total_exposure:
            self.logger.warning(
                f"Portfolio exposure {exposure_ratio:.1%} >= max {self.max_total_exposure:.1%}, "
                f"rejecting new entry signals"
            )
            # Allow exit signals, including stop-loss/take-profit (risk_adjusted)
            # Check for risk_adjusted flag or 'exit' keyword in reason
            is_exit_signal = (
                (filtered_signals['signal'] == 0) |  # Hold signals
                (filtered_signals.get('risk_adjusted', False)) |  # Stop-loss/take-profit
                ((filtered_signals['signal'] != 0) & (filtered_signals['reason'].str.contains('exit', case=False, na=False)))  # Exit signals
            )
            filtered_signals = filtered_signals[is_exit_signal]

        # Check max strategies per asset
        if 'asset' in filtered_signals.columns:
            asset_strategy_counts = filtered_signals.groupby('asset')['strategy'].nunique()
            for asset, count in asset_strategy_counts.items():
                if count > self.max_strategies_per_asset:
                    self.logger.warning(
                        f"Asset {asset} has {count} strategies (max {self.max_strategies_per_asset}), "
                        f"filtering"
                    )
                    # Keep only top strategies by strength
                    asset_signals = filtered_signals[filtered_signals['asset'] == asset]
                    top_signals = asset_signals.nlargest(self.max_strategies_per_asset, 'strength')
                    # Remove others
                    filtered_signals = filtered_signals[
                        (filtered_signals['asset'] != asset) |
                        (filtered_signals.index.isin(top_signals.index))
                    ]

        return filtered_signals

    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """
        Get specific strategy instance.

        Args:
            strategy_name: Name of strategy

        Returns:
            Strategy instance or None if not found
        """
        return self.strategies.get(strategy_name)

    def list_strategies(self) -> List[str]:
        """
        List all loaded strategies.

        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())

    def enable_strategy(self, strategy_name: str):
        """
        Enable a specific strategy.

        Args:
            strategy_name: Name of strategy to enable
        """
        if strategy_name in self.strategies:
            self.logger.info(f"Strategy '{strategy_name}' already loaded")
        else:
            try:
                strategy = create_strategy(strategy_name, self.config)
                self.strategies[strategy_name] = strategy
                self.logger.info(f"Enabled strategy: {strategy_name}")
            except Exception as e:
                self.logger.error(f"Failed to enable strategy '{strategy_name}': {e}")

    def disable_strategy(self, strategy_name: str):
        """
        Disable a specific strategy.

        Args:
            strategy_name: Name of strategy to disable
        """
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Disabled strategy: {strategy_name}")
        else:
            self.logger.warning(f"Strategy '{strategy_name}' not loaded")

    def get_strategy_states(self) -> Dict:
        """
        Get state of all strategies for persistence.

        Returns:
            Dictionary mapping strategy_name → state dict
        """
        states = {}
        for name, strategy in self.strategies.items():
            states[name] = strategy.get_strategy_state()

        return states

    def load_strategy_states(self, states: Dict):
        """
        Load previously saved strategy states.

        Args:
            states: Dictionary mapping strategy_name → state dict
        """
        for name, state in states.items():
            if name in self.strategies:
                self.strategies[name].load_strategy_state(state)
                self.logger.info(f"Loaded state for strategy: {name}")
            else:
                self.logger.warning(f"Strategy '{name}' not loaded, skipping state restore")
