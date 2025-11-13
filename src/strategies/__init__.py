"""
Trading Strategies Module

This module provides a comprehensive framework for implementing and managing
trading strategies for energy markets. It includes:

- BaseStrategy: Abstract base class defining the common interface
- MeanReversionStrategy: Bollinger Bands-based mean reversion strategy
- MomentumStrategy: Moving average crossover momentum strategy
- SpreadTradingStrategy: Inter-market spread arbitrage strategy
- RenewableArbitrageStrategy: Renewable generation-aware arbitrage strategy
- StrategyManager: Utility for loading and managing multiple strategies

All strategies support:
- Standardized signal generation (buy/sell/hold with strength)
- Risk management (position sizing, stop-loss, take-profit)
- Integration with backtesting engine
- Configuration via strategies.yaml

Example:
    >>> from src.strategies import MeanReversionStrategy
    >>> strategy = MeanReversionStrategy()
    >>> signals = strategy.generate_signals(price_data)

    >>> from src.strategies import StrategyManager
    >>> manager = StrategyManager()
    >>> manager.load_strategies(['mean_reversion', 'momentum'])
    >>> combined_signals = manager.generate_combined_signals(data)
"""

from src.strategies.base_strategy import BaseStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.spread_trading import SpreadTradingStrategy
from src.strategies.renewable_arbitrage import RenewableArbitrageStrategy
from src.strategies.strategy_loader import (
    StrategyManager,
    load_strategy_config,
    create_strategy
)

__all__ = [
    'BaseStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'SpreadTradingStrategy',
    'RenewableArbitrageStrategy',
    'StrategyManager',
    'load_strategy_config',
    'create_strategy'
]
