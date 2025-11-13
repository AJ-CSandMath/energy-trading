"""
Backtesting Engine Module

Implements event-driven backtesting engine for strategy evaluation with realistic
transaction costs, slippage, and execution delays.

The engine simulates portfolio management by:
- Processing signals chronologically (event-driven)
- Executing orders with transaction costs and market impact
- Tracking portfolio state (cash, positions, equity)
- Logging all trades for analysis
- Calculating performance metrics

Example:
    >>> from src.backtesting import BacktestEngine
    >>> from src.strategies import MeanReversionStrategy
    >>> from src.data import SyntheticPriceGenerator
    >>>
    >>> # Generate price data
    >>> gen = SyntheticPriceGenerator()
    >>> prices = gen.generate_price_series('2023-01-01', '2023-12-31', frequency='D')
    >>>
    >>> # Create strategy
    >>> strategy = MeanReversionStrategy()
    >>> signals = strategy.generate_signals(prices, asset='energy')
    >>>
    >>> # Run backtest
    >>> engine = BacktestEngine(strategies=[strategy])
    >>> result = engine.run(price_data=prices, signals=signals)
    >>>
    >>> # Access results
    >>> print(f"Total Return: {result.total_return:.2%}")
    >>> print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    >>> print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config.load_config import get_config
from src.strategies.base_strategy import BaseStrategy
from src.data.data_manager import DataManager


@dataclass
class BacktestResult:
    """
    Dataclass encapsulating backtest results.

    Attributes:
        equity_curve: Time series of portfolio value over time
        trades: DataFrame with all executed trades
        portfolio_history: DataFrame with positions, cash, equity at each timestamp
        metrics: Dictionary of performance metrics
        config: Backtest configuration used
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        final_equity: Final portfolio value
        total_return: Total return (percentage)
        strategy_names: List of strategy names used
    """
    equity_curve: pd.Series
    trades: pd.DataFrame
    portfolio_history: pd.DataFrame
    metrics: Dict[str, float]
    config: Dict
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    strategy_names: List[str] = field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtesting engine for strategy evaluation.

    Simulates realistic portfolio management with transaction costs, slippage,
    and execution delays. Tracks complete portfolio state and logs all trades.

    Attributes:
        strategies: List of strategy instances or StrategyManager
        config: Configuration dictionary
        data_manager: DataManager for trade persistence
        logger: Logger instance
        cash: Current cash balance
        positions: Current positions dictionary {asset: quantity}
        equity_curve: List of (timestamp, equity) tuples
        trades: List of trade dictionaries
    """

    def __init__(
        self,
        strategies: Union[List[BaseStrategy], BaseStrategy],
        config: Optional[Dict] = None,
        data_manager: Optional[DataManager] = None
    ):
        """
        Initialize backtesting engine.

        Args:
            strategies: List of strategy instances or single strategy
            config: Optional configuration dictionary
            data_manager: Optional DataManager for trade logging
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config is None:
            config = get_config()
        self.config = config

        # Load backtesting config from strategies.yaml
        self.backtest_config = self.config.get('backtesting', {})
        self.initial_capital = self.backtest_config.get('initial_capital', 1000000)
        self.execution_delay = self.backtest_config.get('execution_delay', 1)

        # Trade logging format (default to 'parquet')
        self.default_trade_format = self.backtest_config.get('transaction_log_format', 'parquet')

        # Transaction costs
        transaction_costs = self.backtest_config.get('transaction_costs', {})
        self.fixed_cost = transaction_costs.get('fixed_cost', 0.0)
        self.percentage_cost = transaction_costs.get('percentage_cost', 0.001)
        self.slippage_pct = transaction_costs.get('slippage', 0.0005)

        # Market impact
        market_impact_config = self.backtest_config.get('market_impact', {})
        self.market_impact_enabled = market_impact_config.get('enabled', True)
        self.impact_coefficient = market_impact_config.get('impact_coefficient', 0.1)
        self.max_slippage = market_impact_config.get('max_slippage', 0.05)

        # Initialize DataManager
        if data_manager is None:
            data_manager = DataManager()
        self.data_manager = data_manager

        # Convert single strategy to list
        if isinstance(strategies, BaseStrategy):
            strategies = [strategies]
        self.strategies = strategies
        self.strategy_names = [s.name for s in self.strategies]

        # Initialize portfolio state
        self.cash = self.initial_capital
        self.positions = {}  # {asset: quantity}
        self.equity_curve = []  # [(timestamp, equity), ...]
        self.trades = []  # [trade_dict, ...]
        self.portfolio_history = []  # [{timestamp, positions, cash, equity}, ...]

        self.logger.info(
            f"BacktestEngine initialized: initial_capital=${self.initial_capital:,.0f}, "
            f"execution_delay={self.execution_delay}, strategies={self.strategy_names}"
        )

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_trades: bool = True,
        trade_format: Optional[Union[str, List[str]]] = None
    ) -> BacktestResult:
        """
        Execute backtest simulation.

        Args:
            price_data: DataFrame with DatetimeIndex and price columns
            signals: DataFrame with signals from strategies. Must contain columns:
                     ['timestamp', 'asset', 'signal', 'strength']. When using multiple
                     strategies, 'strategy' column is also required. For single strategy,
                     'strategy' column is added automatically if missing.
            start_date: Optional start date filter
            end_date: Optional end date filter
            save_trades: Whether to save trades to disk
            trade_format: Format for saving trades ('parquet', 'csv', or list of both).
                         If None, uses default from config (transaction_log_format).
                         Examples: 'csv', 'parquet', ['parquet', 'csv']

        Returns:
            BacktestResult with equity curve, trades, metrics

        Raises:
            ValueError: If data validation fails
        """
        self.logger.info("Starting backtest simulation")

        # Validate inputs
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise ValueError("price_data must have DatetimeIndex")

        required_signal_cols = ['timestamp', 'asset', 'signal', 'strength']

        # Add 'strategy' column requirement when multiple strategies are used
        if len(self.strategies) > 1:
            required_signal_cols.append('strategy')

        missing_cols = [col for col in required_signal_cols if col not in signals.columns]
        if missing_cols:
            if 'strategy' in missing_cols and len(self.strategies) == 1:
                # Single strategy case - add strategy column automatically
                signals = signals.copy()
                signals['strategy'] = self.strategies[0].name
                self.logger.info(f"Added 'strategy' column with value '{self.strategies[0].name}'")
                missing_cols.remove('strategy')

            if missing_cols:
                raise ValueError(f"signals missing required columns: {missing_cols}")

        # Normalize and validate signals timestamp
        signals = signals.copy()
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])
        signals = signals.sort_values('timestamp')

        # Ensure 'reason' column exists (required for risk limits)
        if 'reason' not in signals.columns:
            signals['reason'] = ''

        # Optionally check that signal timestamps fall within price date range
        if len(signals) > 0 and len(price_data) > 0:
            price_start = price_data.index.min()
            price_end = price_data.index.max()
            signal_start = signals['timestamp'].min()
            signal_end = signals['timestamp'].max()

            if signal_end < price_start or signal_start > price_end:
                self.logger.warning(
                    f"Signal timestamp range [{signal_start}, {signal_end}] "
                    f"does not overlap with price data range [{price_start}, {price_end}]"
                )

        # Filter by date range
        if start_date:
            price_data = price_data[price_data.index >= pd.to_datetime(start_date)]
            signals = signals[signals['timestamp'] >= pd.to_datetime(start_date)]

        if end_date:
            price_data = price_data[price_data.index <= pd.to_datetime(end_date)]
            signals = signals[signals['timestamp'] <= pd.to_datetime(end_date)]

        if len(price_data) == 0:
            raise ValueError("No price data in specified date range")

        # Reset portfolio state
        self.reset()

        # Get unique timestamps and sort
        timestamps = sorted(price_data.index.unique())
        self.logger.info(f"Backtesting from {timestamps[0]} to {timestamps[-1]} ({len(timestamps)} periods)")

        # Extract unique assets from signals for price mapping
        signal_assets = signals['asset'].unique().tolist() if 'asset' in signals.columns else []

        # Event loop - iterate through time chronologically
        for idx, timestamp in enumerate(timestamps):
            # Get current prices and volumes for all assets
            current_prices = self._get_current_prices(price_data, timestamp, assets=signal_assets)
            current_volumes = self._get_current_volumes(price_data, timestamp, assets=signal_assets)

            # Get signals for this timestamp (accounting for execution delay)
            if idx >= self.execution_delay:
                signal_timestamp = timestamps[idx - self.execution_delay]
                current_signals = signals[signals['timestamp'] == signal_timestamp]
            else:
                current_signals = pd.DataFrame()  # No signals before execution delay

            # Process signals and generate orders
            if not current_signals.empty:
                orders = self._process_signals(current_signals, current_prices, timestamp)

                # Execute orders
                for order in orders:
                    trade = self._execute_order(order, current_prices, current_volumes, timestamp)
                    if trade is not None:
                        self.trades.append(trade)

            # Calculate portfolio value (mark-to-market)
            equity = self._calculate_portfolio_value(self.positions, current_prices, self.cash)

            # Track portfolio state with expanded positions
            portfolio_record = {
                'timestamp': timestamp,
                'cash': self.cash,
                'equity': equity
            }

            # Expand positions into dedicated columns
            for asset, quantity in self.positions.items():
                portfolio_record[f'pos_{asset}'] = quantity
                # Calculate market value for this position
                price = current_prices.get(asset, 0)
                portfolio_record[f'val_{asset}'] = quantity * price

            self.equity_curve.append((timestamp, equity))
            self.portfolio_history.append(portfolio_record)

        # Convert results to DataFrame format
        equity_series = pd.Series(
            [eq for _, eq in self.equity_curve],
            index=[ts for ts, _ in self.equity_curve],
            name='equity'
        )

        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame(columns=[
            'timestamp', 'strategy', 'asset', 'signal', 'quantity', 'execution_price',
            'fixed_cost', 'percentage_cost', 'slippage', 'realized_pnl'
        ])

        portfolio_df = pd.DataFrame(self.portfolio_history)

        # Calculate metrics
        metrics = self._calculate_metrics(equity_series, trades_df, self.initial_capital)

        # Calculate final statistics
        final_equity = equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Save trades if requested
        if save_trades and not trades_df.empty:
            # Use provided format or default from config
            format_to_use = trade_format if trade_format is not None else self.default_trade_format
            self._save_trades(trades_df, format=format_to_use)

        # Create result
        result = BacktestResult(
            equity_curve=equity_series,
            trades=trades_df,
            portfolio_history=portfolio_df,
            metrics=metrics,
            config=self.backtest_config,
            start_date=str(timestamps[0]),
            end_date=str(timestamps[-1]),
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            strategy_names=self.strategy_names
        )

        self.logger.info(
            f"Backtest complete: total_return={total_return:.2%}, "
            f"final_equity=${final_equity:,.0f}, trades={len(trades_df)}"
        )

        return result

    def _process_signals(
        self,
        signals: pd.DataFrame,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> List[Dict]:
        """
        Process signals and generate orders.

        Args:
            signals: DataFrame with signals for current timestamp
            current_prices: Dictionary of current prices
            timestamp: Current timestamp

        Returns:
            List of order dictionaries
        """
        orders = []

        # Calculate portfolio value once for all strategies
        equity = self._calculate_portfolio_value(self.positions, current_prices, self.cash)

        # Group signals by strategy and apply risk limits per strategy
        if 'strategy' in signals.columns:
            grouped = signals.groupby('strategy')
        else:
            # If no strategy column, treat all as single strategy
            grouped = [('default', signals)]

        filtered_signals_list = []
        for strategy_name, strategy_signals in grouped:
            # Find strategy instance
            strategy = next((s for s in self.strategies if s.name == strategy_name), None)
            if strategy is None:
                # If strategy name doesn't match, use first strategy
                strategy = self.strategies[0] if self.strategies else None

            if strategy is None:
                self.logger.warning(f"No strategy found for '{strategy_name}', skipping signals")
                continue

            # Apply risk limits to filter signals
            try:
                filtered = strategy.apply_risk_limits(
                    signals=strategy_signals,
                    current_positions=strategy.current_positions,
                    account_value=equity,
                    current_prices=current_prices
                )
                filtered_signals_list.append(filtered)
            except Exception as e:
                self.logger.warning(f"Failed to apply risk limits for strategy '{strategy_name}': {e}")
                # Fall back to unfiltered signals if risk limits fail
                filtered_signals_list.append(strategy_signals)

        # Combine all filtered signals
        if filtered_signals_list:
            filtered_signals = pd.concat(filtered_signals_list, ignore_index=True)
        else:
            filtered_signals = pd.DataFrame()

        # Process filtered signals to generate orders
        for _, signal_row in filtered_signals.iterrows():
            asset = signal_row['asset']
            signal = signal_row['signal']
            strength = signal_row['strength']
            strategy_name = signal_row.get('strategy', 'unknown')

            # Skip hold signals
            if signal == 0:
                continue

            # Get current price with fallback to 'default' for backward compatibility
            current_price = current_prices.get(asset)
            if current_price is None and 'default' in current_prices:
                current_price = current_prices['default']

            if current_price is None or current_price <= 0:
                self.logger.warning(f"Invalid price for {asset} at {timestamp}, skipping")
                continue

            # Find strategy instance
            strategy = next((s for s in self.strategies if s.name == strategy_name), None)
            if strategy is None:
                # If strategy name doesn't match, use first strategy
                strategy = self.strategies[0]

            # Calculate position size
            position_size = strategy.calculate_position_size(
                signal_strength=abs(strength),
                account_value=equity,
                current_price=current_price
            )

            # Adjust for signal direction
            if signal < 0:
                position_size = -position_size

            # Get current position from strategy (for multi-strategy attribution)
            current_position = strategy.current_positions.get(asset, 0)

            # Calculate target quantity
            target_quantity = position_size

            # Create order
            order = {
                'timestamp': timestamp,
                'strategy': strategy_name,
                'asset': asset,
                'signal': signal,
                'current_position': current_position,
                'target_quantity': target_quantity,
                'current_price': current_price
            }

            orders.append(order)

        return orders

    def _execute_order(
        self,
        order: Dict,
        current_prices: Dict[str, float],
        current_volumes: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Execute single order with transaction costs and slippage.

        Args:
            order: Order dictionary
            current_prices: Dictionary of current prices
            current_volumes: Dictionary of current volumes
            timestamp: Current timestamp

        Returns:
            Trade record dictionary or None if execution failed
        """
        asset = order['asset']
        target_quantity = order['target_quantity']
        current_position = order['current_position']
        current_price = order['current_price']

        # Calculate quantity to trade
        quantity_to_trade = target_quantity - current_position

        # Skip if quantity is negligible
        if abs(quantity_to_trade) < 1e-6:
            return None

        # Get volume for this asset (if available)
        avg_volume = current_volumes.get(asset, None)

        # Calculate slippage with volume-based market impact if available
        slippage = self._calculate_slippage(quantity_to_trade, current_price, avg_volume)

        # Calculate execution price with slippage
        if quantity_to_trade > 0:
            # Buying - pay slippage
            execution_price = current_price * (1 + slippage)
        else:
            # Selling - receive less due to slippage
            execution_price = current_price * (1 - slippage)

        # Calculate transaction costs
        fixed_cost = self.fixed_cost
        percentage_cost = abs(quantity_to_trade * execution_price * self.percentage_cost)
        total_cost = fixed_cost + percentage_cost

        # Calculate cash impact
        cash_impact = -(quantity_to_trade * execution_price + total_cost)

        # Check if sufficient cash and compute partial fills if needed
        if self.cash + cash_impact < 0:
            # Insufficient cash - attempt partial fill for buy orders
            if quantity_to_trade > 0:
                # Calculate maximum affordable quantity
                # Available cash: self.cash
                # Cost per unit: execution_price + (execution_price * percentage_cost) / quantity
                # Approximate: available_quantity = (cash - fixed_cost) / (price * (1 + percentage_cost))

                if self.cash > fixed_cost:
                    max_affordable = (self.cash - fixed_cost) / (execution_price * (1 + self.percentage_cost))

                    # Set minimum threshold (e.g., 10% of original order or 0.01 units)
                    min_threshold = max(0.1 * abs(quantity_to_trade), 0.01)

                    if max_affordable >= min_threshold:
                        # Execute partial fill
                        quantity_to_trade = max_affordable

                        # Recompute slippage for the reduced quantity
                        slippage = self._calculate_slippage(quantity_to_trade, current_price, avg_volume)

                        # Recompute execution price with updated slippage
                        if quantity_to_trade > 0:
                            execution_price = current_price * (1 + slippage)
                        else:
                            execution_price = current_price * (1 - slippage)

                        # Recalculate costs for partial fill
                        percentage_cost = abs(quantity_to_trade * execution_price * self.percentage_cost)
                        total_cost = fixed_cost + percentage_cost
                        cash_impact = -(quantity_to_trade * execution_price + total_cost)

                        self.logger.info(
                            f"Partial fill: asset={asset}, quantity={quantity_to_trade:.2f} "
                            f"(affordable with cash=${self.cash:.2f})"
                        )
                    else:
                        self.logger.warning(
                            f"Insufficient cash for order: asset={asset}, quantity={quantity_to_trade:.2f}, "
                            f"cash={self.cash:.2f}, max_affordable={max_affordable:.2f} below threshold"
                        )
                        return None
                else:
                    self.logger.warning(
                        f"Insufficient cash for order: asset={asset}, cash={self.cash:.2f} below fixed_cost={fixed_cost:.2f}"
                    )
                    return None
            else:
                # Sell order with insufficient cash (shouldn't happen in normal scenarios)
                self.logger.warning(
                    f"Insufficient cash for sell order: asset={asset}, quantity={quantity_to_trade:.2f}, "
                    f"cash={self.cash:.2f}"
                )
                return None

        # Find the specific strategy for this order
        strategy_name = order['strategy']
        strategy = next((s for s in self.strategies if s.name == strategy_name), None)

        if strategy is None:
            self.logger.warning(
                f"Strategy '{strategy_name}' not found for order, using first strategy as fallback"
            )
            strategy = self.strategies[0] if self.strategies else None

        if strategy is None:
            self.logger.error("No strategies available, cannot execute order")
            return None

        # Calculate realized PnL if closing or reducing position using this strategy's entry price
        realized_pnl = 0.0
        net_realized_pnl = 0.0
        if current_position != 0 and asset in strategy.entry_prices:
            entry_price = strategy.entry_prices[asset]

            # Calculate PnL for the closed portion
            if (current_position > 0 and quantity_to_trade < 0) or \
               (current_position < 0 and quantity_to_trade > 0):
                closed_quantity = min(abs(quantity_to_trade), abs(current_position))
                realized_pnl = closed_quantity * (execution_price - entry_price) * np.sign(current_position)

                # Calculate net realized PnL by subtracting proportional transaction costs
                # Proportional costs for the closed portion
                proportional_cost = (closed_quantity / abs(quantity_to_trade)) * total_cost if abs(quantity_to_trade) > 0 else 0
                net_realized_pnl = realized_pnl - proportional_cost

        # Execute trade
        self.cash += cash_impact
        self.positions[asset] = self.positions.get(asset, 0) + quantity_to_trade

        # Remove position if closed
        if abs(self.positions[asset]) < 1e-6:
            self.positions.pop(asset, None)

        # Update only this strategy's position tracking
        strategy.update_positions(asset, quantity_to_trade, execution_price)

        # Create trade record
        trade = {
            'timestamp': timestamp,
            'strategy': order['strategy'],
            'asset': asset,
            'signal': order['signal'],
            'quantity': quantity_to_trade,
            'execution_price': execution_price,
            'fixed_cost': fixed_cost,
            'percentage_cost': percentage_cost,
            'slippage': slippage,
            'realized_pnl': realized_pnl,
            'net_realized_pnl': net_realized_pnl
        }

        self.logger.debug(
            f"Executed trade: {asset} {quantity_to_trade:+.2f} @ ${execution_price:.2f}, "
            f"costs=${total_cost:.2f}, pnl=${realized_pnl:+.2f}"
        )

        return trade

    def _calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        avg_volume: Optional[float] = None
    ) -> float:
        """
        Calculate market impact slippage.

        Args:
            order_size: Order size (positive for buy, negative for sell)
            current_price: Current price
            avg_volume: Optional average volume for market impact calculation

        Returns:
            Slippage as percentage (e.g., 0.0005 for 0.05%)
        """
        if not self.market_impact_enabled:
            return self.slippage_pct

        # Use volume-based model if average volume provided
        if avg_volume is not None and avg_volume > 0:
            slippage = self.impact_coefficient * np.sqrt(abs(order_size) / avg_volume)
        else:
            # Simple model based on order size
            slippage = self.impact_coefficient * np.sqrt(abs(order_size)) / 1000

        # Add base slippage
        total_slippage = self.slippage_pct + slippage

        # Clamp to sensible range (0 to max_slippage)
        clamped_slippage = min(max(total_slippage, 0), self.max_slippage)

        return clamped_slippage

    def _calculate_portfolio_value(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        cash: float
    ) -> float:
        """
        Calculate current portfolio value (mark-to-market).

        Args:
            positions: Dictionary of positions {asset: quantity}
            current_prices: Dictionary of current prices {asset: price}
            cash: Current cash balance

        Returns:
            Total portfolio equity
        """
        equity = cash

        for asset, quantity in positions.items():
            price = current_prices.get(asset, 0)
            equity += quantity * price

        return equity

    def _get_current_volumes(
        self,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        assets: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract current volumes for all assets from price_data.

        In multi-asset mode, a single 'volume' column will not be broadcast;
        per-asset <asset>_volume columns are required for volume-informed slippage.

        Args:
            price_data: DataFrame with price data (may contain volume columns)
            timestamp: Current timestamp
            assets: Optional list of asset names

        Returns:
            Dictionary mapping asset to volume (empty if no volume data)
        """
        current_volumes = {}

        # Get row for current timestamp
        if timestamp in price_data.index:
            row = price_data.loc[timestamp]

            # Check for volume column(s)
            if 'volume' in price_data.columns:
                # Single asset volume
                volume_value = row['volume'] if not pd.isna(row['volume']) else None
                if volume_value is not None and assets and len(assets) == 1:
                    current_volumes[assets[0]] = volume_value
                elif volume_value is not None and assets and len(assets) > 1:
                    # Multi-asset: do NOT broadcast single volume - log warning
                    self.logger.warning(
                        f"Single 'volume' column detected with {len(assets)} assets. "
                        f"Per-asset volume columns (<asset>_volume) are required for volume-informed slippage. "
                        f"Falling back to non-volume slippage model."
                    )
                    # Do not populate current_volumes for any asset - let them use fallback
            else:
                # Check for per-asset volume columns (e.g., <asset>_volume)
                for col in price_data.columns:
                    if col.endswith('_volume'):
                        asset_name = col.replace('_volume', '')
                        volume = row[col] if not pd.isna(row[col]) else None
                        if volume is not None:
                            current_volumes[asset_name] = volume
        else:
            # Forward-fill volumes
            prior_data = price_data[price_data.index < timestamp]
            if len(prior_data) > 0:
                last_row = prior_data.iloc[-1]
                if 'volume' in price_data.columns:
                    volume_value = last_row['volume'] if not pd.isna(last_row['volume']) else None
                    if volume_value is not None and assets and len(assets) == 1:
                        current_volumes[assets[0]] = volume_value
                    elif volume_value is not None and assets and len(assets) > 1:
                        # Multi-asset: do NOT broadcast single volume - log warning
                        self.logger.warning(
                            f"Single 'volume' column detected with {len(assets)} assets. "
                            f"Per-asset volume columns (<asset>_volume) are required for volume-informed slippage. "
                            f"Falling back to non-volume slippage model."
                        )
                        # Do not populate current_volumes for any asset - let them use fallback
                else:
                    for col in price_data.columns:
                        if col.endswith('_volume'):
                            asset_name = col.replace('_volume', '')
                            volume = last_row[col] if not pd.isna(last_row[col]) else None
                            if volume is not None:
                                current_volumes[asset_name] = volume

        return current_volumes

    def _get_current_prices(
        self,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        assets: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract current prices for all assets from price_data.

        Args:
            price_data: DataFrame with price data
            timestamp: Current timestamp
            assets: Optional list of asset names from signals (for single-asset mapping)

        Returns:
            Dictionary mapping asset to price
        """
        current_prices = {}

        # Get row for current timestamp
        if timestamp in price_data.index:
            row = price_data.loc[timestamp]

            # Handle single or multiple assets
            if 'price' in price_data.columns:
                # Single asset case - map to actual asset name if provided
                price_value = row['price'] if not pd.isna(row['price']) else 0
                if assets and len(assets) == 1:
                    # Use the actual asset name from signals
                    current_prices[assets[0]] = price_value
                elif assets and len(assets) > 1:
                    # Multiple assets but single price column - raise error
                    raise ValueError(
                        f"Multi-asset backtest detected ({len(assets)} assets: {assets}) but price_data "
                        f"contains only a single 'price' column. Per-asset price columns are required "
                        f"for accurate multi-asset backtesting. Use column naming convention: "
                        f"{', '.join([f'{asset}' for asset in assets])} or '{assets[0]}_price', etc."
                    )
                else:
                    # Fallback to 'default' for backward compatibility
                    current_prices['default'] = price_value
            else:
                # Multiple assets - each column is an asset
                for col in price_data.columns:
                    # Exclude non-price columns
                    if col != 'timestamp' and col != 'volume' and not col.endswith('_volume'):
                        price = row[col] if not pd.isna(row[col]) else 0
                        current_prices[col] = price
        else:
            # Forward-fill - use last known price
            prior_data = price_data[price_data.index < timestamp]
            if len(prior_data) > 0:
                last_row = prior_data.iloc[-1]
                if 'price' in price_data.columns:
                    price_value = last_row['price'] if not pd.isna(last_row['price']) else 0
                    if assets and len(assets) == 1:
                        current_prices[assets[0]] = price_value
                    elif assets and len(assets) > 1:
                        # Multiple assets but single price column - raise error
                        raise ValueError(
                            f"Multi-asset backtest detected ({len(assets)} assets: {assets}) but price_data "
                            f"contains only a single 'price' column. Per-asset price columns are required "
                            f"for accurate multi-asset backtesting. Use column naming convention: "
                            f"{', '.join([f'{asset}' for asset in assets])} or '{assets[0]}_price', etc."
                        )
                    else:
                        current_prices['default'] = price_value
                else:
                    for col in price_data.columns:
                        # Exclude non-price columns
                        if col != 'timestamp' and col != 'volume' and not col.endswith('_volume'):
                            price = last_row[col] if not pd.isna(last_row[col]) else 0
                            current_prices[col] = price

        return current_prices

    def _save_trades(self, trades: pd.DataFrame, format: Union[str, List[str]] = 'parquet'):
        """
        Save trade log to storage in specified format(s).

        Args:
            trades: DataFrame with trade records
            format: Format(s) to save trades in ('parquet', 'csv', or list of both)
        """
        # Build validated list of formats
        validated = [f for f in (format if isinstance(format, list) else [format]) if f in ['parquet', 'csv']]
        if not validated:
            self.logger.error("No valid formats specified, defaulting to parquet")
            validated = ['parquet']

        # Set index to timestamp for proper partitioning and date range extraction
        trades_to_save = trades.copy()
        if 'timestamp' in trades_to_save.columns:
            trades_to_save['timestamp'] = pd.to_datetime(trades_to_save['timestamp'])
            trades_to_save = trades_to_save.set_index('timestamp').sort_index()

            # Extract date range from index
            start_date = trades_to_save.index.min().strftime('%Y-%m-%d')
            end_date = trades_to_save.index.max().strftime('%Y-%m-%d')
        else:
            start_date = None
            end_date = None

        # Save in each format
        saved_paths = []
        for fmt in validated:
            try:
                # Use DataManager to save trades
                output_path = self.data_manager.save_processed_data(
                    trades_to_save,
                    source='backtest',
                    dataset='trades',
                    start_date=start_date,
                    end_date=end_date,
                    format=fmt
                )
                saved_paths.append((fmt, output_path))
                self.logger.info(f"Saved {len(trades_to_save)} trades to {output_path} ({fmt} format)")
            except Exception as e:
                self.logger.error(f"Failed to save trades in {fmt} format: {e}")

        # Summary log
        if len(saved_paths) > 1:
            format_list = ', '.join([fmt for fmt, _ in saved_paths])
            self.logger.info(f"Trades saved in {len(saved_paths)} formats: {format_list}")

        # Optional database logging
        db_enabled = self.backtest_config.get('trade_logging', {}).get('db_enabled', False)
        if db_enabled:
            try:
                # Reset index to pass DataFrame with timestamp column to DB
                trades_for_db = trades_to_save.reset_index()
                table_name = self.backtest_config.get('trade_logging', {}).get('db_table', 'trades')
                success = self.data_manager.save_trades_to_db(trades_for_db, table=table_name)
                if success:
                    self.logger.info(f"Trades also saved to database table '{table_name}'")
            except Exception as e:
                self.logger.error(f"Database trade logging failed: {e}. Continuing without DB persistence.")

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.

        Args:
            equity_curve: Series with portfolio equity over time
            trades: DataFrame with trade records
            initial_capital: Starting capital

        Returns:
            Dictionary with performance metrics
        """
        # Import here to avoid circular dependency
        from src.backtesting.metrics import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_max_drawdown,
            calculate_calmar_ratio,
            calculate_win_rate,
            calculate_profit_factor
        )

        metrics = {}

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Calculate metrics
        try:
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
        except Exception as e:
            self.logger.warning(f"Failed to calculate Sharpe ratio: {e}")
            metrics['sharpe_ratio'] = 0.0

        try:
            metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
        except Exception as e:
            self.logger.warning(f"Failed to calculate Sortino ratio: {e}")
            metrics['sortino_ratio'] = 0.0

        try:
            metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)
        except Exception as e:
            self.logger.warning(f"Failed to calculate max drawdown: {e}")
            metrics['max_drawdown'] = 0.0

        try:
            metrics['calmar_ratio'] = calculate_calmar_ratio(
                equity_curve=equity_curve,
                max_drawdown=metrics.get('max_drawdown')
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate Calmar ratio: {e}")
            metrics['calmar_ratio'] = 0.0

        if not trades.empty and 'realized_pnl' in trades.columns:
            try:
                metrics['win_rate'] = calculate_win_rate(trades)
            except Exception as e:
                self.logger.warning(f"Failed to calculate win rate: {e}")
                metrics['win_rate'] = 0.0

            try:
                metrics['profit_factor'] = calculate_profit_factor(trades)
            except Exception as e:
                self.logger.warning(f"Failed to calculate profit factor: {e}")
                metrics['profit_factor'] = 0.0
        else:
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0

        return metrics

    def reset(self):
        """
        Reset engine state for new backtest.
        """
        self.cash = self.initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trades = []
        self.portfolio_history = []

        # Reset strategy positions
        for strategy in self.strategies:
            strategy.current_positions.clear()
            strategy.entry_prices.clear()

        self.logger.debug("Engine state reset")
