"""
Performance Metrics Module

Provides performance metrics for backtesting evaluation including risk-adjusted returns,
drawdown analysis, and trade statistics.

Functions calculate standard trading metrics:
- Sharpe Ratio: Risk-adjusted return
- Sortino Ratio: Downside risk-adjusted return
- Max Drawdown: Maximum peak-to-trough decline
- Calmar Ratio: Return / max drawdown
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profit / gross loss

Example:
    >>> from src.backtesting import BacktestEngine, PerformanceMetrics
    >>>
    >>> # Run backtest
    >>> result = engine.run(price_data, signals)
    >>>
    >>> # Calculate metrics
    >>> metrics = PerformanceMetrics(
    >>>     equity_curve=result.equity_curve,
    >>>     trades=result.trades,
    >>>     initial_capital=result.initial_capital
    >>> )
    >>>
    >>> # Access individual metrics
    >>> print(f"Sharpe Ratio: {metrics.metrics['sharpe_ratio']:.2f}")
    >>> print(f"Max Drawdown: {metrics.metrics['max_drawdown']:.2%}")
"""

import logging
from typing import Optional, Union
import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).

    Args:
        returns: Series or array of period returns
        risk_free_rate: Risk-free rate (annual)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Sharpe ratio (annualized)
    """
    returns = _validate_returns(returns)

    if len(returns) < 2:
        return np.nan

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    # Convert risk-free rate to period rate
    period_rf = risk_free_rate / periods_per_year

    sharpe = (mean_return - period_rf) / std_return * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).

    Args:
        returns: Series or array of period returns
        risk_free_rate: Risk-free rate (annual)
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio (annualized)
    """
    returns = _validate_returns(returns)

    if len(returns) < 2:
        return np.nan

    mean_return = returns.mean()

    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    # Convert risk-free rate to period rate
    period_rf = risk_free_rate / periods_per_year

    sortino = (mean_return - period_rf) / downside_std * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: Series with portfolio values

    Returns:
        Max drawdown as negative percentage (e.g., -0.15 for 15% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0

    # Calculate running maximum
    cummax = equity_curve.cummax()

    # Calculate drawdown
    drawdown = (equity_curve - cummax) / cummax

    # Max drawdown is most negative value
    max_dd = drawdown.min()

    return max_dd


def calculate_calmar_ratio(
    returns: Optional[Union[pd.Series, np.ndarray]] = None,
    max_drawdown: Optional[float] = None,
    periods_per_year: int = 252,
    equity_curve: Optional[pd.Series] = None,
    annualized_return: Optional[float] = None
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Optional series or array of period returns (deprecated, use equity_curve)
        max_drawdown: Optional pre-calculated max drawdown
        periods_per_year: Number of periods per year
        equity_curve: Optional equity curve for calculating annualized return
        annualized_return: Optional pre-calculated annualized return

    Returns:
        Calmar ratio
    """
    # Determine annualized return from available inputs
    if annualized_return is not None:
        ann_return = annualized_return
    elif equity_curve is not None:
        # Calculate annualized return from equity curve using geometric compounding
        ann_return = calculate_annualized_return(equity_curve, periods_per_year)
    elif returns is not None:
        # Fall back to mean return calculation
        returns = _validate_returns(returns)
        if len(returns) < 2:
            return np.nan
        ann_return = returns.mean() * periods_per_year
    else:
        return np.nan

    # Calculate max drawdown if equity curve provided and max_drawdown not given
    if max_drawdown is None:
        if equity_curve is not None:
            max_drawdown = calculate_max_drawdown(equity_curve)
        else:
            # Cannot calculate Calmar without drawdown
            return np.nan

    if max_drawdown == 0:
        return np.inf

    calmar = ann_return / abs(max_drawdown)

    return calmar


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate percentage of winning trades.

    Uses net_realized_pnl if available (includes transaction costs),
    otherwise falls back to realized_pnl.

    Args:
        trades: DataFrame with 'realized_pnl' or 'net_realized_pnl' column

    Returns:
        Win rate as percentage (e.g., 0.55 for 55%)
    """
    trades = _validate_trades(trades)

    # Use net_realized_pnl if available, otherwise use realized_pnl
    pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in trades.columns else 'realized_pnl'

    # Filter trades with realized PnL (closed positions)
    closed_trades = trades[trades[pnl_col] != 0]

    if len(closed_trades) == 0:
        return 0.0

    winning_trades = closed_trades[closed_trades[pnl_col] > 0]
    win_rate = len(winning_trades) / len(closed_trades)

    return win_rate


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Uses net_realized_pnl if available (includes transaction costs),
    otherwise falls back to realized_pnl.

    Args:
        trades: DataFrame with 'realized_pnl' or 'net_realized_pnl' column

    Returns:
        Profit factor (e.g., 1.5 means $1.50 profit per $1 loss)
    """
    trades = _validate_trades(trades)

    # Use net_realized_pnl if available, otherwise use realized_pnl
    pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in trades.columns else 'realized_pnl'

    # Filter trades with realized PnL
    closed_trades = trades[trades[pnl_col] != 0]

    if len(closed_trades) == 0:
        return 0.0

    gross_profit = closed_trades[closed_trades[pnl_col] > 0][pnl_col].sum()
    gross_loss = abs(closed_trades[closed_trades[pnl_col] < 0][pnl_col].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    profit_factor = gross_profit / gross_loss

    return profit_factor


def calculate_average_win(trades: pd.DataFrame) -> float:
    """
    Calculate average winning trade.

    Uses net_realized_pnl if available (includes transaction costs),
    otherwise falls back to realized_pnl.

    Args:
        trades: DataFrame with 'realized_pnl' or 'net_realized_pnl' column

    Returns:
        Average profit of winning trades
    """
    trades = _validate_trades(trades)

    # Use net_realized_pnl if available, otherwise use realized_pnl
    pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in trades.columns else 'realized_pnl'

    winning_trades = trades[trades[pnl_col] > 0]

    if len(winning_trades) == 0:
        return 0.0

    return winning_trades[pnl_col].mean()


def calculate_average_loss(trades: pd.DataFrame) -> float:
    """
    Calculate average losing trade.

    Uses net_realized_pnl if available (includes transaction costs),
    otherwise falls back to realized_pnl.

    Args:
        trades: DataFrame with 'realized_pnl' or 'net_realized_pnl' column

    Returns:
        Average loss of losing trades (as negative number)
    """
    trades = _validate_trades(trades)

    # Use net_realized_pnl if available, otherwise use realized_pnl
    pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in trades.columns else 'realized_pnl'

    losing_trades = trades[trades[pnl_col] < 0]

    if len(losing_trades) == 0:
        return 0.0

    return losing_trades[pnl_col].mean()


def calculate_expectancy(trades: pd.DataFrame) -> float:
    """
    Calculate trade expectancy.

    Uses net_realized_pnl if available (includes transaction costs),
    otherwise falls back to realized_pnl.

    Args:
        trades: DataFrame with 'realized_pnl' or 'net_realized_pnl' column

    Returns:
        Expected profit per trade
    """
    trades = _validate_trades(trades)

    win_rate = calculate_win_rate(trades)
    avg_win = calculate_average_win(trades)
    avg_loss = calculate_average_loss(trades)

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return expectancy


def calculate_recovery_factor(
    equity_curve: pd.Series,
    initial_capital: float
) -> float:
    """
    Calculate recovery factor (net profit / max drawdown).

    Args:
        equity_curve: Series with portfolio values
        initial_capital: Starting capital

    Returns:
        Recovery factor
    """
    if len(equity_curve) < 2:
        return 0.0

    # Calculate net profit
    final_equity = equity_curve.iloc[-1]
    net_profit = final_equity - initial_capital

    # Calculate max drawdown in dollars
    cummax = equity_curve.cummax()
    drawdown_dollars = (equity_curve - cummax).min()

    if drawdown_dollars == 0:
        return np.inf if net_profit > 0 else 0.0

    recovery = net_profit / abs(drawdown_dollars)

    return recovery


def calculate_annualized_return(
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized return.

    Args:
        equity_curve: Series with portfolio values
        periods_per_year: Number of periods per year

    Returns:
        Annualized return as percentage
    """
    if len(equity_curve) < 2:
        return 0.0

    initial_equity = equity_curve.iloc[0]
    final_equity = equity_curve.iloc[-1]

    total_return = (final_equity / initial_equity) - 1
    years = len(equity_curve) / periods_per_year

    if years == 0:
        return 0.0

    annualized = (1 + total_return) ** (1 / years) - 1

    return annualized


def calculate_annualized_volatility(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Series or array of period returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized volatility
    """
    returns = _validate_returns(returns)

    if len(returns) < 2:
        return 0.0

    volatility = returns.std() * np.sqrt(periods_per_year)

    return volatility


class PerformanceMetrics:
    """
    Convenience class to calculate and store all performance metrics.

    Calculates comprehensive trading metrics from backtest results.

    Attributes:
        equity_curve: Portfolio equity over time
        trades: Trade records
        returns: Period returns
        initial_capital: Starting capital
        metrics: Dictionary of calculated metrics
        logger: Logger instance
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        """
        Initialize with backtest results.

        Args:
            equity_curve: Series with portfolio values
            trades: DataFrame with trade records
            returns: Optional pre-calculated returns
            initial_capital: Starting capital
            risk_free_rate: Risk-free rate (annual)
            periods_per_year: Number of periods per year
        """
        self.logger = logging.getLogger(__name__)

        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Calculate returns if not provided
        if returns is None:
            self.returns = equity_curve.pct_change().dropna()
        else:
            self.returns = returns

        # Calculate all metrics
        self.metrics = self.calculate_all()

    def calculate_all(self) -> dict:
        """
        Calculate all performance metrics.

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Risk-adjusted returns
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(
            self.returns, self.risk_free_rate, self.periods_per_year
        )
        metrics['sortino_ratio'] = calculate_sortino_ratio(
            self.returns, self.risk_free_rate, self.periods_per_year
        )

        # Drawdown metrics
        metrics['max_drawdown'] = calculate_max_drawdown(self.equity_curve)
        metrics['calmar_ratio'] = calculate_calmar_ratio(
            self.returns, metrics['max_drawdown'], self.periods_per_year
        )
        metrics['recovery_factor'] = calculate_recovery_factor(
            self.equity_curve, self.initial_capital
        )

        # Return metrics
        metrics['annualized_return'] = calculate_annualized_return(
            self.equity_curve, self.periods_per_year
        )
        metrics['annualized_volatility'] = calculate_annualized_volatility(
            self.returns, self.periods_per_year
        )
        metrics['total_return'] = (self.equity_curve.iloc[-1] / self.initial_capital) - 1

        # Trade metrics (use net_realized_pnl if available, otherwise realized_pnl)
        pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in self.trades.columns else 'realized_pnl'
        if not self.trades.empty and pnl_col in self.trades.columns:
            metrics['win_rate'] = calculate_win_rate(self.trades)
            metrics['profit_factor'] = calculate_profit_factor(self.trades)
            metrics['avg_win'] = calculate_average_win(self.trades)
            metrics['avg_loss'] = calculate_average_loss(self.trades)
            metrics['expectancy'] = calculate_expectancy(self.trades)

            # Trade counts (using same pnl_col as win_rate/profit_factor)
            closed_trades = self.trades[self.trades[pnl_col] != 0]
            metrics['total_trades'] = len(closed_trades)
            metrics['winning_trades'] = len(closed_trades[closed_trades[pnl_col] > 0])
            metrics['losing_trades'] = len(closed_trades[closed_trades[pnl_col] < 0])
        else:
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['avg_win'] = 0.0
            metrics['avg_loss'] = 0.0
            metrics['expectancy'] = 0.0
            metrics['total_trades'] = 0
            metrics['winning_trades'] = 0
            metrics['losing_trades'] = 0

        # Log summary
        self.logger.info(
            f"Metrics calculated: Sharpe={metrics['sharpe_ratio']:.2f}, "
            f"MaxDD={metrics['max_drawdown']:.2%}, WinRate={metrics['win_rate']:.2%}"
        )

        return metrics

    def get_summary(self) -> dict:
        """
        Get formatted summary of key metrics.

        Returns:
            Dictionary with formatted strings
        """
        summary = {
            'Total Return': f"{self.metrics['total_return']:.2%}",
            'Annualized Return': f"{self.metrics['annualized_return']:.2%}",
            'Annualized Volatility': f"{self.metrics['annualized_volatility']:.2%}",
            'Sharpe Ratio': f"{self.metrics['sharpe_ratio']:.2f}",
            'Sortino Ratio': f"{self.metrics['sortino_ratio']:.2f}",
            'Calmar Ratio': f"{self.metrics['calmar_ratio']:.2f}",
            'Max Drawdown': f"{self.metrics['max_drawdown']:.2%}",
            'Recovery Factor': f"{self.metrics['recovery_factor']:.2f}",
            'Win Rate': f"{self.metrics['win_rate']:.2%}",
            'Profit Factor': f"{self.metrics['profit_factor']:.2f}",
            'Total Trades': f"{self.metrics['total_trades']:.0f}",
            'Expectancy': f"${self.metrics['expectancy']:.2f}"
        }

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to DataFrame.

        Returns:
            DataFrame with metric names and values
        """
        df = pd.DataFrame([self.metrics]).T
        df.columns = ['Value']
        df.index.name = 'Metric'

        return df

    @classmethod
    def compare(cls, results: list) -> pd.DataFrame:
        """
        Compare multiple backtest results.

        Args:
            results: List of BacktestResult objects

        Returns:
            Comparison DataFrame
        """
        comparison = []

        for i, result in enumerate(results):
            metrics_obj = cls(
                equity_curve=result.equity_curve,
                trades=result.trades,
                initial_capital=result.initial_capital
            )
            metrics_obj.metrics['name'] = f"Backtest_{i+1}"
            comparison.append(metrics_obj.metrics)

        df = pd.DataFrame(comparison)
        df = df.set_index('name')

        return df


# Helper functions

def _validate_returns(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Validate returns series.

    Args:
        returns: Returns series or array

    Returns:
        Validated returns as Series

    Raises:
        ValueError: If returns are invalid
    """
    # Convert to Series if array
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Drop NaN and inf values
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2:
        logging.getLogger(__name__).warning("Insufficient return data for metric calculation")

    return returns


def _validate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Validate trades DataFrame.

    Args:
        trades: DataFrame with trade records

    Returns:
        Validated trades

    Raises:
        ValueError: If trades are invalid
    """
    if 'realized_pnl' not in trades.columns:
        raise ValueError("Trades DataFrame must have 'realized_pnl' column")

    # Drop rows with invalid PnL
    trades = trades[trades['realized_pnl'].notna()]

    return trades
