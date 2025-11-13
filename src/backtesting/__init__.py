"""
Backtesting Module

Provides comprehensive backtesting framework for trading strategy evaluation
with realistic execution modeling, performance metrics, and visualization.

Main Components:
- BacktestEngine: Event-driven backtesting engine with realistic execution
- BacktestResult: Dataclass containing backtest results
- PerformanceMetrics: Performance metric calculations
- BacktestReport: Interactive reporting with Plotly visualizations

Example:
    >>> from src.backtesting import BacktestEngine, BacktestReport
    >>> from src.strategies import MeanReversionStrategy
    >>>
    >>> strategy = MeanReversionStrategy()
    >>> signals = strategy.generate_signals(price_data, asset='energy')
    >>>
    >>> engine = BacktestEngine(strategies=[strategy])
    >>> result = engine.run(price_data=price_data, signals=signals)
    >>>
    >>> report = BacktestReport(result)
    >>> report.save_report('backtest_report.html')
"""

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.metrics import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor
)
from src.backtesting.reporting import BacktestReport

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'PerformanceMetrics',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    'BacktestReport'
]
