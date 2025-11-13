"""
Dashboard Module

This module provides the Streamlit dashboard components for the Energy Trading
& Portfolio Optimization System.
"""

from src.dashboard.utils import (
    load_backtest_data,
    load_optimization_results,
    create_date_range_selector,
    create_strategy_selector,
    apply_dashboard_theme
)

__all__ = [
    'load_backtest_data',
    'load_optimization_results',
    'create_date_range_selector',
    'create_strategy_selector',
    'apply_dashboard_theme'
]
