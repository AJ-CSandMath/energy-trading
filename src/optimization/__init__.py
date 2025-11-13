"""
Optimization Module

This module provides portfolio optimization and risk analytics functionality
for energy trading strategies.
"""

from src.optimization.risk_analytics import (
    RiskAnalytics,
    calculate_var,
    calculate_cvar,
    calculate_information_ratio,
    calculate_correlation_matrix,
    calculate_implied_volatility,
    calculate_calmar_ratio
)
from src.optimization.risk_reporting import RiskReport
from src.optimization.optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
    BaseOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    BlackLittermanOptimizer,
    MinimumCVaROptimizer,
    RenewableConstraints,
    MultiPeriodOptimizer,
    calculate_portfolio_return,
    calculate_portfolio_risk,
    create_picking_matrix,
    validate_optimization_inputs
)

__all__ = [
    # Risk analytics
    'RiskAnalytics',
    'calculate_var',
    'calculate_cvar',
    'calculate_information_ratio',
    'calculate_correlation_matrix',
    'calculate_implied_volatility',
    'calculate_calmar_ratio',
    'RiskReport',
    # Portfolio optimization
    'PortfolioOptimizer',
    'OptimizationResult',
    'BaseOptimizer',
    'MeanVarianceOptimizer',
    'RiskParityOptimizer',
    'BlackLittermanOptimizer',
    'MinimumCVaROptimizer',
    'RenewableConstraints',
    'MultiPeriodOptimizer',
    'calculate_portfolio_return',
    'calculate_portfolio_risk',
    'create_picking_matrix',
    'validate_optimization_inputs'
]
