"""
Risk Analytics Module

Comprehensive risk analytics framework for portfolio risk measurement including:
- Value at Risk (VaR) and Conditional Value at Risk (CVaR)
- Correlation matrices for assets and strategies
- Risk decomposition (marginal VaR, component VaR)
- Stress testing with renewable energy scenarios
- Information ratio calculation

This module consumes BacktestResult from the backtesting engine and provides
detailed risk metrics for portfolio analysis and optimization.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.backtesting.engine import BacktestResult
from src.backtesting.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)
from src.config.load_config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Standalone Functions
# =============================================================================

def calculate_var(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR).

    VaR represents the maximum expected loss at a given confidence level.
    For example, 95% VaR of $100k means there's a 5% chance of losing more than $100k.

    Args:
        returns: Return series or array
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: 'historical' (empirical quantile) or 'parametric' (normal distribution)

    Returns:
        VaR as positive number (e.g., 0.05 for 5% loss)

    Examples:
        >>> returns = pd.Series([-0.02, 0.01, -0.03, 0.02, -0.01])
        >>> var = calculate_var(returns, confidence_level=0.95, method='historical')
        >>> print(f"95% VaR: {var:.2%}")
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    returns = np.asarray(returns)

    # Handle edge cases
    if len(returns) < 2:
        logger.warning("Insufficient data for VaR calculation (< 2 observations)")
        return np.nan

    if np.std(returns) == 0:
        logger.warning("Zero volatility in returns, VaR is zero")
        return 0.0

    if method == 'historical':
        # Historical VaR: empirical quantile
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        return float(var)

    elif method == 'parametric':
        # Parametric VaR: assumes normal distribution
        z_score = norm.ppf(1 - confidence_level)
        mean = np.mean(returns)
        std = np.std(returns)
        var = -(mean + z_score * std)
        return float(var)

    else:
        raise ValueError(f"Unknown VaR method: {method}. Use 'historical' or 'parametric'")


def calculate_cvar(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    CVaR represents the expected loss given that the loss exceeds VaR.
    It's a more conservative risk measure than VaR as it accounts for tail risk.

    Args:
        returns: Return series or array
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: 'historical' or 'parametric'

    Returns:
        CVaR as positive number

    Examples:
        >>> returns = pd.Series([-0.02, 0.01, -0.03, 0.02, -0.01])
        >>> cvar = calculate_cvar(returns, confidence_level=0.95, method='historical')
        >>> print(f"95% CVaR: {cvar:.2%}")
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    returns = np.asarray(returns)

    # Handle edge cases
    if len(returns) < 2:
        logger.warning("Insufficient data for CVaR calculation (< 2 observations)")
        return np.nan

    if np.std(returns) == 0:
        logger.warning("Zero volatility in returns, CVaR is zero")
        return 0.0

    if method == 'historical':
        # Historical CVaR: mean of returns below VaR threshold
        var = calculate_var(returns, confidence_level, 'historical')
        tail_returns = returns[returns <= -var]

        if len(tail_returns) == 0:
            # No observations in tail, return VaR
            return var

        cvar = -np.mean(tail_returns)
        return float(cvar)

    elif method == 'parametric':
        # Parametric CVaR: closed-form for normal distribution
        z = norm.ppf(1 - confidence_level)
        mean = np.mean(returns)
        std = np.std(returns)
        cvar = -(mean - std * norm.pdf(z) / (1 - confidence_level))
        return float(cvar)

    else:
        raise ValueError(f"Unknown CVaR method: {method}. Use 'historical' or 'parametric'")


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio.

    Information Ratio measures risk-adjusted returns relative to a benchmark:
    IR = (Mean Excess Return) / Tracking Error

    Higher IR indicates better risk-adjusted performance relative to benchmark.

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series (if None, uses zero returns)
        periods_per_year: Annualization factor (252 for daily, 12 for monthly)

    Returns:
        Annualized Information Ratio

    Examples:
        >>> portfolio_returns = pd.Series([0.01, -0.02, 0.03, 0.01])
        >>> benchmark_returns = pd.Series([0.005, -0.01, 0.02, 0.005])
        >>> ir = calculate_information_ratio(portfolio_returns, benchmark_returns)
        >>> print(f"Information Ratio: {ir:.2f}")
    """
    portfolio_returns = portfolio_returns.dropna()

    if len(portfolio_returns) < 2:
        logger.warning("Insufficient data for Information Ratio calculation")
        return np.nan

    # If no benchmark provided, use zero returns
    if benchmark_returns is None:
        excess_returns = portfolio_returns
    else:
        # Align indices
        benchmark_returns = benchmark_returns.reindex(portfolio_returns.index)
        excess_returns = portfolio_returns - benchmark_returns
        excess_returns = excess_returns.dropna()

    if len(excess_returns) < 2:
        logger.warning("Insufficient data for Information Ratio calculation after alignment")
        return np.nan

    # Calculate tracking error (std of excess returns)
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        logger.warning("Zero tracking error, Information Ratio is undefined")
        return np.nan

    # Calculate annualized Information Ratio
    mean_excess_return = excess_returns.mean()
    ir = (mean_excess_return / tracking_error) * np.sqrt(periods_per_year)

    return float(ir)


def calculate_correlation_matrix(
    returns_df: pd.DataFrame,
    method: str = 'pearson',
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Calculate correlation matrix for asset returns.

    Args:
        returns_df: DataFrame with asset returns as columns
        method: 'pearson', 'spearman', or 'kendall'
        min_periods: Minimum number of observations required

    Returns:
        Correlation matrix DataFrame

    Examples:
        >>> returns_df = pd.DataFrame({
        ...     'asset1': [0.01, -0.02, 0.03],
        ...     'asset2': [-0.01, 0.02, -0.01]
        ... })
        >>> corr = calculate_correlation_matrix(returns_df)
        >>> print(corr)
    """
    if returns_df.empty or len(returns_df) < min_periods:
        logger.warning(f"Insufficient data for correlation calculation (< {min_periods} observations)")
        return pd.DataFrame()

    # Calculate correlation matrix
    corr_matrix = returns_df.corr(method=method, min_periods=min_periods)

    return corr_matrix


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 252,
    annualize: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate realized volatility (rolling standard deviation).

    Args:
        returns: Return series
        window: Rolling window size
        annualize: Whether to annualize the volatility
        periods_per_year: Periods per year for annualization

    Returns:
        Series of realized volatility

    Examples:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> vol = calculate_realized_volatility(returns, window=3)
        >>> print(vol)
    """
    returns = returns.dropna()

    if len(returns) < window:
        logger.warning(f"Insufficient data for volatility calculation (< {window} observations)")
        return pd.Series(dtype=float)

    # Calculate rolling standard deviation
    rolling_vol = returns.rolling(window=window).std()

    # Annualize if requested
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(periods_per_year)

    return rolling_vol


def calculate_implied_volatility(
    returns: pd.Series,
    lambda_ewma: float = 0.94,
    annualize: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate implied volatility proxy using EWMA (Exponentially Weighted Moving Average).

    Since option market data is typically unavailable for energy trading portfolios,
    this function provides an EWMA volatility estimate as a proxy for implied volatility.
    EWMA gives more weight to recent observations, making it responsive to changing
    market conditions.

    Args:
        returns: Return series
        lambda_ewma: EWMA decay factor (default 0.94, RiskMetrics standard)
                     Higher values give more weight to recent observations
        annualize: Whether to annualize the volatility
        periods_per_year: Periods per year for annualization

    Returns:
        Series of EWMA volatility (implied volatility proxy)

    Examples:
        >>> returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> implied_vol = calculate_implied_volatility(returns, lambda_ewma=0.94)
        >>> print(implied_vol)

    Note:
        This is a proxy for implied volatility. True implied volatility would require
        option prices and Black-Scholes inversion. EWMA volatility is commonly used
        in risk management as a forward-looking volatility estimate.
    """
    returns = returns.dropna()

    if len(returns) < 2:
        logger.warning("Insufficient data for implied volatility calculation (< 2 observations)")
        return pd.Series(dtype=float)

    # Calculate squared returns
    squared_returns = returns ** 2

    # Apply EWMA to squared returns to get variance
    ewma_variance = squared_returns.ewm(alpha=(1 - lambda_ewma), adjust=False).mean()

    # Take square root to get volatility
    ewma_vol = np.sqrt(ewma_variance)

    # Annualize if requested
    if annualize:
        ewma_vol = ewma_vol * np.sqrt(periods_per_year)

    return ewma_vol


def calculate_calmar_ratio(
    equity_curve: pd.Series,
    max_drawdown: Optional[float] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio (CAGR / Absolute Max Drawdown).

    The Calmar Ratio measures risk-adjusted return by comparing the compound
    annual growth rate to the maximum drawdown. Higher values indicate better
    risk-adjusted performance.

    Args:
        equity_curve: Portfolio equity curve (values over time)
        max_drawdown: Maximum drawdown (if None, will be calculated)
        periods_per_year: Periods per year for annualization

    Returns:
        Calmar Ratio (float)

    Examples:
        >>> equity = pd.Series([100, 105, 103, 110, 108])
        >>> calmar = calculate_calmar_ratio(equity)
        >>> print(f"Calmar Ratio: {calmar:.2f}")

    Note:
        - Calmar Ratio = CAGR / |Max Drawdown|
        - Higher is better (more return per unit of drawdown risk)
        - Undefined if max drawdown is zero (returns NaN)
    """
    if len(equity_curve) < 2:
        logger.warning("Insufficient data for Calmar Ratio calculation (< 2 observations)")
        return np.nan

    # Calculate CAGR (Compound Annual Growth Rate)
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    n_periods = len(equity_curve) - 1

    if initial_value <= 0 or final_value <= 0:
        logger.warning("Non-positive equity values, cannot calculate CAGR")
        return np.nan

    # CAGR = (final_value / initial_value) ^ (periods_per_year / n_periods) - 1
    cagr = (final_value / initial_value) ** (periods_per_year / n_periods) - 1

    # Calculate max drawdown if not provided
    if max_drawdown is None:
        max_drawdown = calculate_max_drawdown(equity_curve)

    # Take absolute value of max drawdown
    abs_max_dd = abs(max_drawdown)

    if abs_max_dd == 0:
        logger.warning("Zero max drawdown, Calmar Ratio is undefined")
        return np.nan

    # Calculate Calmar Ratio
    calmar = cagr / abs_max_dd

    return float(calmar)


# =============================================================================
# RiskAnalytics Class
# =============================================================================

class RiskAnalytics:
    """
    Comprehensive risk analytics for portfolio backtests.

    This class provides methods for calculating:
    - Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    - Marginal and component VaR for risk decomposition
    - Correlation matrices for assets and strategies
    - Stress testing with renewable energy scenarios
    - Information ratio

    Args:
        result: BacktestResult from backtesting engine
        config: Optional configuration dict (if None, loads from get_config())
        benchmark_returns: Optional benchmark return series for Information Ratio

    Examples:
        >>> from src.optimization import RiskAnalytics
        >>> from src.backtesting import BacktestEngine
        >>>
        >>> # Run backtest
        >>> result = engine.run(price_data, signals)
        >>>
        >>> # Create risk analytics
        >>> risk = RiskAnalytics(result)
        >>>
        >>> # Calculate VaR
        >>> var = risk.calculate_portfolio_var(confidence_level=0.95)
        >>> print(f"95% VaR: ${var['var_dollars']:,.2f}")
        >>>
        >>> # Risk decomposition
        >>> decomp = risk.calculate_risk_decomposition()
        >>> print(decomp)
        >>>
        >>> # Stress testing
        >>> scenarios = risk.run_scenario_analysis(['renewable_drought', 'price_spike'])
        >>> print(scenarios)
    """

    def __init__(
        self,
        result: BacktestResult,
        config: Optional[Dict] = None,
        benchmark_returns: Optional[pd.Series] = None
    ):
        """Initialize risk analytics."""
        self.result = result
        self.benchmark_returns = benchmark_returns

        # Load configuration
        if config is None:
            config = get_config()
        self.config = config
        self.risk_config = config.get('risk', {})

        # Store BacktestResult components
        self.equity_curve = result.equity_curve
        self.trades = result.trades
        self.portfolio_history = result.portfolio_history

        # Validate required data
        if self.equity_curve is None or len(self.equity_curve) == 0:
            raise ValueError("BacktestResult must contain equity_curve data")

        # Calculate returns
        self.returns = self.equity_curve.pct_change().dropna()

        # Extract asset and strategy returns
        self.asset_returns = self._extract_asset_returns()
        self.strategy_returns = self._extract_strategy_returns()

        # Get current portfolio state
        self.current_equity = self.equity_curve.iloc[-1] if len(self.equity_curve) > 0 else 0
        self.current_weights = self._get_current_weights()

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RiskAnalytics initialized with {len(self.returns)} return observations")

    def calculate_portfolio_var(
        self,
        confidence_level: Optional[float] = None,
        method: Optional[str] = None,
        horizon: int = 1
    ) -> Dict:
        """
        Calculate portfolio-level Value at Risk.

        Args:
            confidence_level: Confidence level (default from config)
            method: 'historical' or 'parametric' (default from config)
            horizon: Time horizon in periods (default 1)

        Returns:
            Dict with 'var_pct', 'var_dollars', 'confidence_level', 'method', 'horizon'
        """
        # Use config defaults if not specified
        if confidence_level is None:
            confidence_level = self.risk_config.get('var', {}).get('default_confidence', 0.95)
        if method is None:
            method = self.risk_config.get('var', {}).get('method', 'historical')

        # Calculate VaR
        var_pct = calculate_var(self.returns, confidence_level, method)

        # Scale to horizon for parametric method
        if method == 'parametric' and horizon > 1:
            var_pct = var_pct * np.sqrt(horizon)

        # Convert to dollar terms
        var_dollars = var_pct * self.current_equity

        result = {
            'var_pct': var_pct,
            'var_dollars': var_dollars,
            'confidence_level': confidence_level,
            'method': method,
            'horizon': horizon
        }

        self.logger.info(
            f"Portfolio VaR ({confidence_level:.0%}, {method}): "
            f"{var_pct:.2%} (${var_dollars:,.2f})"
        )

        return result

    def calculate_portfolio_cvar(
        self,
        confidence_level: Optional[float] = None,
        method: Optional[str] = None
    ) -> Dict:
        """
        Calculate portfolio-level Conditional Value at Risk (CVaR).

        Args:
            confidence_level: Confidence level (default from config)
            method: 'historical' or 'parametric' (default from config)

        Returns:
            Dict with 'cvar_pct', 'cvar_dollars', 'confidence_level', 'method'
        """
        # Use config defaults if not specified
        if confidence_level is None:
            confidence_level = self.risk_config.get('cvar', {}).get('default_confidence', 0.95)
        if method is None:
            method = self.risk_config.get('cvar', {}).get('method', 'historical')

        # Calculate CVaR
        cvar_pct = calculate_cvar(self.returns, confidence_level, method)

        # Convert to dollar terms
        cvar_dollars = cvar_pct * self.current_equity

        result = {
            'cvar_pct': cvar_pct,
            'cvar_dollars': cvar_dollars,
            'confidence_level': confidence_level,
            'method': method
        }

        self.logger.info(
            f"Portfolio CVaR ({confidence_level:.0%}, {method}): "
            f"{cvar_pct:.2%} (${cvar_dollars:,.2f})"
        )

        return result

    def calculate_marginal_var(
        self,
        confidence_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate marginal VaR for each asset.

        Marginal VaR represents the sensitivity of portfolio VaR to a small
        change in asset weight. It's approximated using a beta-based approach:
        marginal_var = asset_beta * portfolio_var, where beta measures the
        asset's contribution to portfolio risk via correlation and relative volatility.

        This is computationally efficient and provides a reasonable approximation
        for marginal risk contribution.

        Args:
            confidence_level: Confidence level (default from config)

        Returns:
            Dict mapping asset → marginal_var
        """
        if confidence_level is None:
            confidence_level = self.risk_config.get('var', {}).get('default_confidence', 0.95)

        # Calculate base portfolio VaR
        base_var = self.calculate_portfolio_var(confidence_level)['var_pct']

        marginal_vars = {}

        # Calculate marginal VaR for each asset using beta-based approximation
        for asset in self.asset_returns.columns:
            if asset not in self.current_weights or self.current_weights[asset] == 0:
                marginal_vars[asset] = 0.0
                continue

            # Get asset returns
            asset_ret = self.asset_returns[asset].dropna()

            # Align with portfolio returns
            aligned_returns = pd.concat([self.returns, asset_ret], axis=1, join='inner')
            if len(aligned_returns) < 2:
                marginal_vars[asset] = 0.0
                continue

            port_ret = aligned_returns.iloc[:, 0]
            asset_ret_aligned = aligned_returns.iloc[:, 1]

            # Marginal VaR approximation: Cov(asset, portfolio) / Var(portfolio) * portfolio_var
            if port_ret.std() == 0:
                marginal_vars[asset] = 0.0
            else:
                correlation = port_ret.corr(asset_ret_aligned)
                asset_beta = correlation * (asset_ret_aligned.std() / port_ret.std()) if port_ret.std() != 0 else 0
                marginal_var = asset_beta * base_var
                marginal_vars[asset] = marginal_var

        self.logger.info(f"Calculated marginal VaR for {len(marginal_vars)} assets")

        return marginal_vars

    def calculate_component_var(
        self,
        confidence_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate component VaR (risk contribution) for each asset.

        Component VaR = asset_weight * marginal_var

        The sum of component VaRs equals the portfolio VaR.

        Args:
            confidence_level: Confidence level (default from config)

        Returns:
            Dict mapping asset → component_var
        """
        # Calculate marginal VaRs
        marginal_vars = self.calculate_marginal_var(confidence_level)

        # Calculate component VaRs
        component_vars = {}
        for asset, marginal_var in marginal_vars.items():
            weight = self.current_weights.get(asset, 0.0)
            component_vars[asset] = weight * marginal_var

        # Verify sum equals portfolio VaR (approximately)
        total_component_var = sum(component_vars.values())
        portfolio_var = self.calculate_portfolio_var(confidence_level)['var_pct']

        self.logger.info(
            f"Component VaR sum: {total_component_var:.4f}, "
            f"Portfolio VaR: {portfolio_var:.4f}"
        )

        return component_vars

    def calculate_risk_decomposition(
        self,
        confidence_level: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Comprehensive risk decomposition by asset.

        Args:
            confidence_level: Confidence level (default from config)

        Returns:
            DataFrame with columns ['asset', 'weight', 'marginal_var', 'component_var', 'contribution_pct']
        """
        if confidence_level is None:
            confidence_level = self.risk_config.get('var', {}).get('default_confidence', 0.95)

        # Calculate VaR components
        portfolio_var = self.calculate_portfolio_var(confidence_level)['var_pct']
        marginal_vars = self.calculate_marginal_var(confidence_level)
        component_vars = self.calculate_component_var(confidence_level)

        # Build decomposition DataFrame
        decomposition_data = []
        for asset in self.current_weights.keys():
            weight = self.current_weights[asset]
            marginal_var = marginal_vars.get(asset, 0.0)
            component_var = component_vars.get(asset, 0.0)
            contribution_pct = (component_var / portfolio_var * 100) if portfolio_var != 0 else 0.0

            decomposition_data.append({
                'asset': asset,
                'weight': weight,
                'marginal_var': marginal_var,
                'component_var': component_var,
                'contribution_pct': contribution_pct
            })

        decomposition_df = pd.DataFrame(decomposition_data)

        # Sort by contribution percentage
        decomposition_df = decomposition_df.sort_values('contribution_pct', ascending=False)

        self.logger.info(f"Risk decomposition completed for {len(decomposition_df)} assets")

        return decomposition_df

    def calculate_strategy_risk_decomposition(
        self,
        confidence_level: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Risk decomposition by strategy with component VaR.

        Computes strategy-level weights, marginal VaR (using beta to portfolio),
        component VaR (weight * marginal_var), and contribution percentages.

        Args:
            confidence_level: Confidence level (default from config)

        Returns:
            DataFrame with columns ['strategy','weight','marginal_var','component_var','contribution_pct']
        """
        if self.strategy_returns.empty:
            self.logger.warning("No strategy returns available for decomposition")
            return pd.DataFrame()

        if confidence_level is None:
            confidence_level = self.risk_config.get('var', {}).get('default_confidence', 0.95)

        # Calculate portfolio VaR
        portfolio_var = self.calculate_portfolio_var(confidence_level)['var_pct']

        # Calculate strategy weights from latest trades
        strategy_weights = self._get_strategy_weights()

        # Calculate strategy metrics with component VaR
        strategy_metrics = []
        for strategy in self.strategy_returns.columns:
            strategy_ret = self.strategy_returns[strategy].dropna()

            if len(strategy_ret) < 2:
                continue

            # Align with portfolio returns
            aligned_returns = pd.concat([self.returns, strategy_ret], axis=1, join='inner')
            if len(aligned_returns) < 2:
                continue

            port_ret = aligned_returns.iloc[:, 0]
            strat_ret = aligned_returns.iloc[:, 1]

            # Calculate beta to portfolio returns (marginal VaR proxy)
            if port_ret.std() == 0:
                beta = 0.0
            else:
                correlation = port_ret.corr(strat_ret)
                beta = correlation * (strat_ret.std() / port_ret.std())

            # Marginal VaR = beta * portfolio_var
            marginal_var = beta * portfolio_var

            # Get strategy weight
            weight = strategy_weights.get(strategy, 0.0)

            # Component VaR = weight * marginal_var
            component_var = weight * marginal_var

            # Contribution percentage
            contribution_pct = (component_var / portfolio_var * 100) if portfolio_var != 0 else 0.0

            strategy_metrics.append({
                'strategy': strategy,
                'weight': weight,
                'marginal_var': marginal_var,
                'component_var': component_var,
                'contribution_pct': contribution_pct
            })

        strategy_df = pd.DataFrame(strategy_metrics)

        # Sort by contribution percentage
        strategy_df = strategy_df.sort_values('contribution_pct', ascending=False)

        self.logger.info(f"Strategy risk decomposition completed for {len(strategy_df)} strategies")

        return strategy_df

    def calculate_correlation_matrices(
        self,
        window: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlation matrices for assets and strategies.

        Args:
            window: Rolling window size (default from config)

        Returns:
            Dict with:
            - 'asset_corr': Static correlation matrix (n_assets x n_assets DataFrame)
            - 'strategy_corr': Static correlation matrix (n_strategies x n_strategies DataFrame)
            - 'rolling_corr': Time-varying correlation matrix with MultiIndex
                              (timestamp, asset) x (asset) structure.
                              Use get_rolling_correlation() helper to extract pairwise series.

        Note:
            rolling_corr is a MultiIndex DataFrame produced by pandas rolling().corr().
            To extract a single pairwise rolling correlation, use:
            rolling_corr.xs(asset1, level=1)[asset2]
        """
        if window is None:
            window = self.risk_config.get('correlation', {}).get('window', 60)

        method = self.risk_config.get('correlation', {}).get('method', 'pearson')
        min_periods = self.risk_config.get('correlation', {}).get('min_periods', 30)

        result = {}

        # Asset correlation matrix
        if not self.asset_returns.empty:
            result['asset_corr'] = calculate_correlation_matrix(
                self.asset_returns, method, min_periods
            )
        else:
            result['asset_corr'] = pd.DataFrame()

        # Strategy correlation matrix
        if not self.strategy_returns.empty:
            result['strategy_corr'] = calculate_correlation_matrix(
                self.strategy_returns, method, min_periods
            )
        else:
            result['strategy_corr'] = pd.DataFrame()

        # Rolling correlation (for asset returns)
        if not self.asset_returns.empty and len(self.asset_returns) >= window:
            rolling_corr = self.asset_returns.rolling(window=window).corr()
            result['rolling_corr'] = rolling_corr
        else:
            result['rolling_corr'] = pd.DataFrame()

        self.logger.info("Correlation matrices calculated")

        return result

    def get_rolling_correlation(
        self,
        asset1: str,
        asset2: str,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Extract pairwise rolling correlation between two assets.

        This is a convenience helper to extract a simple time series from the
        complex MultiIndex rolling_corr structure.

        Args:
            asset1: First asset name
            asset2: Second asset name
            window: Rolling window size (if None, uses correlation matrices)

        Returns:
            Series of rolling correlation values indexed by timestamp

        Examples:
            >>> risk = RiskAnalytics(result)
            >>> rolling_corr = risk.get_rolling_correlation('asset1', 'asset2')
            >>> rolling_corr.plot()
        """
        # Get correlation matrices
        corr_matrices = self.calculate_correlation_matrices(window)
        rolling_corr = corr_matrices.get('rolling_corr', pd.DataFrame())

        if rolling_corr.empty:
            self.logger.warning("No rolling correlation data available")
            return pd.Series(dtype=float)

        # Extract pairwise correlation using MultiIndex
        try:
            pairwise_corr = rolling_corr.xs(asset1, level=1)[asset2]
            return pairwise_corr
        except (KeyError, IndexError) as e:
            self.logger.error(f"Cannot extract rolling correlation for {asset1} and {asset2}: {e}")
            return pd.Series(dtype=float)

    def run_stress_test(
        self,
        scenario_name: str,
        scenario_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run stress test scenario.

        Args:
            scenario_name: Name of scenario (e.g., 'renewable_drought', 'price_spike')
            scenario_params: Scenario parameters (if None, loads from config)

        Returns:
            Dict with stressed metrics (var, cvar, max_drawdown, total_return, etc.)
        """
        # Load scenario parameters from config if not provided
        if scenario_params is None:
            stress_scenarios = self.risk_config.get('stress_scenarios', {})

            # Check in historical and hypothetical sections
            if scenario_name in stress_scenarios.get('historical', {}):
                scenario_params = stress_scenarios['historical'][scenario_name]
            elif scenario_name in stress_scenarios.get('hypothetical', {}):
                scenario_params = stress_scenarios['hypothetical'][scenario_name]
            else:
                raise ValueError(
                    f"Scenario '{scenario_name}' not found in config. "
                    f"Available: {list(stress_scenarios.get('historical', {}).keys()) + list(stress_scenarios.get('hypothetical', {}).keys())}"
                )

        self.logger.info(f"Running stress test: {scenario_name}")

        # Apply shocks to returns
        stressed_returns = self._apply_scenario_shock(
            self.returns.copy(),
            scenario_params
        )

        # Calculate stressed metrics
        method = self.risk_config.get('var', {}).get('method', 'historical')
        confidence_level = self.risk_config.get('var', {}).get('default_confidence', 0.95)

        stressed_var = calculate_var(stressed_returns, confidence_level, method)
        stressed_cvar = calculate_cvar(stressed_returns, confidence_level, method)

        # Calculate drawdown on stressed equity curve
        stressed_equity = (1 + stressed_returns).cumprod() * self.equity_curve.iloc[0]
        stressed_max_dd = calculate_max_drawdown(stressed_equity)

        # Calculate total return
        stressed_total_return = stressed_equity.iloc[-1] / stressed_equity.iloc[0] - 1

        # Calculate Sharpe ratio
        stressed_sharpe = calculate_sharpe_ratio(stressed_returns)

        result = {
            'scenario': scenario_name,
            'description': scenario_params.get('description', ''),
            'var': stressed_var,
            'cvar': stressed_cvar,
            'max_drawdown': stressed_max_dd,
            'total_return': stressed_total_return,
            'sharpe_ratio': stressed_sharpe,
            'volatility': stressed_returns.std(),
            'final_equity': stressed_equity.iloc[-1]
        }

        self.logger.info(f"Stress test {scenario_name} complete: VaR={stressed_var:.2%}, DD={stressed_max_dd:.2%}")

        return result

    def run_scenario_analysis(
        self,
        scenario_names: List[str]
    ) -> pd.DataFrame:
        """
        Run multiple scenarios and compare results.

        Args:
            scenario_names: List of scenario names to run

        Returns:
            DataFrame with scenarios as rows, metrics as columns
        """
        scenarios = []

        for scenario_name in scenario_names:
            try:
                result = self.run_stress_test(scenario_name)
                scenarios.append(result)
            except Exception as e:
                self.logger.error(f"Error running scenario {scenario_name}: {e}")
                continue

        if not scenarios:
            self.logger.warning("No scenarios completed successfully")
            return pd.DataFrame()

        scenarios_df = pd.DataFrame(scenarios)

        self.logger.info(f"Scenario analysis complete: {len(scenarios)} scenarios")

        return scenarios_df

    def calculate_information_ratio(self) -> float:
        """
        Calculate Information Ratio for portfolio.

        Returns:
            Information Ratio (float)
        """
        periods_per_year = self.risk_config.get('information_ratio', {}).get('periods_per_year', 252)

        ir = calculate_information_ratio(
            self.returns,
            self.benchmark_returns,
            periods_per_year
        )

        self.logger.info(f"Information Ratio: {ir:.2f}")

        return ir

    def calculate_all_metrics(self) -> Dict:
        """
        Calculate comprehensive risk metrics.

        Returns:
            Dict with all risk metrics
        """
        self.logger.info("Calculating all risk metrics...")

        # Get confidence levels from config
        confidence_levels = self.risk_config.get('var', {}).get('confidence_levels', [0.90, 0.95, 0.99])

        metrics = {}

        # VaR and CVaR for each confidence level
        metrics['var'] = {}
        metrics['cvar'] = {}
        for cl in confidence_levels:
            var_result = self.calculate_portfolio_var(confidence_level=cl)
            metrics['var'][f'{cl:.0%}'] = var_result

            cvar_result = self.calculate_portfolio_cvar(confidence_level=cl)
            metrics['cvar'][f'{cl:.0%}'] = cvar_result

        # Risk decomposition
        metrics['risk_decomposition'] = self.calculate_risk_decomposition()
        metrics['strategy_risk_decomposition'] = self.calculate_strategy_risk_decomposition()

        # Correlation matrices
        metrics['correlation'] = self.calculate_correlation_matrices()

        # Information ratio (always compute, uses zero benchmark if None)
        metrics['information_ratio'] = self.calculate_information_ratio()

        # Realized volatility
        vol_window = self.risk_config.get('volatility', {}).get('window', 252)
        metrics['realized_volatility'] = calculate_realized_volatility(
            self.returns,
            window=vol_window
        )

        # Implied volatility (EWMA proxy)
        lambda_ewma = self.risk_config.get('volatility', {}).get('lambda_ewma', 0.94)
        metrics['implied_volatility'] = calculate_implied_volatility(
            self.returns,
            lambda_ewma=lambda_ewma
        )

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(self.returns)
        metrics['sortino_ratio'] = calculate_sortino_ratio(self.returns)
        metrics['max_drawdown'] = calculate_max_drawdown(self.equity_curve)

        # Calmar ratio
        metrics['calmar_ratio'] = calculate_calmar_ratio(
            self.equity_curve,
            max_drawdown=metrics['max_drawdown']
        )

        # Current portfolio state
        metrics['current_equity'] = self.current_equity
        metrics['current_weights'] = self.current_weights

        self.logger.info("All risk metrics calculated")

        return metrics

    def get_risk_summary(self) -> Dict[str, str]:
        """
        Get formatted risk summary.

        Returns:
            Dict with formatted summary strings
        """
        metrics = self.calculate_all_metrics()

        # Format key metrics
        summary = {}

        # VaR
        var_95 = metrics['var'].get('95%', {})
        if var_95:
            summary['95% VaR'] = f"{var_95['var_pct']:.2%} (${var_95['var_dollars']:,.2f})"

        # CVaR
        cvar_95 = metrics['cvar'].get('95%', {})
        if cvar_95:
            summary['95% CVaR'] = f"{cvar_95['cvar_pct']:.2%} (${cvar_95['cvar_dollars']:,.2f})"

        # Sharpe
        if metrics.get('sharpe_ratio') is not None:
            summary['Sharpe Ratio'] = f"{metrics['sharpe_ratio']:.2f}"

        # Sortino
        if metrics.get('sortino_ratio') is not None:
            summary['Sortino Ratio'] = f"{metrics['sortino_ratio']:.2f}"

        # Max Drawdown
        if metrics.get('max_drawdown') is not None:
            summary['Max Drawdown'] = f"{metrics['max_drawdown']:.2%}"

        # Information Ratio
        if metrics.get('information_ratio') is not None and not np.isnan(metrics['information_ratio']):
            summary['Information Ratio'] = f"{metrics['information_ratio']:.2f}"

        # Calmar Ratio
        if metrics.get('calmar_ratio') is not None and not np.isnan(metrics['calmar_ratio']):
            summary['Calmar Ratio'] = f"{metrics['calmar_ratio']:.2f}"

        # Implied Volatility (current value)
        if 'implied_volatility' in metrics and not metrics['implied_volatility'].empty:
            current_impl_vol = metrics['implied_volatility'].iloc[-1]
            if not np.isnan(current_impl_vol):
                summary['Implied Volatility'] = f"{current_impl_vol:.2%}"

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert risk metrics to DataFrame.

        Returns:
            DataFrame with metric names and values
        """
        summary = self.get_risk_summary()

        df = pd.DataFrame([
            {'Metric': key, 'Value': value}
            for key, value in summary.items()
        ])

        return df

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_asset_returns(self) -> pd.DataFrame:
        """
        Extract returns for each asset from portfolio_history.

        Returns:
            DataFrame with asset returns as columns
        """
        if self.portfolio_history is None or self.portfolio_history.empty:
            self.logger.warning("No portfolio_history available for asset returns")
            return pd.DataFrame()

        # Find val_{asset} columns
        val_columns = [col for col in self.portfolio_history.columns if col.startswith('val_')]

        if not val_columns:
            self.logger.warning("No val_ columns found in portfolio_history")
            return pd.DataFrame()

        # Extract asset values
        asset_values = self.portfolio_history[val_columns].copy()

        # Rename columns to remove 'val_' prefix
        asset_values.columns = [col.replace('val_', '') for col in asset_values.columns]

        # Calculate returns
        asset_returns = asset_values.pct_change().dropna()

        return asset_returns

    def _extract_strategy_returns(self) -> pd.DataFrame:
        """
        Extract returns for each strategy from trades.

        Returns:
            DataFrame with strategy returns as columns
        """
        if self.trades is None or self.trades.empty:
            self.logger.warning("No trades available for strategy returns")
            return pd.DataFrame()

        if 'strategy' not in self.trades.columns:
            self.logger.warning("No 'strategy' column in trades")
            return pd.DataFrame()

        # Use net_realized_pnl if available, otherwise realized_pnl
        pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in self.trades.columns else 'realized_pnl'

        # Group trades by strategy and timestamp
        strategy_pnl = self.trades.groupby(['timestamp', 'strategy'])[pnl_col].sum().unstack(fill_value=0)

        # Reindex to equity curve timestamps
        strategy_pnl = strategy_pnl.reindex(self.equity_curve.index, fill_value=0)

        # Calculate cumulative PnL per strategy
        strategy_cum_pnl = strategy_pnl.cumsum()

        # Calculate returns from PnL
        # Initialize with starting equity divided by number of strategies
        initial_value = self.equity_curve.iloc[0] / len(strategy_pnl.columns) if len(strategy_pnl.columns) > 0 else 1.0
        strategy_values = strategy_cum_pnl + initial_value

        # Calculate returns
        strategy_returns = strategy_values.pct_change().dropna()

        return strategy_returns

    def _get_current_weights(self) -> Dict[str, float]:
        """
        Get current portfolio weights.

        Returns:
            Dict mapping asset → weight
        """
        if self.portfolio_history is None or self.portfolio_history.empty:
            self.logger.warning("No portfolio_history available for weights")
            return {}

        # Get latest row
        latest = self.portfolio_history.iloc[-1]

        # Find val_{asset} columns
        val_columns = [col for col in self.portfolio_history.columns if col.startswith('val_')]

        if not val_columns:
            self.logger.warning("No val_ columns found for weights")
            return {}

        # Get equity
        equity = latest.get('equity', 0)

        if equity == 0:
            self.logger.warning("Zero equity, cannot calculate weights")
            return {}

        # Calculate weights
        weights = {}
        for col in val_columns:
            asset = col.replace('val_', '')
            value = latest[col]
            weight = value / equity
            weights[asset] = weight

        return weights

    def _get_strategy_weights(self) -> Dict[str, float]:
        """
        Get current strategy weights based on latest strategy values.

        Returns:
            Dict mapping strategy → weight
        """
        if self.trades is None or self.trades.empty:
            self.logger.warning("No trades available for strategy weights")
            return {}

        if 'strategy' not in self.trades.columns:
            self.logger.warning("No 'strategy' column in trades")
            return {}

        # Use net_realized_pnl if available, otherwise realized_pnl
        pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in self.trades.columns else 'realized_pnl'

        # Group trades by strategy and calculate cumulative PnL
        strategy_pnl = self.trades.groupby(['timestamp', 'strategy'])[pnl_col].sum().unstack(fill_value=0)

        # Reindex to equity curve timestamps
        strategy_pnl = strategy_pnl.reindex(self.equity_curve.index, fill_value=0)

        # Calculate cumulative PnL per strategy
        strategy_cum_pnl = strategy_pnl.cumsum()

        # Get latest cumulative PnL per strategy
        latest_pnl = strategy_cum_pnl.iloc[-1] if len(strategy_cum_pnl) > 0 else pd.Series()

        # Initialize with starting equity divided by number of strategies
        initial_value = self.equity_curve.iloc[0] / len(strategy_pnl.columns) if len(strategy_pnl.columns) > 0 else 1.0

        # Calculate strategy values
        strategy_values = latest_pnl + initial_value

        # Calculate total value
        total_value = strategy_values.sum()

        if total_value == 0:
            self.logger.warning("Zero total strategy value, cannot calculate weights")
            # Return equal weights
            n_strategies = len(strategy_values)
            return {strategy: 1.0 / n_strategies for strategy in strategy_values.index}

        # Calculate weights
        weights = {strategy: value / total_value for strategy, value in strategy_values.items()}

        return weights

    def _apply_scenario_shock(
        self,
        base_returns: pd.Series,
        scenario_params: Dict
    ) -> pd.Series:
        """
        Apply shock to returns based on scenario parameters.

        Shocks are applied only within the first duration_days observations.
        Price shocks are applied multiplicatively (suitable for spikes).
        Volatility multiplier scales deviations from mean only within the shock period.
        Wind/solar shocks are mapped to corresponding asset returns if available.

        Args:
            base_returns: Base return series
            scenario_params: Scenario parameters

        Returns:
            Shocked return series
        """
        shocked_returns = base_returns.copy()

        # Determine shock duration
        duration_days = scenario_params.get('duration_days')
        if duration_days is None:
            # If no duration specified, apply to entire series
            n = len(shocked_returns)
        else:
            # Apply shock only to first duration_days observations
            n = min(duration_days, len(shocked_returns))

        # Create boolean mask for shock period
        mask = pd.Series(False, index=shocked_returns.index)
        mask.iloc[:n] = True

        # Price shock (multiplicative for spikes)
        if 'price_shock' in scenario_params:
            price_shock = scenario_params['price_shock']
            # Multiplicative shock: return = (1 + original_return) * (1 + shock) - 1
            shocked_returns.loc[mask] = (1 + shocked_returns.loc[mask]) * (1 + price_shock) - 1
            self.logger.debug(f"Applied price shock {price_shock:.2%} multiplicatively to first {n} periods")

        # Volatility multiplier (scale deviations from mean only within mask)
        if 'volatility_multiplier' in scenario_params:
            vol_mult = scenario_params['volatility_multiplier']
            # Calculate mean from shock period only
            mean_shock_period = shocked_returns.loc[mask].mean()
            # Scale deviations within mask
            shocked_returns.loc[mask] = mean_shock_period + (shocked_returns.loc[mask] - mean_shock_period) * vol_mult
            self.logger.debug(f"Applied volatility multiplier {vol_mult}x to first {n} periods")

        # Wind/solar shocks (if asset-level data exists)
        # Check if we have asset returns to apply renewable shocks
        if 'wind_shock' in scenario_params or 'solar_shock' in scenario_params:
            if not self.asset_returns.empty:
                # Attempt to map renewable shocks to asset columns
                wind_assets = [col for col in self.asset_returns.columns if 'wind' in col.lower()]
                solar_assets = [col for col in self.asset_returns.columns if 'solar' in col.lower()]

                if 'wind_shock' in scenario_params and wind_assets:
                    wind_shock = scenario_params['wind_shock']
                    # Apply shock to wind asset returns (multiplicative)
                    for asset in wind_assets:
                        asset_ret = self.asset_returns[asset].reindex(shocked_returns.index, fill_value=0)
                        # Infer portfolio sensitivity to this asset
                        weight = self.current_weights.get(asset, 0)
                        # Adjust portfolio returns proportionally
                        shocked_returns.loc[mask] += weight * asset_ret.loc[mask] * wind_shock
                    self.logger.debug(f"Applied wind shock {wind_shock:.2%} to {len(wind_assets)} wind assets")
                elif 'wind_shock' in scenario_params:
                    self.logger.warning("wind_shock specified but no wind assets found in portfolio")

                if 'solar_shock' in scenario_params and solar_assets:
                    solar_shock = scenario_params['solar_shock']
                    # Apply shock to solar asset returns (multiplicative)
                    for asset in solar_assets:
                        asset_ret = self.asset_returns[asset].reindex(shocked_returns.index, fill_value=0)
                        # Infer portfolio sensitivity to this asset
                        weight = self.current_weights.get(asset, 0)
                        # Adjust portfolio returns proportionally
                        shocked_returns.loc[mask] += weight * asset_ret.loc[mask] * solar_shock
                    self.logger.debug(f"Applied solar shock {solar_shock:.2%} to {len(solar_assets)} solar assets")
                elif 'solar_shock' in scenario_params:
                    self.logger.warning("solar_shock specified but no solar assets found in portfolio")
            else:
                self.logger.warning(
                    "wind_shock/solar_shock specified but asset-level data not available. "
                    "Renewable shocks cannot be applied without asset mapping."
                )

        # Wind/solar curtailment (if specified)
        if 'wind_curtailment' in scenario_params or 'solar_curtailment' in scenario_params:
            if not self.asset_returns.empty:
                wind_assets = [col for col in self.asset_returns.columns if 'wind' in col.lower()]
                solar_assets = [col for col in self.asset_returns.columns if 'solar' in col.lower()]

                if 'wind_curtailment' in scenario_params and wind_assets:
                    wind_curtailment = scenario_params['wind_curtailment']
                    for asset in wind_assets:
                        asset_ret = self.asset_returns[asset].reindex(shocked_returns.index, fill_value=0)
                        weight = self.current_weights.get(asset, 0)
                        # Curtailment reduces generation (negative impact)
                        shocked_returns.loc[mask] -= weight * asset_ret.loc[mask] * wind_curtailment
                    self.logger.debug(f"Applied wind curtailment {wind_curtailment:.2%} to {len(wind_assets)} assets")

                if 'solar_curtailment' in scenario_params and solar_assets:
                    solar_curtailment = scenario_params['solar_curtailment']
                    for asset in solar_assets:
                        asset_ret = self.asset_returns[asset].reindex(shocked_returns.index, fill_value=0)
                        weight = self.current_weights.get(asset, 0)
                        # Curtailment reduces generation (negative impact)
                        shocked_returns.loc[mask] -= weight * asset_ret.loc[mask] * solar_curtailment
                    self.logger.debug(f"Applied solar curtailment {solar_curtailment:.2%} to {len(solar_assets)} assets")
            else:
                self.logger.warning("Curtailment shocks specified but asset-level data not available")

        return shocked_returns
