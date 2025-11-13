"""
Portfolio Optimization Module

Comprehensive portfolio optimization framework supporting:
- Mean-Variance (Markowitz) optimization
- Risk Parity (equal risk contribution)
- Black-Litterman model with views
- Minimum CVaR (Conditional Value at Risk) optimization
- Renewable energy-specific constraints and objectives
- Multi-period optimization with transaction costs and rebalancing

This module integrates with BacktestResult for historical returns, RiskAnalytics
for covariance matrices and risk metrics, and DataManager for persistence.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import sqrtm
import cvxpy as cp
import joblib

from src.backtesting.engine import BacktestResult
from src.optimization.risk_analytics import RiskAnalytics, calculate_correlation_matrix
from src.config.load_config import get_config
from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)


# =============================================================================
# OptimizationResult Dataclass
# =============================================================================

@dataclass
class OptimizationResult:
    """
    Encapsulates portfolio optimization results.

    Attributes:
        weights: Asset -> weight mapping
        expected_return: Expected portfolio return
        expected_risk: Expected portfolio volatility or CVaR
        sharpe_ratio: Risk-adjusted return
        method: Optimization method used
        constraints_satisfied: Whether all constraints met
        optimization_status: cvxpy status (optimal, infeasible, etc.)
        efficient_frontier: Risk-return points if computed
        metadata: Additional method-specific data
    """
    weights: pd.Series
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    method: str
    constraints_satisfied: bool
    optimization_status: str
    efficient_frontier: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """
    Calculate portfolio return from weights.

    Args:
        weights: Asset weights array
        expected_returns: Expected returns array

    Returns:
        Portfolio return
    """
    return float(weights @ expected_returns)


def calculate_portfolio_risk(weights: np.ndarray, covariance: np.ndarray) -> float:
    """
    Calculate portfolio risk (volatility) from weights.

    Args:
        weights: Asset weights array
        covariance: Covariance matrix

    Returns:
        Portfolio volatility
    """
    return float(np.sqrt(weights @ covariance @ weights))


def create_picking_matrix(views: Dict[str, float], asset_names: List[str]) -> np.ndarray:
    """
    Create picking matrix for Black-Litterman views.

    Args:
        views: Dict mapping asset name -> view return
        asset_names: List of all asset names

    Returns:
        P matrix (n_views x n_assets)
    """
    n_views = len(views)
    n_assets = len(asset_names)
    P = np.zeros((n_views, n_assets))

    for i, (asset, _) in enumerate(views.items()):
        if asset in asset_names:
            j = asset_names.index(asset)
            P[i, j] = 1.0

    return P


def validate_optimization_inputs(returns: pd.DataFrame, covariance: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate inputs for optimization.

    Args:
        returns: DataFrame with asset returns
        covariance: Covariance matrix

    Returns:
        Tuple of (is_valid, issues list)
    """
    issues = []

    # Check returns DataFrame
    if returns.empty:
        issues.append("Returns DataFrame is empty")

    if returns.isna().any().any():
        issues.append("Returns contain NaN values")

    if len(returns) < 30:
        issues.append(f"Insufficient data: {len(returns)} observations (need at least 30)")

    # Check covariance matrix
    if covariance.shape[0] != covariance.shape[1]:
        issues.append("Covariance matrix is not square")

    # Check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues < -1e-8):
        issues.append("Covariance matrix is not positive semi-definite")

    is_valid = len(issues) == 0
    return is_valid, issues


# =============================================================================
# BaseOptimizer Abstract Class
# =============================================================================

class BaseOptimizer(ABC):
    """
    Abstract base class for portfolio optimizers.

    Defines common interface and shared functionality for all optimization methods.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    ):
        """
        Initialize base optimizer.

        Args:
            returns: DataFrame with asset returns (columns = assets, index = timestamps)
            config: Optional configuration dict
            risk_analytics: Optional RiskAnalytics instance
        """
        self.returns = returns
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)

        # Load config
        if config is None:
            config = get_config()
        self.config = config
        self.opt_config = config.get('optimization', {})

        # Calculate expected returns (mean)
        self.expected_returns = returns.mean().values

        # Get covariance matrix
        if risk_analytics is not None:
            # Use RiskAnalytics covariance
            self.risk_analytics = risk_analytics
            asset_returns_df = risk_analytics.asset_returns
            if not asset_returns_df.empty:
                self.covariance = asset_returns_df.cov().values
            else:
                self.covariance = returns.cov().values
        else:
            self.covariance = returns.cov().values
            self.risk_analytics = None

        # Add regularization for numerical stability
        reg = self.opt_config.get('regularization', 1e-5)
        self.covariance += np.eye(self.n_assets) * reg

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize constraints list
        self.constraints = []

        self.logger.info(f"Initialized {self.__class__.__name__} with {self.n_assets} assets")

    @abstractmethod
    def optimize(self, **kwargs) -> OptimizationResult:
        """
        Run optimization.

        Must be implemented by concrete optimizers.

        Returns:
            OptimizationResult
        """
        raise NotImplementedError("Subclasses must implement optimize()")

    def add_constraint(self, constraint_type: str, params: Dict) -> 'BaseOptimizer':
        """
        Add constraint to optimization problem.

        Args:
            constraint_type: Type of constraint
            params: Constraint parameters

        Returns:
            self (for method chaining)
        """
        self.constraints.append({'type': constraint_type, 'params': params})
        self.logger.debug(f"Added constraint: {constraint_type}")
        return self

    def calculate_portfolio_metrics(self, weights: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Calculate portfolio metrics from weights.

        Args:
            weights: Asset weights

        Returns:
            Dict with metrics
        """
        if isinstance(weights, pd.Series):
            weights = weights.values

        expected_return = calculate_portfolio_return(weights, self.expected_returns)
        expected_risk = calculate_portfolio_risk(weights, self.covariance)

        # Sharpe ratio (assuming risk-free rate = 0 or from config)
        risk_free_rate = self.opt_config.get('mean_variance', {}).get('risk_free_rate', 0.0)
        sharpe_ratio = (expected_return - risk_free_rate) / expected_risk if expected_risk > 0 else 0.0

        return {
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio
        }

    def validate_weights(self, weights: np.ndarray, tolerance: float = 1e-4) -> Tuple[bool, List[str]]:
        """
        Validate optimized weights.

        Args:
            weights: Optimized weights
            tolerance: Tolerance for sum constraint

        Returns:
            Tuple of (is_valid, issues list)
        """
        issues = []

        # Check sum to 1
        if abs(weights.sum() - 1.0) > tolerance:
            issues.append(f"Weights sum to {weights.sum():.4f}, expected 1.0")

        # Check for NaN or inf
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            issues.append("Weights contain NaN or inf values")

        # Check weight bounds from config
        min_weight = self.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_weight = self.opt_config.get('constraints', {}).get('max_weight', 1.0)

        if np.any(weights < min_weight - tolerance):
            issues.append(f"Some weights below minimum {min_weight}")

        if np.any(weights > max_weight + tolerance):
            issues.append(f"Some weights above maximum {max_weight}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def _build_constraints(self, w, w_prev: Optional[np.ndarray] = None) -> List:
        """
        Build cvxpy constraints from config and added constraints.

        Args:
            w: cvxpy Variable for weights
            w_prev: Previous weights (for turnover constraint)

        Returns:
            List of cvxpy constraints
        """
        constraints = []
        constraints_config = self.opt_config.get('constraints', {})

        # 1. Leverage constraints
        max_leverage = constraints_config.get('max_leverage')
        min_leverage = constraints_config.get('min_leverage')
        if max_leverage is not None:
            constraints.append(cp.norm1(w) <= max_leverage)
        if min_leverage is not None:
            constraints.append(cp.norm1(w) >= min_leverage)

        # 2. Sector constraints
        sector_mapping = constraints_config.get('sector_mapping', {})
        max_sector_weight = constraints_config.get('max_sector_weight')
        if sector_mapping and max_sector_weight is not None:
            # Group assets by sector
            sectors = {}
            for asset, sector in sector_mapping.items():
                if sector not in sectors:
                    sectors[sector] = []
                if asset in self.asset_names:
                    sectors[sector].append(self.asset_names.index(asset))

            # Add constraint for each sector
            for sector, indices in sectors.items():
                if indices:
                    constraints.append(cp.sum(w[indices]) <= max_sector_weight)

        # 3. Liquidity constraints
        liquidity_data = constraints_config.get('liquidity_data', {})
        max_position_vs_adv = constraints_config.get('max_position_vs_adv')
        if liquidity_data and max_position_vs_adv is not None:
            for asset, adv in liquidity_data.items():
                if asset in self.asset_names:
                    idx = self.asset_names.index(asset)
                    # Cap weight based on ADV
                    max_weight_adv = max_position_vs_adv * adv
                    constraints.append(w[idx] <= max_weight_adv)

        # 4. Turnover constraint
        max_turnover = constraints_config.get('max_turnover')
        if max_turnover is not None and w_prev is not None:
            constraints.append(cp.norm1(w - w_prev) <= max_turnover)

        # 5. Concentration constraint
        max_concentration = constraints_config.get('max_concentration')
        concentration_n = constraints_config.get('concentration_n', 3)
        if max_concentration is not None:
            # Note: This is approximate since we can't directly enforce "top N" in cvxpy
            # We'll log a warning and skip this constraint
            self.logger.warning(
                "Concentration constraint requires mixed-integer programming, "
                "which is not supported. Skipping concentration constraint."
            )

        # 6. Renewable constraints from config
        renewable_config = self.opt_config.get('renewable_constraints', {})

        # Identify renewable assets from config or by name
        renewable_assets_list = renewable_config.get('renewable_assets', [])
        if not renewable_assets_list:
            # Fallback: detect by name
            renewable_assets_list = [name for name in self.asset_names if 'wind' in name.lower() or 'solar' in name.lower()]

        renewable_idx = [i for i, name in enumerate(self.asset_names) if name in renewable_assets_list]

        # 6a. Capacity factor constraints
        cf_config = renewable_config.get('capacity_factors', {})
        if cf_config.get('enabled', False) and renewable_idx:
            asset_capacities = cf_config.get('asset_capacities', {})
            min_cf = cf_config.get('min_capacity_factor', 0.15)
            max_cf = cf_config.get('max_capacity_factor', 0.95)

            # Compute total capacity
            total_capacity = sum(asset_capacities.get(asset, 100.0) for asset in renewable_assets_list)
            if total_capacity > 0:
                for asset in renewable_assets_list:
                    if asset in self.asset_names:
                        idx = self.asset_names.index(asset)
                        C_a = asset_capacities.get(asset, 100.0)
                        # Approximate min/max generation shares
                        # s_min = (C_a * cf_min) / sum(C_b * cf_max)
                        sum_max_gen = sum(asset_capacities.get(a, 100.0) * max_cf for a in renewable_assets_list)
                        if sum_max_gen > 0:
                            s_min = (C_a * min_cf) / sum_max_gen
                            s_max = (C_a * max_cf) / sum_max_gen
                            constraints.append(w[idx] >= s_min)
                            constraints.append(w[idx] <= s_max)
                            self.logger.debug(f"Capacity factor constraint for {asset}: [{s_min:.4f}, {s_max:.4f}]")

        # 6b. Grid injection limits
        grid_config = renewable_config.get('grid_limits', {})
        if grid_config.get('enabled', False) and renewable_idx:
            max_injection_mw = grid_config.get('max_injection_mw', 1000.0)
            grid_renewable_assets = grid_config.get('renewable_assets', renewable_assets_list)
            grid_renewable_idx = [i for i, name in enumerate(self.asset_names) if name in grid_renewable_assets]

            if grid_renewable_idx:
                # Compute S_max = max_injection_mw / sum(C_b * cf_max)
                cf_config = renewable_config.get('capacity_factors', {})
                asset_capacities = cf_config.get('asset_capacities', {})
                max_cf = cf_config.get('max_capacity_factor', 0.95)
                sum_max_gen = sum(asset_capacities.get(asset, 100.0) * max_cf for asset in grid_renewable_assets)
                if sum_max_gen > 0:
                    S_max = max_injection_mw / sum_max_gen
                    constraints.append(cp.sum(w[grid_renewable_idx]) <= min(1.0, S_max))
                    self.logger.debug(f"Grid injection limit: sum(w[renewable]) <= {min(1.0, S_max):.4f}")
                else:
                    # Fallback: use configured max_renewable_weight
                    max_renewable_weight = grid_config.get('max_renewable_weight', 0.8)
                    constraints.append(cp.sum(w[grid_renewable_idx]) <= max_renewable_weight)
                    self.logger.debug(f"Grid injection limit (fallback): sum(w[renewable]) <= {max_renewable_weight}")

        # 6c. Curtailment constraints
        curtailment_config = renewable_config.get('curtailment', {})
        if curtailment_config.get('enabled', False) and renewable_idx:
            max_curtailment_pct = curtailment_config.get('max_curtailment_pct', 0.20)
            enforce_as_constraint = curtailment_config.get('enforce_as_constraint', True)

            if enforce_as_constraint:
                # Add hard constraint: sum(w[renewable]) <= 1 - max_curtailment_pct
                constraints.append(cp.sum(w[renewable_idx]) <= (1.0 - max_curtailment_pct))
                self.logger.debug(f"Curtailment constraint: sum(w[renewable]) <= {1.0 - max_curtailment_pct:.4f}")
            # If not enforced as constraint, it can be handled as objective penalty (see renewable objectives)

        # 6d. REC minimum percentage
        recs_config = renewable_config.get('recs', {})
        if recs_config.get('enabled', False) and renewable_idx:
            min_rec_percentage = recs_config.get('min_rec_percentage', 0.0)
            if min_rec_percentage > 0:
                constraints.append(cp.sum(w[renewable_idx]) >= min_rec_percentage)
                self.logger.debug(f"REC constraint: sum(w[renewable]) >= {min_rec_percentage}")

        # 7. Constraints added via add_constraint()
        for constraint in self.constraints:
            constraint_type = constraint['type']
            params = constraint['params']

            if constraint_type == 'max_turnover':
                # Turnover constraint
                current_weights = params.get('current_weights')
                max_turn = params.get('max_turnover', 0.3)
                if current_weights is not None:
                    w_prev_arr = np.array(current_weights)
                    constraints.append(cp.norm1(w - w_prev_arr) <= max_turn)

            elif constraint_type == 'capacity_factor':
                # Per-asset capacity factor constraint override
                asset = params.get('asset')
                min_gen = params.get('min_generation', 0.0)
                max_gen = params.get('max_generation', 1.0)
                capacity_mw = params.get('capacity_mw', 100.0)

                if asset in self.asset_names:
                    idx = self.asset_names.index(asset)
                    # Normalize by some total capacity or use as direct weight bounds
                    # For simplicity, treat as weight bounds
                    if capacity_mw > 0:
                        # Approximate: weight proportional to generation/capacity
                        w_min = min_gen / (capacity_mw * 10)  # Scaling factor
                        w_max = max_gen / (capacity_mw * 10)
                        constraints.append(w[idx] >= w_min)
                        constraints.append(w[idx] <= w_max)
                        self.logger.debug(f"Capacity factor override for {asset}: [{w_min:.4f}, {w_max:.4f}]")

            elif constraint_type == 'rec_minimum':
                # REC minimum constraint
                min_percentage = params.get('min_percentage', 0.0)
                ren_idx = [i for i, name in enumerate(self.asset_names) if 'wind' in name.lower() or 'solar' in name.lower()]
                if ren_idx and min_percentage > 0:
                    constraints.append(cp.sum(w[ren_idx]) >= min_percentage)

            elif constraint_type == 'grid_injection':
                # Grid injection constraint
                renewable_assets_param = params.get('assets', [])
                grid_capacity_mw = params.get('grid_capacity_mw', 1000.0)
                ren_idx = [i for i, name in enumerate(self.asset_names) if name in renewable_assets_param]
                if ren_idx:
                    # Use simplified upper bound
                    constraints.append(cp.sum(w[ren_idx]) <= 0.8)
                    self.logger.debug(f"Grid injection (add_constraint): sum(w[{len(ren_idx)} assets]) <= 0.8")

            elif constraint_type == 'curtailment_limit':
                # Curtailment limit constraint
                curtail_assets = params.get('assets', [])
                max_curtail_pct = params.get('max_curtailment_pct', 0.20)
                curtail_idx = [i for i, name in enumerate(self.asset_names) if name in curtail_assets]
                if curtail_idx:
                    constraints.append(cp.sum(w[curtail_idx]) <= (1.0 - max_curtail_pct))
                    self.logger.debug(f"Curtailment limit (add_constraint): sum(w[{len(curtail_idx)} assets]) <= {1.0 - max_curtail_pct:.4f}")

        return constraints

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save optimizer state and results.

        Args:
            filepath: Path to save file

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)

        state = {
            'asset_names': self.asset_names,
            'expected_returns': self.expected_returns,
            'covariance': self.covariance,
            'constraints': self.constraints,
            'config': self.config
        }

        joblib.dump(state, filepath)
        self.logger.info(f"Saved optimizer state to {filepath}")

        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseOptimizer':
        """
        Load saved optimizer.

        Args:
            filepath: Path to saved file

        Returns:
            Optimizer instance with restored state
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Reconstruct returns DataFrame
        returns = pd.DataFrame(
            columns=state['asset_names']
        )

        # Create instance
        instance = cls(returns, config=state['config'])
        instance.expected_returns = state['expected_returns']
        instance.covariance = state['covariance']
        instance.constraints = state['constraints']

        logger.info(f"Loaded optimizer state from {filepath}")

        return instance


# =============================================================================
# MeanVarianceOptimizer Class
# =============================================================================

class MeanVarianceOptimizer(BaseOptimizer):
    """
    Markowitz mean-variance optimization.

    Solves portfolio optimization using mean-variance framework with various objectives.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    ):
        """Initialize mean-variance optimizer."""
        super().__init__(returns, config, risk_analytics)

        # Load mean-variance config
        self.mv_config = self.opt_config.get('mean_variance', {})
        self.risk_aversion = self.mv_config.get('risk_aversion', 2.0)
        self.target_return = self.mv_config.get('target_return')
        self.target_risk = self.mv_config.get('target_risk')

    def optimize(
        self,
        objective: str = 'max_sharpe',
        risk_aversion: Optional[float] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Solve mean-variance optimization.

        Args:
            objective: 'max_sharpe', 'min_risk', 'max_return', 'efficient_frontier'
            risk_aversion: Risk aversion parameter (overrides config)
            w_prev: Previous weights for turnover penalty (optional)
            turnover_penalty: Penalty coefficient for turnover (optional)

        Returns:
            OptimizationResult
        """
        if risk_aversion is None:
            risk_aversion = self.risk_aversion

        # Extract turnover penalty parameters
        w_prev = kwargs.get('w_prev')
        turnover_penalty = kwargs.get('turnover_penalty')

        # Get penalty from config if not provided
        if turnover_penalty is None and w_prev is not None:
            # First check constraints.turnover_penalty
            turnover_penalty = self.opt_config.get('constraints', {}).get('turnover_penalty', 0.0)
            if turnover_penalty == 0.0:
                # Fall back to multi_period transaction costs
                mp_config = self.opt_config.get('multi_period', {})
                tc_config = mp_config.get('transaction_costs', {})
                turnover_penalty = tc_config.get('proportional_cost', 0.0)

        if w_prev is not None and turnover_penalty > 0:
            self.logger.info(f"Applying turnover penalty: {turnover_penalty}")

        self.logger.info(f"Running mean-variance optimization with objective: {objective}")

        if objective == 'efficient_frontier':
            return self._optimize_efficient_frontier(**kwargs)
        elif objective == 'max_sharpe':
            return self.find_tangency_portfolio()
        elif objective == 'min_risk':
            return self._optimize_min_risk()
        elif objective == 'max_return':
            return self._optimize_max_return()
        else:
            # Default: risk-return tradeoff
            return self._optimize_risk_return_tradeoff(risk_aversion, w_prev, turnover_penalty)

    def _optimize_risk_return_tradeoff(
        self,
        risk_aversion: float,
        w_prev: Optional[np.ndarray] = None,
        turnover_penalty: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize with risk-return tradeoff and optional turnover penalty."""
        # Decision variable
        w = cp.Variable(self.n_assets)

        # Objective: maximize return - risk_aversion * variance
        returns_part = self.expected_returns @ w
        risk_part = cp.quad_form(w, self.covariance)
        objective_value = returns_part - risk_aversion * risk_part

        # Add transaction cost penalty if w_prev provided
        penalty_applied = 0.0
        if w_prev is not None and turnover_penalty is not None and turnover_penalty > 0:
            # Use epigraph form for numerical stability
            d = cp.Variable(self.n_assets)
            objective_value -= turnover_penalty * cp.sum(d)
            penalty_applied = turnover_penalty
        else:
            d = None

        objective = cp.Maximize(objective_value)

        # Constraints
        constraints = [cp.sum(w) == 1]

        # Epigraph constraints for turnover penalty
        if d is not None:
            constraints.append(d >= w - w_prev)
            constraints.append(d >= -(w - w_prev))

        # Weight bounds
        min_w = self.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_w = self.opt_config.get('constraints', {}).get('max_weight', 1.0)
        constraints.append(w >= min_w)
        constraints.append(w <= max_w)

        # Add constraints from _build_constraints
        constraints.extend(self._build_constraints(w, w_prev))

        # Solve
        problem = cp.Problem(objective, constraints)
        solver = self.opt_config.get('solver', 'ECOS')
        problem.solve(solver=solver)

        # Extract results
        weights = w.value
        if weights is None:
            self.logger.error("Optimization failed")
            return self._create_failed_result(problem.status)

        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(weights)

        # Calculate realized turnover if w_prev provided
        realized_turnover = None
        if w_prev is not None:
            realized_turnover = np.sum(np.abs(weights - w_prev))

        # Validate
        is_valid, issues = self.validate_weights(weights)

        weights_series = pd.Series(weights, index=self.asset_names)

        metadata = {
            'risk_aversion': risk_aversion,
            'issues': issues
        }
        if penalty_applied > 0:
            metadata['turnover_penalty'] = penalty_applied
        if realized_turnover is not None:
            metadata['realized_turnover'] = realized_turnover

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='mean_variance',
            constraints_satisfied=is_valid,
            optimization_status=problem.status,
            metadata=metadata
        )

    def _optimize_min_risk(self) -> OptimizationResult:
        """Minimize risk subject to target return."""
        w = cp.Variable(self.n_assets)

        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, self.covariance))

        # Constraints
        constraints = [cp.sum(w) == 1]

        # Target return constraint
        if self.target_return is not None:
            constraints.append(self.expected_returns @ w >= self.target_return)

        # Weight bounds
        min_w = self.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_w = self.opt_config.get('constraints', {}).get('max_weight', 1.0)
        constraints.append(w >= min_w)
        constraints.append(w <= max_w)

        # Add constraints from _build_constraints
        constraints.extend(self._build_constraints(w))

        # Solve
        problem = cp.Problem(objective, constraints)
        solver = self.opt_config.get('solver', 'ECOS')
        problem.solve(solver=solver)

        weights = w.value
        if weights is None:
            return self._create_failed_result(problem.status)

        metrics = self.calculate_portfolio_metrics(weights)
        is_valid, issues = self.validate_weights(weights)

        weights_series = pd.Series(weights, index=self.asset_names)

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='mean_variance_min_risk',
            constraints_satisfied=is_valid,
            optimization_status=problem.status,
            metadata={'target_return': self.target_return, 'issues': issues}
        )

    def _optimize_max_return(self) -> OptimizationResult:
        """Maximize return subject to target risk."""
        w = cp.Variable(self.n_assets)

        # Objective: maximize return
        objective = cp.Maximize(self.expected_returns @ w)

        # Constraints
        constraints = [cp.sum(w) == 1]

        # Target risk constraint
        if self.target_risk is not None:
            constraints.append(cp.quad_form(w, self.covariance) <= self.target_risk ** 2)

        # Weight bounds
        min_w = self.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_w = self.opt_config.get('constraints', {}).get('max_weight', 1.0)
        constraints.append(w >= min_w)
        constraints.append(w <= max_w)

        # Add constraints from _build_constraints
        constraints.extend(self._build_constraints(w))

        # Solve
        problem = cp.Problem(objective, constraints)
        solver = self.opt_config.get('solver', 'ECOS')
        problem.solve(solver=solver)

        weights = w.value
        if weights is None:
            return self._create_failed_result(problem.status)

        metrics = self.calculate_portfolio_metrics(weights)
        is_valid, issues = self.validate_weights(weights)

        weights_series = pd.Series(weights, index=self.asset_names)

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='mean_variance_max_return',
            constraints_satisfied=is_valid,
            optimization_status=problem.status,
            metadata={'target_risk': self.target_risk, 'issues': issues}
        )

    def find_tangency_portfolio(self) -> OptimizationResult:
        """Find maximum Sharpe ratio portfolio."""
        # Use scipy optimization for Sharpe maximization
        def neg_sharpe(weights):
            ret = calculate_portfolio_return(weights, self.expected_returns)
            risk = calculate_portfolio_risk(weights, self.covariance)
            risk_free = self.opt_config.get('mean_variance', {}).get('risk_free_rate', 0.0)
            return -(ret - risk_free) / risk if risk > 0 else 1e10

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        # Bounds
        min_w = self.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_w = self.opt_config.get('constraints', {}).get('max_weight', 1.0)
        bounds = [(min_w, max_w)] * self.n_assets

        # Initial guess
        w0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            self.logger.warning(f"Sharpe maximization did not converge: {result.message}")

        weights = result.x
        metrics = self.calculate_portfolio_metrics(weights)
        is_valid, issues = self.validate_weights(weights)

        weights_series = pd.Series(weights, index=self.asset_names)

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='mean_variance_max_sharpe',
            constraints_satisfied=is_valid,
            optimization_status='optimal' if result.success else 'suboptimal',
            metadata={'scipy_result': result, 'issues': issues}
        )

    def compute_efficient_frontier(
        self,
        n_points: int = 50,
        return_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Compute efficient frontier.

        Args:
            n_points: Number of points to compute
            return_range: (min_return, max_return) or None for auto

        Returns:
            DataFrame with columns ['return', 'risk', 'sharpe', 'weights']
        """
        self.logger.info(f"Computing efficient frontier with {n_points} points")

        # Determine return range
        if return_range is None:
            min_ret = self.expected_returns.min()
            max_ret = self.expected_returns.max()
        else:
            min_ret, max_ret = return_range

        target_returns = np.linspace(min_ret, max_ret, n_points)

        results = []
        for target_ret in target_returns:
            self.target_return = target_ret
            try:
                opt_result = self._optimize_min_risk()
                if opt_result.optimization_status == 'optimal':
                    results.append({
                        'return': opt_result.expected_return,
                        'risk': opt_result.expected_risk,
                        'sharpe': opt_result.sharpe_ratio,
                        'weights': opt_result.weights.values
                    })
            except Exception as e:
                self.logger.warning(f"Failed to optimize for target return {target_ret:.4f}: {e}")
                continue

        if not results:
            self.logger.error("Failed to compute efficient frontier")
            return pd.DataFrame()

        frontier_df = pd.DataFrame(results)
        self.logger.info(f"Computed {len(frontier_df)} points on efficient frontier")

        return frontier_df

    def _optimize_efficient_frontier(self, **kwargs) -> OptimizationResult:
        """Optimize with efficient frontier computation."""
        n_points = kwargs.get('n_points', 50)
        return_range = kwargs.get('return_range')

        frontier = self.compute_efficient_frontier(n_points, return_range)

        if frontier.empty:
            return self._create_failed_result('infeasible')

        # Return the maximum Sharpe ratio point
        max_sharpe_idx = frontier['sharpe'].idxmax()
        best_point = frontier.loc[max_sharpe_idx]

        weights_series = pd.Series(best_point['weights'], index=self.asset_names)

        return OptimizationResult(
            weights=weights_series,
            expected_return=best_point['return'],
            expected_risk=best_point['risk'],
            sharpe_ratio=best_point['sharpe'],
            method='mean_variance_efficient_frontier',
            constraints_satisfied=True,
            optimization_status='optimal',
            efficient_frontier=frontier,
            metadata={'n_points': n_points}
        )

    def _create_failed_result(self, status: str) -> OptimizationResult:
        """Create failed optimization result."""
        weights = np.ones(self.n_assets) / self.n_assets
        weights_series = pd.Series(weights, index=self.asset_names)
        metrics = self.calculate_portfolio_metrics(weights)

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='mean_variance_failed',
            constraints_satisfied=False,
            optimization_status=status,
            metadata={'error': f'Optimization failed with status: {status}'}
        )


# =============================================================================
# RiskParityOptimizer Class
# =============================================================================

class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity (equal risk contribution) optimization.

    Allocates capital so that each asset contributes equally to portfolio risk.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    ):
        """Initialize risk parity optimizer."""
        super().__init__(returns, config, risk_analytics)

        # Load risk parity config
        self.rp_config = self.opt_config.get('risk_parity', {})
        self.tolerance = self.rp_config.get('tolerance', 1e-6)
        self.max_iter = self.rp_config.get('max_iterations', 1000)

    def optimize(self, **kwargs) -> OptimizationResult:
        """
        Solve risk parity optimization with sector, turnover, liquidity, and leverage constraints.

        Returns:
            OptimizationResult
        """
        self.logger.info("Running risk parity optimization")

        # Read constraints configuration
        constraints_cfg = self.opt_config.get('constraints', {})

        # Initial guess
        initial_weights = self.rp_config.get('initial_weights')
        if initial_weights is None:
            w0 = np.ones(self.n_assets) / self.n_assets
        else:
            w0 = np.array(initial_weights)

        # Base constraints: sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        # Base bounds from config
        min_w = constraints_cfg.get('min_weight', 0.0)
        max_w = constraints_cfg.get('max_weight', 1.0)
        bounds = [(min_w, max_w)] * self.n_assets

        # Track applied constraints for logging
        applied_constraints = []

        # 1. Sector constraints
        sector_mapping = constraints_cfg.get('sector_mapping', {})
        max_sector_weight = constraints_cfg.get('max_sector_weight')
        if sector_mapping and max_sector_weight is not None:
            # Precompute index lists per sector
            sectors = {}
            for asset, sector in sector_mapping.items():
                if sector not in sectors:
                    sectors[sector] = []
                if asset in self.asset_names:
                    sectors[sector].append(self.asset_names.index(asset))

            # Add inequality constraint for each sector
            for sector, indices in sectors.items():
                if indices:
                    def sector_constraint(w, idx=indices, max_w=max_sector_weight):
                        return max_w - np.sum(w[idx])
                    constraints.append({'type': 'ineq', 'fun': sector_constraint})
                    applied_constraints.append(f"Sector '{sector}': max {max_sector_weight} ({len(indices)} assets)")

        # 2. Turnover constraint from add_constraint()
        w_prev = None
        max_turnover = None
        for constraint in self.constraints:
            if constraint['type'] == 'max_turnover':
                params = constraint['params']
                w_prev = params.get('current_weights')
                max_turnover = params.get('max_turnover', 0.3)
                if w_prev is not None:
                    w_prev = np.array(w_prev)
                    # Smooth L1 norm approximation
                    def turnover_constraint(w, w_p=w_prev, max_t=max_turnover):
                        # Smooth approximation: sum(sqrt((w - w_p)**2 + eps))
                        diff = w - w_p
                        smooth_l1 = np.sum(np.sqrt(diff**2 + 1e-12))
                        return max_t - smooth_l1
                    constraints.append({'type': 'ineq', 'fun': turnover_constraint})
                    applied_constraints.append(f"Turnover: max {max_turnover}")
                break

        # 3. Liquidity constraints - translate to tighter per-asset bounds
        liquidity_data = constraints_cfg.get('liquidity_data', {})
        max_position_vs_adv = constraints_cfg.get('max_position_vs_adv')
        if liquidity_data and max_position_vs_adv is not None:
            bounds_list = list(bounds)  # Convert to list for modification
            for asset, adv in liquidity_data.items():
                if asset in self.asset_names:
                    idx = self.asset_names.index(asset)
                    # Compute tighter upper bound
                    ub_i = min(bounds_list[idx][1], max_position_vs_adv * adv)
                    bounds_list[idx] = (bounds_list[idx][0], ub_i)
                    applied_constraints.append(f"Liquidity '{asset}': max {ub_i:.4f} (ADV={adv})")
            bounds = bounds_list

        # 4. Leverage constraints (for long-only, leverage is fixed at 1; skip unless allowing shorts)
        max_leverage = constraints_cfg.get('max_leverage')
        min_leverage = constraints_cfg.get('min_leverage')
        if max_leverage is not None and max_leverage != 1.0:
            # Only add if different from default long-only
            def leverage_max_constraint(w, max_lev=max_leverage):
                return max_lev - np.sum(np.abs(w))
            constraints.append({'type': 'ineq', 'fun': leverage_max_constraint})
            applied_constraints.append(f"Leverage: max {max_leverage}")
        if min_leverage is not None and min_leverage != 1.0:
            def leverage_min_constraint(w, min_lev=min_leverage):
                return np.sum(np.abs(w)) - min_lev
            constraints.append({'type': 'ineq', 'fun': leverage_min_constraint})
            applied_constraints.append(f"Leverage: min {min_leverage}")

        # 5. Ensure initial guess respects new bounds
        w0_clipped = np.array([
            np.clip(w0[i], bounds[i][0], bounds[i][1])
            for i in range(self.n_assets)
        ])
        # Re-normalize to sum to 1
        w0 = w0_clipped / w0_clipped.sum()

        # Log applied constraints
        if applied_constraints:
            self.logger.info(f"Applied {len(applied_constraints)} additional constraints:")
            for constraint_desc in applied_constraints:
                self.logger.info(f"  - {constraint_desc}")

        # Optimize
        result = minimize(
            self._risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tolerance}
        )

        if not result.success:
            self.logger.warning(f"Risk parity optimization did not converge: {result.message}")

        weights = result.x

        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(weights)

        # Calculate risk contributions
        risk_contributions = self.calculate_risk_contributions(weights)

        # Validate
        is_valid, issues = self.validate_weights(weights)

        weights_series = pd.Series(weights, index=self.asset_names)

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='risk_parity',
            constraints_satisfied=is_valid,
            optimization_status='optimal' if result.success else 'suboptimal',
            metadata={
                'risk_contributions': risk_contributions,
                'scipy_result': result,
                'issues': issues,
                'applied_constraints': applied_constraints
            }
        )

    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution for each asset.

        Args:
            weights: Portfolio weights

        Returns:
            Array of risk contributions
        """
        portfolio_vol = calculate_portfolio_risk(weights, self.covariance)

        if portfolio_vol == 0:
            return np.zeros(self.n_assets)

        # Marginal contribution to risk: (Σ @ w) / σ_p
        marginal_contrib = (self.covariance @ weights) / portfolio_vol

        # Risk contribution: w_i * MRC_i
        risk_contrib = weights * marginal_contrib

        return risk_contrib

    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """
        Objective function for risk parity optimization.

        Minimizes sum of squared deviations from equal risk contribution.

        Args:
            weights: Portfolio weights

        Returns:
            Objective value
        """
        risk_contrib = self.calculate_risk_contributions(weights)

        # Target: equal risk contribution (1/n for each asset)
        target = 1.0 / self.n_assets

        # Sum of squared deviations
        deviations = risk_contrib - target
        objective = np.sum(deviations ** 2)

        return objective


# =============================================================================
# BlackLittermanOptimizer Class
# =============================================================================

class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman model with investor views.

    Combines market equilibrium with investor views to generate optimal portfolio.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    ):
        """Initialize Black-Litterman optimizer."""
        super().__init__(returns, config, risk_analytics)

        # Load Black-Litterman config
        self.bl_config = self.opt_config.get('black_litterman', {})
        self.tau = self.bl_config.get('tau', 0.05)
        self.risk_aversion = self.bl_config.get('risk_aversion', 2.5)

        # Views storage
        self.views = {}  # asset -> expected_return
        self.view_confidences = {}  # asset -> confidence

    def optimize(
        self,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Solve Black-Litterman optimization.

        Args:
            views: Dict mapping asset -> expected return view
            view_confidences: Dict mapping asset -> confidence (0-1)
            w_prev: Previous weights for turnover penalty (optional)
            turnover_penalty: Penalty coefficient for turnover (optional)

        Returns:
            OptimizationResult
        """
        self.logger.info("Running Black-Litterman optimization")

        # Extract turnover penalty parameters
        w_prev = kwargs.get('w_prev')
        turnover_penalty = kwargs.get('turnover_penalty')

        # Get penalty from config if not provided
        if turnover_penalty is None and w_prev is not None:
            turnover_penalty = self.opt_config.get('constraints', {}).get('turnover_penalty', 0.0)
            if turnover_penalty == 0.0:
                mp_config = self.opt_config.get('multi_period', {})
                tc_config = mp_config.get('transaction_costs', {})
                turnover_penalty = tc_config.get('proportional_cost', 0.0)

        if w_prev is not None and turnover_penalty > 0:
            self.logger.info(f"Applying turnover penalty: {turnover_penalty}")

        # Use provided views or stored views
        if views is not None:
            self.views = views
        if view_confidences is not None:
            self.view_confidences = view_confidences

        if not self.views:
            self.logger.warning("No views provided, using market equilibrium")
            # Fall back to mean-variance with market equilibrium returns
            market_weights = self.bl_config.get('market_weights')
            if market_weights is None:
                market_weights = np.ones(self.n_assets) / self.n_assets
            else:
                market_weights = np.array(market_weights)

            implied_returns = self.calculate_implied_returns(market_weights, self.risk_aversion)
            self.expected_returns = implied_returns
        else:
            # Calculate posterior returns with views
            posterior_returns, posterior_cov = self._calculate_posterior_distribution()
            self.expected_returns = posterior_returns
            self.covariance = posterior_cov

        # Optimize using mean-variance with posterior distribution
        w = cp.Variable(self.n_assets)

        # Objective: maximize return - risk_aversion * variance
        objective_value = self.expected_returns @ w - (self.risk_aversion / 2) * cp.quad_form(w, self.covariance)

        # Add transaction cost penalty if w_prev provided
        penalty_applied = 0.0
        if w_prev is not None and turnover_penalty is not None and turnover_penalty > 0:
            # Use epigraph form for numerical stability
            d = cp.Variable(self.n_assets)
            objective_value -= turnover_penalty * cp.sum(d)
            penalty_applied = turnover_penalty
        else:
            d = None

        objective = cp.Maximize(objective_value)

        # Constraints
        constraints = [cp.sum(w) == 1]

        # Epigraph constraints for turnover penalty
        if d is not None:
            constraints.append(d >= w - w_prev)
            constraints.append(d >= -(w - w_prev))

        # Weight bounds
        min_w = self.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_w = self.opt_config.get('constraints', {}).get('max_weight', 1.0)
        constraints.append(w >= min_w)
        constraints.append(w <= max_w)

        # Add constraints from _build_constraints
        constraints.extend(self._build_constraints(w, w_prev))

        # Solve
        problem = cp.Problem(objective, constraints)
        solver = self.opt_config.get('solver', 'ECOS')
        problem.solve(solver=solver)

        weights = w.value
        if weights is None:
            self.logger.error("Black-Litterman optimization failed")
            weights = np.ones(self.n_assets) / self.n_assets

        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(weights)

        # Calculate realized turnover if w_prev provided
        realized_turnover = None
        if w_prev is not None:
            realized_turnover = np.sum(np.abs(weights - w_prev))

        # Validate
        is_valid, issues = self.validate_weights(weights)

        weights_series = pd.Series(weights, index=self.asset_names)

        metadata = {
            'views': self.views,
            'view_confidences': self.view_confidences,
            'tau': self.tau,
            'issues': issues
        }
        if penalty_applied > 0:
            metadata['turnover_penalty'] = penalty_applied
        if realized_turnover is not None:
            metadata['realized_turnover'] = realized_turnover

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='black_litterman',
            constraints_satisfied=is_valid,
            optimization_status=problem.status if weights is not None else 'failed',
            metadata=metadata
        )

    def add_view(self, asset: str, expected_return: float, confidence: float) -> 'BlackLittermanOptimizer':
        """
        Add investor view.

        Args:
            asset: Asset name
            expected_return: Expected return view
            confidence: Confidence level (0-1)

        Returns:
            self (for method chaining)
        """
        self.views[asset] = expected_return
        self.view_confidences[asset] = confidence
        self.logger.debug(f"Added view for {asset}: return={expected_return:.4f}, confidence={confidence:.2f}")
        return self

    def add_renewable_view(
        self,
        asset: str,
        generation_forecast: float,
        price_forecast: float,
        correlation: float = -0.6
    ) -> 'BlackLittermanOptimizer':
        """
        Add renewable generation-based view.

        High renewable generation typically leads to lower prices, which is
        positive for generation assets.

        Args:
            asset: Asset name (e.g., 'wind', 'solar')
            generation_forecast: Expected capacity factor (0-1)
            price_forecast: Expected price ($/MWh)
            correlation: Generation-price correlation (default -0.6)

        Returns:
            self (for method chaining)
        """
        # Calculate expected return adjustment
        # Higher generation with negative correlation to price suggests
        # favorable conditions for the generator
        base_return = self.expected_returns[self.asset_names.index(asset)] if asset in self.asset_names else 0.05

        # Adjust based on generation forecast (higher = better)
        generation_adjustment = (generation_forecast - 0.3) * 0.1  # 30% baseline

        # Adjust based on price forecast (depends on correlation)
        price_adjustment = (price_forecast - 50.0) / 100.0  # $50/MWh baseline

        expected_return = base_return + generation_adjustment + (correlation * price_adjustment)

        # Confidence based on generation forecast (more certain about high/low generation)
        confidence = min(0.8, abs(generation_forecast - 0.3) * 2)

        self.add_view(asset, expected_return, confidence)

        self.logger.info(
            f"Added renewable view for {asset}: gen={generation_forecast:.2f}, "
            f"price={price_forecast:.2f}, return={expected_return:.4f}"
        )

        return self

    def calculate_implied_returns(self, market_weights: np.ndarray, risk_aversion: float) -> np.ndarray:
        """
        Calculate market equilibrium (implied) returns.

        Reverse optimization: π = λ * Σ @ w_market

        Args:
            market_weights: Market capitalization weights
            risk_aversion: Risk aversion parameter

        Returns:
            Array of implied returns
        """
        implied_returns = risk_aversion * (self.covariance @ market_weights)
        return implied_returns

    def _calculate_posterior_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate posterior returns and covariance incorporating views.

        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        """
        # Market equilibrium
        market_weights = self.bl_config.get('market_weights')
        if market_weights is None:
            market_weights = np.ones(self.n_assets) / self.n_assets
        else:
            market_weights = np.array(market_weights)

        pi = self.calculate_implied_returns(market_weights, self.risk_aversion)

        # Construct P (picking matrix) and Q (view returns)
        P = create_picking_matrix(self.views, self.asset_names)
        Q = np.array([self.views[asset] for asset in self.views.keys()])

        # Construct Ω (view uncertainty matrix)
        n_views = len(self.views)
        Omega = np.zeros((n_views, n_views))

        default_uncertainty = self.bl_config.get('default_view_uncertainty', 0.1)
        view_confidence_scale = self.bl_config.get('view_confidence_scale', 1.0)

        for i, asset in enumerate(self.views.keys()):
            confidence = self.view_confidences.get(asset, 0.5)
            # Lower confidence = higher uncertainty
            uncertainty = (1 - confidence) * default_uncertainty * view_confidence_scale
            Omega[i, i] = uncertainty ** 2

        # Add small diagonal jitter to Omega for numerical stability (safeguard for singular Ω)
        Omega += np.eye(n_views) * 1e-8

        # Black-Litterman formula
        # Posterior returns: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1π + P'Ω^-1Q]
        tau_sigma = self.tau * self.covariance
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        # Posterior precision
        posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
        posterior_covariance = np.linalg.inv(posterior_precision)

        # Posterior mean
        posterior_returns = posterior_covariance @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        return posterior_returns, posterior_covariance


# =============================================================================
# MinimumCVaROptimizer Class
# =============================================================================

class MinimumCVaROptimizer(BaseOptimizer):
    """
    Minimum Conditional Value at Risk (CVaR) optimization.

    Minimizes the expected loss in the worst-case scenarios.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    ):
        """Initialize CVaR optimizer."""
        super().__init__(returns, config, risk_analytics)

        # Load CVaR config
        self.cvar_config = self.opt_config.get('min_cvar', {})
        self.confidence_level = self.cvar_config.get('confidence_level', 0.95)
        self.min_return = self.cvar_config.get('min_return')  # Read min_return from config

        # Scenario generation settings
        self.scenario_method = self.cvar_config.get('scenario_method', 'historical_bootstrap')
        self.config_n_scenarios = self.cvar_config.get('n_scenarios', 10000)
        self.block_size = self.cvar_config.get('block_size', 5)

        # Validate scenario method
        supported_methods = ['historical_bootstrap']
        if self.scenario_method not in supported_methods:
            self.logger.warning(
                f"Scenario method '{self.scenario_method}' not fully implemented. "
                f"Supported methods: {supported_methods}. Using historical data."
            )
            self.scenario_method = 'historical_bootstrap'

        # Generate scenarios based on method
        self._generate_scenarios(returns)

    def _generate_scenarios(self, returns: pd.DataFrame):
        """
        Generate return scenarios based on scenario_method.

        Args:
            returns: Historical returns DataFrame
        """
        if self.scenario_method == 'historical_bootstrap':
            # Use historical bootstrap (sampling with replacement)
            historical_returns = returns.values  # T x n_assets
            n_historical = len(historical_returns)

            # If config_n_scenarios exceeds historical data, use bootstrap
            if self.config_n_scenarios > n_historical:
                # Bootstrap sampling with replacement
                indices = np.random.choice(n_historical, size=self.config_n_scenarios, replace=True)
                self.return_scenarios = historical_returns[indices]
            else:
                # Use all historical data if n_scenarios not specified or <= historical length
                self.return_scenarios = historical_returns

            self.n_scenarios = len(self.return_scenarios)
            self.logger.info(f"Generated {self.n_scenarios} scenarios using {self.scenario_method}")
        else:
            # Fallback: use historical data as-is
            self.return_scenarios = returns.values
            self.n_scenarios = len(self.return_scenarios)
            self.logger.info(f"Using {self.n_scenarios} historical scenarios")

    def optimize(self, **kwargs) -> OptimizationResult:
        """
        Solve minimum CVaR optimization.

        Args:
            w_prev: Previous weights for turnover penalty (optional)
            turnover_penalty: Penalty coefficient for turnover (optional)

        Returns:
            OptimizationResult
        """
        self.logger.info(f"Running minimum CVaR optimization (α={self.confidence_level})")

        # Extract turnover penalty parameters
        w_prev = kwargs.get('w_prev')
        turnover_penalty = kwargs.get('turnover_penalty')

        # Get penalty from config if not provided
        if turnover_penalty is None and w_prev is not None:
            turnover_penalty = self.opt_config.get('constraints', {}).get('turnover_penalty', 0.0)
            if turnover_penalty == 0.0:
                mp_config = self.opt_config.get('multi_period', {})
                tc_config = mp_config.get('transaction_costs', {})
                turnover_penalty = tc_config.get('proportional_cost', 0.0)

        if w_prev is not None and turnover_penalty > 0:
            self.logger.info(f"Applying turnover penalty: {turnover_penalty}")

        # Decision variables
        w = cp.Variable(self.n_assets)  # Portfolio weights
        z = cp.Variable()  # VaR
        u = cp.Variable(self.n_scenarios)  # CVaR auxiliary variables

        # Objective: minimize CVaR
        alpha = self.confidence_level
        objective_value = z + (1 / (self.n_scenarios * (1 - alpha))) * cp.sum(u)

        # Add transaction cost penalty if w_prev provided (minimize, so add penalty)
        penalty_applied = 0.0
        if w_prev is not None and turnover_penalty is not None and turnover_penalty > 0:
            # Use epigraph form for numerical stability
            d = cp.Variable(self.n_assets)
            objective_value += turnover_penalty * cp.sum(d)
            penalty_applied = turnover_penalty
        else:
            d = None

        objective = cp.Minimize(objective_value)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            u >= 0
        ]

        # Epigraph constraints for turnover penalty
        if d is not None:
            constraints.append(d >= w - w_prev)
            constraints.append(d >= -(w - w_prev))

        # CVaR constraint: u >= -R @ w - z
        for t in range(self.n_scenarios):
            constraints.append(u[t] >= -(self.return_scenarios[t, :] @ w) - z)

        # Minimum return constraint
        if self.min_return is not None:
            constraints.append(self.expected_returns @ w >= self.min_return)

        # Weight bounds
        min_w = self.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_w = self.opt_config.get('constraints', {}).get('max_weight', 1.0)
        constraints.append(w >= min_w)
        constraints.append(w <= max_w)

        # Add constraints from _build_constraints
        constraints.extend(self._build_constraints(w, w_prev))

        # Solve
        problem = cp.Problem(objective, constraints)
        solver = self.opt_config.get('solver', 'ECOS')
        problem.solve(solver=solver)

        weights = w.value
        if weights is None:
            self.logger.error("CVaR optimization failed")
            weights = np.ones(self.n_assets) / self.n_assets

        # Calculate metrics
        var_value = z.value if z.value is not None else 0.0
        cvar_value = objective.value if problem.value is not None else 0.0

        expected_return = calculate_portfolio_return(weights, self.expected_returns)

        # Calculate realized turnover if w_prev provided
        realized_turnover = None
        if w_prev is not None:
            realized_turnover = np.sum(np.abs(weights - w_prev))

        # Use CVaR as the risk measure
        sharpe_ratio = expected_return / abs(cvar_value) if cvar_value != 0 else 0.0

        # Validate
        is_valid, issues = self.validate_weights(weights)

        weights_series = pd.Series(weights, index=self.asset_names)

        metadata = {
            'var': var_value,
            'cvar': cvar_value,
            'confidence_level': self.confidence_level,
            'n_scenarios': self.n_scenarios,
            'issues': issues
        }
        if penalty_applied > 0:
            metadata['turnover_penalty'] = penalty_applied
        if realized_turnover is not None:
            metadata['realized_turnover'] = realized_turnover

        return OptimizationResult(
            weights=weights_series,
            expected_return=expected_return,
            expected_risk=abs(cvar_value),  # CVaR as risk measure
            sharpe_ratio=sharpe_ratio,
            method='min_cvar',
            constraints_satisfied=is_valid,
            optimization_status=problem.status,
            metadata=metadata
        )

    def calculate_portfolio_cvar(self, weights: np.ndarray, confidence_level: float) -> float:
        """
        Calculate portfolio CVaR from historical returns.

        Args:
            weights: Portfolio weights
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        # Calculate portfolio returns for each scenario
        portfolio_returns = self.return_scenarios @ weights

        # Calculate VaR (quantile)
        var_value = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # Calculate CVaR (mean of returns below VaR)
        tail_returns = portfolio_returns[portfolio_returns <= -var_value]
        if len(tail_returns) > 0:
            cvar_value = -np.mean(tail_returns)
        else:
            cvar_value = var_value

        return cvar_value


# =============================================================================
# RenewableConstraints Class
# =============================================================================

class RenewableConstraints:
    """
    Helper class for renewable energy-specific constraints.
    """

    def __init__(self, config: Dict):
        """
        Initialize renewable constraints.

        Args:
            config: Configuration dict
        """
        self.config = config
        self.renewable_config = config.get('optimization', {}).get('renewable_constraints', {})
        self.logger = logging.getLogger(f"{__name__}.RenewableConstraints")

    def capacity_factor_constraint(
        self,
        asset: str,
        min_cf: float,
        max_cf: float,
        capacity_mw: float
    ) -> Dict:
        """
        Create capacity factor constraint.

        Args:
            asset: Asset name
            min_cf: Minimum capacity factor
            max_cf: Maximum capacity factor
            capacity_mw: Installed capacity (MW)

        Returns:
            Constraint dict
        """
        return {
            'type': 'capacity_factor',
            'asset': asset,
            'min_generation': min_cf * capacity_mw,
            'max_generation': max_cf * capacity_mw,
            'capacity_mw': capacity_mw
        }

    def curtailment_limit_constraint(
        self,
        renewable_assets: List[str],
        max_curtailment_pct: float
    ) -> Dict:
        """
        Create curtailment limit constraint.

        Args:
            renewable_assets: List of renewable asset names
            max_curtailment_pct: Maximum curtailment percentage

        Returns:
            Constraint dict
        """
        return {
            'type': 'curtailment_limit',
            'assets': renewable_assets,
            'max_curtailment_pct': max_curtailment_pct
        }

    def grid_injection_constraint(
        self,
        renewable_assets: List[str],
        grid_capacity_mw: float
    ) -> Dict:
        """
        Create grid injection limit constraint.

        Args:
            renewable_assets: List of renewable asset names
            grid_capacity_mw: Grid injection capacity (MW)

        Returns:
            Constraint dict
        """
        return {
            'type': 'grid_injection',
            'assets': renewable_assets,
            'grid_capacity_mw': grid_capacity_mw
        }

    def rec_constraint(self, min_rec_percentage: float) -> Dict:
        """
        Create Renewable Energy Credit constraint.

        Args:
            min_rec_percentage: Minimum percentage in REC-eligible assets

        Returns:
            Constraint dict
        """
        return {
            'type': 'rec_minimum',
            'min_percentage': min_rec_percentage
        }


# =============================================================================
# MultiPeriodOptimizer Class
# =============================================================================

class MultiPeriodOptimizer:
    """
    Multi-period optimization with rebalancing and transaction costs.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        base_optimizer: BaseOptimizer,
        config: Optional[Dict] = None
    ):
        """
        Initialize multi-period optimizer.

        Args:
            returns: DataFrame with asset returns
            base_optimizer: Base optimizer instance
            config: Optional configuration dict
        """
        self.returns = returns
        self.base_optimizer = base_optimizer
        self.config = config if config is not None else get_config()

        # Load multi-period config
        self.mp_config = self.config.get('optimization', {}).get('multi_period', {})
        self.n_periods = self.mp_config.get('n_periods', 12)
        self.rebalancing_frequency = self.mp_config.get('rebalancing_frequency', 'monthly')
        self.transaction_cost_pct = self.mp_config.get('transaction_cost_pct', 0.001)
        self.max_turnover = self.mp_config.get('max_turnover_per_period', 0.3)

        self.logger = logging.getLogger(f"{__name__}.MultiPeriodOptimizer")

    def optimize(self, **kwargs) -> List[OptimizationResult]:
        """
        Solve multi-period optimization.

        Returns:
            List of OptimizationResult for each period
        """
        self.logger.info(f"Running multi-period optimization for {self.n_periods} periods")

        results = []
        current_weights = None

        for period in range(self.n_periods):
            self.logger.debug(f"Optimizing period {period + 1}/{self.n_periods}")

            # Optimize for current period
            if current_weights is None:
                # First period: no transaction cost constraint
                result = self.base_optimizer.optimize(**kwargs)
            else:
                # Subsequent periods: constrain turnover
                result = self.optimize_with_turnover_constraint(
                    current_weights, self.max_turnover, **kwargs
                )

            results.append(result)
            current_weights = result.weights

        self.logger.info(f"Multi-period optimization complete: {len(results)} periods")

        return results

    def calculate_rebalancing_schedule(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        frequency: str
    ) -> List[pd.Timestamp]:
        """
        Calculate rebalancing dates.

        Args:
            start_date: Start date
            end_date: End date
            frequency: 'daily', 'weekly', 'monthly', 'quarterly'

        Returns:
            List of rebalancing dates
        """
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'MS',
            'quarterly': 'QS'
        }

        freq_str = freq_map.get(frequency, 'MS')
        dates = pd.date_range(start=start_date, end=end_date, freq=freq_str)

        return list(dates)

    def calculate_transaction_costs(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        portfolio_value: float
    ) -> float:
        """
        Calculate transaction costs for rebalancing.

        Args:
            old_weights: Current portfolio weights
            new_weights: Target portfolio weights
            portfolio_value: Current portfolio value

        Returns:
            Transaction cost in dollars
        """
        # Align weights
        old_weights = old_weights.reindex(new_weights.index, fill_value=0)

        # Calculate turnover
        turnover = np.abs(new_weights - old_weights).sum()

        # Calculate costs
        variable_cost = self.transaction_cost_pct * portfolio_value * turnover
        fixed_cost = self.mp_config.get('fixed_transaction_cost', 0.0)

        total_cost = variable_cost + fixed_cost

        self.logger.debug(f"Transaction cost: ${total_cost:.2f} (turnover: {turnover:.2%})")

        return total_cost

    def optimize_with_turnover_constraint(
        self,
        current_weights: pd.Series,
        max_turnover: float,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize with turnover constraint and transaction cost penalty in objective.

        Args:
            current_weights: Current portfolio weights
            max_turnover: Maximum allowed turnover

        Returns:
            OptimizationResult

        Note:
            Turnover constraint is enforced via _build_constraints().
            Transaction cost penalty is incorporated directly into the objective function.
        """
        # Add turnover constraint to base optimizer
        self.base_optimizer.add_constraint(
            'max_turnover',
            {'current_weights': current_weights.values, 'max_turnover': max_turnover}
        )

        # Retrieve penalty coefficient from config
        penalty_coeff = self.mp_config.get('transaction_costs', {}).get('proportional_cost', 0.0)
        if penalty_coeff == 0.0:
            # Fall back to constraints.turnover_penalty
            penalty_coeff = self.config.get('optimization', {}).get('constraints', {}).get('turnover_penalty', 0.0)

        # Pass w_prev and turnover_penalty to optimizer
        kwargs_with_penalty = kwargs.copy()
        kwargs_with_penalty['w_prev'] = current_weights.values
        if penalty_coeff > 0:
            kwargs_with_penalty['turnover_penalty'] = penalty_coeff
            self.logger.info(f"Multi-period turnover penalty activated: {penalty_coeff}")

        # Optimize (penalty is now in objective, not post-hoc)
        result = self.base_optimizer.optimize(**kwargs_with_penalty)

        # Record realized turnover (already in metadata if penalty was applied)
        if 'realized_turnover' not in result.metadata:
            result.metadata['realized_turnover'] = np.abs(result.weights - current_weights).sum()

        # Log penalty application
        if penalty_coeff > 0:
            self.logger.info(
                f"Period optimized with turnover penalty {penalty_coeff:.6f}, "
                f"realized turnover: {result.metadata.get('realized_turnover', 0):.4f}"
            )

        return result


# =============================================================================
# PortfolioOptimizer Class (Main Orchestrator)
# =============================================================================

class PortfolioOptimizer:
    """
    Main portfolio optimizer class.

    Orchestrates different optimization methods and provides high-level interface.

    Examples:
        >>> from src.optimization import PortfolioOptimizer
        >>> from src.backtesting import BacktestEngine
        >>>
        >>> # Run backtest to get historical data
        >>> result = engine.run(price_data, signals)
        >>>
        >>> # Create optimizer
        >>> optimizer = PortfolioOptimizer(result)
        >>>
        >>> # Mean-variance optimization
        >>> mv_result = optimizer.optimize(method='mean_variance', objective='max_sharpe')
        >>> print(f"Optimal weights: {mv_result.weights}")
        >>> print(f"Expected return: {mv_result.expected_return:.2%}")
        >>> print(f"Expected risk: {mv_result.expected_risk:.2%}")
        >>> print(f"Sharpe ratio: {mv_result.sharpe_ratio:.2f}")
        >>>
        >>> # Risk parity optimization
        >>> rp_result = optimizer.optimize(method='risk_parity')
        >>>
        >>> # Compare methods
        >>> comparison = optimizer.compare_methods(['mean_variance', 'risk_parity', 'min_cvar'])
        >>> print(comparison)
    """

    def __init__(
        self,
        returns: Union[pd.DataFrame, BacktestResult],
        config: Optional[Dict] = None,
        risk_analytics: Optional[RiskAnalytics] = None
    ):
        """
        Initialize portfolio optimizer.

        Args:
            returns: DataFrame with asset returns or BacktestResult
            config: Optional configuration dict
            risk_analytics: Optional RiskAnalytics instance
        """
        # Extract returns from BacktestResult if provided
        if isinstance(returns, BacktestResult):
            self.backtest_result = returns
            # Extract asset returns from portfolio_history
            portfolio_history = returns.portfolio_history
            if portfolio_history is not None and not portfolio_history.empty:
                # Find val_{asset} columns
                val_columns = [col for col in portfolio_history.columns if col.startswith('val_')]
                if val_columns:
                    asset_values = portfolio_history[val_columns].copy()
                    asset_values.columns = [col.replace('val_', '') for col in asset_values.columns]
                    self.returns = asset_values.pct_change().dropna()
                else:
                    raise ValueError("No val_ columns found in portfolio_history")
            else:
                raise ValueError("BacktestResult must contain portfolio_history")
        else:
            self.returns = returns
            self.backtest_result = None

        # Load config
        if config is None:
            config = get_config()
        self.config = config

        # Initialize or use provided RiskAnalytics
        if risk_analytics is None and self.backtest_result is not None:
            self.risk_analytics = RiskAnalytics(self.backtest_result, config)
        else:
            self.risk_analytics = risk_analytics

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.PortfolioOptimizer")

        # Data manager for persistence
        self.data_manager = DataManager(config=config)

        self.logger.info(f"Initialized PortfolioOptimizer with {len(self.returns.columns)} assets")

    def optimize(
        self,
        method: str = 'mean_variance',
        **kwargs
    ) -> OptimizationResult:
        """
        Run optimization with specified method.

        Args:
            method: 'mean_variance', 'risk_parity', 'black_litterman', 'min_cvar'
            **kwargs: Method-specific parameters

        Returns:
            OptimizationResult
        """
        self.logger.info(f"Running optimization with method: {method}")

        # Validate inputs
        is_valid, issues = validate_optimization_inputs(self.returns, self.returns.cov().values)
        if not is_valid:
            self.logger.error(f"Invalid optimization inputs: {issues}")
            raise ValueError(f"Invalid optimization inputs: {issues}")

        # Create optimizer instance
        if method == 'mean_variance':
            optimizer = MeanVarianceOptimizer(self.returns, self.config, self.risk_analytics)
        elif method == 'risk_parity':
            optimizer = RiskParityOptimizer(self.returns, self.config, self.risk_analytics)
        elif method == 'black_litterman':
            optimizer = BlackLittermanOptimizer(self.returns, self.config, self.risk_analytics)
        elif method == 'min_cvar':
            optimizer = MinimumCVaROptimizer(self.returns, self.config, self.risk_analytics)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Apply constraints from config
        constraints_config = self.config.get('optimization', {}).get('constraints', {})
        for constraint_type, params in constraints_config.items():
            if isinstance(params, dict) and params.get('enabled', True):
                optimizer.add_constraint(constraint_type, params)

        # Run optimization
        result = optimizer.optimize(**kwargs)

        # Validate results
        if not result.constraints_satisfied:
            self.logger.warning(f"Optimization constraints not fully satisfied: {result.metadata.get('issues', [])}")

        return result

    def optimize_with_renewable_objectives(
        self,
        base_method: str = 'mean_variance',
        objectives: Optional[List[str]] = None,
        capacity_factors: Optional[Dict[str, float]] = None,
        rec_values: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize with renewable-specific objectives incorporated into cvxpy objective.

        Args:
            base_method: Base optimization method (currently only 'mean_variance' supported)
            objectives: List of objectives: ['max_utilization', 'min_curtailment', 'max_recs']
            capacity_factors: Dict mapping asset -> capacity factor (for utilization objective)
            rec_values: Dict mapping asset -> REC value ($/MWh) (for REC objective)

        Returns:
            OptimizationResult
        """
        if objectives is None:
            objectives = ['min_curtailment']

        if base_method != 'mean_variance':
            self.logger.warning(
                f"Renewable objectives only supported for mean_variance method. "
                f"Using mean_variance instead of {base_method}."
            )

        self.logger.info(f"Optimizing with renewable objectives: {objectives}")

        # Get renewable objectives config
        renewable_obj_config = self.config.get('optimization', {}).get('renewable_objectives', {})

        # Create mean-variance optimizer
        optimizer = MeanVarianceOptimizer(self.returns, self.config, self.risk_analytics)

        # Get risk aversion from kwargs or config
        risk_aversion = kwargs.get('risk_aversion', optimizer.risk_aversion)

        # Build custom objective with renewable terms
        w = cp.Variable(optimizer.n_assets)

        # Base objective: return - risk_aversion * variance
        returns_part = optimizer.expected_returns @ w
        risk_part = cp.quad_form(w, optimizer.covariance)
        objective_value = returns_part - risk_aversion * risk_part

        # Identify renewable assets
        renewable_idx = [
            i for i, name in enumerate(optimizer.asset_names)
            if 'wind' in name.lower() or 'solar' in name.lower()
        ]

        # Add renewable objective terms
        for obj in objectives:
            if obj == 'max_utilization' and renewable_obj_config.get('max_utilization', {}).get('enabled', False):
                # Add utilization bonus: λ_util * sum(w_ren * cf)
                lambda_util = renewable_obj_config.get('max_utilization', {}).get('weight', 0.1)
                if capacity_factors:
                    cf_values = np.array([capacity_factors.get(name, 0.3) for name in optimizer.asset_names])
                    objective_value += lambda_util * cp.sum(cp.multiply(w, cf_values))
                else:
                    # Default: assume 0.35 capacity factor for renewables
                    if renewable_idx:
                        objective_value += lambda_util * 0.35 * cp.sum(w[renewable_idx])

            elif obj == 'min_curtailment' and renewable_obj_config.get('min_curtailment', {}).get('enabled', True):
                # Subtract curtailment penalty: - λ_curt * curtailment_risk(w)
                # Simplified: penalize high renewable concentration
                lambda_curt = renewable_obj_config.get('min_curtailment', {}).get('weight', 0.15)
                if renewable_idx:
                    # Penalize sum of squared renewable weights (encourage diversification)
                    objective_value -= lambda_curt * cp.sum_squares(w[renewable_idx])

            elif obj == 'max_recs' and renewable_obj_config.get('max_recs', {}).get('enabled', False):
                # Add REC value bonus: λ_rec * sum(w_ren * rec_value)
                lambda_rec = renewable_obj_config.get('max_recs', {}).get('weight', 0.05)
                if rec_values:
                    rec_vals = np.array([rec_values.get(name, 0.0) for name in optimizer.asset_names])
                    objective_value += lambda_rec * cp.sum(cp.multiply(w, rec_vals))
                else:
                    # Default REC value
                    default_rec_value = renewable_obj_config.get('max_recs', {}).get('rec_value', 10.0)
                    if renewable_idx:
                        objective_value += lambda_rec * (default_rec_value / 100.0) * cp.sum(w[renewable_idx])

        # Constraints
        constraints = [cp.sum(w) == 1]

        # Weight bounds
        min_w = optimizer.opt_config.get('constraints', {}).get('min_weight', 0.0)
        max_w = optimizer.opt_config.get('constraints', {}).get('max_weight', 1.0)
        constraints.append(w >= min_w)
        constraints.append(w <= max_w)

        # Add other constraints from _build_constraints
        constraints.extend(optimizer._build_constraints(w))

        # Solve
        objective = cp.Maximize(objective_value)
        problem = cp.Problem(objective, constraints)
        solver = optimizer.opt_config.get('solver', 'ECOS')
        problem.solve(solver=solver)

        weights = w.value
        if weights is None:
            self.logger.error("Renewable objectives optimization failed")
            weights = np.ones(optimizer.n_assets) / optimizer.n_assets

        # Calculate metrics
        metrics = optimizer.calculate_portfolio_metrics(weights)
        is_valid, issues = optimizer.validate_weights(weights)

        weights_series = pd.Series(weights, index=optimizer.asset_names)

        return OptimizationResult(
            weights=weights_series,
            expected_return=metrics['expected_return'],
            expected_risk=metrics['expected_risk'],
            sharpe_ratio=metrics['sharpe_ratio'],
            method='mean_variance_renewable',
            constraints_satisfied=is_valid,
            optimization_status=problem.status,
            metadata={
                'renewable_objectives': objectives,
                'risk_aversion': risk_aversion,
                'issues': issues
            }
        )

    def compare_methods(self, methods: List[str]) -> pd.DataFrame:
        """
        Compare multiple optimization methods.

        Args:
            methods: List of method names

        Returns:
            DataFrame comparing methods
        """
        self.logger.info(f"Comparing optimization methods: {methods}")

        results = []
        for method in methods:
            try:
                result = self.optimize(method=method)
                results.append({
                    'method': method,
                    'expected_return': result.expected_return,
                    'expected_risk': result.expected_risk,
                    'sharpe_ratio': result.sharpe_ratio,
                    'status': result.optimization_status,
                    'weights': result.weights.to_dict()
                })
            except Exception as e:
                self.logger.error(f"Failed to optimize with method {method}: {e}")
                results.append({
                    'method': method,
                    'expected_return': np.nan,
                    'expected_risk': np.nan,
                    'sharpe_ratio': np.nan,
                    'status': 'failed',
                    'weights': {}
                })

        comparison_df = pd.DataFrame(results)

        self.logger.info("Method comparison complete")

        return comparison_df

    def backtest_strategy(
        self,
        optimization_result: OptimizationResult,
        price_data: pd.DataFrame,
        rebalancing_frequency: str = 'monthly'
    ) -> BacktestResult:
        """
        Backtest optimized portfolio.

        Args:
            optimization_result: OptimizationResult with weights
            price_data: Historical price data
            rebalancing_frequency: Rebalancing frequency

        Returns:
            BacktestResult
        """
        self.logger.info("Backtesting optimized portfolio")

        # This is a placeholder - full implementation would simulate
        # portfolio performance with periodic rebalancing and transaction costs

        # For now, return a warning
        self.logger.warning("backtest_strategy() is not fully implemented")

        # Return None or minimal BacktestResult
        return None

    def save_results(
        self,
        result: OptimizationResult,
        filepath: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Save optimization results.

        Args:
            result: OptimizationResult to save
            filepath: Base filepath (without extension)

        Returns:
            Dict mapping data type to saved path
        """
        filepath = Path(filepath)
        saved_paths = {}

        # Save weights as CSV
        weights_path = filepath.parent / f"{filepath.stem}_weights.csv"
        result.weights.to_csv(weights_path, header=['weight'])
        saved_paths['weights'] = weights_path
        self.logger.info(f"Saved weights to {weights_path}")

        # Save metrics as CSV
        metrics_data = {
            'expected_return': [result.expected_return],
            'expected_risk': [result.expected_risk],
            'sharpe_ratio': [result.sharpe_ratio],
            'method': [result.method],
            'status': [result.optimization_status]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = filepath.parent / f"{filepath.stem}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        saved_paths['metrics'] = metrics_path
        self.logger.info(f"Saved metrics to {metrics_path}")

        # Save efficient frontier if available
        if result.efficient_frontier is not None:
            frontier_path = filepath.parent / f"{filepath.stem}_efficient_frontier.csv"
            result.efficient_frontier.to_csv(frontier_path, index=False)
            saved_paths['efficient_frontier'] = frontier_path
            self.logger.info(f"Saved efficient frontier to {frontier_path}")

        return saved_paths

    @classmethod
    def load_results(cls, filepath: Union[str, Path]) -> OptimizationResult:
        """
        Load saved optimization results.

        Args:
            filepath: Base filepath (without extension)

        Returns:
            OptimizationResult
        """
        filepath = Path(filepath)

        # Load weights
        weights_path = filepath.parent / f"{filepath.stem}_weights.csv"
        df = pd.read_csv(weights_path, index_col=0)
        weights = pd.Series(df['weight'].values, index=df.index)

        # Load metrics
        metrics_path = filepath.parent / f"{filepath.stem}_metrics.csv"
        metrics_df = pd.read_csv(metrics_path)

        # Load efficient frontier if exists
        frontier_path = filepath.parent / f"{filepath.stem}_efficient_frontier.csv"
        if frontier_path.exists():
            efficient_frontier = pd.read_csv(frontier_path)
        else:
            efficient_frontier = None

        result = OptimizationResult(
            weights=weights,
            expected_return=metrics_df['expected_return'].iloc[0],
            expected_risk=metrics_df['expected_risk'].iloc[0],
            sharpe_ratio=metrics_df['sharpe_ratio'].iloc[0],
            method=metrics_df['method'].iloc[0],
            constraints_satisfied=True,
            optimization_status=metrics_df['status'].iloc[0],
            efficient_frontier=efficient_frontier
        )

        logger.info(f"Loaded optimization results from {filepath}")

        return result
