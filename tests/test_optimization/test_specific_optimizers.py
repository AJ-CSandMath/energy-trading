"""
Unit tests for specific optimizer classes.

Tests individual optimizer implementations for correct weights,
bounds compliance, and method-specific properties. Deterministic
tests with small datasets for CI stability.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.optimizer import (
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    BlackLittermanOptimizer,
    MinimumCVaROptimizer,
    OptimizationResult,
)


# =============================================================================
# Test Helpers
# =============================================================================

def create_test_returns(n_assets=3, n_samples=100, seed=42):
    """Create deterministic test returns DataFrame."""
    np.random.seed(seed)
    asset_names = [f"ASSET_{chr(65+i)}" for i in range(n_assets)]

    # Small magnitudes for stability
    returns = pd.DataFrame({
        asset_names[0]: np.random.randn(n_samples) * 0.01 + 0.001,
        asset_names[1]: np.random.randn(n_samples) * 0.015 + 0.0005,
        asset_names[2]: np.random.randn(n_samples) * 0.012 + 0.0008,
    }[:n_assets])

    return returns


# =============================================================================
# MeanVarianceOptimizer Tests
# =============================================================================

class TestMeanVarianceOptimizer:
    """Test Mean-Variance optimizer."""

    def test_optimize_max_sharpe(self):
        """Test max Sharpe optimization."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = MeanVarianceOptimizer(returns)
        result = optimizer.optimize(objective='max_sharpe')

        # Verify OptimizationResult structure
        assert isinstance(result, OptimizationResult)
        assert result.weights is not None
        assert result.method == 'mean_variance'

        # Verify weights sum to 1
        weight_sum = result.weights.sum()
        assert abs(weight_sum - 1.0) < 1e-6

        # Verify bounds (0 to 1 for long-only)
        assert (result.weights >= -1e-6).all()
        assert (result.weights <= 1.0 + 1e-6).all()

        # Verify Sharpe ratio is finite
        assert np.isfinite(result.sharpe_ratio)
        assert result.sharpe_ratio > 0  # Should be positive for reasonable data

    def test_optimize_min_risk(self):
        """Test minimum risk optimization."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = MeanVarianceOptimizer(returns)
        result = optimizer.optimize(objective='min_risk')

        # Verify weights sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6

        # Verify risk is minimized relative to naive equal weights
        equal_weights = np.ones(3) / 3
        equal_risk = np.sqrt(equal_weights @ optimizer.covariance @ equal_weights)

        assert result.expected_risk <= equal_risk * 1.01  # Allow small tolerance

    def test_compute_efficient_frontier(self):
        """Test efficient frontier computation."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = MeanVarianceOptimizer(returns)
        frontier = optimizer.compute_efficient_frontier(n_points=10)

        # Verify DataFrame structure
        assert isinstance(frontier, pd.DataFrame)
        assert 'return' in frontier.columns
        assert 'risk' in frontier.columns
        assert len(frontier) == 10

        # Verify monotonic relationship: higher risk -> higher return
        # (allowing for small numerical errors)
        for i in range(1, len(frontier)):
            if frontier.loc[i, 'risk'] > frontier.loc[i-1, 'risk']:
                assert frontier.loc[i, 'return'] >= frontier.loc[i-1, 'return'] - 1e-6

    def test_constraints_applied(self):
        """Test that weight constraints are respected."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        config = {
            'optimization': {
                'constraints': {
                    'min_weight': 0.1,
                    'max_weight': 0.5,
                }
            }
        }

        optimizer = MeanVarianceOptimizer(returns, config=config)
        result = optimizer.optimize(objective='max_sharpe')

        # Verify all weights respect bounds
        assert (result.weights >= 0.1 - 1e-6).all()
        assert (result.weights <= 0.5 + 1e-6).all()
        assert abs(result.weights.sum() - 1.0) < 1e-6


# =============================================================================
# RiskParityOptimizer Tests
# =============================================================================

class TestRiskParityOptimizer:
    """Test Risk Parity optimizer."""

    def test_optimize_equal_risk_contributions(self):
        """Test that risk parity produces approximately equal risk contributions."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = RiskParityOptimizer(returns)
        result = optimizer.optimize()

        # Verify weights sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6

        # Verify all weights are positive (risk parity is long-only)
        assert (result.weights > 0).all()

        # Compute risk contributions
        weights_array = result.weights.values
        risk_contributions = optimizer.calculate_risk_contributions(weights_array)

        # Verify risk contributions are approximately equal
        # Allow 10% tolerance for numerical optimization
        mean_rc = np.mean(risk_contributions)
        for rc in risk_contributions:
            assert abs(rc - mean_rc) / mean_rc < 0.10

    def test_calculate_risk_contributions(self):
        """Test risk contribution calculation formula."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=50)

        optimizer = RiskParityOptimizer(returns)

        # Use known weights for deterministic test
        weights = np.array([0.4, 0.35, 0.25])
        risk_contributions = optimizer.calculate_risk_contributions(weights)

        # Verify formula: RC_i = w_i * (Σ @ w)_i / σ_p
        cov = optimizer.covariance
        marginal_risk = cov @ weights
        portfolio_risk = np.sqrt(weights @ cov @ weights)

        expected_rc = weights * marginal_risk / portfolio_risk

        np.testing.assert_allclose(risk_contributions, expected_rc, rtol=1e-6)

        # Verify risk contributions sum to portfolio risk
        assert abs(risk_contributions.sum() - portfolio_risk) < 1e-10


# =============================================================================
# BlackLittermanOptimizer Tests
# =============================================================================

class TestBlackLittermanOptimizer:
    """Test Black-Litterman optimizer."""

    def test_add_view_and_optimize(self):
        """Test adding views and optimizing."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        # Create market weights (equal for simplicity)
        market_weights = np.ones(3) / 3

        optimizer = BlackLittermanOptimizer(returns)

        # Optimize without views first (baseline)
        result_no_views = optimizer.optimize(market_weights=market_weights)

        # Add a bullish view on ASSET_A
        optimizer.add_view('ASSET_A', expected_return=0.05, confidence=0.8)

        # Optimize with views
        result_with_views = optimizer.optimize(
            market_weights=market_weights,
            views={'ASSET_A': 0.05},
            view_confidences={'ASSET_A': 0.8}
        )

        # Verify weights sum to 1
        assert abs(result_with_views.weights.sum() - 1.0) < 1e-6

        # Verify ASSET_A gets higher weight with positive view
        assert result_with_views.weights['ASSET_A'] > result_no_views.weights['ASSET_A']

    def test_calculate_implied_returns(self):
        """Test implied returns calculation."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = BlackLittermanOptimizer(returns)

        # Market weights
        market_weights = np.array([0.5, 0.3, 0.2])
        risk_aversion = 2.5

        implied_returns = optimizer.calculate_implied_returns(market_weights, risk_aversion)

        # Verify correct length
        assert len(implied_returns) == 3

        # Verify non-NaN
        assert not np.isnan(implied_returns).any()

        # Verify formula: π = λ * Σ @ w_market
        expected_implied = risk_aversion * (optimizer.covariance @ market_weights)
        np.testing.assert_allclose(implied_returns, expected_implied, rtol=1e-6)


# =============================================================================
# MinimumCVaROptimizer Tests
# =============================================================================

class TestMinimumCVaROptimizer:
    """Test Minimum CVaR optimizer."""

    def test_optimize_cvar_minimization(self):
        """Test CVaR minimization."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = MinimumCVaROptimizer(returns)
        result = optimizer.optimize()

        # Verify weights sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6

        # Verify bounds
        assert (result.weights >= -1e-6).all()
        assert (result.weights <= 1.0 + 1e-6).all()

        # Compute portfolio CVaR
        weights_array = result.weights.values
        portfolio_cvar = optimizer.calculate_portfolio_cvar(weights_array, confidence_level=0.95)

        # Verify CVaR is lower than equal weights baseline
        equal_weights = np.ones(3) / 3
        equal_cvar = optimizer.calculate_portfolio_cvar(equal_weights, confidence_level=0.95)

        assert portfolio_cvar <= equal_cvar * 1.05  # Allow small tolerance

    def test_calculate_portfolio_cvar_from_scenarios(self):
        """Test CVaR calculation from scenarios."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = MinimumCVaROptimizer(returns)

        # Use known weights
        weights = np.array([0.4, 0.35, 0.25])
        confidence_level = 0.95

        cvar = optimizer.calculate_portfolio_cvar(weights, confidence_level)

        # Verify CVaR is positive (loss)
        assert cvar > 0

        # Manually compute CVaR for verification
        # Portfolio returns for each scenario
        scenario_returns = optimizer.return_scenarios @ weights

        # VaR at confidence level
        var_level = np.percentile(scenario_returns, (1 - confidence_level) * 100)

        # CVaR = expected loss beyond VaR
        tail_losses = scenario_returns[scenario_returns <= var_level]
        if len(tail_losses) > 0:
            expected_cvar = -tail_losses.mean()
        else:
            expected_cvar = -var_level

        # CVaR should be close to manually computed value
        assert abs(cvar - expected_cvar) / expected_cvar < 0.15  # Allow 15% tolerance for numerical differences


# =============================================================================
# Cross-Optimizer Tests
# =============================================================================

class TestAllOptimizers:
    """Tests applicable to all optimizers."""

    @pytest.mark.parametrize("optimizer_class", [
        MeanVarianceOptimizer,
        RiskParityOptimizer,
        BlackLittermanOptimizer,
        MinimumCVaROptimizer,
    ])
    def test_weights_sum_to_one(self, optimizer_class):
        """Test that all optimizers produce weights summing to 1."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = optimizer_class(returns)

        # Call optimize with appropriate parameters
        if optimizer_class == BlackLittermanOptimizer:
            market_weights = np.ones(3) / 3
            result = optimizer.optimize(market_weights=market_weights)
        else:
            result = optimizer.optimize()

        # Verify weights sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6

    @pytest.mark.parametrize("optimizer_class", [
        MeanVarianceOptimizer,
        RiskParityOptimizer,
        BlackLittermanOptimizer,
        MinimumCVaROptimizer,
    ])
    def test_optimization_result_structure(self, optimizer_class):
        """Test that all optimizers return properly structured OptimizationResult."""
        np.random.seed(42)
        returns = create_test_returns(n_assets=3, n_samples=100)

        optimizer = optimizer_class(returns)

        if optimizer_class == BlackLittermanOptimizer:
            market_weights = np.ones(3) / 3
            result = optimizer.optimize(market_weights=market_weights)
        else:
            result = optimizer.optimize()

        # Verify OptimizationResult structure
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.weights, pd.Series)
        assert np.isfinite(result.expected_return)
        assert np.isfinite(result.expected_risk)
        assert result.expected_risk >= 0
        assert result.constraints_satisfied is True
        assert result.optimization_status in ['optimal', 'Optimal']
