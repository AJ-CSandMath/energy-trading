"""
Risk Reporting Module

Risk reporting with interactive Plotly charts including:
- Correlation heatmaps (assets and strategies)
- VaR and CVaR charts over time
- Risk contribution breakdowns (stacked bar and pie charts)
- Scenario comparison charts
- Risk metrics dashboards

This module consumes RiskAnalytics output and generates publication-ready
visualizations for portfolio risk analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.optimization.risk_analytics import RiskAnalytics
from src.backtesting.engine import BacktestResult
from src.config.load_config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Styling Constants
# =============================================================================

COLORS = {
    'var': '#d62728',  # Red for VaR
    'cvar': '#ff7f0e',  # Orange for CVaR
    'positive': '#2ca02c',  # Green
    'negative': '#d62728',  # Red
    'neutral': '#7f7f7f',  # Gray
    'correlation_high': '#d62728',  # Red for high correlation
    'correlation_low': '#2ca02c',  # Green for low correlation
    'equity': '#1f77b4',  # Blue for equity
    'benchmark': '#ff7f0e'  # Orange for benchmark
}

LAYOUT_DEFAULTS = {
    'height': 400,
    'template': 'plotly_white',
    'hovermode': 'x unified',
    'showlegend': True
}


# =============================================================================
# Helper Functions
# =============================================================================

def _format_currency(value: float) -> str:
    """
    Format value as currency.

    Args:
        value: Numeric value

    Returns:
        Formatted string (e.g., "$1,234.56")
    """
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def _format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.

    Args:
        value: Numeric value (0.1234 = 12.34%)
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "12.34%")
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def _get_chart_theme() -> str:
    """
    Get Plotly theme from config.

    Returns:
        Theme name (e.g., 'plotly_white', 'plotly_dark')
    """
    try:
        config = get_config()
        theme = config.get('dashboard', {}).get('charts', {}).get('theme', 'plotly_white')
        return theme
    except Exception:
        return 'plotly_white'


def _create_diverging_colorscale() -> List:
    """
    Create diverging colorscale for correlations.

    Red (negative) → White (zero) → Green (positive)

    Returns:
        Colorscale list
    """
    return [
        [0.0, '#d62728'],  # Red (strong negative)
        [0.25, '#ff9999'],  # Light red
        [0.5, '#ffffff'],  # White (zero)
        [0.75, '#90ee90'],  # Light green
        [1.0, '#2ca02c']  # Green (strong positive)
    ]


def _calculate_rolling_var_vectorized(
    returns: pd.Series,
    window: int,
    confidence_level: float,
    min_periods: int = 30
) -> pd.Series:
    """
    Calculate rolling VaR using optimized vectorized operations.

    This function precomputes rolling windows and uses vectorized quantile
    operations for better performance on long series.

    Args:
        returns: Return series
        window: Rolling window size
        confidence_level: Confidence level (e.g., 0.95)
        min_periods: Minimum observations required per window

    Returns:
        Series of rolling VaR values
    """
    # Preallocate result array
    n = len(returns)
    var_values = np.full(n, np.nan)

    # Convert to numpy for faster indexing
    returns_array = returns.values

    # Calculate quantile threshold once
    quantile = (1 - confidence_level)

    # Vectorized calculation using rolling windows
    for i in range(window - 1, n):
        window_data = returns_array[i - window + 1:i + 1]
        # Remove NaN values
        window_data = window_data[~np.isnan(window_data)]

        if len(window_data) >= min_periods:
            # Historical VaR: negative of quantile
            var_values[i] = -np.percentile(window_data, quantile * 100)

    return pd.Series(var_values, index=returns.index)


def _calculate_rolling_cvar_vectorized(
    returns: pd.Series,
    window: int,
    confidence_level: float,
    min_periods: int = 30
) -> pd.Series:
    """
    Calculate rolling CVaR using optimized vectorized operations.

    Args:
        returns: Return series
        window: Rolling window size
        confidence_level: Confidence level (e.g., 0.95)
        min_periods: Minimum observations required per window

    Returns:
        Series of rolling CVaR values
    """
    # Preallocate result array
    n = len(returns)
    cvar_values = np.full(n, np.nan)

    # Convert to numpy for faster indexing
    returns_array = returns.values

    # Calculate quantile threshold once
    quantile = (1 - confidence_level)

    # Vectorized calculation using rolling windows
    for i in range(window - 1, n):
        window_data = returns_array[i - window + 1:i + 1]
        # Remove NaN values
        window_data = window_data[~np.isnan(window_data)]

        if len(window_data) >= min_periods:
            # Calculate VaR threshold
            var_threshold = -np.percentile(window_data, quantile * 100)
            # CVaR: mean of returns below VaR
            tail_returns = window_data[window_data <= -var_threshold]
            if len(tail_returns) > 0:
                cvar_values[i] = -np.mean(tail_returns)
            else:
                cvar_values[i] = var_threshold

    return pd.Series(cvar_values, index=returns.index)


# =============================================================================
# RiskReport Class
# =============================================================================

class RiskReport:
    """
    Risk report generator with interactive Plotly visualizations.

    This class creates comprehensive risk analytics reports including:
    - VaR and CVaR charts
    - Correlation heatmaps
    - Risk decomposition charts
    - Scenario comparison charts
    - Risk metrics dashboards

    Args:
        risk_analytics: RiskAnalytics instance
        result: Optional BacktestResult (for additional context)

    Examples:
        >>> from src.optimization import RiskAnalytics, RiskReport
        >>>
        >>> # Create risk analytics
        >>> risk = RiskAnalytics(backtest_result)
        >>>
        >>> # Generate report
        >>> report = RiskReport(risk)
        >>>
        >>> # Create individual charts
        >>> var_fig = report.plot_var_cvar()
        >>> var_fig.show()
        >>>
        >>> corr_fig = report.plot_correlation_heatmap('asset')
        >>> corr_fig.show()
        >>>
        >>> # Generate full report
        >>> full_report = report.generate_full_report()
        >>>
        >>> # Save to HTML
        >>> report.save_report('risk_report.html')
        >>>
        >>> # Export data
        >>> report.export_to_csv('risk_results/')
    """

    def __init__(
        self,
        risk_analytics: RiskAnalytics,
        result: Optional[BacktestResult] = None
    ):
        """Initialize risk report generator."""
        self.risk_analytics = risk_analytics
        self.result = result

        # Calculate all metrics
        self.metrics = risk_analytics.calculate_all_metrics()

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize figure storage
        self.figures = {}

        # Apply theme
        LAYOUT_DEFAULTS['template'] = _get_chart_theme()

        self.logger.info("RiskReport initialized")

    def generate_full_report(self) -> Dict:
        """
        Generate complete risk report.

        Honors risk.reporting config flags to conditionally include charts.
        Uses color settings from risk.reporting.charts config.

        Returns:
            Dict with 'figures' (dict of plotly figures) and 'summary' (DataFrame)
        """
        self.logger.info("Generating full risk report...")

        # Get reporting config
        config = get_config()
        reporting_config = config.get('risk', {}).get('reporting', {})

        # Get flags
        include_var_backtest = reporting_config.get('include_var_backtest', True)
        include_correlation_heatmap = reporting_config.get('include_correlation_heatmap', True)
        include_risk_decomposition = reporting_config.get('include_risk_decomposition', True)
        include_scenario_analysis = reporting_config.get('include_scenario_analysis', True)

        # Get color settings
        chart_config = reporting_config.get('charts', {})
        if 'var_color' in chart_config:
            COLORS['var'] = chart_config['var_color']
        if 'cvar_color' in chart_config:
            COLORS['cvar'] = chart_config['cvar_color']

        # VaR/CVaR chart (always include)
        self.figures['var_cvar'] = self.plot_var_cvar()

        # VaR backtest
        if include_var_backtest:
            self.figures['var_backtest'] = self.plot_var_backtest()

        # Correlation heatmaps
        if include_correlation_heatmap:
            self.figures['correlation_heatmap_asset'] = self.plot_correlation_heatmap('asset')
            # Strategy correlation (if available)
            if not self.risk_analytics.strategy_returns.empty:
                self.figures['correlation_heatmap_strategy'] = self.plot_correlation_heatmap('strategy')

        # Risk decomposition
        if include_risk_decomposition:
            self.figures['risk_decomposition_asset'] = self.plot_risk_decomposition('asset')
            # Strategy decomposition (if available)
            if not self.risk_analytics.strategy_returns.empty:
                self.figures['risk_decomposition_strategy'] = self.plot_risk_decomposition('strategy')

        # Scenario analysis
        if include_scenario_analysis:
            # Run default scenarios
            stress_scenarios = config.get('risk', {}).get('stress_scenarios', {})
            scenario_names = []
            for category in ['historical', 'hypothetical']:
                scenario_names.extend(stress_scenarios.get(category, {}).keys())

            if scenario_names:
                self.figures['scenario_comparison'] = self.plot_scenario_comparison()

        # Summary table
        summary = self.create_risk_summary_table()

        self.logger.info(f"Full risk report generated with {len(self.figures)} figures")

        return {
            'figures': self.figures,
            'summary': summary
        }

    def plot_var_cvar(
        self,
        confidence_levels: Optional[List[float]] = None
    ) -> go.Figure:
        """
        Plot VaR and CVaR over time.

        Uses optimized vectorized calculations for better performance on long series.
        Respects config settings for rolling window size and data length limits.

        Args:
            confidence_levels: List of confidence levels (default [0.90, 0.95, 0.99])

        Returns:
            plotly.graph_objects.Figure
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]

        fig = go.Figure()

        # Get config settings
        config = get_config()
        rolling_config = config.get('risk', {}).get('reporting', {}).get('rolling_var', {})

        # Check if rolling calculations are enabled
        enabled = rolling_config.get('enabled', True)
        window = rolling_config.get('window', 60)
        max_data_points = rolling_config.get('max_data_points', 10000)
        min_periods = rolling_config.get('min_periods', 30)

        returns = self.risk_analytics.returns

        # Check if data is too long
        if not enabled or len(returns) > max_data_points:
            self.logger.info(
                f"Rolling VaR/CVaR calculation skipped: "
                f"enabled={enabled}, data_points={len(returns)}, max={max_data_points}"
            )
            fig.add_annotation(
                text=f"Rolling VaR/CVaR calculation skipped (series length: {len(returns)} exceeds max: {max_data_points})",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="gray")
            )
        elif len(returns) < window:
            self.logger.warning(f"Insufficient data for rolling VaR/CVaR (need {window}, have {len(returns)})")
            fig.add_annotation(
                text="Insufficient data for rolling VaR/CVaR calculation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
        else:
            # Use optimized vectorized calculations
            for cl in confidence_levels:
                # Rolling VaR (vectorized)
                rolling_var = _calculate_rolling_var_vectorized(
                    returns, window, cl, min_periods
                )

                # Rolling CVaR (vectorized)
                rolling_cvar = _calculate_rolling_cvar_vectorized(
                    returns, window, cl, min_periods
                )

                # Add VaR trace
                fig.add_trace(go.Scatter(
                    x=rolling_var.index,
                    y=rolling_var * 100,  # Convert to percentage
                    name=f'VaR {cl:.0%}',
                    mode='lines',
                    line=dict(width=2)
                ))

                # Add CVaR trace
                fig.add_trace(go.Scatter(
                    x=rolling_cvar.index,
                    y=rolling_cvar * 100,
                    name=f'CVaR {cl:.0%}',
                    mode='lines',
                    line=dict(width=2, dash='dash')
                ))

        fig.update_layout(
            title='Value at Risk (VaR) and Conditional VaR Over Time',
            xaxis_title='Date',
            yaxis_title='Risk (%)',
            **LAYOUT_DEFAULTS
        )

        return fig

    def plot_correlation_heatmap(
        self,
        correlation_type: str = 'asset'
    ) -> go.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            correlation_type: 'asset' or 'strategy'

        Returns:
            plotly.graph_objects.Figure
        """
        # Get correlation matrix
        corr_matrices = self.metrics.get('correlation', {})

        if correlation_type == 'asset':
            corr_matrix = corr_matrices.get('asset_corr', pd.DataFrame())
            title = 'Asset Correlation Matrix'
        elif correlation_type == 'strategy':
            corr_matrix = corr_matrices.get('strategy_corr', pd.DataFrame())
            title = 'Strategy Correlation Matrix'
        else:
            raise ValueError(f"Unknown correlation_type: {correlation_type}. Use 'asset' or 'strategy'")

        fig = go.Figure()

        if corr_matrix.empty:
            # Show empty chart with message
            fig.add_annotation(
                text=f"No {correlation_type} correlation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        else:
            # Create heatmap
            fig.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdYlGn',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=10),
                colorbar=dict(title='Correlation')
            ))

        fig.update_layout(
            title=title,
            xaxis_title='',
            yaxis_title='',
            **LAYOUT_DEFAULTS
        )

        return fig

    def plot_risk_decomposition(
        self,
        decomposition_type: str = 'asset'
    ) -> go.Figure:
        """
        Plot risk contribution by asset or strategy.

        Args:
            decomposition_type: 'asset' or 'strategy'

        Returns:
            plotly.graph_objects.Figure with subplots (bar + pie)
        """
        # Get risk decomposition
        if decomposition_type == 'asset':
            decomp = self.metrics.get('risk_decomposition', pd.DataFrame())
            title = 'Risk Contribution by Asset'
        elif decomposition_type == 'strategy':
            decomp = self.metrics.get('strategy_risk_decomposition', pd.DataFrame())
            title = 'Risk Contribution by Strategy'
        else:
            raise ValueError(f"Unknown decomposition_type: {decomposition_type}. Use 'asset' or 'strategy'")

        # Create subplot figure (1 row, 2 columns)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Component VaR', 'Contribution %'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )

        if decomp.empty:
            # Show empty chart with message
            fig.add_annotation(
                text=f"No {decomposition_type} risk decomposition data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        else:
            # Sort by contribution
            decomp = decomp.sort_values('contribution_pct', ascending=False)

            # Bar chart - Component VaR
            entity_col = 'asset' if decomposition_type == 'asset' else 'strategy'
            fig.add_trace(
                go.Bar(
                    x=decomp[entity_col],
                    y=decomp['component_var'] * 100,  # Convert to percentage
                    name='Component VaR',
                    marker_color=COLORS['var']
                ),
                row=1, col=1
            )

            # Pie chart - Contribution %
            fig.add_trace(
                go.Pie(
                    labels=decomp[entity_col],
                    values=decomp['contribution_pct'],
                    name='Contribution %'
                ),
                row=1, col=2
            )

        fig.update_layout(
            title_text=title,
            showlegend=False,
            **LAYOUT_DEFAULTS
        )

        fig.update_xaxes(title_text='', row=1, col=1)
        fig.update_yaxes(title_text='Component VaR (%)', row=1, col=1)

        return fig

    def plot_marginal_var(self) -> go.Figure:
        """
        Plot marginal VaR for each asset.

        Returns:
            plotly.graph_objects.Figure
        """
        decomp = self.metrics.get('risk_decomposition', pd.DataFrame())

        fig = go.Figure()

        if decomp.empty:
            fig.add_annotation(
                text="No asset risk decomposition data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        else:
            # Sort by marginal VaR
            decomp = decomp.sort_values('marginal_var', ascending=True)

            # Color bars by sign
            colors = [COLORS['positive'] if x > 0 else COLORS['negative']
                      for x in decomp['marginal_var']]

            fig.add_trace(go.Bar(
                x=decomp['marginal_var'] * 100,
                y=decomp['asset'],
                orientation='h',
                marker_color=colors,
                name='Marginal VaR'
            ))

        fig.update_layout(
            title='Marginal VaR by Asset',
            xaxis_title='Marginal VaR (%)',
            yaxis_title='',
            **LAYOUT_DEFAULTS
        )

        return fig

    def plot_scenario_comparison(
        self,
        scenarios: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Plot stress test scenario comparison.

        Args:
            scenarios: DataFrame from run_scenario_analysis() (if None, generates comparison)

        Returns:
            plotly.graph_objects.Figure
        """
        if scenarios is None:
            # Run default scenarios
            config = get_config()
            stress_scenarios = config.get('risk', {}).get('stress_scenarios', {})

            scenario_names = []
            for category in ['historical', 'hypothetical']:
                scenario_names.extend(stress_scenarios.get(category, {}).keys())

            if not scenario_names:
                self.logger.warning("No stress scenarios defined in config")
                scenarios = pd.DataFrame()
            else:
                scenarios = self.risk_analytics.run_scenario_analysis(scenario_names[:5])  # Limit to 5

        fig = go.Figure()

        if scenarios.empty:
            fig.add_annotation(
                text="No scenario analysis data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        else:
            # Create grouped bar chart
            metrics_to_plot = ['var', 'cvar', 'max_drawdown', 'total_return']

            for metric in metrics_to_plot:
                if metric in scenarios.columns:
                    fig.add_trace(go.Bar(
                        x=scenarios['scenario'],
                        y=scenarios[metric] * 100,  # Convert to percentage
                        name=metric.replace('_', ' ').title()
                    ))

        fig.update_layout(
            title='Stress Test Scenario Comparison',
            xaxis_title='Scenario',
            yaxis_title='Metric Value (%)',
            barmode='group',
            **LAYOUT_DEFAULTS
        )

        return fig

    def plot_rolling_correlation(
        self,
        asset_pair: Tuple[str, str],
        window: int = 60
    ) -> go.Figure:
        """
        Plot rolling correlation over time.

        Args:
            asset_pair: Tuple of (asset1, asset2)
            window: Rolling window size

        Returns:
            plotly.graph_objects.Figure
        """
        asset1, asset2 = asset_pair

        # Get asset returns
        asset_returns = self.risk_analytics.asset_returns

        if asset1 not in asset_returns.columns or asset2 not in asset_returns.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Assets {asset1} or {asset2} not found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

        # Calculate rolling correlation
        rolling_corr = asset_returns[asset1].rolling(window=window).corr(asset_returns[asset2])

        fig = go.Figure()

        # Add correlation line
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode='lines',
            name=f'{asset1} vs {asset2}',
            line=dict(width=2, color=COLORS['equity'])
        ))

        # Add high correlation threshold
        high_corr_threshold = 0.7
        fig.add_hline(
            y=high_corr_threshold,
            line_dash="dash",
            line_color=COLORS['correlation_high'],
            annotation_text="High Correlation Threshold"
        )

        fig.add_hline(
            y=-high_corr_threshold,
            line_dash="dash",
            line_color=COLORS['correlation_high']
        )

        fig.update_layout(
            title=f'Rolling Correlation: {asset1} vs {asset2}',
            xaxis_title='Date',
            yaxis_title='Correlation',
            yaxis=dict(range=[-1, 1]),
            **LAYOUT_DEFAULTS
        )

        return fig

    def plot_var_backtest(self) -> go.Figure:
        """
        Plot VaR backtesting (violations).

        Compares actual returns to VaR threshold and marks violations.

        Returns:
            plotly.graph_objects.Figure
        """
        returns = self.risk_analytics.returns
        confidence_level = 0.95

        # Calculate VaR
        var = self.risk_analytics.calculate_portfolio_var(confidence_level)['var_pct']

        # Identify violations (returns < -VaR)
        violations = returns[returns < -var]

        # Calculate violation rate
        violation_rate = len(violations) / len(returns)
        expected_rate = 1 - confidence_level

        fig = go.Figure()

        # Plot returns
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns * 100,
            mode='lines',
            name='Returns',
            line=dict(color=COLORS['equity'])
        ))

        # Plot VaR threshold
        fig.add_hline(
            y=-var * 100,
            line_dash="dash",
            line_color=COLORS['var'],
            annotation_text=f'VaR {confidence_level:.0%}'
        )

        # Mark violations
        if not violations.empty:
            fig.add_trace(go.Scatter(
                x=violations.index,
                y=violations * 100,
                mode='markers',
                name='VaR Violations',
                marker=dict(color=COLORS['negative'], size=10, symbol='x')
            ))

        # Add annotation with violation rate
        fig.add_annotation(
            text=f"Violation Rate: {violation_rate:.2%} (Expected: {expected_rate:.2%})",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        fig.update_layout(
            title='VaR Backtesting',
            xaxis_title='Date',
            yaxis_title='Returns (%)',
            **LAYOUT_DEFAULTS
        )

        return fig

    def plot_risk_metrics_dashboard(self) -> go.Figure:
        """
        Create comprehensive risk dashboard.

        Returns:
            plotly.graph_objects.Figure with 2x2 subplots
        """
        # Create 2x2 subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'VaR/CVaR Over Time',
                'Asset Correlation',
                'Risk Decomposition',
                'Scenario Comparison'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'heatmap'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )

        # 1. VaR/CVaR (simplified)
        returns = self.risk_analytics.returns
        window = 60
        if len(returns) >= window:
            rolling_var = returns.rolling(window=window).apply(
                lambda x: calculate_var(x, 0.95, 'historical') if len(x.dropna()) >= 30 else np.nan,
                raw=False
            )
            fig.add_trace(
                go.Scatter(x=rolling_var.index, y=rolling_var * 100, name='VaR 95%', line=dict(color=COLORS['var'])),
                row=1, col=1
            )

        # 2. Correlation heatmap
        corr_matrix = self.metrics.get('correlation', {}).get('asset_corr', pd.DataFrame())
        if not corr_matrix.empty:
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    showscale=False
                ),
                row=1, col=2
            )

        # 3. Risk decomposition
        decomp = self.metrics.get('risk_decomposition', pd.DataFrame())
        if not decomp.empty:
            decomp_sorted = decomp.sort_values('contribution_pct', ascending=False).head(5)
            fig.add_trace(
                go.Bar(x=decomp_sorted['asset'], y=decomp_sorted['contribution_pct'], marker_color=COLORS['var'], showlegend=False),
                row=2, col=1
            )

        # 4. Scenario comparison (placeholder - would need scenario data)
        fig.add_annotation(
            text="Run scenario analysis to populate",
            xref="x4", yref="y4",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=10, color="gray"),
            row=2, col=2
        )

        fig.update_layout(
            title_text='Risk Metrics Dashboard',
            showlegend=False,
            height=800,
            **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != 'height'}
        )

        return fig

    def create_risk_summary_table(self) -> pd.DataFrame:
        """
        Create formatted risk summary table.

        Returns:
            DataFrame with risk metrics
        """
        summary_data = []

        # VaR/CVaR section
        for cl_str, var_data in self.metrics.get('var', {}).items():
            summary_data.append({
                'Category': 'VaR',
                'Metric': f'VaR {cl_str}',
                'Value': _format_percentage(var_data['var_pct']),
                'Dollar Value': _format_currency(var_data['var_dollars'])
            })

        for cl_str, cvar_data in self.metrics.get('cvar', {}).items():
            summary_data.append({
                'Category': 'CVaR',
                'Metric': f'CVaR {cl_str}',
                'Value': _format_percentage(cvar_data['cvar_pct']),
                'Dollar Value': _format_currency(cvar_data['cvar_dollars'])
            })

        # Risk-adjusted metrics
        summary_data.append({
            'Category': 'Risk-Adjusted',
            'Metric': 'Sharpe Ratio',
            'Value': f"{self.metrics.get('sharpe_ratio', np.nan):.2f}",
            'Dollar Value': 'N/A'
        })

        summary_data.append({
            'Category': 'Risk-Adjusted',
            'Metric': 'Sortino Ratio',
            'Value': f"{self.metrics.get('sortino_ratio', np.nan):.2f}",
            'Dollar Value': 'N/A'
        })

        summary_data.append({
            'Category': 'Drawdown',
            'Metric': 'Max Drawdown',
            'Value': _format_percentage(self.metrics.get('max_drawdown', np.nan)),
            'Dollar Value': 'N/A'
        })

        # Information ratio
        if self.metrics.get('information_ratio') is not None and not np.isnan(self.metrics['information_ratio']):
            summary_data.append({
                'Category': 'Risk-Adjusted',
                'Metric': 'Information Ratio',
                'Value': f"{self.metrics['information_ratio']:.2f}",
                'Dollar Value': 'N/A'
            })

        # Calmar ratio
        if self.metrics.get('calmar_ratio') is not None and not np.isnan(self.metrics['calmar_ratio']):
            summary_data.append({
                'Category': 'Risk-Adjusted',
                'Metric': 'Calmar Ratio',
                'Value': f"{self.metrics['calmar_ratio']:.2f}",
                'Dollar Value': 'N/A'
            })

        # Implied Volatility (current value)
        if 'implied_volatility' in self.metrics and not self.metrics['implied_volatility'].empty:
            current_impl_vol = self.metrics['implied_volatility'].iloc[-1]
            if not np.isnan(current_impl_vol):
                summary_data.append({
                    'Category': 'Volatility',
                    'Metric': 'Implied Volatility (EWMA)',
                    'Value': _format_percentage(current_impl_vol),
                    'Dollar Value': 'N/A'
                })

        # Risk decomposition (top 3 contributors)
        decomp = self.metrics.get('risk_decomposition', pd.DataFrame())
        if not decomp.empty:
            top_contributors = decomp.nlargest(3, 'contribution_pct')
            for _, row in top_contributors.iterrows():
                summary_data.append({
                    'Category': 'Risk Contribution',
                    'Metric': f'{row["asset"]} Contribution',
                    'Value': _format_percentage(row['contribution_pct'] / 100),
                    'Dollar Value': 'N/A'
                })

        # Correlation statistics
        corr_matrix = self.metrics.get('correlation', {}).get('asset_corr', pd.DataFrame())
        if not corr_matrix.empty:
            # Get upper triangle (exclude diagonal)
            corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            corr_flat = corr_values.values.flatten()
            corr_flat = corr_flat[~np.isnan(corr_flat)]

            if len(corr_flat) > 0:
                summary_data.append({
                    'Category': 'Correlation',
                    'Metric': 'Average Correlation',
                    'Value': f"{np.mean(corr_flat):.2f}",
                    'Dollar Value': 'N/A'
                })

                summary_data.append({
                    'Category': 'Correlation',
                    'Metric': 'Max Correlation',
                    'Value': f"{np.max(corr_flat):.2f}",
                    'Dollar Value': 'N/A'
                })

        summary_df = pd.DataFrame(summary_data)

        return summary_df

    def create_scenario_summary_table(
        self,
        scenarios: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create scenario analysis summary table.

        Args:
            scenarios: DataFrame from run_scenario_analysis()

        Returns:
            DataFrame with scenario metrics
        """
        if scenarios is None or scenarios.empty:
            self.logger.warning("No scenario data available for summary table")
            return pd.DataFrame()

        # Format scenario results
        summary = scenarios[['scenario', 'description', 'var', 'cvar', 'max_drawdown', 'total_return']].copy()

        # Format numeric columns as percentages
        for col in ['var', 'cvar', 'max_drawdown', 'total_return']:
            if col in summary.columns:
                summary[col] = summary[col].apply(lambda x: _format_percentage(x))

        # Rename columns
        summary.columns = ['Scenario', 'Description', 'VaR', 'CVaR', 'Max Drawdown', 'Total Return']

        return summary

    def save_report(
        self,
        output_path: Union[str, Path],
        include_all: bool = True
    ) -> Path:
        """
        Save report to HTML file.

        Args:
            output_path: Path to save HTML file
            include_all: If True, generates all figures before saving

        Returns:
            Path to saved HTML file
        """
        output_path = Path(output_path)

        # Generate all figures if requested
        if include_all and not self.figures:
            self.generate_full_report()

        # Create HTML content
        html_parts = []

        # Add title
        html_parts.append("<html><head><title>Risk Analytics Report</title></head><body>")
        html_parts.append("<h1>Risk Analytics Report</h1>")

        # Add summary table
        summary = self.create_risk_summary_table()
        html_parts.append("<h2>Risk Summary</h2>")
        html_parts.append(summary.to_html(index=False))

        # Add figures
        html_parts.append("<h2>Risk Charts</h2>")
        for name, fig in self.figures.items():
            html_parts.append(f"<h3>{name.replace('_', ' ').title()}</h3>")
            html_parts.append(fig.to_html(include_plotlyjs='cdn', full_html=False))

        html_parts.append("</body></html>")

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_parts))

        self.logger.info(f"Risk report saved to {output_path}")

        return output_path

    def save_figures(
        self,
        output_dir: Path,
        format: str = 'png'
    ) -> List[Path]:
        """
        Save individual figures as images.

        Requires kaleido package.

        Args:
            output_dir: Directory to save figures
            format: Image format ('png', 'jpg', 'svg', 'pdf')

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for name, fig in self.figures.items():
            output_path = output_dir / f"{name}.{format}"
            try:
                fig.write_image(output_path)
                saved_paths.append(output_path)
                self.logger.info(f"Figure saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving figure {name}: {e}")

        return saved_paths

    def export_to_csv(
        self,
        output_dir: Union[str, Path],
        scenarios: Optional[pd.DataFrame] = None
    ) -> Dict[str, Path]:
        """
        Export risk metrics to CSV files.

        Args:
            output_dir: Directory to save CSV files
            scenarios: Optional scenarios DataFrame (if None, generates default scenarios)

        Returns:
            Dict mapping filename to path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Risk metrics summary
        summary = self.create_risk_summary_table()
        if not summary.empty:
            path = output_dir / 'risk_metrics.csv'
            summary.to_csv(path, index=False)
            saved_files['risk_metrics'] = path
            self.logger.info(f"Risk metrics saved to {path}")

        # Risk decomposition
        decomp = self.metrics.get('risk_decomposition', pd.DataFrame())
        if not decomp.empty:
            path = output_dir / 'risk_decomposition.csv'
            decomp.to_csv(path, index=False)
            saved_files['risk_decomposition'] = path
            self.logger.info(f"Risk decomposition saved to {path}")

        # Correlation matrix
        corr_matrix = self.metrics.get('correlation', {}).get('asset_corr', pd.DataFrame())
        if not corr_matrix.empty:
            path = output_dir / 'correlation_matrix.csv'
            corr_matrix.to_csv(path)
            saved_files['correlation_matrix'] = path
            self.logger.info(f"Correlation matrix saved to {path}")

        # Scenario analysis
        if scenarios is None:
            # Generate default scenarios
            config = get_config()
            stress_scenarios = config.get('risk', {}).get('stress_scenarios', {})

            scenario_names = []
            for category in ['historical', 'hypothetical']:
                scenario_names.extend(stress_scenarios.get(category, {}).keys())

            if scenario_names:
                try:
                    scenarios = self.risk_analytics.run_scenario_analysis(scenario_names[:10])  # Limit to 10
                except Exception as e:
                    self.logger.error(f"Error running scenario analysis: {e}")
                    scenarios = None

        if scenarios is not None and not scenarios.empty:
            path = output_dir / 'scenario_analysis.csv'
            scenarios.to_csv(path, index=False)
            saved_files['scenario_analysis'] = path
            self.logger.info(f"Scenario analysis saved to {path}")

        return saved_files


# Import calculate_var and calculate_cvar for use in plotting
from src.optimization.risk_analytics import calculate_var, calculate_cvar
