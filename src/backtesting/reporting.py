"""
Backtesting Report Generation Module

Provides report generation with interactive Plotly charts (equity curves, drawdown,
trade distributions) and summary tables for backtest analysis.

Features:
- Interactive Plotly visualizations (equity curves, drawdowns, trade analysis)
- Comprehensive summary tables
- HTML report generation for standalone viewing
- CSV export for external analysis
- Compatible with Streamlit dashboard integration

Example:
    >>> from src.backtesting import BacktestEngine, BacktestReport
    >>>
    >>> # Run backtest
    >>> result = engine.run(price_data, signals)
    >>>
    >>> # Generate report
    >>> report = BacktestReport(result)
    >>>
    >>> # Create individual charts
    >>> equity_fig = report.plot_equity_curve(show_drawdown=True, show_trades=True)
    >>> equity_fig.show()
    >>>
    >>> # Generate full report
    >>> full_report = report.generate_full_report()
    >>>
    >>> # Save to HTML
    >>> report.save_report('backtest_report.html')
    >>>
    >>> # Export data
    >>> report.export_to_csv('backtest_results/')
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtesting.engine import BacktestResult
from src.backtesting.metrics import PerformanceMetrics
from src.config.load_config import get_config


# Styling constants
COLORS = {
    'equity': '#1f77b4',  # Blue
    'drawdown': '#d62728',  # Red
    'win': '#2ca02c',  # Green
    'loss': '#d62728',  # Red
    'neutral': '#7f7f7f'  # Gray
}

LAYOUT_DEFAULTS = {
    'height': 400,
    'template': 'plotly_white',
    'hovermode': 'x unified',
    'showlegend': True
}


class BacktestReport:
    """
    Backtesting report generator with interactive visualizations.

    Creates comprehensive reports including equity curves, drawdown analysis,
    trade distributions, and performance summaries.

    Attributes:
        result: BacktestResult object
        metrics: PerformanceMetrics object
        figures: Dictionary of generated figures
        logger: Logger instance
    """

    def __init__(
        self,
        result: BacktestResult,
        metrics: Optional[PerformanceMetrics] = None
    ):
        """
        Initialize report generator.

        Args:
            result: BacktestResult from backtest engine
            metrics: Optional PerformanceMetrics (calculated if not provided)
        """
        self.logger = logging.getLogger(__name__)
        self.result = result

        # Calculate metrics if not provided
        if metrics is None:
            self.metrics = PerformanceMetrics(
                equity_curve=result.equity_curve,
                trades=result.trades,
                initial_capital=result.initial_capital
            )
        else:
            self.metrics = metrics

        # Storage for generated figures
        self.figures = {}

        self.logger.info("BacktestReport initialized")

    def generate_full_report(self) -> Dict:
        """
        Generate complete report with all visualizations.

        Returns:
            Dictionary with all figures and summary
        """
        self.logger.info("Generating full report")

        # Generate all charts
        self.figures['equity_curve'] = self.plot_equity_curve(
            show_drawdown=True,
            show_trades=True
        )
        self.figures['drawdown'] = self.plot_drawdown_chart()
        self.figures['trade_distribution'] = self.plot_trade_distribution()
        self.figures['monthly_returns'] = self.plot_monthly_returns()
        self.figures['rolling_metrics'] = self.plot_rolling_metrics()
        self.figures['trade_analysis'] = self.plot_trade_analysis()
        self.figures['position_exposure'] = self.plot_position_exposure()

        # Generate summary table
        summary = self.create_summary_table()
        trade_summary = self.create_trade_summary()

        full_report = {
            'figures': self.figures,
            'summary': summary,
            'trade_summary': trade_summary
        }

        self.logger.info("Full report generated")

        return full_report

    def plot_equity_curve(
        self,
        show_drawdown: bool = True,
        show_trades: bool = True
    ) -> go.Figure:
        """
        Plot portfolio equity curve over time.

        Args:
            show_drawdown: Whether to show drawdown as secondary plot
            show_trades: Whether to show trade markers

        Returns:
            Plotly Figure
        """
        # Create figure with secondary y-axis for drawdown
        fig = make_subplots(
            rows=2 if show_drawdown else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Portfolio Equity', 'Drawdown') if show_drawdown else ('Portfolio Equity',),
            row_heights=[0.7, 0.3] if show_drawdown else [1.0]
        )

        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=self.result.equity_curve.index,
                y=self.result.equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color=COLORS['equity'], width=2),
                hovertemplate='Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>'
            ),
            row=1,
            col=1
        )

        # Add trade markers if requested
        if show_trades and not self.result.trades.empty:
            # Select PnL column: prefer net_realized_pnl if available
            pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in self.result.trades.columns else 'realized_pnl'
            pnl_label = 'Net PnL' if pnl_col == 'net_realized_pnl' else 'PnL'

            # Winning trades
            winning_trades = self.result.trades[self.result.trades[pnl_col] > 0]
            if not winning_trades.empty:
                # Map trades to equity curve
                trade_equity = []
                for timestamp in winning_trades['timestamp']:
                    if timestamp in self.result.equity_curve.index:
                        trade_equity.append(self.result.equity_curve.loc[timestamp])
                    else:
                        trade_equity.append(None)

                fig.add_trace(
                    go.Scatter(
                        x=winning_trades['timestamp'],
                        y=trade_equity,
                        mode='markers',
                        name='Winning Trades',
                        marker=dict(color=COLORS['win'], size=8, symbol='triangle-up'),
                        hovertemplate=f'Date: %{{x}}<br>{pnl_label}: +$%{{text:.2f}}<extra></extra>',
                        text=winning_trades[pnl_col]
                    ),
                    row=1,
                    col=1
                )

            # Losing trades
            losing_trades = self.result.trades[self.result.trades[pnl_col] < 0]
            if not losing_trades.empty:
                trade_equity = []
                for timestamp in losing_trades['timestamp']:
                    if timestamp in self.result.equity_curve.index:
                        trade_equity.append(self.result.equity_curve.loc[timestamp])
                    else:
                        trade_equity.append(None)

                fig.add_trace(
                    go.Scatter(
                        x=losing_trades['timestamp'],
                        y=trade_equity,
                        mode='markers',
                        name='Losing Trades',
                        marker=dict(color=COLORS['loss'], size=8, symbol='triangle-down'),
                        hovertemplate=f'Date: %{{x}}<br>{pnl_label}: $%{{text:.2f}}<extra></extra>',
                        text=losing_trades[pnl_col]
                    ),
                    row=1,
                    col=1
                )

        # Plot drawdown if requested
        if show_drawdown:
            # Calculate drawdown
            cummax = self.result.equity_curve.cummax()
            drawdown = (self.result.equity_curve - cummax) / cummax * 100

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color=COLORS['drawdown'], width=1),
                    hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=2,
                col=1
            )

            # Update y-axis for drawdown
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        # Update layout
        fig.update_xaxes(title_text="Date", row=2 if show_drawdown else 1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)

        fig.update_layout(
            title=f"Backtest Results: {self.result.start_date} to {self.result.end_date}",
            height=600 if show_drawdown else 400,
            template=LAYOUT_DEFAULTS['template'],
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def plot_drawdown_chart(self) -> go.Figure:
        """
        Plot drawdown over time.

        Returns:
            Plotly Figure
        """
        # Calculate drawdown
        cummax = self.result.equity_curve.cummax()
        drawdown = (self.result.equity_curve - cummax) / cummax * 100

        fig = go.Figure()

        # Plot drawdown as area
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color=COLORS['drawdown'], width=2),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            )
        )

        # Add horizontal line at max drawdown
        max_dd = self.metrics.metrics['max_drawdown'] * 100
        fig.add_hline(
            y=max_dd,
            line_dash="dash",
            line_color=COLORS['loss'],
            annotation_text=f"Max DD: {max_dd:.2f}%",
            annotation_position="right"
        )

        # Update layout
        fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=LAYOUT_DEFAULTS['height'],
            template=LAYOUT_DEFAULTS['template'],
            hovermode='x unified',
            showlegend=False
        )

        return fig

    def plot_trade_distribution(self) -> go.Figure:
        """
        Plot distribution of trade PnLs.

        Returns:
            Plotly Figure
        """
        # Select PnL column: prefer net_realized_pnl if available
        pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in self.result.trades.columns else 'realized_pnl'

        if self.result.trades.empty or pnl_col not in self.result.trades.columns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

        # Filter closed trades
        closed_trades = self.result.trades[self.result.trades[pnl_col] != 0]

        if closed_trades.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No closed trades",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

        fig = go.Figure()

        # Plot winning trades
        winning_pnl = closed_trades[closed_trades[pnl_col] > 0][pnl_col]
        if len(winning_pnl) > 0:
            fig.add_trace(
                go.Histogram(
                    x=winning_pnl,
                    name='Winning Trades',
                    marker_color=COLORS['win'],
                    opacity=0.7,
                    hovertemplate='PnL: $%{x:.2f}<br>Count: %{y}<extra></extra>'
                )
            )

        # Plot losing trades
        losing_pnl = closed_trades[closed_trades[pnl_col] < 0][pnl_col]
        if len(losing_pnl) > 0:
            fig.add_trace(
                go.Histogram(
                    x=losing_pnl,
                    name='Losing Trades',
                    marker_color=COLORS['loss'],
                    opacity=0.7,
                    hovertemplate='PnL: $%{x:.2f}<br>Count: %{y}<extra></extra>'
                )
            )

        # Add vertical line at zero
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=COLORS['neutral'],
            annotation_text="Break-even"
        )

        # Add statistics annotation
        mean_pnl = closed_trades[pnl_col].mean()
        median_pnl = closed_trades[pnl_col].median()
        std_pnl = closed_trades[pnl_col].std()

        fig.add_annotation(
            text=f"Mean: ${mean_pnl:.2f}<br>Median: ${median_pnl:.2f}<br>Std: ${std_pnl:.2f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            align="right",
            bgcolor="white",
            bordercolor=COLORS['neutral'],
            borderwidth=1
        )

        # Update layout
        fig.update_layout(
            title="Trade PnL Distribution",
            xaxis_title="Realized PnL ($)",
            yaxis_title="Count",
            height=LAYOUT_DEFAULTS['height'],
            template=LAYOUT_DEFAULTS['template'],
            barmode='overlay',
            showlegend=True
        )

        return fig

    def plot_monthly_returns(self) -> go.Figure:
        """
        Plot monthly returns heatmap.

        Returns:
            Plotly Figure
        """
        # Resample to monthly
        monthly_equity = self.result.equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100  # Convert to percentage

        # Create year-month grid
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })

        # Pivot to create heatmap
        heatmap_data = df.pivot(index='Year', columns='Month', values='Return')

        # Reindex columns to ensure all months 1-12 exist (fill missing with NaN)
        heatmap_data = heatmap_data.reindex(columns=range(1, 13))

        # Month names for x-axis (aligned with columns)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=month_names,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=heatmap_data.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title="Monthly Returns (%)",
            xaxis_title="Month",
            yaxis_title="Year",
            height=LAYOUT_DEFAULTS['height'],
            template=LAYOUT_DEFAULTS['template']
        )

        return fig

    def plot_rolling_metrics(self, window: int = 60) -> go.Figure:
        """
        Plot rolling Sharpe and drawdown.

        Args:
            window: Rolling window size

        Returns:
            Plotly Figure
        """
        # Calculate rolling metrics
        returns = self.result.equity_curve.pct_change()
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)

        # Rolling drawdown
        rolling_max = self.result.equity_curve.rolling(window, min_periods=1).max()
        rolling_dd = (self.result.equity_curve - rolling_max) / rolling_max * 100

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(f'Rolling Sharpe Ratio ({window}-day)', f'Rolling Max Drawdown ({window}-day)'),
            vertical_spacing=0.1
        )

        # Plot rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=COLORS['equity'], width=2),
                hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=1,
            col=1
        )

        # Plot rolling drawdown
        fig.add_trace(
            go.Scatter(
                x=rolling_dd.index,
                y=rolling_dd.values,
                mode='lines',
                name='Rolling Drawdown',
                fill='tozeroy',
                line=dict(color=COLORS['drawdown'], width=2),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2,
            col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.update_layout(
            title="Rolling Performance Metrics",
            height=600,
            template=LAYOUT_DEFAULTS['template'],
            hovermode='x unified',
            showlegend=False
        )

        return fig

    def plot_trade_analysis(self) -> go.Figure:
        """
        Plot trade statistics over time.

        Returns:
            Plotly Figure with subplots
        """
        if self.result.trades.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

        # Select PnL column: prefer net_realized_pnl if available
        pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in self.result.trades.columns else 'realized_pnl'

        # Calculate cumulative PnL
        trades_with_pnl = self.result.trades[self.result.trades[pnl_col] != 0].copy()
        trades_with_pnl['cumulative_pnl'] = trades_with_pnl[pnl_col].cumsum()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=('Cumulative PnL', 'Trade Size', 'Win Rate (Rolling)', 'Trade Frequency'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Cumulative PnL
        fig.add_trace(
            go.Scatter(
                x=trades_with_pnl['timestamp'],
                y=trades_with_pnl['cumulative_pnl'],
                mode='lines',
                name='Cumulative PnL',
                line=dict(color=COLORS['equity'], width=2),
                hovertemplate='Date: %{x}<br>PnL: $%{y:,.2f}<extra></extra>'
            ),
            row=1,
            col=1
        )

        # Trade size distribution
        if 'quantity' in self.result.trades.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.result.trades['timestamp'],
                    y=abs(self.result.trades['quantity']),
                    mode='markers',
                    name='Trade Size',
                    marker=dict(color=COLORS['neutral'], size=5),
                    hovertemplate='Date: %{x}<br>Size: %{y:.2f}<extra></extra>'
                ),
                row=1,
                col=2
            )

        # Rolling win rate
        if len(trades_with_pnl) >= 10:
            window = min(20, len(trades_with_pnl) // 2)
            trades_with_pnl['is_win'] = (trades_with_pnl[pnl_col] > 0).astype(int)
            rolling_win_rate = trades_with_pnl['is_win'].rolling(window).mean() * 100

            fig.add_trace(
                go.Scatter(
                    x=trades_with_pnl['timestamp'],
                    y=rolling_win_rate,
                    mode='lines',
                    name=f'Win Rate ({window}-trade)',
                    line=dict(color=COLORS['win'], width=2),
                    hovertemplate='Date: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>'
                ),
                row=2,
                col=1
            )

        # Trade frequency (trades per day)
        trade_counts = self.result.trades.groupby(self.result.trades['timestamp'].dt.date).size()
        fig.add_trace(
            go.Bar(
                x=trade_counts.index,
                y=trade_counts.values,
                name='Trades per Day',
                marker_color=COLORS['neutral'],
                hovertemplate='Date: %{x}<br>Trades: %{y}<extra></extra>'
            ),
            row=2,
            col=2
        )

        # Update layout
        fig.update_layout(
            title="Trade Analysis",
            height=600,
            template=LAYOUT_DEFAULTS['template'],
            showlegend=False
        )

        return fig

    def plot_position_exposure(self) -> go.Figure:
        """
        Plot position exposure over time.

        Returns:
            Plotly Figure with stacked area chart
        """
        if self.result.portfolio_history.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio history available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

        portfolio_df = self.result.portfolio_history

        # Extract value columns (val_<asset>)
        value_cols = [col for col in portfolio_df.columns if col.startswith('val_')]

        if not value_cols:
            fig = go.Figure()
            fig.add_annotation(
                text="No position value data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

        fig = go.Figure()

        # Add stacked area for each asset
        for val_col in value_cols:
            asset_name = val_col.replace('val_', '')
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['timestamp'],
                    y=portfolio_df[val_col].fillna(0),
                    mode='lines',
                    name=asset_name,
                    stackgroup='one',
                    hovertemplate=f'{asset_name}: $%{{y:,.2f}}<extra></extra>'
                )
            )

        # Add cash as separate area
        if 'cash' in portfolio_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['timestamp'],
                    y=portfolio_df['cash'],
                    mode='lines',
                    name='Cash',
                    stackgroup='one',
                    line=dict(color=COLORS['neutral']),
                    hovertemplate='Cash: $%{y:,.2f}<extra></extra>'
                )
            )

        # Update layout
        fig.update_layout(
            title="Position Exposure Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=LAYOUT_DEFAULTS['height'],
            template=LAYOUT_DEFAULTS['template'],
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create formatted summary table.

        Returns:
            DataFrame with performance summary
        """
        summary_data = {
            'Metric': [
                'Initial Capital',
                'Final Equity',
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                '',  # Separator
                'Sharpe Ratio',
                'Sortino Ratio',
                'Calmar Ratio',
                '',  # Separator
                'Max Drawdown',
                'Recovery Factor',
                '',  # Separator
                'Total Trades',
                'Winning Trades',
                'Losing Trades',
                'Win Rate',
                'Profit Factor',
                'Average Win',
                'Average Loss',
                'Expectancy'
            ],
            'Value': [
                _format_currency(self.result.initial_capital),
                _format_currency(self.result.final_equity),
                _format_percentage(self.metrics.metrics['total_return']),
                _format_percentage(self.metrics.metrics['annualized_return']),
                _format_percentage(self.metrics.metrics['annualized_volatility']),
                '',
                f"{self.metrics.metrics['sharpe_ratio']:.2f}",
                f"{self.metrics.metrics['sortino_ratio']:.2f}",
                f"{self.metrics.metrics['calmar_ratio']:.2f}",
                '',
                _format_percentage(self.metrics.metrics['max_drawdown']),
                f"{self.metrics.metrics['recovery_factor']:.2f}",
                '',
                f"{self.metrics.metrics['total_trades']:.0f}",
                f"{self.metrics.metrics['winning_trades']:.0f}",
                f"{self.metrics.metrics['losing_trades']:.0f}",
                _format_percentage(self.metrics.metrics['win_rate']),
                f"{self.metrics.metrics['profit_factor']:.2f}",
                _format_currency(self.metrics.metrics['avg_win']),
                _format_currency(self.metrics.metrics['avg_loss']),
                _format_currency(self.metrics.metrics['expectancy'])
            ]
        }

        df = pd.DataFrame(summary_data)
        return df

    def create_trade_summary(self) -> pd.DataFrame:
        """
        Create trade-level summary statistics.

        Returns:
            DataFrame with strategy-level breakdown
        """
        if self.result.trades.empty:
            return pd.DataFrame()

        # Select PnL column: prefer net_realized_pnl if available
        pnl_col = 'net_realized_pnl' if 'net_realized_pnl' in self.result.trades.columns else 'realized_pnl'

        # Group by strategy
        if 'strategy' in self.result.trades.columns:
            summary = self.result.trades.groupby('strategy').agg({
                pnl_col: ['count', 'sum', 'mean'],
                'quantity': 'sum'
            }).round(2)

            # Flatten column names
            summary.columns = ['Total Trades', 'Total PnL', 'Avg PnL', 'Total Volume']

            return summary
        else:
            # No strategy column, return overall stats
            summary = pd.DataFrame({
                'Total Trades': [len(self.result.trades)],
                'Total PnL': [self.result.trades[pnl_col].sum()],
                'Avg PnL': [self.result.trades[pnl_col].mean()],
                'Total Volume': [self.result.trades['quantity'].sum()]
            })

            return summary

    def save_report(
        self,
        output_path: Union[Path, str],
        include_all: bool = True
    ) -> Path:
        """
        Save report to HTML file.

        Args:
            output_path: Path to save HTML report
            include_all: Whether to include all charts

        Returns:
            Path to saved HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving report to {output_path}")

        # Generate figures if not already created
        if include_all and not self.figures:
            self.generate_full_report()

        # Create HTML content
        html_parts = []

        # Add title and summary
        html_parts.append(f"""
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <h2>Period: {self.result.start_date} to {self.result.end_date}</h2>
        """)

        # Add summary table
        summary_df = self.create_summary_table()
        html_parts.append("<h2>Performance Summary</h2>")
        html_parts.append(summary_df.to_html(index=False))

        # Add charts
        for name, fig in self.figures.items():
            html_parts.append(f"<h2>{name.replace('_', ' ').title()}</h2>")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        html_parts.append("</body></html>")

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_parts))

        self.logger.info(f"Report saved to {output_path}")

        return output_path

    def save_figures(
        self,
        output_dir: Path,
        format: str = 'png'
    ) -> List[Path]:
        """
        Save individual figures as images.

        Requires kaleido package for image export.

        Args:
            output_dir: Directory to save figures
            format: Image format ('png', 'jpg', 'svg')

        Returns:
            List of saved paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        # Generate figures if not already created
        if not self.figures:
            self.generate_full_report()

        for name, fig in self.figures.items():
            output_path = output_dir / f"{name}.{format}"
            try:
                fig.write_image(str(output_path))
                saved_paths.append(output_path)
                self.logger.debug(f"Saved figure: {output_path}")
            except (ValueError, ImportError) as e:
                self.logger.error(
                    f"Failed to save figure '{name}': {e}. "
                    "Please install kaleido: pip install kaleido"
                )

        if saved_paths:
            self.logger.info(f"Saved {len(saved_paths)} figures to {output_dir}")
        else:
            self.logger.warning("No figures were saved. Kaleido may not be installed.")

        return saved_paths

    def export_to_csv(self, output_dir: Union[Path, str]) -> Dict[str, Path]:
        """
        Export data to CSV files.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping filename to path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Export equity curve
        equity_path = output_dir / 'equity_curve.csv'
        self.result.equity_curve.to_csv(equity_path)
        saved_files['equity_curve'] = equity_path

        # Export trades
        if not self.result.trades.empty:
            trades_path = output_dir / 'trades.csv'
            self.result.trades.to_csv(trades_path, index=False)
            saved_files['trades'] = trades_path

        # Export metrics
        metrics_path = output_dir / 'metrics.csv'
        metrics_df = pd.DataFrame([self.metrics.metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv(metrics_path)
        saved_files['metrics'] = metrics_path

        # Export portfolio history
        if not self.result.portfolio_history.empty:
            portfolio_path = output_dir / 'portfolio_history.csv'
            self.result.portfolio_history.to_csv(portfolio_path, index=False)
            saved_files['portfolio'] = portfolio_path

        self.logger.info(f"Exported {len(saved_files)} CSV files to {output_dir}")

        return saved_files


# Helper functions

def _format_currency(value: float) -> str:
    """
    Format value as currency string.

    Args:
        value: Numeric value

    Returns:
        Formatted string (e.g., "$1,234.56")
    """
    return f"${value:,.2f}"


def _format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.

    Args:
        value: Numeric value (e.g., 0.1234 for 12.34%)
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "12.34%")
    """
    return f"{value * 100:.{decimals}f}%"


def _get_chart_theme() -> str:
    """
    Get Plotly chart theme from config.

    Returns:
        Theme name
    """
    try:
        config = get_config()
        theme = config.get('dashboard', {}).get('charts', {}).get('theme', 'plotly_white')
        return theme
    except Exception:
        # Fallback to default if config fails
        return 'plotly_white'


# Apply config-driven theme to layout defaults
LAYOUT_DEFAULTS['template'] = _get_chart_theme()


def _add_watermark(fig: go.Figure) -> go.Figure:
    """
    Add watermark to figure.

    Args:
        fig: Plotly Figure

    Returns:
        Modified figure
    """
    fig.add_annotation(
        text="Generated by Energy Trading System",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.1,
        showarrow=False,
        font=dict(size=10, color="gray"),
        opacity=0.5
    )

    return fig
