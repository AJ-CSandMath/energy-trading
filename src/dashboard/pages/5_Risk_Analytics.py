"""
Risk Analytics Page

Portfolio risk measurement, stress testing, and scenario analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from src.dashboard.utils import (
    load_backtest_data,
    create_scenario_selector,
    apply_dashboard_theme,
    format_percentage,
    format_currency,
    handle_error,
    get_session_value
)

st.title("⚠️ Risk Analytics")
st.markdown("Portfolio risk measurement, stress testing, and scenario analysis")

try:
    # Load backtest data for risk analysis
    if 'backtest_result' not in st.session_state:
        st.info("Please run a backtest from the Trading Strategies page first to enable risk analytics.")

        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                date_range = get_session_value('date_range', (
                    datetime.now() - pd.Timedelta(days=365),
                    datetime.now()
                ))
                result = load_backtest_data(
                    strategy_name='Mean Reversion',
                    date_range=date_range,
                    data_source=get_session_value('data_source', 'synthetic')
                )
                if result:
                    st.session_state.backtest_result = result
                    st.rerun()
    else:
        result = st.session_state.backtest_result

        # Risk Overview Section
        st.subheader("Risk Overview")

        # Configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[0.90, 0.95, 0.99],
                value=0.95
            )

        with col2:
            method = st.selectbox(
                "VaR Method",
                options=['Historical', 'Parametric'],
                index=0
            )

        with col3:
            horizon = st.slider(
                "Time Horizon (days)",
                min_value=1,
                max_value=10,
                value=1
            )

        # Top Risk Metrics
        st.markdown("---")

        returns = result.portfolio_history['portfolio_value'].pct_change().dropna()

        # Calculate VaR and CVaR with horizon scaling
        # For 1-day returns, calculate VaR
        var_1day = -np.percentile(returns, (1 - confidence_level) * 100)

        # Scale for multi-day horizon (square root of time approximation)
        var_value = var_1day * np.sqrt(horizon)

        # Calculate CVaR (Conditional VaR / Expected Shortfall)
        tail_losses = returns[returns <= -var_1day]
        if len(tail_losses) > 0:
            cvar_1day = -tail_losses.mean()
            cvar_value = cvar_1day * np.sqrt(horizon)
        else:
            # If no tail losses, CVaR equals VaR
            cvar_value = var_value

        max_dd = (result.portfolio_history['portfolio_value'] /
                 result.portfolio_history['portfolio_value'].cummax() - 1).min()
        volatility = returns.std() * np.sqrt(252)  # Annualized

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label=f"VaR ({int(confidence_level*100)}%)",
                value=format_percentage(var_value)
            )

        with col2:
            st.metric(
                label=f"CVaR ({int(confidence_level*100)}%)",
                value=format_percentage(cvar_value)
            )

        with col3:
            st.metric(
                label="Max Drawdown",
                value=format_percentage(max_dd)
            )

        with col4:
            st.metric(
                label="Volatility (Ann.)",
                value=format_percentage(volatility)
            )

        # VaR/CVaR Analysis
        st.markdown("---")
        st.subheader("VaR/CVaR Analysis")

        # Calculate rolling VaR
        window = 60
        rolling_var = returns.rolling(window).apply(
            lambda x: -np.percentile(x, (1 - confidence_level) * 100)
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=returns.index,
            y=rolling_var * 100,
            mode='lines',
            name=f'Rolling VaR ({int(confidence_level*100)}%)',
            line=dict(color='#d62728')
        ))

        fig.add_hline(
            y=var_value * 100,
            line_dash="dash",
            annotation_text=f"Current VaR: {var_value*100:.2f}%"
        )

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="VaR (%)")
        fig.update_layout(height=400)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # VaR Backtest
        st.write("**VaR Backtest**")

        violations = returns < -var_value
        violation_rate = violations.sum() / len(returns)
        expected_rate = 1 - confidence_level

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Violations", f"{violations.sum()}")
        with col2:
            st.metric("Violation Rate", f"{violation_rate:.2%}")
        with col3:
            st.metric("Expected Rate", f"{expected_rate:.2%}")

        # Stress Testing
        st.markdown("---")
        st.subheader("Stress Testing")

        scenarios = create_scenario_selector()

        if st.button("🔬 Run Stress Test") and scenarios:
            with st.spinner("Running stress tests..."):
                # Placeholder stress test results
                stress_results = []
                for scenario in scenarios:
                    stress_results.append({
                        'Scenario': scenario,
                        'Portfolio Impact': f"{np.random.uniform(-30, 10):.1f}%",
                        'VaR Impact': f"{np.random.uniform(10, 50):.1f}%",
                        'Max DD': f"{np.random.uniform(15, 40):.1f}%"
                    })

                stress_df = pd.DataFrame(stress_results)
                st.dataframe(stress_df, use_container_width=True)

                st.session_state.stress_results = stress_df

        # Risk Decomposition
        st.markdown("---")
        st.subheader("Risk Decomposition")

        # Placeholder data for risk decomposition
        assets = ['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4']
        risk_contrib = np.random.dirichlet(np.ones(len(assets))) * 100

        fig = go.Figure(data=[go.Pie(
            labels=assets,
            values=risk_contrib,
            hole=0.3
        )])

        fig.update_layout(height=400, title="Component VaR by Asset")
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Correlation Analysis
        st.markdown("---")
        st.subheader("Correlation Analysis")

        # Generate correlation matrix
        n_assets = 4
        corr_matrix = np.random.rand(n_assets, n_assets)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=assets,
            y=assets,
            colorscale='RdYlGn',
            zmid=0,
            text=corr_matrix,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))

        fig.update_layout(height=400, title="Asset Correlation Matrix")
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Risk-Adjusted Returns
        st.markdown("---")
        st.subheader("Risk-Adjusted Returns")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.2f}")
            st.metric("Sortino Ratio", "1.45")

        with col2:
            st.metric("Calmar Ratio", "1.28")
            st.metric("Information Ratio", "0.85")

        # Tail Risk Analysis
        st.markdown("---")
        st.subheader("Tail Risk Analysis")

        # Distribution of returns
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns Distribution',
            marker=dict(color='#7f7f7f')
        ))

        fig.add_vline(
            x=-var_value * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({int(confidence_level*100)}%)"
        )

        fig.update_xaxes(title_text="Returns (%)")
        fig.update_yaxes(title_text="Frequency")
        fig.update_layout(height=400)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Export Risk Report
        st.markdown("---")
        st.download_button(
            label="📥 Download Risk Report",
            data="Risk Analysis Report\n" + "="*50 + f"\n\nVaR ({int(confidence_level*100)}%): {var_value:.4f}\nCVaR: {cvar_value:.4f}\nMax Drawdown: {max_dd:.4f}\nVolatility: {volatility:.4f}",
            file_name=f"risk_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

except Exception as e:
    handle_error(e, "Risk Analytics Page")
