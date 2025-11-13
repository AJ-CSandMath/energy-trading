"""
Portfolio Optimization Page

Modern portfolio theory and renewable-aware optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from src.dashboard.utils import (
    load_optimization_results,
    create_optimization_controls,
    apply_dashboard_theme,
    format_percentage,
    handle_error
)

st.title("⚖️ Portfolio Optimization")
st.markdown("Modern portfolio theory and renewable-aware optimization")

try:
    # Optimization Method Selection
    st.subheader("Optimization Method")

    method = st.radio(
        "Select Method",
        options=['Mean-Variance (Markowitz)', 'Risk Parity', 'Black-Litterman', 'Minimum CVaR', 'Compare All Methods'],
        index=0
    )

    # Map display names to method names
    method_map = {
        'Mean-Variance (Markowitz)': 'mean_variance',
        'Risk Parity': 'risk_parity',
        'Black-Litterman': 'black_litterman',
        'Minimum CVaR': 'min_cvar'
    }

    if method != 'Compare All Methods':
        # Objective Selector (if Mean-Variance)
        objective = None
        if method == 'Mean-Variance (Markowitz)':
            objective = st.radio(
                "Objective",
                options=['Max Sharpe', 'Min Risk', 'Max Return'],
                index=0
            )

        # Optimization Constraints
        st.markdown("---")
        st.subheader("Optimization Constraints")

        with st.expander("⚙️ Configure Constraints"):
            constraints = create_optimization_controls()

            # Add objective to constraints for Mean-Variance
            if objective is not None:
                # Map display name to optimizer parameter
                objective_map = {
                    'Max Sharpe': 'max_sharpe',
                    'Min Risk': 'min_risk',
                    'Max Return': 'max_return'
                }
                constraints['objective'] = objective_map[objective]

            # Renewable Constraints
            st.write("**Renewable Constraints**")
            enable_renewable = st.checkbox("Enable Renewable Constraints")

            if enable_renewable:
                col1, col2 = st.columns(2)
                with col1:
                    wind_cf_min = st.slider("Wind CF Min", 0.0, 1.0, 0.2)
                    solar_cf_min = st.slider("Solar CF Min", 0.0, 1.0, 0.1)
                with col2:
                    wind_cf_max = st.slider("Wind CF Max", 0.0, 1.0, 0.5)
                    solar_cf_max = st.slider("Solar CF Max", 0.0, 1.0, 0.4)

                grid_limit = st.number_input("Grid Injection Limit (MW)", value=1000.0)
                max_curtailment = st.slider("Max Curtailment (%)", 0.0, 50.0, 20.0) / 100

                # Add renewable constraints to constraints dict
                constraints['renewable_enabled'] = True
                constraints['wind_cf_min'] = wind_cf_min
                constraints['wind_cf_max'] = wind_cf_max
                constraints['solar_cf_min'] = solar_cf_min
                constraints['solar_cf_max'] = solar_cf_max
                constraints['grid_limit'] = grid_limit
                constraints['max_curtailment'] = max_curtailment
            else:
                constraints['renewable_enabled'] = False

        # Black-Litterman Views (only for Black-Litterman method)
        if method == 'Black-Litterman':
            st.markdown("---")
            with st.expander("💡 Black-Litterman Market Views"):
                st.write("**Specify your market views on expected returns**")
                st.info("Add subjective views on asset returns to combine with market equilibrium")

                # Number of views
                num_views = st.number_input("Number of Views", min_value=0, max_value=10, value=2)

                if num_views > 0:
                    views_data = []

                    for i in range(num_views):
                        st.write(f"**View {i+1}**")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            asset = st.selectbox(
                                f"Asset (View {i+1})",
                                options=['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4'],
                                key=f"bl_asset_{i}"
                            )

                        with col2:
                            view_type = st.selectbox(
                                f"View Type",
                                options=['Absolute', 'Relative'],
                                key=f"bl_type_{i}"
                            )

                        with col3:
                            expected_return = st.number_input(
                                f"Expected Return (%)",
                                min_value=-50.0,
                                max_value=100.0,
                                value=5.0,
                                step=0.5,
                                key=f"bl_return_{i}"
                            )

                        confidence = st.slider(
                            f"View Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            key=f"bl_conf_{i}"
                        )

                        views_data.append({
                            'asset': asset,
                            'type': view_type,
                            'return': expected_return / 100,
                            'confidence': confidence
                        })

                    constraints['bl_views'] = views_data
                    st.success(f"Added {num_views} market views")
                else:
                    constraints['bl_views'] = []

        # Multi-Period Optimization
        st.markdown("---")
        with st.expander("📅 Multi-Period Optimization"):
            st.write("**Optimize portfolio allocation over multiple time periods**")

            enable_multiperiod = st.checkbox("Enable Multi-Period Optimization")

            if enable_multiperiod:
                col1, col2 = st.columns(2)

                with col1:
                    num_periods = st.slider(
                        "Number of Periods",
                        min_value=2,
                        max_value=12,
                        value=4,
                        help="Number of rebalancing periods (e.g., quarters)"
                    )

                    rebalancing_cost = st.number_input(
                        "Rebalancing Cost (%)",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.1,
                        step=0.05,
                        help="Transaction cost per rebalancing"
                    )

                with col2:
                    period_length = st.selectbox(
                        "Period Length",
                        options=['Daily', 'Weekly', 'Monthly', 'Quarterly'],
                        index=3
                    )

                    discount_rate = st.number_input(
                        "Discount Rate (%/year)",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5,
                        help="Discount rate for future returns"
                    )

                # Forecast uncertainty
                st.write("**Forecast Uncertainty**")
                uncertainty_growth = st.slider(
                    "Uncertainty Growth Rate",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    help="How much forecast uncertainty increases per period"
                )

                # Add to constraints
                constraints['multiperiod_enabled'] = True
                constraints['num_periods'] = num_periods
                constraints['rebalancing_cost'] = rebalancing_cost / 100
                constraints['period_length'] = period_length.lower()
                constraints['discount_rate'] = discount_rate / 100
                constraints['uncertainty_growth'] = uncertainty_growth

                st.success(f"Multi-period optimization enabled for {num_periods} {period_length.lower()} periods")
            else:
                constraints['multiperiod_enabled'] = False

        # Run Optimization
        st.markdown("---")
        if st.button("⚖️ Run Optimization"):
            with st.spinner("Optimizing portfolio..."):
                result = load_optimization_results(
                    method_map[method],
                    constraints
                )

                if result is not None:
                    st.session_state.opt_result = result
                    st.success("Optimization completed successfully!")

        # Display Results
        if 'opt_result' in st.session_state:
            result = st.session_state.opt_result

            st.markdown("---")
            st.subheader("Optimal Portfolio")

            # Top Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Expected Return",
                    value=format_percentage(result.expected_return)
                )
            with col2:
                st.metric(
                    label="Expected Risk",
                    value=format_percentage(result.expected_risk)
                )
            with col3:
                st.metric(
                    label="Sharpe Ratio",
                    value=f"{result.sharpe_ratio:.2f}"
                )
            with col4:
                method_display = result.method.replace('_', ' ').title() if hasattr(result, 'method') else method_map.get(method, 'Unknown')
                st.metric(
                    label="Method",
                    value=method_display
                )

            # Optimal Weights Visualization
            st.markdown("---")
            st.write("**Portfolio Allocation**")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Pie Chart
                # Ensure weights is a Series
                weights = pd.Series(result.weights) if not isinstance(result.weights, pd.Series) else result.weights

                fig = go.Figure(data=[go.Pie(
                    labels=weights.index,
                    values=weights.values,
                    hole=0.3
                )])
                fig.update_layout(height=400)
                fig = apply_dashboard_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Weights Table
                weights = pd.Series(result.weights) if not isinstance(result.weights, pd.Series) else result.weights

                weights_df = pd.DataFrame({
                    'Asset': weights.index,
                    'Weight (%)': weights.values * 100,
                    'Dollar Amount': weights.values * 1000000  # Assuming $1M portfolio
                })
                st.dataframe(weights_df, use_container_width=True)

            # Efficient Frontier (if applicable)
            if method == 'Mean-Variance (Markowitz)' and hasattr(result, 'efficient_frontier') and result.efficient_frontier is not None:
                st.markdown("---")
                st.subheader("Efficient Frontier")

                fig = go.Figure()

                # Efficient Frontier curve
                ef_data = result.efficient_frontier
                fig.add_trace(go.Scatter(
                    x=ef_data['risk'],
                    y=ef_data['return'],
                    mode='lines+markers',
                    name='Efficient Frontier',
                    line=dict(color='#1f77b4')
                ))

                # Optimal Portfolio
                fig.add_trace(go.Scatter(
                    x=[result.expected_risk],
                    y=[result.expected_return],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(size=15, color='#d62728', symbol='star')
                ))

                fig.update_xaxes(title_text="Risk (Volatility)")
                fig.update_yaxes(title_text="Expected Return")
                fig.update_layout(height=500)
                fig = apply_dashboard_theme(fig)

                st.plotly_chart(fig, use_container_width=True)

            # Export Results
            st.markdown("---")
            weights_export = pd.Series(result.weights) if not isinstance(result.weights, pd.Series) else result.weights
            csv = weights_export.to_csv()
            st.download_button(
                label="📥 Download Optimal Weights",
                data=csv,
                file_name=f"optimal_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        else:
            st.info("👆 Configure constraints and click 'Run Optimization' to see results")

    else:
        # Compare All Methods
        st.markdown("---")
        if st.button("⚖️ Compare All Methods"):
            with st.spinner("Running optimizations..."):
                results = {}
                for display_name, method_name in method_map.items():
                    result = load_optimization_results(method_name, {})
                    if result is not None:
                        results[display_name] = result

                if results:
                    st.session_state.comparison_results = results
                    st.success("Comparison completed successfully!")

        if 'comparison_results' in st.session_state:
            results = st.session_state.comparison_results

            st.markdown("---")
            st.subheader("Method Comparison")

            # Comparison Table
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Method': name,
                    'Expected Return': format_percentage(result.expected_return),
                    'Expected Risk': format_percentage(result.expected_risk),
                    'Sharpe Ratio': f"{result.sharpe_ratio:.2f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Risk-Return Comparison with Efficient Frontiers
            st.markdown("---")
            st.write("**Risk-Return Comparison**")

            fig = go.Figure()

            # Add efficient frontiers if available
            for name, result in results.items():
                if hasattr(result, 'efficient_frontier') and result.efficient_frontier is not None:
                    ef_data = result.efficient_frontier
                    fig.add_trace(go.Scatter(
                        x=ef_data['risk'],
                        y=ef_data['return'],
                        mode='lines',
                        name=f'{name} - Frontier',
                        line=dict(width=2, dash='dot'),
                        showlegend=True
                    ))

            # Add optimal portfolio points
            for name, result in results.items():
                fig.add_trace(go.Scatter(
                    x=[result.expected_risk],
                    y=[result.expected_return],
                    mode='markers+text',
                    name=name,
                    text=[name.replace(' (Markowitz)', '')],
                    textposition='top center',
                    marker=dict(size=12, symbol='star')
                ))

            fig.update_xaxes(title_text="Risk (Volatility)")
            fig.update_yaxes(title_text="Expected Return")
            fig.update_layout(height=500, hovermode='closest')
            fig = apply_dashboard_theme(fig)

            st.plotly_chart(fig, use_container_width=True)

            # Weights Comparison Bar Chart
            st.markdown("---")
            st.write("**Portfolio Weights Comparison**")

            # Collect all unique assets
            all_assets = set()
            for result in results.values():
                weights = pd.Series(result.weights) if not isinstance(result.weights, pd.Series) else result.weights
                all_assets.update(weights.index.tolist())
            all_assets = sorted(list(all_assets))

            # Create grouped bar chart
            fig = go.Figure()

            for name, result in results.items():
                weights = pd.Series(result.weights) if not isinstance(result.weights, pd.Series) else result.weights
                weights_values = []
                for asset in all_assets:
                    if asset in weights.index:
                        weights_values.append(weights[asset] * 100)
                    else:
                        weights_values.append(0)

                fig.add_trace(go.Bar(
                    name=name.replace(' (Markowitz)', ''),
                    x=all_assets,
                    y=weights_values,
                    text=[f"{w:.1f}%" for w in weights_values],
                    textposition='auto'
                ))

            fig.update_xaxes(title_text="Asset")
            fig.update_yaxes(title_text="Weight (%)")
            fig.update_layout(
                barmode='group',
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig = apply_dashboard_theme(fig)

            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    handle_error(e, "Portfolio Optimization Page")
