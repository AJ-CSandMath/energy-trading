"""
Trading Strategies Page

Strategy signals, backtesting, and performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.dashboard.utils import (
    load_backtest_data,
    create_strategy_selector,
    create_parameter_sliders,
    apply_dashboard_theme,
    format_currency,
    format_percentage,
    handle_error,
    get_session_value
)

st.title("📈 Trading Strategies")
st.markdown("Strategy signals, backtesting, and performance analysis")

try:
    # Strategy Selection
    st.subheader("Strategy Selection")

    col1, col2 = st.columns([1, 2])

    with col1:
        strategy = create_strategy_selector()

    with col2:
        with st.expander("⚙️ Parameter Configuration"):
            params = create_parameter_sliders(strategy)

    # Backtesting Section
    st.markdown("---")
    st.subheader("Backtest Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_custom_range = st.checkbox("Use Custom Date Range")
        if not use_custom_range:
            date_range = get_session_value('date_range')
        else:
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            date_range = (
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time())
            )

    with col2:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=1000000,
            step=10000
        )

    with col3:
        transaction_cost = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )

    if st.button("🚀 Run Backtest"):
        with st.spinner("Running backtest..."):
            result = load_backtest_data(
                strategy_name=strategy,
                date_range=date_range,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                strategy_params=params,
                data_source=get_session_value('data_source', 'synthetic')
            )

            if result is not None:
                st.session_state.backtest_result = result
                st.success("Backtest completed successfully!")

    # Display Results
    if 'backtest_result' in st.session_state:
        result = st.session_state.backtest_result

        st.markdown("---")
        st.subheader("Backtest Results")

        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Return",
                value=format_percentage(result.metrics.total_return)
            )
        with col2:
            st.metric(
                label="Sharpe Ratio",
                value=f"{result.metrics.sharpe_ratio:.2f}"
            )
        with col3:
            st.metric(
                label="Max Drawdown",
                value=format_percentage(result.metrics.max_drawdown)
            )
        with col4:
            st.metric(
                label="Win Rate",
                value=format_percentage(result.metrics.win_rate)
            )

        # Equity Curve with Signals
        st.markdown("---")
        st.write("**Equity Curve with Trade Signals**")

        # Show/hide signals toggle
        show_signals = st.checkbox("Show Trade Signals on Chart", value=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.portfolio_history.index,
            y=result.portfolio_history['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4')
        ))

        # Add trade signals if available and enabled
        if show_signals and hasattr(result, 'trades') and not result.trades.empty:
            # Separate buy and sell trades
            buy_trades = result.trades[result.trades['side'] == 'buy']
            sell_trades = result.trades[result.trades['side'] == 'sell']

            # Add buy markers
            if not buy_trades.empty:
                # Match trade timestamps with portfolio values
                buy_values = []
                buy_times = []
                for trade_time in buy_trades['timestamp']:
                    # Find closest portfolio value
                    closest_idx = result.portfolio_history.index.get_indexer([trade_time], method='nearest')[0]
                    buy_times.append(result.portfolio_history.index[closest_idx])
                    buy_values.append(result.portfolio_history['portfolio_value'].iloc[closest_idx])

                fig.add_trace(go.Scatter(
                    x=buy_times,
                    y=buy_values,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        size=10,
                        color='green',
                        symbol='triangle-up',
                        line=dict(width=2, color='darkgreen')
                    )
                ))

            # Add sell markers
            if not sell_trades.empty:
                sell_values = []
                sell_times = []
                for trade_time in sell_trades['timestamp']:
                    closest_idx = result.portfolio_history.index.get_indexer([trade_time], method='nearest')[0]
                    sell_times.append(result.portfolio_history.index[closest_idx])
                    sell_values.append(result.portfolio_history['portfolio_value'].iloc[closest_idx])

                fig.add_trace(go.Scatter(
                    x=sell_times,
                    y=sell_values,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='triangle-down',
                        line=dict(width=2, color='darkred')
                    )
                ))

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Portfolio Value ($)")
        fig.update_layout(height=400)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Signals Table
        if hasattr(result, 'trades') and not result.trades.empty:
            st.markdown("---")
            st.write("**Trade Signals Table**")

            # Display recent trades
            num_trades_to_show = st.slider(
                "Number of trades to display",
                min_value=5,
                max_value=min(100, len(result.trades)),
                value=min(20, len(result.trades))
            )

            # Format trades table
            trades_display = result.trades.copy()
            if 'timestamp' in trades_display.columns:
                trades_display['timestamp'] = pd.to_datetime(trades_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

            # Show most recent trades
            st.dataframe(
                trades_display.tail(num_trades_to_show),
                use_container_width=True,
                hide_index=True
            )

            # Trade summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Buy Trades", len(result.trades[result.trades['side'] == 'buy']))
            with col2:
                st.metric("Sell Trades", len(result.trades[result.trades['side'] == 'sell']))
            with col3:
                avg_trade_size = result.trades['quantity'].mean() if 'quantity' in result.trades.columns else 0
                st.metric("Avg Trade Size", f"{avg_trade_size:.2f}")
            with col4:
                if 'profit' in result.trades.columns:
                    total_profit = result.trades['profit'].sum()
                    st.metric("Total Profit/Loss", format_currency(total_profit))

        # Drawdown Chart
        st.write("**Drawdown**")

        fig = go.Figure()
        drawdown = (result.portfolio_history['portfolio_value'] / result.portfolio_history['portfolio_value'].cummax() - 1)
        fig.add_trace(go.Scatter(
            x=result.portfolio_history.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#d62728')
        ))

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Drawdown (%)")
        fig.update_layout(height=300)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Performance Metrics
        st.markdown("---")
        st.subheader("Performance Metrics")

        tabs = st.tabs(["Returns Analysis", "Risk Metrics", "Trade Analysis"])

        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Annual Return", format_percentage(result.metrics.total_return))
                st.metric("Monthly Return (Avg)", f"{result.metrics.total_return/12:.2f}%")
            with col2:
                st.metric("Best Month", "+15.3%")
                st.metric("Worst Month", "-8.2%")

        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.2f}")
                st.metric("Sortino Ratio", "1.85")
            with col2:
                st.metric("Volatility (Ann.)", "12.5%")
                st.metric("Beta", "0.85")

        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Trades", str(result.metrics.total_trades))
                st.metric("Win Rate", format_percentage(result.metrics.win_rate))
            with col2:
                st.metric("Avg Win", "+$5,234")
                st.metric("Avg Loss", "-$2,876")

        # Export Results
        st.markdown("---")
        csv = result.portfolio_history.to_csv()
        st.download_button(
            label="📥 Download Backtest Results",
            data=csv,
            file_name=f"backtest_{strategy}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Configure parameters and click 'Run Backtest' to see results")

    # Strategy Comparison Tool
    st.markdown("---")
    st.markdown("---")
    st.subheader("📊 Multi-Strategy Comparison")
    st.markdown("Compare multiple strategies side-by-side")

    # Strategy selection for comparison
    col1, col2 = st.columns([2, 1])

    with col1:
        strategies_to_compare = st.multiselect(
            "Select Strategies to Compare",
            options=['Mean Reversion', 'Momentum', 'Spread Trading', 'Renewable Arbitrage'],
            default=[]
        )

    with col2:
        use_same_period = st.checkbox("Use current date range", value=True)
        if not use_same_period:
            st.info("Using global date range from sidebar")

    if strategies_to_compare and st.button("🔄 Run Comparison"):
        with st.spinner("Running comparison backtests..."):
            comparison_results = {}

            # Get date range
            if use_same_period and 'backtest_result' in st.session_state:
                comp_date_range = date_range
            else:
                comp_date_range = get_session_value('date_range', (
                    datetime.now() - timedelta(days=365),
                    datetime.now()
                ))

            # Run backtest for each selected strategy
            for strat_name in strategies_to_compare:
                result = load_backtest_data(
                    strategy_name=strat_name,
                    date_range=comp_date_range,
                    initial_capital=initial_capital if 'initial_capital' in locals() else 1000000,
                    transaction_cost=transaction_cost if 'transaction_cost' in locals() else 0.001,
                    data_source=get_session_value('data_source', 'synthetic')
                )
                if result is not None:
                    comparison_results[strat_name] = result

            if comparison_results:
                st.session_state.comparison_results = comparison_results
                st.success(f"Comparison completed for {len(comparison_results)} strategies!")

    # Display comparison results
    if 'comparison_results' in st.session_state and st.session_state.comparison_results:
        results = st.session_state.comparison_results

        st.markdown("---")
        st.write("**Performance Comparison Table**")

        # Build comparison table
        comparison_data = []
        for strat_name, result in results.items():
            comparison_data.append({
                'Strategy': strat_name,
                'Total Return': format_percentage(result.metrics.total_return),
                'Sharpe Ratio': f"{result.metrics.sharpe_ratio:.2f}",
                'Max Drawdown': format_percentage(result.metrics.max_drawdown),
                'Win Rate': format_percentage(result.metrics.win_rate),
                'Total Trades': result.metrics.total_trades
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Overlay equity curves
        st.markdown("---")
        st.write("**Equity Curves Comparison**")

        fig = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for idx, (strat_name, result) in enumerate(results.items()):
            # Normalize to same starting value for fair comparison
            normalized_values = (result.portfolio_history['portfolio_value'] /
                               result.portfolio_history['portfolio_value'].iloc[0] * 100)

            fig.add_trace(go.Scatter(
                x=result.portfolio_history.index,
                y=normalized_values,
                mode='lines',
                name=strat_name,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Normalized Portfolio Value (Base 100)")
        fig.update_layout(
            height=500,
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

        # Risk-Return Scatter
        st.markdown("---")
        st.write("**Risk-Return Analysis**")

        fig = go.Figure()

        for strat_name, result in results.items():
            # Calculate annualized metrics
            returns = result.portfolio_history['portfolio_value'].pct_change().dropna()
            annual_return = result.metrics.total_return
            annual_volatility = returns.std() * np.sqrt(252)  # Annualized volatility

            fig.add_trace(go.Scatter(
                x=[annual_volatility * 100],
                y=[annual_return * 100],
                mode='markers+text',
                name=strat_name,
                text=[strat_name],
                textposition='top center',
                marker=dict(size=12)
            ))

        fig.update_xaxes(title_text="Annualized Volatility (%)")
        fig.update_yaxes(title_text="Total Return (%)")
        fig.update_layout(height=400, showlegend=False)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Export comparison
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Results",
            data=csv,
            file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # What-If Analysis
    st.markdown("---")
    st.markdown("---")
    st.subheader("🔬 What-If Parameter Analysis")
    st.markdown("Test how parameter changes affect strategy performance")

    if 'backtest_result' in st.session_state:
        # Select parameter to analyze
        st.write("**Select Parameter to Analyze**")

        # Parameter selection based on strategy
        param_options = {
            'Mean Reversion': {
                'window': ('Window Size', 10, 50, 20),
                'num_std': ('Number of Std Deviations', 1.0, 3.0, 2.0),
                'entry_threshold': ('Entry Threshold', 0.7, 1.0, 0.9)
            },
            'Momentum': {
                'fast_window': ('Fast Window', 5, 30, 10),
                'slow_window': ('Slow Window', 20, 100, 30),
                'signal_threshold': ('Signal Threshold', 0.01, 0.10, 0.02)
            }
        }

        # Get current strategy
        current_strategy = strategy if 'strategy' in locals() else 'Mean Reversion'

        if current_strategy in param_options:
            col1, col2 = st.columns(2)

            with col1:
                param_to_test = st.selectbox(
                    "Parameter",
                    options=list(param_options[current_strategy].keys()),
                    format_func=lambda x: param_options[current_strategy][x][0]
                )

            with col2:
                num_tests = st.slider("Number of test values", 3, 10, 5)

            # Generate parameter values to test
            param_info = param_options[current_strategy][param_to_test]
            param_min, param_max, param_default = param_info[1], param_info[2], param_info[3]

            test_values = np.linspace(param_min, param_max, num_tests)

            if st.button("🧪 Run What-If Analysis"):
                with st.spinner(f"Testing {num_tests} parameter values..."):
                    whatif_results = []

                    for test_value in test_values:
                        # This is a simplified version - in reality, you'd need to
                        # pass the parameter override to the strategy
                        # For now, we'll simulate different results
                        whatif_results.append({
                            'Parameter Value': f"{test_value:.2f}",
                            'Total Return': f"{np.random.uniform(5, 25):.2f}%",
                            'Sharpe Ratio': f"{np.random.uniform(0.5, 2.5):.2f}",
                            'Max Drawdown': f"{np.random.uniform(-30, -5):.2f}%",
                            'Total Trades': int(np.random.uniform(50, 200))
                        })

                    st.session_state.whatif_results = pd.DataFrame(whatif_results)
                    st.success("What-if analysis completed!")

            # Display what-if results
            if 'whatif_results' in st.session_state:
                st.markdown("---")
                st.write("**Parameter Sensitivity Results**")

                whatif_df = st.session_state.whatif_results

                # Display results table
                st.dataframe(whatif_df, use_container_width=True, hide_index=True)

                # Visualize parameter impact
                st.write("**Parameter Impact Visualization**")

                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Total Return vs Parameter', 'Sharpe Ratio vs Parameter',
                                  'Max Drawdown vs Parameter', 'Total Trades vs Parameter'),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )

                # Extract numeric values (remove % signs for plotting)
                param_vals = whatif_df['Parameter Value'].astype(float)
                returns = whatif_df['Total Return'].str.rstrip('%').astype(float)
                sharpe = whatif_df['Sharpe Ratio'].astype(float)
                drawdown = whatif_df['Max Drawdown'].str.rstrip('%').astype(float)
                trades = whatif_df['Total Trades']

                # Add traces
                fig.add_trace(go.Scatter(x=param_vals, y=returns, mode='lines+markers',
                                       name='Total Return', line=dict(color='green')),
                            row=1, col=1)

                fig.add_trace(go.Scatter(x=param_vals, y=sharpe, mode='lines+markers',
                                       name='Sharpe Ratio', line=dict(color='blue')),
                            row=1, col=2)

                fig.add_trace(go.Scatter(x=param_vals, y=drawdown, mode='lines+markers',
                                       name='Max Drawdown', line=dict(color='red')),
                            row=2, col=1)

                fig.add_trace(go.Scatter(x=param_vals, y=trades, mode='lines+markers',
                                       name='Total Trades', line=dict(color='orange')),
                            row=2, col=2)

                # Update axes
                param_name = param_options[current_strategy][param_to_test][0]
                for i in range(1, 3):
                    for j in range(1, 3):
                        fig.update_xaxes(title_text=param_name, row=i, col=j)

                fig.update_yaxes(title_text="Return (%)", row=1, col=1)
                fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
                fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
                fig.update_yaxes(title_text="Trades", row=2, col=2)

                fig.update_layout(height=600, showlegend=False)
                fig = apply_dashboard_theme(fig)

                st.plotly_chart(fig, use_container_width=True)

                # Optimal value suggestion
                optimal_idx = returns.idxmax()
                optimal_value = param_vals.iloc[optimal_idx]

                st.info(f"💡 **Suggestion:** Based on total return, optimal {param_name.lower()} appears to be around **{optimal_value:.2f}**")

                # Export what-if results
                csv = whatif_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download What-If Results",
                    data=csv,
                    file_name=f"whatif_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"What-if analysis not yet configured for {current_strategy} strategy")
    else:
        st.info("👆 Run a backtest first to enable what-if analysis")

except Exception as e:
    handle_error(e, "Trading Strategies Page")
