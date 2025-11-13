"""
Market Overview Page

Real-time and historical energy market data visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.dashboard.utils import (
    load_price_data,
    apply_dashboard_theme,
    format_currency,
    format_percentage,
    handle_error,
    get_session_value
)

st.title("📊 Market Overview")
st.markdown("Real-time and historical energy market data visualization")

try:
    # Get date range from session state
    date_range = get_session_value('date_range', (
        datetime.now() - timedelta(days=365),
        datetime.now()
    ))

    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    # Load price data using global data source setting
    data_source = get_session_value('data_source', 'synthetic')
    data = load_price_data(data_source, date_range)

    if not data.empty:
        current_price = data.iloc[-1, 0]
        prev_price = data.iloc[-2, 0]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        with col1:
            st.metric(
                label="Current Price",
                value=format_currency(current_price),
                delta=f"{price_change_pct:.2f}%"
            )
        with col2:
            st.metric(
                label="24h Volume",
                value="1.2M MWh"
            )
        with col3:
            st.metric(
                label="24h High/Low",
                value=f"{format_currency(data.iloc[-24:, 0].max())} / {format_currency(data.iloc[-24:, 0].min())}"
            )
        with col4:
            st.metric(
                label="Market Status",
                value="Open"
            )

        st.markdown("---")

        # Price Chart Section
        st.subheader("Price Chart")

        # Controls Row
        col1, col2, col3 = st.columns(3)
        with col1:
            asset = st.selectbox("Asset", ['Energy', 'Wind', 'Solar'])
        with col2:
            timeframe = st.radio("Timeframe", ['1D', '1W', '1M', '3M', '1Y', 'All'], index=2)
        with col3:
            chart_type = st.radio("Chart Type", ['Line', 'Candlestick', 'Area'], index=0)

        # Apply asset selection to data (select appropriate column if multiple assets)
        # If data has multiple columns, select the one matching the asset
        if len(data.columns) > 1:
            # Try to find matching column
            asset_lower = asset.lower()
            matching_cols = [col for col in data.columns if asset_lower in str(col).lower()]
            if matching_cols:
                selected_data = data[matching_cols[0]]
            else:
                # Default to first column
                selected_data = data.iloc[:, 0]
        else:
            selected_data = data.iloc[:, 0]

        # Filter data by timeframe
        end_date = data.index[-1]
        if timeframe == '1D':
            start_date = end_date - timedelta(days=1)
        elif timeframe == '1W':
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == '1M':
            start_date = end_date - timedelta(days=30)
        elif timeframe == '3M':
            start_date = end_date - timedelta(days=90)
        elif timeframe == '1Y':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = data.index[0]

        filtered_data = selected_data[selected_data.index >= start_date]

        # Create chart
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price', 'Volume'),
            vertical_spacing=0.1
        )

        if chart_type == 'Line':
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data.values,
                    mode='lines',
                    name='Price',
                    line=dict(color='#1f77b4')
                ),
                row=1, col=1
            )
        elif chart_type == 'Area':
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data.values,
                    mode='lines',
                    name='Price',
                    fill='tozeroy',
                    line=dict(color='#1f77b4')
                ),
                row=1, col=1
            )
        elif chart_type == 'Candlestick':
            # Generate OHLC data from price data
            # Resample to daily or hourly bars for candlestick
            resample_freq = 'D' if len(filtered_data) > 500 else 'H'
            # Use list aggregation for Series
            ohlc_data = filtered_data.resample(resample_freq).agg(['first', 'max', 'min', 'last']).dropna()

            # Rename columns to OHLC
            ohlc_data.columns = ['Open', 'High', 'Low', 'Close']

            fig.add_trace(
                go.Candlestick(
                    x=ohlc_data.index,
                    open=ohlc_data['Open'],
                    high=ohlc_data['High'],
                    low=ohlc_data['Low'],
                    close=ohlc_data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )

        # Add volume bars
        volume = np.random.randint(10000, 50000, size=len(filtered_data))
        fig.add_trace(
            go.Bar(
                x=filtered_data.index,
                y=volume,
                name='Volume',
                marker=dict(color='#7f7f7f')
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (MWh)", row=2, col=1)

        fig.update_layout(height=600, showlegend=True)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Market Statistics Section
        st.subheader("Market Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Price Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    format_currency(filtered_data.mean()),
                    format_currency(filtered_data.median()),
                    format_currency(filtered_data.std()),
                    format_currency(filtered_data.min()),
                    format_currency(filtered_data.max()),
                    format_currency(filtered_data.max() - filtered_data.min())
                ]
            })
            st.dataframe(stats_df, use_container_width=True)

        with col2:
            st.write("**Volume Statistics**")
            vol_stats_df = pd.DataFrame({
                'Metric': ['Total Volume', 'Avg Daily Volume', 'Peak Volume'],
                'Value': [
                    f"{volume.sum():,.0f} MWh",
                    f"{volume.mean():,.0f} MWh",
                    f"{volume.max():,.0f} MWh"
                ]
            })
            st.dataframe(vol_stats_df, use_container_width=True)

        st.markdown("---")

        # Price Distribution
        st.subheader("Price Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=filtered_data.values,
            name='Price Distribution',
            marker=dict(color='#7f7f7f'),
            nbinsx=50
        ))

        # Add mean and median lines
        mean_price = filtered_data.mean()
        median_price = filtered_data.median()

        fig.add_vline(x=mean_price, line_dash="dash", line_color="red", annotation_text="Mean")
        fig.add_vline(x=median_price, line_dash="dash", line_color="green", annotation_text="Median")

        fig.update_xaxes(title_text="Price ($/MWh)")
        fig.update_yaxes(title_text="Frequency")
        fig.update_layout(height=400, showlegend=True)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Recent Price Movements
        st.subheader("Recent Price Movements")

        recent_data_series = selected_data.tail(10)
        changes = recent_data_series.diff()
        changes_pct = recent_data_series.pct_change() * 100

        display_data = pd.DataFrame({
            'Timestamp': recent_data_series.index.strftime('%Y-%m-%d %H:%M'),
            'Price': recent_data_series.apply(lambda x: format_currency(x)),
            'Change': changes.apply(lambda x: format_currency(x) if pd.notnull(x) else 'N/A'),
            'Change %': changes_pct.apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else 'N/A')
        })

        st.dataframe(display_data, use_container_width=True)

        st.markdown("---")

        # Correlation Heatmap Section
        st.subheader("Asset Correlation Analysis")

        # Load data for multiple assets if available
        # For now, we'll generate synthetic data for multiple assets for demonstration
        if len(data.columns) > 1:
            # Use existing multi-asset data
            corr_data = data
        else:
            # Generate synthetic multi-asset data for correlation
            with st.spinner("Loading multi-asset data for correlation analysis..."):
                multi_asset_data = load_price_data(data_source, date_range)

                # If still single column, create synthetic multi-asset dataset
                if len(multi_asset_data.columns) == 1:
                    # Create correlated assets based on the primary asset
                    base_prices = multi_asset_data.iloc[:, 0]

                    # Generate related assets with different correlation levels
                    np.random.seed(42)
                    corr_data = pd.DataFrame({
                        'Energy': base_prices,
                        'Wind': base_prices * (1 + np.random.normal(0, 0.1, len(base_prices))),
                        'Solar': base_prices * (1 + np.random.normal(0, 0.15, len(base_prices))),
                        'Gas': base_prices * (1 + np.random.normal(0, 0.12, len(base_prices)))
                    }, index=base_prices.index)
                else:
                    corr_data = multi_asset_data

        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Asset Price Correlation Matrix",
            height=400,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Correlation insights
        col1, col2 = st.columns(2)
        with col1:
            # Find highest correlation (excluding diagonal)
            corr_values = correlation_matrix.values
            np.fill_diagonal(corr_values, -np.inf)
            max_corr_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
            max_corr = correlation_matrix.iloc[max_corr_idx[0], max_corr_idx[1]]
            max_corr_pair = f"{correlation_matrix.index[max_corr_idx[0]]} - {correlation_matrix.columns[max_corr_idx[1]]}"

            st.metric(
                "Highest Correlation",
                f"{max_corr:.2f}",
                delta=max_corr_pair
            )

        with col2:
            # Find lowest correlation (excluding diagonal)
            corr_values = correlation_matrix.values.copy()
            np.fill_diagonal(corr_values, np.inf)
            min_corr_idx = np.unravel_index(np.argmin(corr_values), corr_values.shape)
            min_corr = correlation_matrix.iloc[min_corr_idx[0], min_corr_idx[1]]
            min_corr_pair = f"{correlation_matrix.index[min_corr_idx[0]]} - {correlation_matrix.columns[min_corr_idx[1]]}"

            st.metric(
                "Lowest Correlation",
                f"{min_corr:.2f}",
                delta=min_corr_pair
            )

        st.markdown("---")

        # Download button
        csv = data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        st.warning("No data available for the selected date range.")

except Exception as e:
    handle_error(e, "Market Overview Page")
