"""
Main Streamlit Application

Energy Trading & Portfolio Optimization System Dashboard
Main entry point with navigation sidebar and welcome page.
"""

import streamlit as st
from datetime import datetime, timedelta
import logging

from src.dashboard.utils import (
    initialize_session_state,
    create_date_range_selector,
    refresh_data,
    auto_refresh_check,
    format_currency,
    format_percentage,
    handle_error
)
from src.config.load_config import get_config

# Get logger
logger = logging.getLogger(__name__)

# =============================================================================
# Page Configuration (Must be first Streamlit command)
# =============================================================================

st.set_page_config(
    page_title="Energy Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Load Configuration
# =============================================================================

try:
    config = get_config()
    dashboard_config = config.get('dashboard', {})
except Exception as e:
    st.error(f"Error loading configuration: {str(e)}")
    logger.error(f"Configuration error: {str(e)}", exc_info=True)
    st.stop()

# =============================================================================
# Session State Initialization
# =============================================================================

initialize_session_state()

# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application function."""

    # Title and Header
    st.title(dashboard_config.get('title', 'Energy Trading & Portfolio Optimization Dashboard'))
    st.markdown(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if 'last_update' in st.session_state:
        st.caption(f"Last Data Refresh: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown("---")

    # Welcome Page Content
    st.header("Welcome to the Energy Trading Dashboard")

    st.markdown("""
    This comprehensive dashboard provides real-time analytics and insights for energy trading
    and portfolio optimization. Navigate to different pages using the sidebar to access:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Market Overview")
        st.write("""
        - Real-time and historical price charts
        - Volume analysis and statistics
        - Price distribution and correlation
        - Market status and indicators
        """)

        st.subheader("🔮 Price Forecasts")
        st.write("""
        - Machine learning price predictions
        - Multiple forecasting models
        - Model performance comparison
        - Scenario analysis
        """)

        st.subheader("📈 Trading Strategies")
        st.write("""
        - Strategy signal generation
        - Backtesting and performance analysis
        - Strategy comparison tools
        - What-if analysis
        """)

    with col2:
        st.subheader("⚖️ Portfolio Optimization")
        st.write("""
        - Modern portfolio theory optimization
        - Efficient frontier visualization
        - Multiple optimization methods
        - Renewable-aware constraints
        """)

        st.subheader("⚠️ Risk Analytics")
        st.write("""
        - VaR and CVaR analysis
        - Stress testing and scenarios
        - Risk decomposition
        - Comprehensive risk metrics
        """)

    st.markdown("---")

    # Quick Stats (placeholder)
    st.subheader("Quick Stats")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📊 Total Portfolio Value", value="$1,000,000")
    with col2:
        st.metric(label="📈 Today's Return", value="+2.5%", delta="2.5%")
    with col3:
        st.metric(label="📍 Active Positions", value="8")
    with col4:
        st.metric(label="⚡ Active Strategies", value="4")

    st.markdown("---")

    # Getting Started Guide
    st.subheader("Getting Started")

    st.markdown("""
    1. **Configure Date Range**: Use the sidebar to select your analysis period
    2. **Choose Data Source**: Select between synthetic or real market data
    3. **Navigate**: Use the sidebar menu to explore different analysis pages
    4. **Refresh Data**: Click the refresh button in the sidebar to update data
    5. **Export Results**: Download data and charts from individual pages
    """)

    # System Information
    with st.expander("ℹ️ System Information"):
        st.write(f"**Dashboard Version:** 1.0.0")
        st.write(f"**Configuration File:** config/config.yaml")
        st.write(f"**Data Update Frequency:** {dashboard_config.get('update_frequency', 300)} seconds")
        st.write(f"**Auto-Refresh Enabled:** {dashboard_config.get('enable_auto_refresh', False)}")

    # Footer
    if dashboard_config.get('ui', {}).get('show_footer', True):
        st.markdown("---")
        st.caption("Energy Trading & Portfolio Optimization System | © 2024")
        st.caption("⚡ Powered by Streamlit, Plotly, and Python")


def sidebar_content():
    """Create sidebar content."""

    st.sidebar.title("⚡ Energy Trading")

    st.sidebar.markdown("---")

    # Data Source Selector
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        options=['Synthetic Data', 'Real Data (EIA/CAISO)'],
        index=0
    )
    st.session_state.data_source = 'synthetic' if 'Synthetic' in data_source else 'real'

    st.sidebar.markdown("---")

    # Global Date Range Selector
    st.sidebar.subheader("Global Date Range")
    date_range = create_date_range_selector(container=st.sidebar)
    st.session_state.date_range = date_range

    st.sidebar.markdown("---")

    # Refresh Button
    if st.sidebar.button("🔄 Refresh Data", key="refresh_btn"):
        refresh_data()

    st.sidebar.markdown("---")

    # Settings Expander
    with st.sidebar.expander("⚙️ Settings"):
        # Theme selector
        theme = st.selectbox(
            "Chart Theme",
            options=['plotly_white', 'plotly', 'plotly_dark', 'ggplot2', 'seaborn'],
            index=0
        )
        st.session_state.chart_theme = theme

        # Chart height
        chart_height = st.slider(
            "Chart Height (px)",
            min_value=300,
            max_value=800,
            value=st.session_state.get('chart_height', 400),
            step=50
        )
        st.session_state.chart_height = chart_height

        # Auto-refresh toggle
        enable_auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.get('enable_auto_refresh', False)
        )
        st.session_state.enable_auto_refresh = enable_auto_refresh

        if enable_auto_refresh:
            update_frequency = st.slider(
                "Update Frequency (seconds)",
                min_value=30,
                max_value=600,
                value=st.session_state.get('update_frequency', 300),
                step=30
            )
            st.session_state.update_frequency = update_frequency

        # Clear cache button
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    st.sidebar.markdown("---")

    # Navigation Guide
    st.sidebar.info("""
    **Quick Navigation:**
    - 📊 Market Overview
    - 🔮 Price Forecasts
    - 📈 Trading Strategies
    - ⚖️ Portfolio Optimization
    - ⚠️ Risk Analytics
    """)


# =============================================================================
# Run Application
# =============================================================================

if __name__ == "__main__":
    try:
        # Create sidebar content
        sidebar_content()

        # Auto-refresh check
        if auto_refresh_check():
            st.rerun()

        # Run main application
        main()

    except Exception as e:
        handle_error(e, "Dashboard Application")
        logger.error("Dashboard error", exc_info=True)
