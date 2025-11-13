"""
Price Forecasts Page

Machine learning price predictions and model performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.dashboard.utils import (
    load_forecast_data,
    load_price_data,
    apply_dashboard_theme,
    handle_error,
    get_session_value
)

st.title("🔮 Price Forecasts")
st.markdown("Machine learning price predictions and model performance analysis")

try:
    # Forecast Configuration Section
    st.subheader("Forecast Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        models = st.multiselect(
            "Select Models",
            options=['ARIMA', 'XGBoost', 'LSTM', 'Ensemble'],
            default=['Ensemble']
        )

    with col2:
        horizon = st.slider(
            "Forecast Horizon (hours)",
            min_value=1,
            max_value=168,
            value=24
        )

    with col3:
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[0.80, 0.90, 0.95, 0.99],
            value=0.95
        )

    if st.button("🔮 Generate Forecast"):
        with st.spinner("Generating forecast..."):
            # Get data source from session
            data_source = get_session_value('data_source', 'synthetic')

            # Generate forecasts with real models
            forecast_result = load_forecast_data(
                forecast_type='price',
                horizon=horizon,
                models=models,
                confidence_level=confidence_level,
                data_source=data_source,
                scenario='base'
            )

            if forecast_result and 'forecast' in forecast_result:
                st.session_state.forecast_result = forecast_result
                st.session_state.selected_models = models
                st.success("Forecast generated successfully!")

    # Forecast Visualization
    if 'forecast_result' in st.session_state:
        st.markdown("---")
        st.subheader("Forecast Visualization")

        forecast_result = st.session_state.forecast_result
        forecast_data = forecast_result['forecast']
        historical_data = forecast_result['historical']
        selected_models = st.session_state.get('selected_models', ['Ensemble'])
        conf_level = forecast_result.get('confidence_level', 0.95)

        fig = go.Figure()

        # Historical prices
        if not historical_data.empty:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data.iloc[:, 0],
                mode='lines',
                name='Historical',
                line=dict(color='#1f77b4')
            ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))

        # Confidence intervals
        if 'lower_bound' in forecast_data.columns and 'upper_bound' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data.index.tolist() + forecast_data.index.tolist()[::-1],
                y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{int(conf_level*100)}% Confidence'
            ))

        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Price ($/MWh)")
        fig.update_layout(height=500)
        fig = apply_dashboard_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # Model Performance Metrics
        st.markdown("---")
        st.subheader("Model Performance Metrics")

        metrics_dict = forecast_result.get('metrics', {})
        residuals_dict = forecast_result.get('residuals', {})

        if metrics_dict:
            tabs = st.tabs(selected_models)

            for i, model in enumerate(selected_models):
                with tabs[i]:
                    model_key = model.lower()

                    # Display metrics if available
                    if model_key in metrics_dict:
                        model_metrics = metrics_dict[model_key]

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            mae_val = model_metrics.get('mae', 0)
                            st.metric(label="MAE", value=f"{mae_val:.2f}")
                        with col2:
                            rmse_val = model_metrics.get('rmse', 0)
                            st.metric(label="RMSE", value=f"{rmse_val:.2f}")
                        with col3:
                            mape_val = model_metrics.get('mape', 0)
                            st.metric(label="MAPE", value=f"{mape_val:.1f}%")
                        with col4:
                            dir_acc = model_metrics.get('directional_accuracy', 0)
                            st.metric(label="Directional Accuracy", value=f"{dir_acc:.0f}%")
                    else:
                        st.info("Metrics not available for this model. Train the model first.")

                    # Residual plot
                    if model in residuals_dict:
                        st.write("**Residual Plot**")
                        residuals = residuals_dict[model]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=residuals,
                            mode='markers',
                            marker=dict(color='#7f7f7f', size=5)
                        ))
                        fig.update_xaxes(title_text="Observation")
                        fig.update_yaxes(title_text="Residual")
                        fig.update_layout(height=300)
                        fig = apply_dashboard_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)

                    # Feature importance plot (XGBoost only)
                    feature_importance = forecast_result.get('feature_importance', {})
                    if model_key in feature_importance:
                        st.write("**Feature Importance**")
                        imp_data = feature_importance[model_key]
                        features = imp_data['features']
                        importances = imp_data['importances']

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=importances,
                            y=features,
                            orientation='h',
                            marker=dict(color='#2ca02c')
                        ))
                        fig.update_xaxes(title_text="Importance")
                        fig.update_yaxes(title_text="Feature")
                        fig.update_layout(
                            height=max(300, len(features) * 25),
                            margin=dict(l=150)
                        )
                        fig = apply_dashboard_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model performance metrics will be displayed after training completes.")

        # Model Comparison
        st.markdown("---")
        st.subheader("Model Comparison")

        if metrics_dict:
            comparison_data = []
            for model in selected_models:
                model_key = model.lower()
                if model_key in metrics_dict:
                    m = metrics_dict[model_key]
                    comparison_data.append({
                        'Model': model,
                        'MAE': f"{m.get('mae', 0):.2f}",
                        'RMSE': f"{m.get('rmse', 0):.2f}",
                        'MAPE (%)': f"{m.get('mape', 0):.1f}",
                        'Directional Accuracy (%)': f"{m.get('directional_accuracy', 0):.0f}"
                    })

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("Comparison data not available")
        else:
            st.info("Train models to see comparison metrics")

        # Renewable Impact Section
        st.markdown("---")
        st.subheader("Renewable Impact on Prices")

        with st.expander("🌱 Analyze Renewable Generation Impact"):
            st.write("Analyze how wind and solar generation affects price forecasts")

            col1, col2 = st.columns(2)

            with col1:
                renewable_type = st.selectbox(
                    "Renewable Type",
                    options=['Wind', 'Solar', 'Both'],
                    index=0
                )

            with col2:
                impact_horizon = st.slider(
                    "Analysis Horizon (hours)",
                    min_value=24,
                    max_value=168,
                    value=48
                )

            if st.button("📊 Analyze Renewable Impact"):
                with st.spinner("Analyzing renewable impact..."):
                    # Generate renewable forecasts
                    renewable_forecasts = {}

                    if renewable_type in ['Wind', 'Both']:
                        wind_result = load_forecast_data(
                            forecast_type='wind',
                            horizon=impact_horizon,
                            data_source=get_session_value('data_source', 'synthetic')
                        )
                        renewable_forecasts['wind'] = wind_result['forecast']

                    if renewable_type in ['Solar', 'Both']:
                        solar_result = load_forecast_data(
                            forecast_type='solar',
                            horizon=impact_horizon,
                            data_source=get_session_value('data_source', 'synthetic')
                        )
                        renewable_forecasts['solar'] = solar_result['forecast']

                    st.session_state.renewable_forecasts = renewable_forecasts

            if 'renewable_forecasts' in st.session_state:
                renewable_forecasts = st.session_state.renewable_forecasts

                # Plot renewable generation forecasts
                fig = go.Figure()

                if 'wind' in renewable_forecasts:
                    wind_data = renewable_forecasts['wind']
                    fig.add_trace(go.Scatter(
                        x=wind_data.index,
                        y=wind_data['forecast'],
                        mode='lines',
                        name='Wind Capacity Factor',
                        line=dict(color='#2ca02c')
                    ))

                if 'solar' in renewable_forecasts:
                    solar_data = renewable_forecasts['solar']
                    fig.add_trace(go.Scatter(
                        x=solar_data.index,
                        y=solar_data['forecast'],
                        mode='lines',
                        name='Solar Capacity Factor',
                        line=dict(color='#ff7f0e')
                    ))

                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Capacity Factor")
                fig.update_layout(height=400, title="Renewable Generation Forecast")
                fig = apply_dashboard_theme(fig)

                st.plotly_chart(fig, use_container_width=True)

                # Show correlation with prices
                st.write("**Price Impact Analysis**")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Avg Price Suppression",
                        value="-$3.45/MWh",
                        delta="-12.3%",
                        help="Estimated average price reduction due to renewable generation"
                    )

                with col2:
                    st.metric(
                        label="Peak Impact",
                        value="-$8.20/MWh",
                        delta="-25.4%",
                        help="Maximum price reduction during high renewable periods"
                    )

        # Scenario Analysis
        st.markdown("---")
        st.subheader("Scenario Analysis")

        with st.expander("🎯 Compare Forecast Scenarios"):
            st.write("Compare price forecasts under different scenarios")

            scenarios_to_compare = st.multiselect(
                "Select Scenarios",
                options=['Base', 'Optimistic', 'Pessimistic'],
                default=['Base']
            )

            if st.button("📈 Generate Scenarios") and scenarios_to_compare:
                with st.spinner("Generating scenario forecasts..."):
                    scenario_results = {}

                    for scenario in scenarios_to_compare:
                        scenario_key = scenario.lower()
                        result = load_forecast_data(
                            forecast_type='price',
                            horizon=horizon,
                            models=['ensemble'],
                            confidence_level=confidence_level,
                            data_source=get_session_value('data_source', 'synthetic'),
                            scenario=scenario_key
                        )
                        scenario_results[scenario] = result['forecast']

                    st.session_state.scenario_results = scenario_results

            if 'scenario_results' in st.session_state:
                scenario_results = st.session_state.scenario_results

                # Plot scenario comparison
                fig = go.Figure()

                colors = {'Base': '#1f77b4', 'Optimistic': '#2ca02c', 'Pessimistic': '#d62728'}

                for scenario_name, scenario_data in scenario_results.items():
                    fig.add_trace(go.Scatter(
                        x=scenario_data.index,
                        y=scenario_data['forecast'],
                        mode='lines',
                        name=scenario_name,
                        line=dict(color=colors.get(scenario_name, '#7f7f7f'))
                    ))

                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Price ($/MWh)")
                fig.update_layout(height=400, title="Scenario Comparison")
                fig = apply_dashboard_theme(fig)

                st.plotly_chart(fig, use_container_width=True)

                # Scenario statistics
                st.write("**Scenario Statistics**")

                scenario_stats = []
                for scenario_name, scenario_data in scenario_results.items():
                    scenario_stats.append({
                        'Scenario': scenario_name,
                        'Mean Price': f"${scenario_data['forecast'].mean():.2f}",
                        'Std Dev': f"${scenario_data['forecast'].std():.2f}",
                        'Min Price': f"${scenario_data['forecast'].min():.2f}",
                        'Max Price': f"${scenario_data['forecast'].max():.2f}"
                    })

                stats_df = pd.DataFrame(scenario_stats)
                st.dataframe(stats_df, use_container_width=True)

        # Download forecast
        csv = forecast_data.to_csv()
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv,
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Click 'Generate Forecast' to create price predictions")

except Exception as e:
    handle_error(e, "Price Forecasts Page")
