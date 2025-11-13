"""
Streamlit web interface for hierarchical time series forecasting.

This module provides an interactive web interface for exploring hierarchical
time series forecasting results and running experiments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import logging

# Import our modules
import sys
sys.path.append('src')
from src.main import HierarchicalForecastingPipeline
from src.data_generator import DataGenerator, HierarchyConfig
from src.forecasters import ARIMAForecaster, ProphetForecaster
from src.reconciliation import ReconciliationEngine
from src.evaluation import ModelEvaluator
from src.visualization import Plotter


# Page configuration
st.set_page_config(
    page_title="Hierarchical Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'data' not in st.session_state:
    st.session_state.data = None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Hierarchical Time Series Forecasting</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.header("Data Generation")
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=10000)
    periods = st.sidebar.number_input("Number of Periods", value=60, min_value=12, max_value=200)
    forecast_horizon = st.sidebar.number_input("Forecast Horizon", value=12, min_value=1, max_value=24)
    
    # Hierarchy configuration
    st.sidebar.header("Hierarchy Configuration")
    n_regions = st.sidebar.number_input("Number of Regions", value=3, min_value=2, max_value=5)
    
    regions = []
    for i in range(n_regions):
        with st.sidebar.expander(f"Region {i+1}"):
            name = st.text_input(f"Name", value=f"Region{i+1}", key=f"region_{i}_name")
            base_value = st.number_input(f"Base Value", value=100 + i*20, min_value=10, key=f"region_{i}_base")
            volatility = st.number_input(f"Volatility", value=5.0, min_value=0.1, max_value=20.0, key=f"region_{i}_vol")
            trend = st.number_input(f"Trend", value=0.02, min_value=-0.1, max_value=0.1, step=0.001, key=f"region_{i}_trend")
            regions.append(HierarchyConfig(name, base_value, volatility, trend))
    
    # Forecasting methods
    st.sidebar.header("Forecasting Methods")
    use_arima = st.sidebar.checkbox("ARIMA", value=True)
    use_prophet = st.sidebar.checkbox("Prophet", value=True)
    use_lstm = st.sidebar.checkbox("LSTM", value=False)
    
    # Reconciliation methods
    st.sidebar.header("Reconciliation Methods")
    use_bottom_up = st.sidebar.checkbox("Bottom-Up", value=True)
    use_top_down = st.sidebar.checkbox("Top-Down", value=True)
    use_middle_out = st.sidebar.checkbox("Middle-Out", value=False)
    use_optimal = st.sidebar.checkbox("Optimal", value=False)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔮 Forecasting", "📈 Results", "⚙️ Configuration"])
    
    with tab1:
        show_data_overview(regions, periods, seed)
    
    with tab2:
        show_forecasting_interface(regions, periods, seed, forecast_horizon, 
                                 use_arima, use_prophet, use_lstm,
                                 use_bottom_up, use_top_down, use_middle_out, use_optimal)
    
    with tab3:
        show_results()
    
    with tab4:
        show_configuration()


def show_data_overview(regions, periods, seed):
    """Show data overview tab."""
    st.header("📊 Data Overview")
    
    if st.button("Generate Sample Data", key="generate_data"):
        with st.spinner("Generating hierarchical time series data..."):
            # Generate data
            data_generator = DataGenerator(seed=seed)
            data = data_generator.generate_hierarchical_data(
                periods=periods,
                start_date="2020-01-01",
                frequency="M",
                hierarchy_configs=regions
            )
            
            st.session_state.data = data
            
            # Display data info
            st.success(f"Generated data with {len(data)} periods and {len(data.columns)} series")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
            # Show data statistics
            st.subheader("Data Statistics")
            st.dataframe(data.describe())
            
            # Plot time series
            st.subheader("Time Series Plot")
            fig = go.Figure()
            
            for column in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Hierarchical Time Series Data",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show hierarchy structure
            st.subheader("Hierarchy Structure")
            hierarchy_structure = data_generator.get_hierarchy_structure(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Bottom Level:**")
                for series in hierarchy_structure['bottom_level']:
                    st.write(f"- {series}")
            
            with col2:
                st.write("**Top Level:**")
                for series in hierarchy_structure['top_level']:
                    st.write(f"- {series}")


def show_forecasting_interface(regions, periods, seed, forecast_horizon, 
                               use_arima, use_prophet, use_lstm,
                               use_bottom_up, use_top_down, use_middle_out, use_optimal):
    """Show forecasting interface tab."""
    st.header("🔮 Forecasting Interface")
    
    if st.session_state.data is None:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    if st.button("Run Forecasting Experiment", key="run_forecast"):
        with st.spinner("Running forecasting experiment..."):
            try:
                # Prepare configuration
                config = {
                    'data': {
                        'seed': seed,
                        'periods': periods,
                        'start_date': '2020-01-01',
                        'frequency': 'M',
                        'hierarchy': [
                            {'name': r.name, 'base_value': r.base_value, 
                             'volatility': r.volatility, 'trend': r.trend} 
                            for r in regions
                        ]
                    },
                    'forecasting': {
                        'horizon': forecast_horizon,
                        'methods': [],
                        'arima': {'order': [1, 1, 0], 'seasonal_order': [0, 0, 0, 0]},
                        'prophet': {'yearly_seasonality': True, 'weekly_seasonality': False, 'daily_seasonality': False}
                    },
                    'reconciliation': {
                        'methods': []
                    }
                }
                
                # Add selected methods
                if use_arima:
                    config['forecasting']['methods'].append('arima')
                if use_prophet:
                    config['forecasting']['methods'].append('prophet')
                if use_lstm:
                    config['forecasting']['methods'].append('lstm')
                
                # Add reconciliation methods
                if use_bottom_up:
                    config['reconciliation']['methods'].append('bottom_up')
                if use_top_down:
                    config['reconciliation']['methods'].append('top_down')
                if use_middle_out:
                    config['reconciliation']['methods'].append('middle_out')
                if use_optimal:
                    config['reconciliation']['methods'].append('optimal')
                
                # Initialize pipeline
                pipeline = HierarchicalForecastingPipeline()
                pipeline.config = config
                pipeline.forecasters = pipeline._initialize_forecasters()
                
                # Run experiment
                data = st.session_state.data
                hierarchy_structure = pipeline.data_generator.get_hierarchy_structure(data)
                results = pipeline.run_forecasting_experiment(data, hierarchy_structure)
                
                st.session_state.results = results
                st.success("Forecasting experiment completed successfully!")
                
            except Exception as e:
                st.error(f"Forecasting experiment failed: {str(e)}")


def show_results():
    """Show results tab."""
    st.header("📈 Results")
    
    if st.session_state.results is None:
        st.warning("Please run a forecasting experiment first.")
        return
    
    results = st.session_state.results
    
    # Show experiment summary
    st.subheader("Experiment Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Periods", len(results['train_data']))
    with col2:
        st.metric("Test Periods", len(results['test_data']))
    with col3:
        st.metric("Forecast Horizon", len(results['forecast_dates']))
    with col4:
        st.metric("Hierarchy Levels", len(results['hierarchy_structure']['bottom_level']))
    
    # Show forecasts
    st.subheader("Forecast Results")
    
    for method_name, method_forecasts in results['forecast_results'].items():
        st.write(f"**{method_name.upper()} Method:**")
        
        for recon_name, forecasts in method_forecasts.items():
            with st.expander(f"{recon_name.replace('_', ' ').title()} Forecasts"):
                # Create forecast plot
                fig = make_subplots(
                    rows=len(forecasts), cols=1,
                    subplot_titles=list(forecasts.keys()),
                    vertical_spacing=0.1
                )
                
                for i, (series_name, forecast_values) in enumerate(forecasts.items(), 1):
                    # Historical data
                    if series_name in results['train_data'].columns:
                        fig.add_trace(
                            go.Scatter(
                                x=results['train_data'].index,
                                y=results['train_data'][series_name],
                                mode='lines',
                                name=f'{series_name} (Historical)',
                                line=dict(color='#1f77b4', width=2)
                            ),
                            row=i, col=1
                        )
                    
                    # Forecasts
                    fig.add_trace(
                        go.Scatter(
                            x=results['forecast_dates'],
                            y=forecast_values,
                            mode='lines',
                            name=f'{series_name} (Forecast)',
                            line=dict(color='#ff7f0e', width=2, dash='dash')
                        ),
                        row=i, col=1
                    )
                
                fig.update_layout(
                    title=f"{recon_name.replace('_', ' ').title()} Forecasts",
                    height=300 * len(forecasts),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Show evaluation metrics
    if results['evaluation_results']:
        st.subheader("Evaluation Metrics")
        
        for recon_name, evaluation in results['evaluation_results'].items():
            if evaluation:
                with st.expander(f"{recon_name.replace('_', ' ').title()} Metrics"):
                    # Create metrics dataframe
                    metrics_df = pd.DataFrame(evaluation).T
                    st.dataframe(metrics_df)
                    
                    # Create metrics plot
                    fig = go.Figure()
                    
                    for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                        if metric in metrics_df.columns:
                            fig.add_trace(go.Bar(
                                name=metric,
                                x=metrics_df.index,
                                y=metrics_df[metric]
                            ))
                    
                    fig.update_layout(
                        title=f"{recon_name.replace('_', ' ').title()} - Evaluation Metrics",
                        xaxis_title="Series",
                        yaxis_title="Metric Value",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)


def show_configuration():
    """Show configuration tab."""
    st.header("⚙️ Configuration")
    
    st.subheader("Current Configuration")
    
    # Display current config
    config_text = """
    # Hierarchical Time Series Forecasting Configuration
    
    ## Data Generation
    - Random Seed: 42
    - Number of Periods: 60
    - Start Date: 2020-01-01
    - Frequency: Monthly
    
    ## Forecasting Methods
    - ARIMA: (1,1,0) order
    - Prophet: With yearly seasonality
    - LSTM: 2 layers, 50 hidden units
    
    ## Reconciliation Methods
    - Bottom-Up: Aggregate from bottom level
    - Top-Down: Disaggregate from top level
    - Middle-Out: Start from middle level
    - Optimal: Least squares reconciliation
    
    ## Evaluation Metrics
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - sMAPE: Symmetric MAPE
    """
    
    st.code(config_text, language='yaml')
    
    # Download configuration
    st.subheader("Download Configuration")
    
    config_dict = {
        'data': {
            'seed': 42,
            'periods': 60,
            'start_date': '2020-01-01',
            'frequency': 'M'
        },
        'forecasting': {
            'horizon': 12,
            'methods': ['arima', 'prophet'],
            'arima': {'order': [1, 1, 0], 'seasonal_order': [0, 0, 0, 0]},
            'prophet': {'yearly_seasonality': True, 'weekly_seasonality': False, 'daily_seasonality': False}
        },
        'reconciliation': {
            'methods': ['bottom_up', 'top_down']
        }
    }
    
    config_yaml = yaml.dump(config_dict, default_flow_style=False)
    
    st.download_button(
        label="Download Configuration YAML",
        data=config_yaml,
        file_name="hts_config.yaml",
        mime="text/yaml"
    )


if __name__ == "__main__":
    main()
