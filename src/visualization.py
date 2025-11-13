"""
Visualization module for hierarchical time series forecasting.

This module provides comprehensive plotting capabilities for visualizing
time series data, forecasts, and evaluation results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


class Plotter:
    """
    Comprehensive plotting class for hierarchical time series analysis.
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the plotter.
        
        Args:
            style: Matplotlib style
            figure_size: Default figure size
        """
        self.logger = logging.getLogger(__name__)
        self.style = style
        self.figure_size = figure_size
        
        # Set style
        try:
            plt.style.use(style)
        except OSError:
            self.logger.warning(f"Style {style} not found, using default")
            plt.style.use('default')
        
        # Set default colors
        self.colors = {
            'historical': '#1f77b4',
            'forecast': '#ff7f0e',
            'confidence': '#d3d3d3',
            'actual': '#2ca02c',
            'error': '#d62728'
        }
    
    def plot_time_series(
        self, 
        data: pd.DataFrame, 
        title: str = "Time Series Data",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot time series data.
        
        Args:
            data: DataFrame with time series data
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        figsize = figsize or self.figure_size
        fig, ax = plt.subplots(figsize=figsize)
        
        for column in data.columns:
            ax.plot(data.index, data[column], label=column, linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_forecasts(
        self, 
        historical_data: pd.DataFrame,
        forecasts: Dict[str, np.ndarray],
        forecast_dates: pd.DatetimeIndex,
        confidence_intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        title: str = "Hierarchical Time Series Forecasts",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot historical data and forecasts.
        
        Args:
            historical_data: Historical time series data
            forecasts: Dictionary of forecasts for each series
            forecast_dates: Dates for forecast period
            confidence_intervals: Confidence intervals for forecasts
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        figsize = figsize or self.figure_size
        n_series = len(forecasts)
        
        if n_series == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(n_series, 1, figsize=(figsize[0], figsize[1] * n_series))
            if n_series == 1:
                axes = [axes]
        
        for i, (series_name, forecast_values) in enumerate(forecasts.items()):
            ax = axes[i] if n_series > 1 else axes[0]
            
            # Plot historical data
            if series_name in historical_data.columns:
                ax.plot(historical_data.index, historical_data[series_name], 
                       color=self.colors['historical'], label=f'{series_name} (Historical)', 
                       linewidth=2)
            
            # Plot forecasts
            ax.plot(forecast_dates, forecast_values, 
                   color=self.colors['forecast'], label=f'{series_name} (Forecast)', 
                   linewidth=2, linestyle='--')
            
            # Plot confidence intervals if available
            if confidence_intervals and series_name in confidence_intervals:
                lower, upper = confidence_intervals[series_name]
                ax.fill_between(forecast_dates, lower, upper, 
                               color=self.colors['confidence'], alpha=0.3, 
                               label=f'{series_name} (95% CI)')
            
            ax.set_title(f'{series_name} Forecast', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_hierarchy_structure(
        self, 
        hierarchy_structure: Dict[str, List[str]],
        title: str = "Hierarchy Structure"
    ) -> plt.Figure:
        """
        Plot the hierarchy structure as a tree diagram.
        
        Args:
            hierarchy_structure: Structure of the hierarchy
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple tree visualization
        top_level = hierarchy_structure.get('top_level', [])
        bottom_level = hierarchy_structure.get('bottom_level', [])
        
        # Plot top level
        if top_level:
            ax.text(0.5, 0.8, top_level[0], ha='center', va='center', 
                   fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Plot bottom level
        n_bottom = len(bottom_level)
        for i, series in enumerate(bottom_level):
            x_pos = 0.2 + (i * 0.6) / max(1, n_bottom - 1)
            ax.text(x_pos, 0.2, series, ha='center', va='center', 
                   fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            
            # Draw connection line
            if top_level:
                ax.plot([0.5, x_pos], [0.7, 0.3], 'k-', alpha=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_evaluation_metrics(
        self, 
        evaluation_results: Dict[str, Dict[str, float]],
        title: str = "Model Evaluation Metrics",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot evaluation metrics comparison.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        figsize = figsize or self.figure_size
        
        # Prepare data for plotting
        plot_data = []
        for series_name, metrics in evaluation_results.items():
            for metric_name, metric_value in metrics.items():
                plot_data.append({
                    'Series': series_name,
                    'Metric': metric_name,
                    'Value': metric_value
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots for different metrics
        metrics = df['Metric'].unique()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0] * n_metrics, figsize[1]))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            metric_data = df[df['Metric'] == metric]
            
            axes[i].bar(metric_data['Series'], metric_data['Value'], 
                       color=self.colors['historical'], alpha=0.7)
            axes[i].set_title(f'{metric}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Series', fontsize=10)
            axes[i].set_ylabel('Value', fontsize=10)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_forecast_comparison(
        self, 
        actual_data: Dict[str, np.ndarray],
        forecast_results: Dict[str, Dict[str, np.ndarray]],
        forecast_dates: pd.DatetimeIndex,
        title: str = "Forecast Method Comparison",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Compare different forecasting methods.
        
        Args:
            actual_data: Dictionary of actual values
            forecast_results: Dictionary of forecast results for each method
            forecast_dates: Dates for forecast period
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        figsize = figsize or self.figure_size
        n_series = len(actual_data)
        
        fig, axes = plt.subplots(n_series, 1, figsize=(figsize[0], figsize[1] * n_series))
        if n_series == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(forecast_results)))
        
        for i, (series_name, actual_values) in enumerate(actual_data.items()):
            ax = axes[i]
            
            # Plot actual values
            ax.plot(forecast_dates, actual_values, 
                   color=self.colors['actual'], label='Actual', 
                   linewidth=3, marker='o')
            
            # Plot forecasts for each method
            for j, (method_name, forecasts) in enumerate(forecast_results.items()):
                if series_name in forecasts:
                    ax.plot(forecast_dates, forecasts[series_name], 
                           color=colors[j], label=method_name, 
                           linewidth=2, linestyle='--')
            
            ax.set_title(f'{series_name} - Method Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_interactive_plot(
        self, 
        historical_data: pd.DataFrame,
        forecasts: Dict[str, np.ndarray],
        forecast_dates: pd.DatetimeIndex,
        title: str = "Interactive Hierarchical Forecasts"
    ) -> go.Figure:
        """
        Create interactive Plotly plot.
        
        Args:
            historical_data: Historical time series data
            forecasts: Dictionary of forecasts for each series
            forecast_dates: Dates for forecast period
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=len(forecasts), cols=1,
            subplot_titles=list(forecasts.keys()),
            vertical_spacing=0.1
        )
        
        for i, (series_name, forecast_values) in enumerate(forecasts.items(), 1):
            # Add historical data
            if series_name in historical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data.index,
                        y=historical_data[series_name],
                        mode='lines',
                        name=f'{series_name} (Historical)',
                        line=dict(color=self.colors['historical'], width=2)
                    ),
                    row=i, col=1
                )
            
            # Add forecasts
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode='lines',
                    name=f'{series_name} (Forecast)',
                    line=dict(color=self.colors['forecast'], width=2, dash='dash')
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title=title,
            height=300 * len(forecasts),
            showlegend=True
        )
        
        return fig
    
    def plot_anomaly_detection(
        self, 
        data: pd.DataFrame,
        anomalies: Dict[str, np.ndarray],
        title: str = "Anomaly Detection Results",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot anomaly detection results.
        
        Args:
            data: Time series data
            anomalies: Dictionary of anomaly flags for each series
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        figsize = figsize or self.figure_size
        n_series = len(data.columns)
        
        fig, axes = plt.subplots(n_series, 1, figsize=(figsize[0], figsize[1] * n_series))
        if n_series == 1:
            axes = [axes]
        
        for i, (series_name, series_data) in enumerate(data.items()):
            ax = axes[i]
            
            # Plot normal data
            normal_mask = ~anomalies.get(series_name, np.zeros(len(series_data), dtype=bool))
            ax.plot(data.index[normal_mask], series_data[normal_mask], 
                   color=self.colors['historical'], label='Normal', alpha=0.7)
            
            # Plot anomalies
            anomaly_mask = anomalies.get(series_name, np.zeros(len(series_data), dtype=bool))
            if np.any(anomaly_mask):
                ax.scatter(data.index[anomaly_mask], series_data[anomaly_mask], 
                          color=self.colors['error'], label='Anomaly', s=50, zorder=5)
            
            ax.set_title(f'{series_name} - Anomaly Detection', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution for saved image
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        self.logger.info(f"Plot saved to {filename}")
    
    def show_plot(self, fig: plt.Figure) -> None:
        """
        Display plot.
        
        Args:
            fig: Matplotlib figure
        """
        plt.show()
