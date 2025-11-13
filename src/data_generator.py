"""
Data generation module for hierarchical time series.

This module provides functionality to generate synthetic hierarchical time series
data with realistic patterns including trends, seasonality, and noise.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class HierarchyConfig:
    """Configuration for hierarchy structure."""
    name: str
    base_value: float
    volatility: float
    trend: float


class DataGenerator:
    """
    Generate synthetic hierarchical time series data.
    
    This class creates realistic time series data with hierarchical structure,
    including trends, seasonality, and noise patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        np.random.seed(seed)
    
    def generate_hierarchical_data(
        self,
        periods: int,
        start_date: str,
        frequency: str,
        hierarchy_configs: List[HierarchyConfig],
        add_seasonality: bool = True,
        seasonal_period: int = 12
    ) -> pd.DataFrame:
        """
        Generate hierarchical time series data.
        
        Args:
            periods: Number of time periods to generate
            start_date: Start date for the time series
            frequency: Frequency of the time series ('M', 'D', 'W', etc.)
            hierarchy_configs: List of hierarchy configurations
            add_seasonality: Whether to add seasonal patterns
            seasonal_period: Period of seasonal pattern
            
        Returns:
            DataFrame with hierarchical time series data
        """
        self.logger.info(f"Generating {periods} periods of hierarchical data")
        
        # Create time index
        time_index = pd.date_range(
            start=start_date, 
            periods=periods, 
            freq=frequency
        )
        
        # Initialize DataFrame
        df = pd.DataFrame(index=time_index)
        
        # Generate data for each level in hierarchy
        hierarchy_data = {}
        for config in hierarchy_configs:
            data = self._generate_series(
                periods=periods,
                base_value=config.base_value,
                volatility=config.volatility,
                trend=config.trend,
                add_seasonality=add_seasonality,
                seasonal_period=seasonal_period
            )
            hierarchy_data[config.name] = data
            df[config.name] = data
        
        # Calculate total (sum of all hierarchy levels)
        df['Total'] = df[hierarchy_configs[0].name]
        for config in hierarchy_configs[1:]:
            df['Total'] += df[config.name]
        
        self.logger.info("Hierarchical data generation completed")
        return df
    
    def _generate_series(
        self,
        periods: int,
        base_value: float,
        volatility: float,
        trend: float,
        add_seasonality: bool = True,
        seasonal_period: int = 12
    ) -> np.ndarray:
        """
        Generate a single time series with trend, seasonality, and noise.
        
        Args:
            periods: Number of periods
            base_value: Base value for the series
            volatility: Volatility (standard deviation of noise)
            trend: Trend coefficient
            add_seasonality: Whether to add seasonal component
            seasonal_period: Period of seasonal pattern
            
        Returns:
            Generated time series as numpy array
        """
        # Generate trend component
        trend_component = np.arange(periods) * trend
        
        # Generate seasonal component
        seasonal_component = np.zeros(periods)
        if add_seasonality:
            seasonal_component = (
                np.sin(2 * np.pi * np.arange(periods) / seasonal_period) * 
                base_value * 0.1
            )
        
        # Generate noise component
        noise_component = np.random.normal(0, volatility, periods)
        
        # Combine components
        series = base_value + trend_component + seasonal_component + noise_component
        
        # Ensure non-negative values (for sales data)
        series = np.maximum(series, 0)
        
        return series
    
    def add_anomalies(
        self, 
        df: pd.DataFrame, 
        anomaly_probability: float = 0.05,
        anomaly_magnitude: float = 2.0
    ) -> pd.DataFrame:
        """
        Add anomalies to the time series data.
        
        Args:
            df: Input DataFrame
            anomaly_probability: Probability of anomaly at each point
            anomaly_magnitude: Magnitude multiplier for anomalies
            
        Returns:
            DataFrame with added anomalies
        """
        df_anomalous = df.copy()
        
        for column in df.columns:
            if column == 'Total':
                continue  # Skip total column, will be recalculated
            
            # Randomly select points for anomalies
            anomaly_mask = np.random.random(len(df)) < anomaly_probability
            
            # Add anomalies
            anomalies = np.random.normal(0, df[column].std() * anomaly_magnitude, len(df))
            df_anomalous[column] = df[column] + (anomaly_mask * anomalies)
            
            # Ensure non-negative values
            df_anomalous[column] = np.maximum(df_anomalous[column], 0)
        
        # Recalculate total
        hierarchy_columns = [col for col in df.columns if col != 'Total']
        df_anomalous['Total'] = df_anomalous[hierarchy_columns].sum(axis=1)
        
        self.logger.info(f"Added anomalies with probability {anomaly_probability}")
        return df_anomalous
    
    def get_hierarchy_structure(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get the hierarchy structure from the DataFrame.
        
        Args:
            df: DataFrame with hierarchical data
            
        Returns:
            Dictionary mapping hierarchy levels to column names
        """
        hierarchy_columns = [col for col in df.columns if col != 'Total']
        
        return {
            'bottom_level': hierarchy_columns,
            'top_level': ['Total']
        }
