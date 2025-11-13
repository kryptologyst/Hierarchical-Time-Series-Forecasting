"""
Evaluation metrics for hierarchical time series forecasting.

This module provides comprehensive evaluation metrics for assessing
forecasting performance across different hierarchy levels.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class ModelEvaluator:
    """
    Comprehensive evaluator for hierarchical time series forecasting models.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_mae(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            MAE value
        """
        return mean_absolute_error(actual, forecast)
    
    def calculate_mse(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE).
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            MSE value
        """
        return mean_squared_error(actual, forecast)
    
    def calculate_rmse(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(actual, forecast))
    
    def calculate_mape(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return np.inf
        
        actual_masked = actual[mask]
        forecast_masked = forecast[mask]
        
        return np.mean(np.abs((actual_masked - forecast_masked) / actual_masked)) * 100
    
    def calculate_smape(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            sMAPE value (as percentage)
        """
        numerator = np.abs(actual - forecast)
        denominator = (np.abs(actual) + np.abs(forecast)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    def calculate_mase(self, actual: np.ndarray, forecast: np.ndarray, 
                      naive_forecast: np.ndarray) -> float:
        """
        Calculate Mean Absolute Scaled Error (MASE).
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            naive_forecast: Naive forecast (e.g., seasonal naive)
            
        Returns:
            MASE value
        """
        mae_forecast = self.calculate_mae(actual, forecast)
        mae_naive = self.calculate_mae(actual, naive_forecast)
        
        if mae_naive == 0:
            return np.inf
        
        return mae_forecast / mae_naive
    
    def calculate_all_metrics(
        self, 
        actual: np.ndarray, 
        forecast: np.ndarray,
        naive_forecast: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            naive_forecast: Naive forecast for MASE calculation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'MAE': self.calculate_mae(actual, forecast),
            'MSE': self.calculate_mse(actual, forecast),
            'RMSE': self.calculate_rmse(actual, forecast),
            'MAPE': self.calculate_mape(actual, forecast),
            'sMAPE': self.calculate_smape(actual, forecast)
        }
        
        if naive_forecast is not None:
            metrics['MASE'] = self.calculate_mase(actual, forecast, naive_forecast)
        
        return metrics
    
    def evaluate_forecasts(
        self, 
        actual_data: Dict[str, np.ndarray], 
        forecast_data: Dict[str, np.ndarray],
        naive_forecasts: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecasts for multiple series.
        
        Args:
            actual_data: Dictionary of actual values for each series
            forecast_data: Dictionary of forecasted values for each series
            naive_forecasts: Dictionary of naive forecasts for each series
            
        Returns:
            Dictionary of metrics for each series
        """
        self.logger.info("Evaluating forecasts for all series")
        
        results = {}
        
        for series_name in actual_data.keys():
            if series_name not in forecast_data:
                self.logger.warning(f"No forecast found for {series_name}")
                continue
            
            actual = actual_data[series_name]
            forecast = forecast_data[series_name]
            naive = naive_forecasts.get(series_name) if naive_forecasts else None
            
            try:
                metrics = self.calculate_all_metrics(actual, forecast, naive)
                results[series_name] = metrics
                self.logger.info(f"Evaluated {series_name}: MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")
            except Exception as e:
                self.logger.error(f"Failed to evaluate {series_name}: {e}")
                results[series_name] = {}
        
        return results
    
    def cross_validate(
        self, 
        data: pd.DataFrame, 
        forecaster_class, 
        forecaster_params: Dict,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Perform time series cross-validation.
        
        Args:
            data: Time series data
            forecaster_class: Forecaster class to use
            forecaster_params: Parameters for the forecaster
            n_splits: Number of CV splits
            test_size: Size of test set relative to total data
            
        Returns:
            Dictionary of CV results for each series
        """
        self.logger.info(f"Performing {n_splits}-fold time series cross-validation")
        
        # Calculate split sizes
        total_size = len(data)
        test_size_int = int(total_size * test_size)
        train_size = total_size - test_size_int
        
        cv_results = {series: [] for series in data.columns}
        
        for split in range(n_splits):
            self.logger.info(f"CV Split {split + 1}/{n_splits}")
            
            # Calculate split indices
            start_idx = split * (total_size // n_splits)
            end_idx = min(start_idx + train_size, total_size - test_size_int)
            
            train_data = data.iloc[start_idx:end_idx]
            test_data = data.iloc[end_idx:end_idx + test_size_int]
            
            if len(train_data) < 10 or len(test_data) < 2:
                self.logger.warning(f"Skipping split {split + 1} due to insufficient data")
                continue
            
            try:
                # Train forecaster
                forecaster = forecaster_class(**forecaster_params)
                forecaster.fit(train_data)
                
                # Generate forecasts
                forecasts = forecaster.forecast(len(test_data))
                
                # Evaluate forecasts
                actual_data = {series: test_data[series].values for series in test_data.columns}
                forecast_data = {series: forecasts[series] for series in forecasts.keys()}
                
                split_results = self.evaluate_forecasts(actual_data, forecast_data)
                
                # Store results
                for series, metrics in split_results.items():
                    cv_results[series].append(metrics)
                    
            except Exception as e:
                self.logger.error(f"CV split {split + 1} failed: {e}")
        
        return cv_results
    
    def calculate_cv_summary(self, cv_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics from cross-validation results.
        
        Args:
            cv_results: Cross-validation results
            
        Returns:
            Summary statistics for each series
        """
        summary = {}
        
        for series, results in cv_results.items():
            if not results:
                continue
            
            # Calculate mean and std for each metric
            metrics_summary = {}
            for metric in results[0].keys():
                values = [result[metric] for result in results if metric in result]
                if values:
                    metrics_summary[f"{metric}_mean"] = np.mean(values)
                    metrics_summary[f"{metric}_std"] = np.std(values)
            
            summary[series] = metrics_summary
        
        return summary
    
    def compare_methods(
        self, 
        actual_data: Dict[str, np.ndarray], 
        forecast_results: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Compare multiple forecasting methods.
        
        Args:
            actual_data: Dictionary of actual values for each series
            forecast_results: Dictionary of forecast results for each method
            
        Returns:
            DataFrame comparing methods across series
        """
        self.logger.info("Comparing forecasting methods")
        
        comparison_data = []
        
        for method_name, forecasts in forecast_results.items():
            method_results = self.evaluate_forecasts(actual_data, forecasts)
            
            for series_name, metrics in method_results.items():
                for metric_name, metric_value in metrics.items():
                    comparison_data.append({
                        'Method': method_name,
                        'Series': series_name,
                        'Metric': metric_name,
                        'Value': metric_value
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Pivot to get methods as columns
        pivot_df = comparison_df.pivot_table(
            index=['Series', 'Metric'], 
            columns='Method', 
            values='Value'
        ).reset_index()
        
        return pivot_df
