"""
Main application for hierarchical time series forecasting.

This module provides the main interface for running hierarchical time series
forecasting experiments with multiple methods and reconciliation techniques.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .data_generator import DataGenerator, HierarchyConfig
from .forecasters import ARIMAForecaster, ProphetForecaster, LSTMForecaster
from .reconciliation import ReconciliationEngine
from .evaluation import ModelEvaluator
from .visualization import Plotter


class HierarchicalForecastingPipeline:
    """
    Main pipeline for hierarchical time series forecasting.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the forecasting pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
        # Initialize components
        self.data_generator = DataGenerator(seed=self.config['data']['seed'])
        self.reconciliation_engine = ReconciliationEngine()
        self.evaluator = ModelEvaluator()
        self.plotter = Plotter(
            style=self.config['visualization']['style'],
            figure_size=tuple(self.config['visualization']['figure_size'])
        )
        
        # Initialize forecasters
        self.forecasters = self._initialize_forecasters()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Hierarchical forecasting pipeline initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Configuration file {self.config_path} not found. Using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'data': {
                'seed': 42,
                'periods': 60,
                'start_date': '2020-01-01',
                'frequency': 'M',
                'hierarchy': [
                    {'name': 'Region1', 'base_value': 100, 'volatility': 5, 'trend': 0.02},
                    {'name': 'Region2', 'base_value': 120, 'volatility': 6, 'trend': 0.015},
                    {'name': 'Region3', 'base_value': 80, 'volatility': 4, 'trend': 0.025}
                ]
            },
            'forecasting': {
                'horizon': 12,
                'methods': ['arima', 'prophet'],
                'arima': {'order': [1, 1, 0], 'seasonal_order': [0, 0, 0, 0]},
                'prophet': {'yearly_seasonality': True, 'weekly_seasonality': False, 'daily_seasonality': False}
            },
            'reconciliation': {
                'methods': ['bottom_up', 'top_down']
            },
            'visualization': {
                'figure_size': [12, 8],
                'style': 'seaborn-v0_8'
            }
        }
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get('file', 'logs/hts_forecasting.log'))
            ]
        )
    
    def _initialize_forecasters(self) -> Dict:
        """Initialize forecasting models."""
        forecasters = {}
        forecasting_config = self.config['forecasting']
        
        if 'arima' in forecasting_config['methods']:
            arima_config = forecasting_config['arima']
            forecasters['arima'] = ARIMAForecaster(
                order=tuple(arima_config['order']),
                seasonal_order=tuple(arima_config['seasonal_order'])
            )
        
        if 'prophet' in forecasting_config['methods']:
            prophet_config = forecasting_config['prophet']
            forecasters['prophet'] = ProphetForecaster(**prophet_config)
        
        if 'lstm' in forecasting_config['methods']:
            lstm_config = forecasting_config['lstm']
            forecasters['lstm'] = LSTMForecaster(**lstm_config)
        
        return forecasters
    
    def generate_data(self) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Generate hierarchical time series data.
        
        Returns:
            Tuple of (data, hierarchy_structure)
        """
        self.logger.info("Generating hierarchical time series data")
        
        data_config = self.config['data']
        hierarchy_configs = [
            HierarchyConfig(**config) for config in data_config['hierarchy']
        ]
        
        data = self.data_generator.generate_hierarchical_data(
            periods=data_config['periods'],
            start_date=data_config['start_date'],
            frequency=data_config['frequency'],
            hierarchy_configs=hierarchy_configs
        )
        
        hierarchy_structure = self.data_generator.get_hierarchy_structure(data)
        
        self.logger.info(f"Generated data with shape {data.shape}")
        return data, hierarchy_structure
    
    def run_forecasting_experiment(
        self, 
        data: pd.DataFrame, 
        hierarchy_structure: Dict[str, List[str]],
        test_size: float = 0.2
    ) -> Dict[str, Dict]:
        """
        Run complete forecasting experiment.
        
        Args:
            data: Time series data
            hierarchy_structure: Structure of the hierarchy
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary of experiment results
        """
        self.logger.info("Starting forecasting experiment")
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        self.logger.info(f"Train data: {len(train_data)} periods, Test data: {len(test_data)} periods")
        
        # Generate forecasts for each method
        forecast_horizon = self.config['forecasting']['horizon']
        forecast_results = {}
        
        for method_name, forecaster in self.forecasters.items():
            self.logger.info(f"Running {method_name} forecasting")
            
            try:
                # Fit model
                forecaster.fit(train_data)
                
                # Generate forecasts
                forecasts = forecaster.forecast(forecast_horizon)
                
                # Apply reconciliation
                reconciliation_methods = self.config['reconciliation']['methods']
                reconciled_forecasts = {}
                
                for recon_method in reconciliation_methods:
                    reconciled = self.reconciliation_engine.reconcile(
                        forecasts, hierarchy_structure, recon_method
                    )
                    reconciled_forecasts[f"{method_name}_{recon_method}"] = reconciled
                
                forecast_results[method_name] = reconciled_forecasts
                
            except Exception as e:
                self.logger.error(f"Failed to run {method_name} forecasting: {e}")
        
        # Evaluate forecasts
        evaluation_results = {}
        forecast_dates = pd.date_range(
            start=data.index[-1] + pd.DateOffset(months=1),
            periods=forecast_horizon,
            freq='M'
        )
        
        # Use test data for evaluation (truncated to forecast horizon)
        actual_data = {series: test_data[series].values[:forecast_horizon] 
                      for series in test_data.columns}
        
        for method_name, method_forecasts in forecast_results.items():
            for recon_name, forecasts in method_forecasts.items():
                try:
                    evaluation = self.evaluator.evaluate_forecasts(actual_data, forecasts)
                    evaluation_results[recon_name] = evaluation
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {recon_name}: {e}")
        
        # Store results
        results = {
            'train_data': train_data,
            'test_data': test_data,
            'forecast_results': forecast_results,
            'evaluation_results': evaluation_results,
            'forecast_dates': forecast_dates,
            'hierarchy_structure': hierarchy_structure
        }
        
        self.logger.info("Forecasting experiment completed")
        return results
    
    def create_visualizations(self, results: Dict[str, Dict]) -> None:
        """
        Create comprehensive visualizations.
        
        Args:
            results: Experiment results
        """
        self.logger.info("Creating visualizations")
        
        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Plot hierarchy structure
        hierarchy_fig = self.plotter.plot_hierarchy_structure(
            results['hierarchy_structure']
        )
        self.plotter.save_plot(hierarchy_fig, plots_dir / "hierarchy_structure.png")
        
        # Plot historical data
        historical_fig = self.plotter.plot_time_series(
            results['train_data'],
            title="Historical Hierarchical Time Series Data"
        )
        self.plotter.save_plot(historical_fig, plots_dir / "historical_data.png")
        
        # Plot forecasts for each method
        for method_name, method_forecasts in results['forecast_results'].items():
            for recon_name, forecasts in method_forecasts.items():
                forecast_fig = self.plotter.plot_forecasts(
                    results['train_data'],
                    forecasts,
                    results['forecast_dates'],
                    title=f"{recon_name.replace('_', ' ').title()} Forecasts"
                )
                self.plotter.save_plot(forecast_fig, plots_dir / f"{recon_name}_forecasts.png")
        
        # Plot evaluation metrics
        if results['evaluation_results']:
            evaluation_fig = self.plotter.plot_evaluation_metrics(
                results['evaluation_results'],
                title="Model Evaluation Metrics Comparison"
            )
            self.plotter.save_plot(evaluation_fig, plots_dir / "evaluation_metrics.png")
        
        self.logger.info(f"Visualizations saved to {plots_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Dict]:
        """
        Run the complete hierarchical forecasting pipeline.
        
        Returns:
            Complete experiment results
        """
        self.logger.info("Starting complete hierarchical forecasting pipeline")
        
        # Generate data
        data, hierarchy_structure = self.generate_data()
        
        # Run forecasting experiment
        results = self.run_forecasting_experiment(data, hierarchy_structure)
        
        # Create visualizations
        self.create_visualizations(results)
        
        # Print summary
        self._print_experiment_summary(results)
        
        self.logger.info("Complete pipeline finished successfully")
        return results
    
    def _print_experiment_summary(self, results: Dict[str, Dict]) -> None:
        """Print experiment summary."""
        print("\n" + "="*60)
        print("HIERARCHICAL TIME SERIES FORECASTING EXPERIMENT SUMMARY")
        print("="*60)
        
        print(f"\nData Shape: {results['train_data'].shape[0]} training periods, {results['test_data'].shape[0]} test periods")
        print(f"Forecast Horizon: {self.config['forecasting']['horizon']} periods")
        print(f"Hierarchy Levels: {len(results['hierarchy_structure']['bottom_level'])} bottom-level series")
        
        print(f"\nForecasting Methods Used:")
        for method in self.forecasters.keys():
            print(f"  - {method.upper()}")
        
        print(f"\nReconciliation Methods Used:")
        for method in self.config['reconciliation']['methods']:
            print(f"  - {method.replace('_', ' ').title()}")
        
        if results['evaluation_results']:
            print(f"\nBest Performing Methods (by MAE):")
            for recon_name, evaluation in results['evaluation_results'].items():
                if evaluation:
                    # Calculate average MAE across series
                    mae_values = [metrics.get('MAE', np.inf) for metrics in evaluation.values()]
                    avg_mae = np.mean(mae_values)
                    print(f"  - {recon_name.replace('_', ' ').title()}: {avg_mae:.4f}")
        
        print(f"\nVisualizations saved to: plots/")
        print("="*60)


def main():
    """Main function to run the hierarchical forecasting pipeline."""
    try:
        # Initialize pipeline
        pipeline = HierarchicalForecastingPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\nPipeline completed successfully!")
        return results
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
