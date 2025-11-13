"""
Utility functions for hierarchical time series forecasting.

This module provides utility functions for logging, checkpointing, and other
common operations.
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self, 
        model: Any, 
        metadata: Dict[str, Any], 
        checkpoint_name: str
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            metadata: Metadata about the model
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.pkl"
        
        checkpoint_data = {
            'model': model,
            'metadata': metadata,
            'timestamp': timestamp,
            'checkpoint_name': checkpoint_name
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing model and metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pkl"))
        return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def get_latest_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Path]:
        """
        Get the latest checkpoint.
        
        Args:
            checkpoint_name: Optional name filter for checkpoints
            
        Returns:
            Path to latest checkpoint or None if not found
        """
        checkpoints = self.list_checkpoints()
        
        if checkpoint_name:
            checkpoints = [cp for cp in checkpoints if checkpoint_name in cp.name]
        
        return checkpoints[0] if checkpoints else None


class ExperimentLogger:
    """
    Logger for tracking experiments and results.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Log the start of an experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_file = self.log_dir / f"{experiment_id}.log"
        
        # Setup file handler for this experiment
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        experiment_logger = logging.getLogger(f"experiment_{experiment_id}")
        experiment_logger.addHandler(file_handler)
        experiment_logger.setLevel(logging.INFO)
        
        # Log experiment start
        experiment_logger.info(f"Starting experiment: {experiment_name}")
        experiment_logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        return experiment_id
    
    def log_experiment_end(self, experiment_id: str, results: Dict[str, Any]) -> None:
        """
        Log the end of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            results: Experiment results
        """
        experiment_logger = logging.getLogger(f"experiment_{experiment_id}")
        experiment_logger.info(f"Experiment completed: {experiment_id}")
        experiment_logger.info(f"Results summary: {json.dumps(results, indent=2)}")
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        """
        Log evaluation metrics.
        
        Args:
            experiment_id: ID of the experiment
            metrics: Evaluation metrics
        """
        experiment_logger = logging.getLogger(f"experiment_{experiment_id}")
        experiment_logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Optional format string
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )


def save_results(
    results: Dict[str, Any], 
    output_path: Union[str, Path],
    format: str = "json"
) -> None:
    """
    Save experiment results to file.
    
    Args:
        results: Results to save
        output_path: Output file path
        format: Output format ('json', 'pickle')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert numpy arrays to lists for JSON serialization
        json_results = _convert_numpy_to_json(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    elif format == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(
    input_path: Union[str, Path],
    format: str = "json"
) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Args:
        input_path: Input file path
        format: Input format ('json', 'pickle')
        
    Returns:
        Loaded results
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    if format == "json":
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        results = _convert_json_to_numpy(results)
    
    elif format == "pickle":
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return results


def _convert_numpy_to_json(obj: Any) -> Any:
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_json(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


def _convert_json_to_numpy(obj: Any) -> Any:
    """Convert lists back to numpy arrays where appropriate."""
    if isinstance(obj, dict):
        return {key: _convert_json_to_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Check if this looks like a numpy array (list of numbers)
        if obj and isinstance(obj[0], (int, float)):
            return np.array(obj)
        else:
            return [_convert_json_to_numpy(item) for item in obj]
    else:
        return obj


def create_experiment_summary(results: Dict[str, Any]) -> str:
    """
    Create a text summary of experiment results.
    
    Args:
        results: Experiment results
        
    Returns:
        Text summary
    """
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("HIERARCHICAL TIME SERIES FORECASTING EXPERIMENT SUMMARY")
    summary_lines.append("=" * 60)
    
    # Data information
    if 'train_data' in results:
        train_data = results['train_data']
        summary_lines.append(f"\nData Shape: {train_data.shape[0]} training periods, {train_data.shape[1]} series")
    
    if 'test_data' in results:
        test_data = results['test_data']
        summary_lines.append(f"Test Data: {test_data.shape[0]} test periods")
    
    # Forecasting methods
    if 'forecast_results' in results:
        methods = list(results['forecast_results'].keys())
        summary_lines.append(f"\nForecasting Methods Used:")
        for method in methods:
            summary_lines.append(f"  - {method.upper()}")
    
    # Evaluation results
    if 'evaluation_results' in results:
        summary_lines.append(f"\nEvaluation Results:")
        for recon_name, evaluation in results['evaluation_results'].items():
            if evaluation:
                # Calculate average MAE across series
                mae_values = [metrics.get('MAE', np.inf) for metrics in evaluation.values()]
                avg_mae = np.mean(mae_values)
                summary_lines.append(f"  - {recon_name.replace('_', ' ').title()}: MAE = {avg_mae:.4f}")
    
    summary_lines.append("=" * 60)
    
    return "\n".join(summary_lines)
