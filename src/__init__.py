"""
Hierarchical Time Series Forecasting Package

A comprehensive package for hierarchical time series forecasting with support for
multiple forecasting methods, reconciliation techniques, and evaluation metrics.
"""

__version__ = "1.0.0"
__author__ = "Time Series Analysis Team"
__email__ = "team@timeseries.com"

from .data_generator import DataGenerator
from .reconciliation import ReconciliationEngine
from .evaluation import ModelEvaluator
from .visualization import Plotter

# Import forecasters with fallbacks
try:
    from .forecasters import ARIMAForecaster, ProphetForecaster, LSTMForecaster
    __all__ = [
        "DataGenerator",
        "ARIMAForecaster", 
        "ProphetForecaster",
        "LSTMForecaster",
        "ReconciliationEngine",
        "ModelEvaluator",
        "Plotter"
    ]
except ImportError as e:
    # If some forecasters fail to import, include only the ones that work
    available_forecasters = []
    try:
        from .forecasters import ARIMAForecaster
        available_forecasters.append("ARIMAForecaster")
    except ImportError:
        pass
    
    try:
        from .forecasters import ProphetForecaster
        available_forecasters.append("ProphetForecaster")
    except ImportError:
        pass
    
    try:
        from .forecasters import LSTMForecaster
        available_forecasters.append("LSTMForecaster")
    except ImportError:
        pass
    
    __all__ = [
        "DataGenerator",
        "ReconciliationEngine",
        "ModelEvaluator",
        "Plotter"
    ] + available_forecasters
