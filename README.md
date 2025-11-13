# Hierarchical Time Series Forecasting

A comprehensive Python package for hierarchical time series forecasting with support for multiple forecasting methods, reconciliation techniques, and evaluation metrics.

## Overview

This project provides a modern, well-structured implementation of hierarchical time series forecasting methods. It supports various forecasting algorithms (ARIMA, Prophet, LSTM) and reconciliation techniques (bottom-up, top-down, middle-out, optimal) to ensure consistency across different levels of the hierarchy.

## Features

- **Multiple Forecasting Methods**: ARIMA, Prophet, and LSTM models
- **Reconciliation Techniques**: Bottom-up, top-down, middle-out, and optimal reconciliation
- **Comprehensive Evaluation**: MAE, MSE, RMSE, MAPE, sMAPE, and MASE metrics
- **Interactive Web Interface**: Streamlit-based dashboard for exploration
- **Advanced Visualization**: Matplotlib, Seaborn, and Plotly plots
- **Configuration Management**: YAML-based configuration system
- **Logging and Checkpointing**: Experiment tracking and model persistence
- **Unit Testing**: Comprehensive test suite
- **Type Hints**: Full type annotation support
- **Documentation**: Detailed docstrings and examples

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

Run the complete forecasting pipeline:

```bash
python -m src.main
```

### Python API Usage

```python
from src.main import HierarchicalForecastingPipeline

# Initialize pipeline
pipeline = HierarchicalForecastingPipeline()

# Run complete experiment
results = pipeline.run_complete_pipeline()
```

### Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## Project Structure

```
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main pipeline
│   ├── data_generator.py        # Data generation utilities
│   ├── forecasters.py           # Forecasting models
│   ├── reconciliation.py       # Reconciliation methods
│   ├── evaluation.py            # Evaluation metrics
│   ├── visualization.py        # Plotting utilities
│   └── utils.py                # Utility functions
├── tests/                       # Unit tests
│   ├── test_data_generator.py
│   ├── test_forecasters.py
│   └── test_reconciliation.py
├── config/                      # Configuration files
│   └── config.yaml             # Main configuration
├── data/                        # Data storage
├── models/                      # Model checkpoints
├── logs/                        # Log files
├── plots/                       # Generated plots
├── notebooks/                   # Jupyter notebooks
├── streamlit_app.py            # Streamlit interface
├── requirements.txt             # Dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Configuration

The project uses YAML configuration files. See `config/config.yaml` for the main configuration:

```yaml
# Data generation parameters
data:
  seed: 42
  periods: 60
  start_date: "2020-01-01"
  frequency: "M"
  
  hierarchy:
    - name: "Region1"
      base_value: 100
      volatility: 5
      trend: 0.02

# Forecasting parameters
forecasting:
  horizon: 12
  methods: ["arima", "prophet"]
  
  arima:
    order: [1, 1, 0]
    seasonal_order: [0, 0, 0, 0]
  
  prophet:
    yearly_seasonality: true
    weekly_seasonality: false
    daily_seasonality: false

# Reconciliation methods
reconciliation:
  methods: ["bottom_up", "top_down"]
```

## Usage Examples

### Basic Forecasting

```python
from src.data_generator import DataGenerator, HierarchyConfig
from src.forecasters import ARIMAForecaster
from src.reconciliation import ReconciliationEngine

# Generate data
generator = DataGenerator(seed=42)
hierarchy_configs = [
    HierarchyConfig("Region1", 100.0, 5.0, 0.02),
    HierarchyConfig("Region2", 120.0, 6.0, 0.015),
    HierarchyConfig("Region3", 80.0, 4.0, 0.025)
]

data = generator.generate_hierarchical_data(
    periods=60,
    start_date="2020-01-01",
    frequency="M",
    hierarchy_configs=hierarchy_configs
)

# Train forecaster
forecaster = ARIMAForecaster(order=(1, 1, 0))
forecaster.fit(data)

# Generate forecasts
forecasts = forecaster.forecast(horizon=12)

# Reconcile forecasts
reconciler = ReconciliationEngine()
hierarchy_structure = generator.get_hierarchy_structure(data)
reconciled_forecasts = reconciler.reconcile(forecasts, hierarchy_structure, 'bottom_up')
```

### Model Evaluation

```python
from src.evaluation import ModelEvaluator
import numpy as np

# Prepare test data
test_data = {'Region1': np.array([100, 105, 110, 115])}
forecast_data = {'Region1': np.array([102, 107, 112, 117])}

# Evaluate forecasts
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_forecasts(test_data, forecast_data)

print(metrics)
# {'Region1': {'MAE': 2.0, 'MSE': 4.0, 'RMSE': 2.0, 'MAPE': 1.82, 'sMAPE': 1.85}}
```

### Visualization

```python
from src.visualization import Plotter
import matplotlib.pyplot as plt

# Create plots
plotter = Plotter()
fig = plotter.plot_forecasts(
    historical_data=data,
    forecasts=reconciled_forecasts,
    forecast_dates=pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M'),
    title="Hierarchical Forecasts"
)

plt.show()
```

## Forecasting Methods

### ARIMA
- **Description**: AutoRegressive Integrated Moving Average
- **Parameters**: Order (p, d, q) and seasonal order (P, D, Q, s)
- **Use Case**: Linear time series with trends and seasonality

### Prophet
- **Description**: Facebook's forecasting tool
- **Parameters**: Seasonality settings, holidays, changepoints
- **Use Case**: Business time series with strong seasonality

### LSTM
- **Description**: Long Short-Term Memory neural network
- **Parameters**: Sequence length, hidden size, layers, dropout
- **Use Case**: Complex non-linear patterns

## Reconciliation Methods

### Bottom-Up
- Forecast at bottom level, aggregate upward
- Preserves bottom-level patterns
- Simple and intuitive

### Top-Down
- Forecast at top level, disaggregate downward
- Uses historical proportions
- Good for stable hierarchies

### Middle-Out
- Forecast at middle level, reconcile both ways
- Balances top-down and bottom-up approaches
- Flexible hierarchy handling

### Optimal
- Minimizes sum of squared errors
- Maintains hierarchy consistency
- Mathematically optimal

## Evaluation Metrics

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric MAPE
- **MASE**: Mean Absolute Scaled Error

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_forecasters.py

# Run with verbose output
pytest -v
```

## Web Interface

The Streamlit interface provides:

- **Data Overview**: Generate and explore hierarchical time series data
- **Forecasting Interface**: Configure and run forecasting experiments
- **Results Visualization**: Interactive plots and metrics
- **Configuration Management**: Download and modify configurations

Launch with:
```bash
streamlit run streamlit_app.py
```

## Logging and Checkpointing

The system provides comprehensive logging and checkpointing:

```python
from src.utils import CheckpointManager, ExperimentLogger

# Save model checkpoint
checkpoint_manager = CheckpointManager()
checkpoint_manager.save_checkpoint(
    model=forecaster,
    metadata={'method': 'ARIMA', 'order': (1, 1, 0)},
    checkpoint_name='arima_model'
)

# Load checkpoint
checkpoint_data = checkpoint_manager.load_checkpoint('path/to/checkpoint.pkl')
model = checkpoint_data['model']

# Log experiment
logger = ExperimentLogger()
experiment_id = logger.log_experiment_start('my_experiment', config)
logger.log_metrics(experiment_id, metrics)
logger.log_experiment_end(experiment_id, results)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Statsmodels for ARIMA implementation
- Facebook Prophet for Prophet implementation
- PyTorch for LSTM implementation
- Streamlit for web interface
- Plotly for interactive visualizations

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation in the `src/` directory
- Review the test files for usage examples

## Roadmap

- [ ] Add more forecasting methods (XGBoost, NeuralProphet)
- [ ] Implement advanced reconciliation (MinT, ERM)
- [ ] Add anomaly detection capabilities
- [ ] Support for external datasets
- [ ] Docker containerization
- [ ] Cloud deployment options
- [ ] Real-time forecasting API
# Hierarchical-Time-Series-Forecasting
