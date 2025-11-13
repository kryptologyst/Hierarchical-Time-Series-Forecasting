"""
Forecasting models for hierarchical time series.

This module provides various forecasting methods including ARIMA, Prophet, and LSTM
for hierarchical time series forecasting.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

# Import with fallbacks for optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')


class BaseForecaster(ABC):
    """Abstract base class for forecasters."""
    
    def __init__(self, name: str):
        """
        Initialize the forecaster.
        
        Args:
            name: Name of the forecaster
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseForecaster':
        """
        Fit the forecasting model.
        
        Args:
            data: Training data
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def forecast(self, horizon: int) -> Dict[str, np.ndarray]:
        """
        Generate forecasts.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Dictionary of forecasts for each series
        """
        pass
    
    @abstractmethod
    def get_confidence_intervals(self, horizon: int, alpha: float = 0.05) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get confidence intervals for forecasts.
        
        Args:
            horizon: Forecast horizon
            alpha: Significance level
            
        Returns:
            Dictionary of confidence intervals for each series
        """
        pass


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA-based forecaster for hierarchical time series.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 0), seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        """
        super().__init__("ARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}
    
    def fit(self, data: pd.DataFrame) -> 'ARIMAForecaster':
        """
        Fit ARIMA models for each series.
        
        Args:
            data: Training data
            
        Returns:
            Self for method chaining
        """
        if ARIMA is None:
            raise ImportError("statsmodels is required for ARIMA forecasting. Install with: pip install statsmodels")
        
        self.logger.info(f"Fitting ARIMA models with order {self.order}")
        
        for column in data.columns:
            try:
                model = ARIMA(data[column], order=self.order, seasonal_order=self.seasonal_order)
                fitted_model = model.fit()
                self.models[column] = fitted_model
                self.logger.info(f"Fitted ARIMA model for {column}")
            except Exception as e:
                self.logger.error(f"Failed to fit ARIMA model for {column}: {e}")
                # Fallback to simpler model
                try:
                    model = ARIMA(data[column], order=(1, 1, 0))
                    fitted_model = model.fit()
                    self.models[column] = fitted_model
                    self.logger.info(f"Fitted fallback ARIMA model for {column}")
                except Exception as e2:
                    self.logger.error(f"Failed to fit fallback ARIMA model for {column}: {e2}")
        
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> Dict[str, np.ndarray]:
        """
        Generate ARIMA forecasts.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Dictionary of forecasts for each series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = {}
        for column, model in self.models.items():
            try:
                forecast = model.forecast(steps=horizon)
                forecasts[column] = forecast.values if hasattr(forecast, 'values') else forecast
            except Exception as e:
                self.logger.error(f"Failed to forecast for {column}: {e}")
                # Use naive forecast as fallback
                forecasts[column] = np.full(horizon, model.fittedvalues.iloc[-1])
        
        return forecasts
    
    def get_confidence_intervals(self, horizon: int, alpha: float = 0.05) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get ARIMA confidence intervals.
        
        Args:
            horizon: Forecast horizon
            alpha: Significance level
            
        Returns:
            Dictionary of confidence intervals for each series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        intervals = {}
        for column, model in self.models.items():
            try:
                forecast_result = model.get_forecast(steps=horizon)
                ci = forecast_result.conf_int(alpha=alpha)
                intervals[column] = (ci.iloc[:, 0].values, ci.iloc[:, 1].values)
            except Exception as e:
                self.logger.error(f"Failed to get confidence intervals for {column}: {e}")
                # Use naive intervals
                forecast_mean = self.forecast(horizon)[column]
                std_error = np.std(model.resid)
                margin = 1.96 * std_error  # Approximate 95% CI
                intervals[column] = (
                    forecast_mean - margin,
                    forecast_mean + margin
                )
        
        return intervals


class ProphetForecaster(BaseForecaster):
    """
    Prophet-based forecaster for hierarchical time series.
    """
    
    def __init__(self, **prophet_params):
        """
        Initialize Prophet forecaster.
        
        Args:
            **prophet_params: Prophet model parameters
        """
        super().__init__("Prophet")
        self.prophet_params = prophet_params
        self.models = {}
    
    def fit(self, data: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit Prophet models for each series.
        
        Args:
            data: Training data
            
        Returns:
            Self for method chaining
        """
        if Prophet is None:
            raise ImportError("prophet is required for Prophet forecasting. Install with: pip install prophet")
        
        self.logger.info("Fitting Prophet models")
        
        for column in data.columns:
            try:
                # Prepare data for Prophet
                prophet_data = pd.DataFrame({
                    'ds': data.index,
                    'y': data[column]
                })
                
                # Initialize and fit Prophet model
                model = Prophet(**self.prophet_params)
                model.fit(prophet_data)
                self.models[column] = model
                self.logger.info(f"Fitted Prophet model for {column}")
            except Exception as e:
                self.logger.error(f"Failed to fit Prophet model for {column}: {e}")
        
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> Dict[str, np.ndarray]:
        """
        Generate Prophet forecasts.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Dictionary of forecasts for each series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = {}
        for column, model in self.models.items():
            try:
                # Create future dataframe
                future = model.make_future_dataframe(periods=horizon, freq='M')
                
                # Generate forecast
                forecast_result = model.predict(future)
                
                # Extract forecast values
                forecast_values = forecast_result['yhat'].tail(horizon).values
                forecasts[column] = forecast_values
            except Exception as e:
                self.logger.error(f"Failed to forecast for {column}: {e}")
                # Use naive forecast as fallback
                forecasts[column] = np.full(horizon, model.history['y'].iloc[-1])
        
        return forecasts
    
    def get_confidence_intervals(self, horizon: int, alpha: float = 0.05) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get Prophet confidence intervals.
        
        Args:
            horizon: Forecast horizon
            alpha: Significance level
            
        Returns:
            Dictionary of confidence intervals for each series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        intervals = {}
        for column, model in self.models.items():
            try:
                # Create future dataframe
                future = model.make_future_dataframe(periods=horizon, freq='M')
                
                # Generate forecast
                forecast_result = model.predict(future)
                
                # Extract confidence intervals
                lower_col = f'yhat_lower_{int((1-alpha)*100)}'
                upper_col = f'yhat_upper_{int((1-alpha)*100)}'
                
                if lower_col in forecast_result.columns and upper_col in forecast_result.columns:
                    lower = forecast_result[lower_col].tail(horizon).values
                    upper = forecast_result[upper_col].tail(horizon).values
                else:
                    # Fallback to default confidence intervals
                    lower = forecast_result['yhat_lower'].tail(horizon).values
                    upper = forecast_result['yhat_upper'].tail(horizon).values
                
                intervals[column] = (lower, upper)
            except Exception as e:
                self.logger.error(f"Failed to get confidence intervals for {column}: {e}")
                # Use naive intervals
                forecast_mean = self.forecast(horizon)[column]
                std_error = np.std(model.history['y'])
                margin = 1.96 * std_error
                intervals[column] = (
                    forecast_mean - margin,
                    forecast_mean + margin
                )
        
        return intervals


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based forecaster for hierarchical time series.
    """
    
    def __init__(
        self,
        sequence_length: int = 12,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = {}
        self.scalers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _create_lstm_model(self, input_size: int) -> nn.Module:
        """Create LSTM model architecture."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.fc(lstm_out[:, -1, :])
                return output
        
        return LSTMModel(input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def fit(self, data: pd.DataFrame) -> 'LSTMForecaster':
        """
        Fit LSTM models for each series.
        
        Args:
            data: Training data
            
        Returns:
            Self for method chaining
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for LSTM forecasting. Install with: pip install torch")
        
        self.logger.info("Fitting LSTM models")
        
        for column in data.columns:
            try:
                # Prepare data
                series_data = data[column].values.reshape(-1, 1)
                
                # Scale data
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(series_data).flatten()
                self.scalers[column] = scaler
                
                # Create sequences
                X, y = self._create_sequences(scaled_data)
                
                if len(X) == 0:
                    self.logger.warning(f"Not enough data for LSTM training for {column}")
                    continue
                
                # Convert to tensors
                X_tensor = torch.FloatTensor(X).to(self.device)
                y_tensor = torch.FloatTensor(y).to(self.device)
                
                # Create model
                model = self._create_lstm_model(1).to(self.device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                
                # Train model
                model.train()
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs.squeeze(), y_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 20 == 0:
                        self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                
                self.models[column] = model
                self.logger.info(f"Fitted LSTM model for {column}")
                
            except Exception as e:
                self.logger.error(f"Failed to fit LSTM model for {column}: {e}")
        
        self.is_fitted = True
        return self
    
    def forecast(self, horizon: int) -> Dict[str, np.ndarray]:
        """
        Generate LSTM forecasts.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Dictionary of forecasts for each series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = {}
        for column, model in self.models.items():
            try:
                scaler = self.scalers[column]
                
                # Get last sequence for prediction
                last_sequence = scaler.transform(
                    self.last_data[column].values[-self.sequence_length:].reshape(-1, 1)
                ).flatten()
                
                # Generate forecasts recursively
                model.eval()
                with torch.no_grad():
                    current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
                    forecast_values = []
                    
                    for _ in range(horizon):
                        output = model(current_sequence)
                        forecast_values.append(output.item())
                        
                        # Update sequence for next prediction
                        new_value = output.item()
                        current_sequence = torch.cat([
                            current_sequence[:, 1:, :],
                            torch.FloatTensor([[new_value]]).unsqueeze(0).to(self.device)
                        ], dim=1)
                
                # Inverse transform forecasts
                forecast_array = np.array(forecast_values).reshape(-1, 1)
                forecast_scaled = scaler.inverse_transform(forecast_array).flatten()
                forecasts[column] = forecast_scaled
                
            except Exception as e:
                self.logger.error(f"Failed to forecast for {column}: {e}")
                # Use naive forecast as fallback
                forecasts[column] = np.full(horizon, self.last_data[column].iloc[-1])
        
        return forecasts
    
    def get_confidence_intervals(self, horizon: int, alpha: float = 0.05) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get LSTM confidence intervals (simplified approach).
        
        Args:
            horizon: Forecast horizon
            alpha: Significance level
            
        Returns:
            Dictionary of confidence intervals for each series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        intervals = {}
        forecasts = self.forecast(horizon)
        
        for column in forecasts:
            # Simple approach: use historical volatility
            historical_std = self.last_data[column].std()
            margin = 1.96 * historical_std  # Approximate 95% CI
            
            intervals[column] = (
                forecasts[column] - margin,
                forecasts[column] + margin
            )
        
        return intervals
    
    def fit(self, data: pd.DataFrame) -> 'LSTMForecaster':
        """
        Fit LSTM models for each series (updated to store last data).
        
        Args:
            data: Training data
            
        Returns:
            Self for method chaining
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for LSTM forecasting. Install with: pip install torch")
        
        self.last_data = data  # Store for forecasting
        self.logger.info("Fitting LSTM models")
        
        for column in data.columns:
            try:
                # Prepare data
                series_data = data[column].values.reshape(-1, 1)
                
                # Scale data
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(series_data).flatten()
                self.scalers[column] = scaler
                
                # Create sequences
                X, y = self._create_sequences(scaled_data)
                
                if len(X) == 0:
                    self.logger.warning(f"Not enough data for LSTM training for {column}")
                    continue
                
                # Convert to tensors
                X_tensor = torch.FloatTensor(X).to(self.device)
                y_tensor = torch.FloatTensor(y).to(self.device)
                
                # Create model
                model = self._create_lstm_model(1).to(self.device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                
                # Train model
                model.train()
                for epoch in range(self.epochs):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs.squeeze(), y_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 20 == 0:
                        self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                
                self.models[column] = model
                self.logger.info(f"Fitted LSTM model for {column}")
                
            except Exception as e:
                self.logger.error(f"Failed to fit LSTM model for {column}: {e}")
        
        self.is_fitted = True
        return self
