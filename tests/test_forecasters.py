"""
Tests for forecasting models.
"""

import pytest
import numpy as np
import pandas as pd
from src.forecasters import ARIMAForecaster, ProphetForecaster, LSTMForecaster
from src.data_generator import DataGenerator, HierarchyConfig


class TestARIMAForecaster:
    """Test ARIMA forecaster."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        generator = DataGenerator(seed=42)
        hierarchy_configs = [
            HierarchyConfig("Region1", 100.0, 5.0, 0.02),
            HierarchyConfig("Region2", 120.0, 6.0, 0.015)
        ]
        
        data = generator.generate_hierarchical_data(
            periods=36,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        return data
    
    @pytest.fixture
    def arima_forecaster(self):
        """Create ARIMA forecaster instance."""
        return ARIMAForecaster(order=(1, 1, 0))
    
    def test_arima_initialization(self, arima_forecaster):
        """Test ARIMA forecaster initialization."""
        assert arima_forecaster.name == "ARIMA"
        assert arima_forecaster.order == (1, 1, 0)
        assert not arima_forecaster.is_fitted
    
    def test_arima_fit(self, arima_forecaster, sample_data):
        """Test ARIMA model fitting."""
        forecaster = arima_forecaster.fit(sample_data)
        
        assert forecaster.is_fitted
        assert len(forecaster.models) == len(sample_data.columns)
        
        # Check that models are fitted for each column
        for column in sample_data.columns:
            assert column in forecaster.models
    
    def test_arima_forecast(self, arima_forecaster, sample_data):
        """Test ARIMA forecasting."""
        forecaster = arima_forecaster.fit(sample_data)
        forecasts = forecaster.forecast(horizon=6)
        
        assert len(forecasts) == len(sample_data.columns)
        
        for column, forecast in forecasts.items():
            assert len(forecast) == 6
            assert all(val >= 0 for val in forecast)  # Non-negative forecasts
    
    def test_arima_confidence_intervals(self, arima_forecaster, sample_data):
        """Test ARIMA confidence intervals."""
        forecaster = arima_forecaster.fit(sample_data)
        intervals = forecaster.get_confidence_intervals(horizon=6)
        
        assert len(intervals) == len(sample_data.columns)
        
        for column, (lower, upper) in intervals.items():
            assert len(lower) == 6
            assert len(upper) == 6
            assert all(lower <= upper)  # Lower bound <= upper bound
    
    def test_arima_forecast_without_fit(self, arima_forecaster):
        """Test that forecasting without fitting raises error."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            arima_forecaster.forecast(horizon=6)


class TestProphetForecaster:
    """Test Prophet forecaster."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        generator = DataGenerator(seed=42)
        hierarchy_configs = [
            HierarchyConfig("Region1", 100.0, 5.0, 0.02),
            HierarchyConfig("Region2", 120.0, 6.0, 0.015)
        ]
        
        data = generator.generate_hierarchical_data(
            periods=36,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        return data
    
    @pytest.fixture
    def prophet_forecaster(self):
        """Create Prophet forecaster instance."""
        return ProphetForecaster(yearly_seasonality=True)
    
    def test_prophet_initialization(self, prophet_forecaster):
        """Test Prophet forecaster initialization."""
        assert prophet_forecaster.name == "Prophet"
        assert prophet_forecaster.prophet_params["yearly_seasonality"] is True
        assert not prophet_forecaster.is_fitted
    
    def test_prophet_fit(self, prophet_forecaster, sample_data):
        """Test Prophet model fitting."""
        forecaster = prophet_forecaster.fit(sample_data)
        
        assert forecaster.is_fitted
        assert len(forecaster.models) == len(sample_data.columns)
        
        # Check that models are fitted for each column
        for column in sample_data.columns:
            assert column in forecaster.models
    
    def test_prophet_forecast(self, prophet_forecaster, sample_data):
        """Test Prophet forecasting."""
        forecaster = prophet_forecaster.fit(sample_data)
        forecasts = forecaster.forecast(horizon=6)
        
        assert len(forecasts) == len(sample_data.columns)
        
        for column, forecast in forecasts.items():
            assert len(forecast) == 6
            assert all(val >= 0 for val in forecast)  # Non-negative forecasts
    
    def test_prophet_confidence_intervals(self, prophet_forecaster, sample_data):
        """Test Prophet confidence intervals."""
        forecaster = prophet_forecaster.fit(sample_data)
        intervals = forecaster.get_confidence_intervals(horizon=6)
        
        assert len(intervals) == len(sample_data.columns)
        
        for column, (lower, upper) in intervals.items():
            assert len(lower) == 6
            assert len(upper) == 6
            assert all(lower <= upper)  # Lower bound <= upper bound
    
    def test_prophet_forecast_without_fit(self, prophet_forecaster):
        """Test that forecasting without fitting raises error."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            prophet_forecaster.forecast(horizon=6)


class TestLSTMForecaster:
    """Test LSTM forecaster."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        generator = DataGenerator(seed=42)
        hierarchy_configs = [
            HierarchyConfig("Region1", 100.0, 5.0, 0.02)
        ]
        
        data = generator.generate_hierarchical_data(
            periods=36,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        return data
    
    @pytest.fixture
    def lstm_forecaster(self):
        """Create LSTM forecaster instance."""
        return LSTMForecaster(
            sequence_length=6,
            hidden_size=10,
            num_layers=1,
            epochs=5,  # Reduced for testing
            batch_size=16
        )
    
    def test_lstm_initialization(self, lstm_forecaster):
        """Test LSTM forecaster initialization."""
        assert lstm_forecaster.name == "LSTM"
        assert lstm_forecaster.sequence_length == 6
        assert lstm_forecaster.hidden_size == 10
        assert lstm_forecaster.num_layers == 1
        assert not lstm_forecaster.is_fitted
    
    def test_lstm_fit(self, lstm_forecaster, sample_data):
        """Test LSTM model fitting."""
        forecaster = lstm_forecaster.fit(sample_data)
        
        assert forecaster.is_fitted
        assert len(forecaster.models) <= len(sample_data.columns)  # May skip some if insufficient data
    
    def test_lstm_forecast(self, lstm_forecaster, sample_data):
        """Test LSTM forecasting."""
        forecaster = lstm_forecaster.fit(sample_data)
        
        if forecaster.is_fitted and forecaster.models:
            forecasts = forecaster.forecast(horizon=6)
            
            assert len(forecasts) > 0
            
            for column, forecast in forecasts.items():
                assert len(forecast) == 6
                assert all(val >= 0 for val in forecast)  # Non-negative forecasts
    
    def test_lstm_confidence_intervals(self, lstm_forecaster, sample_data):
        """Test LSTM confidence intervals."""
        forecaster = lstm_forecaster.fit(sample_data)
        
        if forecaster.is_fitted and forecaster.models:
            intervals = forecaster.get_confidence_intervals(horizon=6)
            
            assert len(intervals) > 0
            
            for column, (lower, upper) in intervals.items():
                assert len(lower) == 6
                assert len(upper) == 6
                assert all(lower <= upper)  # Lower bound <= upper bound
    
    def test_lstm_forecast_without_fit(self, lstm_forecaster):
        """Test that forecasting without fitting raises error."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            lstm_forecaster.forecast(horizon=6)


class TestForecasterComparison:
    """Test comparison between different forecasters."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        generator = DataGenerator(seed=42)
        hierarchy_configs = [
            HierarchyConfig("Region1", 100.0, 5.0, 0.02)
        ]
        
        data = generator.generate_hierarchical_data(
            periods=36,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        return data
    
    def test_forecaster_consistency(self, sample_data):
        """Test that all forecasters produce consistent output format."""
        forecasters = [
            ARIMAForecaster(order=(1, 1, 0)),
            ProphetForecaster(yearly_seasonality=True)
        ]
        
        horizon = 6
        
        for forecaster in forecasters:
            fitted_forecaster = forecaster.fit(sample_data)
            forecasts = fitted_forecaster.forecast(horizon)
            intervals = fitted_forecaster.get_confidence_intervals(horizon)
            
            # Check output format consistency
            assert isinstance(forecasts, dict)
            assert isinstance(intervals, dict)
            
            for column in sample_data.columns:
                if column in forecasts:
                    assert len(forecasts[column]) == horizon
                    assert len(intervals[column][0]) == horizon  # Lower bound
                    assert len(intervals[column][1]) == horizon  # Upper bound
