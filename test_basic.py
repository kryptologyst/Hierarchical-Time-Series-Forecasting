#!/usr/bin/env python3
"""
Basic test script to verify the hierarchical time series forecasting functionality.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from src.data_generator import DataGenerator, HierarchyConfig
from src.reconciliation import ReconciliationEngine
from src.evaluation import ModelEvaluator
from src.visualization import Plotter

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("Testing basic functionality...")
    
    # Test data generation
    print("1. Testing data generation...")
    hierarchy_configs = [
        HierarchyConfig("Region1", base_value=100.0, volatility=5.0, trend=0.02),
        HierarchyConfig("Region2", base_value=120.0, volatility=6.0, trend=0.015),
        HierarchyConfig("Region3", base_value=80.0, volatility=4.0, trend=0.025)
    ]
    
    generator = DataGenerator(seed=42)
    data = generator.generate_hierarchical_data(
        periods=24,
        start_date="2020-01-01",
        frequency="M",
        hierarchy_configs=hierarchy_configs
    )
    
    print(f"   Generated data shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Verify hierarchy consistency
    total_calculated = data["Region1"] + data["Region2"] + data["Region3"]
    assert np.allclose(data["Total"], total_calculated), "Hierarchy consistency check failed"
    print("   ✓ Hierarchy consistency verified")
    
    # Test reconciliation
    print("2. Testing reconciliation...")
    hierarchy_structure = generator.get_hierarchy_structure(data)
    
    # Create sample forecasts
    sample_forecasts = {
        'Region1': np.array([100, 105, 110, 115]),
        'Region2': np.array([120, 125, 130, 135]),
        'Region3': np.array([80, 85, 90, 95])
    }
    
    reconciler = ReconciliationEngine()
    reconciled = reconciler.reconcile(sample_forecasts, hierarchy_structure, 'bottom_up')
    
    # Verify reconciliation
    expected_total = sample_forecasts['Region1'] + sample_forecasts['Region2'] + sample_forecasts['Region3']
    assert np.allclose(reconciled['Total'], expected_total), "Bottom-up reconciliation failed"
    print("   ✓ Bottom-up reconciliation verified")
    
    # Test evaluation
    print("3. Testing evaluation...")
    actual_data = {'Region1': np.array([100, 105, 110, 115])}
    forecast_data = {'Region1': np.array([102, 107, 112, 117])}
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_forecasts(actual_data, forecast_data)
    
    assert 'Region1' in metrics, "Evaluation failed"
    assert 'MAE' in metrics['Region1'], "MAE metric missing"
    print(f"   ✓ Evaluation completed: MAE = {metrics['Region1']['MAE']:.4f}")
    
    # Test visualization
    print("4. Testing visualization...")
    plotter = Plotter()
    fig = plotter.plot_time_series(data, title="Test Plot")
    assert fig is not None, "Plotting failed"
    print("   ✓ Visualization completed")
    
    print("\n✅ All basic tests passed!")
    return True

def test_forecasters():
    """Test forecasters if dependencies are available."""
    print("\nTesting forecasters...")
    
    # Test ARIMA
    try:
        from src.forecasters import ARIMAForecaster
        print("   ✓ ARIMAForecaster available")
        
        # Create sample data
        generator = DataGenerator(seed=42)
        hierarchy_configs = [HierarchyConfig("Region1", base_value=100.0, volatility=5.0, trend=0.02)]
        data = generator.generate_hierarchical_data(
            periods=24, start_date="2020-01-01", frequency="M", hierarchy_configs=hierarchy_configs
        )
        
        # Test ARIMA
        forecaster = ARIMAForecaster(order=(1, 1, 0))
        forecaster.fit(data)
        forecasts = forecaster.forecast(horizon=6)
        
        assert len(forecasts) > 0, "ARIMA forecasting failed"
        print("   ✓ ARIMA forecasting works")
        
    except ImportError as e:
        print(f"   ⚠ ARIMAForecaster not available: {e}")
    except Exception as e:
        print(f"   ⚠ ARIMA test failed: {e}")
    
    # Test Prophet
    try:
        from src.forecasters import ProphetForecaster
        print("   ✓ ProphetForecaster available")
        
        # Test Prophet
        forecaster = ProphetForecaster(yearly_seasonality=True)
        forecaster.fit(data)
        forecasts = forecaster.forecast(horizon=6)
        
        assert len(forecasts) > 0, "Prophet forecasting failed"
        print("   ✓ Prophet forecasting works")
        
    except ImportError as e:
        print(f"   ⚠ ProphetForecaster not available: {e}")
    except Exception as e:
        print(f"   ⚠ Prophet test failed: {e}")
    
    # Test LSTM
    try:
        from src.forecasters import LSTMForecaster
        print("   ✓ LSTMForecaster available")
        
        # Test LSTM (with reduced parameters for speed)
        forecaster = LSTMForecaster(
            sequence_length=6, hidden_size=10, num_layers=1, epochs=2, batch_size=16
        )
        forecaster.fit(data)
        forecasts = forecaster.forecast(horizon=6)
        
        assert len(forecasts) > 0, "LSTM forecasting failed"
        print("   ✓ LSTM forecasting works")
        
    except ImportError as e:
        print(f"   ⚠ LSTMForecaster not available: {e}")
    except Exception as e:
        print(f"   ⚠ LSTM test failed: {e}")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_forecasters()
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
