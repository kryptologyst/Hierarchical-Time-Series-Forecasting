"""
Tests for data generation module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import DataGenerator, HierarchyConfig


class TestHierarchyConfig:
    """Test HierarchyConfig dataclass."""
    
    def test_hierarchy_config_creation(self):
        """Test creating a hierarchy configuration."""
        config = HierarchyConfig("Region1", 100.0, 5.0, 0.02)
        assert config.name == "Region1"
        assert config.base_value == 100.0
        assert config.volatility == 5.0
        assert config.trend == 0.02


class TestDataGenerator:
    """Test DataGenerator class."""
    
    @pytest.fixture
    def data_generator(self):
        """Create a data generator instance."""
        return DataGenerator(seed=42)
    
    @pytest.fixture
    def hierarchy_configs(self):
        """Create sample hierarchy configurations."""
        return [
            HierarchyConfig("Region1", 100.0, 5.0, 0.02),
            HierarchyConfig("Region2", 120.0, 6.0, 0.015),
            HierarchyConfig("Region3", 80.0, 4.0, 0.025)
        ]
    
    def test_data_generator_initialization(self, data_generator):
        """Test data generator initialization."""
        assert data_generator.seed == 42
        assert data_generator.logger is not None
    
    def test_generate_hierarchical_data(self, data_generator, hierarchy_configs):
        """Test hierarchical data generation."""
        data = data_generator.generate_hierarchical_data(
            periods=24,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        
        # Check data shape
        assert len(data) == 24
        assert len(data.columns) == 4  # 3 regions + Total
        
        # Check column names
        expected_columns = ["Region1", "Region2", "Region3", "Total"]
        assert list(data.columns) == expected_columns
        
        # Check that Total is sum of regions
        total_calculated = data["Region1"] + data["Region2"] + data["Region3"]
        np.testing.assert_array_almost_equal(data["Total"], total_calculated)
        
        # Check that all values are non-negative
        assert (data >= 0).all().all()
    
    def test_generate_series(self, data_generator):
        """Test single series generation."""
        series = data_generator._generate_series(
            periods=12,
            base_value=100.0,
            volatility=5.0,
            trend=0.02,
            add_seasonality=True,
            seasonal_period=12
        )
        
        assert len(series) == 12
        assert all(val >= 0 for val in series)  # Non-negative values
    
    def test_add_anomalies(self, data_generator, hierarchy_configs):
        """Test adding anomalies to data."""
        # Generate base data
        data = data_generator.generate_hierarchical_data(
            periods=24,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        
        # Add anomalies
        anomalous_data = data_generator.add_anomalies(data, anomaly_probability=0.1)
        
        # Check shape is preserved
        assert anomalous_data.shape == data.shape
        
        # Check that Total is still sum of regions
        total_calculated = anomalous_data["Region1"] + anomalous_data["Region2"] + anomalous_data["Region3"]
        np.testing.assert_array_almost_equal(anomalous_data["Total"], total_calculated)
        
        # Check that all values are still non-negative
        assert (anomalous_data >= 0).all().all()
    
    def test_get_hierarchy_structure(self, data_generator, hierarchy_configs):
        """Test getting hierarchy structure."""
        data = data_generator.generate_hierarchical_data(
            periods=24,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        
        structure = data_generator.get_hierarchy_structure(data)
        
        assert "bottom_level" in structure
        assert "top_level" in structure
        assert len(structure["bottom_level"]) == 3
        assert len(structure["top_level"]) == 1
        assert structure["top_level"][0] == "Total"
    
    def test_reproducibility(self):
        """Test that data generation is reproducible with same seed."""
        generator1 = DataGenerator(seed=42)
        generator2 = DataGenerator(seed=42)
        
        hierarchy_configs = [
            HierarchyConfig("Region1", 100.0, 5.0, 0.02)
        ]
        
        data1 = generator1.generate_hierarchical_data(
            periods=12,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        
        data2 = generator2.generate_hierarchical_data(
            periods=12,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        
        np.testing.assert_array_almost_equal(data1.values, data2.values)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        generator1 = DataGenerator(seed=42)
        generator2 = DataGenerator(seed=123)
        
        hierarchy_configs = [
            HierarchyConfig("Region1", 100.0, 5.0, 0.02)
        ]
        
        data1 = generator1.generate_hierarchical_data(
            periods=12,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        
        data2 = generator2.generate_hierarchical_data(
            periods=12,
            start_date="2020-01-01",
            frequency="M",
            hierarchy_configs=hierarchy_configs
        )
        
        # Data should be different (very unlikely to be identical)
        assert not np.allclose(data1.values, data2.values)
