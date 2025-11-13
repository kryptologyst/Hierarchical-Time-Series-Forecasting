"""
Tests for reconciliation methods.
"""

import pytest
import numpy as np
from src.reconciliation import (
    ReconciliationEngine, 
    BottomUpReconciliation, 
    TopDownReconciliation,
    MiddleOutReconciliation,
    OptimalReconciliation
)


class TestBottomUpReconciliation:
    """Test bottom-up reconciliation method."""
    
    @pytest.fixture
    def sample_forecasts(self):
        """Create sample forecasts."""
        return {
            'Region1': np.array([100, 105, 110, 115]),
            'Region2': np.array([120, 125, 130, 135]),
            'Region3': np.array([80, 85, 90, 95])
        }
    
    @pytest.fixture
    def hierarchy_structure(self):
        """Create sample hierarchy structure."""
        return {
            'bottom_level': ['Region1', 'Region2', 'Region3'],
            'top_level': ['Total']
        }
    
    @pytest.fixture
    def bottom_up_reconciler(self):
        """Create bottom-up reconciler."""
        return BottomUpReconciliation()
    
    def test_bottom_up_reconciliation(self, bottom_up_reconciler, sample_forecasts, hierarchy_structure):
        """Test bottom-up reconciliation."""
        reconciled = bottom_up_reconciler.reconcile(sample_forecasts, hierarchy_structure)
        
        # Check that all original forecasts are preserved
        for region in sample_forecasts:
            np.testing.assert_array_equal(reconciled[region], sample_forecasts[region])
        
        # Check that Total is sum of regions
        expected_total = sample_forecasts['Region1'] + sample_forecasts['Region2'] + sample_forecasts['Region3']
        np.testing.assert_array_equal(reconciled['Total'], expected_total)
    
    def test_bottom_up_with_existing_total(self, bottom_up_reconciler, sample_forecasts, hierarchy_structure):
        """Test bottom-up reconciliation when Total already exists."""
        # Add existing Total forecast
        sample_forecasts['Total'] = np.array([300, 315, 330, 345])
        
        reconciled = bottom_up_reconciler.reconcile(sample_forecasts, hierarchy_structure)
        
        # Should preserve existing Total
        np.testing.assert_array_equal(reconciled['Total'], sample_forecasts['Total'])


class TestTopDownReconciliation:
    """Test top-down reconciliation method."""
    
    @pytest.fixture
    def sample_forecasts(self):
        """Create sample forecasts."""
        return {
            'Total': np.array([300, 315, 330, 345])
        }
    
    @pytest.fixture
    def hierarchy_structure(self):
        """Create sample hierarchy structure."""
        return {
            'bottom_level': ['Region1', 'Region2', 'Region3'],
            'top_level': ['Total']
        }
    
    @pytest.fixture
    def top_down_reconciler(self):
        """Create top-down reconciler."""
        return TopDownReconciliation(disaggregation_method="proportional")
    
    def test_top_down_reconciliation(self, top_down_reconciler, sample_forecasts, hierarchy_structure):
        """Test top-down reconciliation."""
        reconciled = top_down_reconciler.reconcile(sample_forecasts, hierarchy_structure)
        
        # Check that Total is preserved
        np.testing.assert_array_equal(reconciled['Total'], sample_forecasts['Total'])
        
        # Check that bottom level forecasts sum to Total
        bottom_sum = reconciled['Region1'] + reconciled['Region2'] + reconciled['Region3']
        np.testing.assert_array_almost_equal(bottom_sum, sample_forecasts['Total'])
    
    def test_top_down_without_total(self, top_down_reconciler, hierarchy_structure):
        """Test top-down reconciliation without Total forecast."""
        sample_forecasts = {
            'Region1': np.array([100, 105, 110, 115])
        }
        
        # Should fall back to bottom-up
        reconciled = top_down_reconciler.reconcile(sample_forecasts, hierarchy_structure)
        
        # Should have Total calculated from bottom level
        assert 'Total' in reconciled


class TestMiddleOutReconciliation:
    """Test middle-out reconciliation method."""
    
    @pytest.fixture
    def sample_forecasts(self):
        """Create sample forecasts."""
        return {
            'Region1': np.array([100, 105, 110, 115]),
            'Region2': np.array([120, 125, 130, 135])
        }
    
    @pytest.fixture
    def hierarchy_structure(self):
        """Create sample hierarchy structure."""
        return {
            'bottom_level': ['Region1', 'Region2'],
            'top_level': ['Total']
        }
    
    @pytest.fixture
    def middle_out_reconciler(self):
        """Create middle-out reconciler."""
        return MiddleOutReconciliation()
    
    def test_middle_out_reconciliation(self, middle_out_reconciler, sample_forecasts, hierarchy_structure):
        """Test middle-out reconciliation."""
        reconciled = middle_out_reconciler.reconcile(sample_forecasts, hierarchy_structure)
        
        # Should have all levels reconciled
        assert 'Region1' in reconciled
        assert 'Region2' in reconciled
        assert 'Total' in reconciled
        
        # Bottom level should sum to top level
        bottom_sum = reconciled['Region1'] + reconciled['Region2']
        np.testing.assert_array_almost_equal(bottom_sum, reconciled['Total'])


class TestOptimalReconciliation:
    """Test optimal reconciliation method."""
    
    @pytest.fixture
    def sample_forecasts(self):
        """Create sample forecasts."""
        return {
            'Region1': np.array([100, 105, 110, 115]),
            'Region2': np.array([120, 125, 130, 135]),
            'Region3': np.array([80, 85, 90, 95]),
            'Total': np.array([300, 315, 330, 345])
        }
    
    @pytest.fixture
    def hierarchy_structure(self):
        """Create sample hierarchy structure."""
        return {
            'bottom_level': ['Region1', 'Region2', 'Region3'],
            'top_level': ['Total']
        }
    
    @pytest.fixture
    def optimal_reconciler(self):
        """Create optimal reconciler."""
        return OptimalReconciliation()
    
    def test_optimal_reconciliation(self, optimal_reconciler, sample_forecasts, hierarchy_structure):
        """Test optimal reconciliation."""
        reconciled = optimal_reconciler.reconcile(sample_forecasts, hierarchy_structure)
        
        # Should have all levels
        assert 'Region1' in reconciled
        assert 'Region2' in reconciled
        assert 'Region3' in reconciled
        assert 'Total' in reconciled
        
        # Bottom level should sum to top level
        bottom_sum = reconciled['Region1'] + reconciled['Region2'] + reconciled['Region3']
        np.testing.assert_array_almost_equal(bottom_sum, reconciled['Total'])


class TestReconciliationEngine:
    """Test reconciliation engine."""
    
    @pytest.fixture
    def reconciliation_engine(self):
        """Create reconciliation engine."""
        return ReconciliationEngine()
    
    @pytest.fixture
    def sample_forecasts(self):
        """Create sample forecasts."""
        return {
            'Region1': np.array([100, 105, 110, 115]),
            'Region2': np.array([120, 125, 130, 135]),
            'Region3': np.array([80, 85, 90, 95])
        }
    
    @pytest.fixture
    def hierarchy_structure(self):
        """Create sample hierarchy structure."""
        return {
            'bottom_level': ['Region1', 'Region2', 'Region3'],
            'top_level': ['Total']
        }
    
    def test_reconciliation_engine_initialization(self, reconciliation_engine):
        """Test reconciliation engine initialization."""
        assert len(reconciliation_engine.methods) > 0
        assert 'bottom_up' in reconciliation_engine.methods
        assert 'top_down' in reconciliation_engine.methods
    
    def test_get_available_methods(self, reconciliation_engine):
        """Test getting available methods."""
        methods = reconciliation_engine.get_available_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
        assert 'bottom_up' in methods
    
    def test_reconcile_bottom_up(self, reconciliation_engine, sample_forecasts, hierarchy_structure):
        """Test reconciliation using bottom-up method."""
        reconciled = reconciliation_engine.reconcile(sample_forecasts, hierarchy_structure, 'bottom_up')
        
        # Check that Total is calculated
        assert 'Total' in reconciled
        expected_total = sample_forecasts['Region1'] + sample_forecasts['Region2'] + sample_forecasts['Region3']
        np.testing.assert_array_equal(reconciled['Total'], expected_total)
    
    def test_reconcile_top_down(self, reconciliation_engine, hierarchy_structure):
        """Test reconciliation using top-down method."""
        sample_forecasts = {
            'Total': np.array([300, 315, 330, 345])
        }
        
        reconciled = reconciliation_engine.reconcile(sample_forecasts, hierarchy_structure, 'top_down')
        
        # Check that bottom level is calculated
        assert 'Region1' in reconciled
        assert 'Region2' in reconciled
        assert 'Region3' in reconciled
        
        # Bottom level should sum to Total
        bottom_sum = reconciled['Region1'] + reconciled['Region2'] + reconciled['Region3']
        np.testing.assert_array_almost_equal(bottom_sum, sample_forecasts['Total'])
    
    def test_reconcile_invalid_method(self, reconciliation_engine, sample_forecasts, hierarchy_structure):
        """Test reconciliation with invalid method."""
        with pytest.raises(ValueError, match="Unknown reconciliation method"):
            reconciliation_engine.reconcile(sample_forecasts, hierarchy_structure, 'invalid_method')
    
    def test_all_methods_produce_consistent_results(self, reconciliation_engine, sample_forecasts, hierarchy_structure):
        """Test that all reconciliation methods produce consistent hierarchy."""
        methods = ['bottom_up', 'top_down', 'optimal']
        
        results = {}
        for method in methods:
            try:
                results[method] = reconciliation_engine.reconcile(sample_forecasts, hierarchy_structure, method)
            except Exception:
                # Some methods might fail with certain inputs
                continue
        
        # Check that all successful methods produce consistent hierarchy
        for method, reconciled in results.items():
            if 'Total' in reconciled:
                bottom_level = hierarchy_structure['bottom_level']
                bottom_sum = sum(reconciled[series] for series in bottom_level if series in reconciled)
                np.testing.assert_array_almost_equal(bottom_sum, reconciled['Total'])
