"""
Reconciliation methods for hierarchical time series forecasting.

This module provides various reconciliation techniques to ensure consistency
across different levels of the hierarchy.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseReconciliation(ABC):
    """Abstract base class for reconciliation methods."""
    
    def __init__(self, name: str):
        """
        Initialize the reconciliation method.
        
        Args:
            name: Name of the reconciliation method
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def reconcile(
        self, 
        forecasts: Dict[str, np.ndarray], 
        hierarchy_structure: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Reconcile forecasts across hierarchy levels.
        
        Args:
            forecasts: Dictionary of forecasts for each series
            hierarchy_structure: Structure of the hierarchy
            
        Returns:
            Dictionary of reconciled forecasts
        """
        pass


class BottomUpReconciliation(BaseReconciliation):
    """
    Bottom-up reconciliation method.
    
    Forecasts are generated at the bottom level and aggregated upward.
    """
    
    def __init__(self):
        super().__init__("Bottom-Up")
    
    def reconcile(
        self, 
        forecasts: Dict[str, np.ndarray], 
        hierarchy_structure: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Perform bottom-up reconciliation.
        
        Args:
            forecasts: Dictionary of forecasts for each series
            hierarchy_structure: Structure of the hierarchy
            
        Returns:
            Dictionary of reconciled forecasts
        """
        self.logger.info("Performing bottom-up reconciliation")
        
        reconciled_forecasts = forecasts.copy()
        
        # Get bottom level forecasts
        bottom_level = hierarchy_structure['bottom_level']
        top_level = hierarchy_structure['top_level']
        
        # Aggregate bottom level forecasts to get top level
        for top_series in top_level:
            if top_series in forecasts:
                continue  # Skip if already forecasted
            
            # Sum bottom level forecasts
            top_forecast = np.zeros_like(list(forecasts.values())[0])
            for bottom_series in bottom_level:
                if bottom_series in forecasts:
                    top_forecast += forecasts[bottom_series]
            
            reconciled_forecasts[top_series] = top_forecast
        
        return reconciled_forecasts


class TopDownReconciliation(BaseReconciliation):
    """
    Top-down reconciliation method.
    
    Forecasts are generated at the top level and disaggregated downward.
    """
    
    def __init__(self, disaggregation_method: str = "proportional"):
        """
        Initialize top-down reconciliation.
        
        Args:
            disaggregation_method: Method for disaggregation ('proportional', 'average')
        """
        super().__init__("Top-Down")
        self.disaggregation_method = disaggregation_method
    
    def reconcile(
        self, 
        forecasts: Dict[str, np.ndarray], 
        hierarchy_structure: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Perform top-down reconciliation.
        
        Args:
            forecasts: Dictionary of forecasts for each series
            hierarchy_structure: Structure of the hierarchy
            
        Returns:
            Dictionary of reconciled forecasts
        """
        self.logger.info(f"Performing top-down reconciliation with {self.disaggregation_method} disaggregation")
        
        reconciled_forecasts = forecasts.copy()
        
        # Get hierarchy levels
        bottom_level = hierarchy_structure['bottom_level']
        top_level = hierarchy_structure['top_level']
        
        # Find top level forecast
        top_forecast = None
        for top_series in top_level:
            if top_series in forecasts:
                top_forecast = forecasts[top_series]
                break
        
        if top_forecast is None:
            self.logger.warning("No top level forecast found, using bottom-up as fallback")
            return BottomUpReconciliation().reconcile(forecasts, hierarchy_structure)
        
        # Disaggregate top level forecast to bottom level
        if self.disaggregation_method == "proportional":
            # Use historical proportions
            historical_totals = {}
            for series in bottom_level:
                if series in forecasts:
                    historical_totals[series] = np.mean(forecasts[series])
            
            total_historical = sum(historical_totals.values())
            
            for series in bottom_level:
                if series not in forecasts:
                    proportion = historical_totals.get(series, 1.0) / total_historical
                    reconciled_forecasts[series] = top_forecast * proportion
        
        elif self.disaggregation_method == "average":
            # Equal distribution
            num_series = len(bottom_level)
            for series in bottom_level:
                if series not in forecasts:
                    reconciled_forecasts[series] = top_forecast / num_series
        
        return reconciled_forecasts


class MiddleOutReconciliation(BaseReconciliation):
    """
    Middle-out reconciliation method.
    
    Forecasts are generated at a middle level and reconciled both upward and downward.
    """
    
    def __init__(self, middle_level: Optional[List[str]] = None):
        """
        Initialize middle-out reconciliation.
        
        Args:
            middle_level: List of series at the middle level
        """
        super().__init__("Middle-Out")
        self.middle_level = middle_level
    
    def reconcile(
        self, 
        forecasts: Dict[str, np.ndarray], 
        hierarchy_structure: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Perform middle-out reconciliation.
        
        Args:
            forecasts: Dictionary of forecasts for each series
            hierarchy_structure: Structure of the hierarchy
            
        Returns:
            Dictionary of reconciled forecasts
        """
        self.logger.info("Performing middle-out reconciliation")
        
        reconciled_forecasts = forecasts.copy()
        
        # Determine middle level if not specified
        if self.middle_level is None:
            # Use bottom level as middle level
            self.middle_level = hierarchy_structure['bottom_level']
        
        # First, reconcile upward (bottom-up from middle level)
        bottom_up = BottomUpReconciliation()
        temp_forecasts = bottom_up.reconcile(forecasts, {
            'bottom_level': self.middle_level,
            'top_level': hierarchy_structure['top_level']
        })
        
        # Then, reconcile downward (top-down from middle level)
        top_down = TopDownReconciliation()
        reconciled_forecasts = top_down.reconcile(temp_forecasts, hierarchy_structure)
        
        return reconciled_forecasts


class OptimalReconciliation(BaseReconciliation):
    """
    Optimal reconciliation method using least squares.
    
    This method finds the optimal reconciliation that minimizes the sum of squared errors
    while maintaining hierarchy consistency.
    """
    
    def __init__(self):
        super().__init__("Optimal")
    
    def reconcile(
        self, 
        forecasts: Dict[str, np.ndarray], 
        hierarchy_structure: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Perform optimal reconciliation.
        
        Args:
            forecasts: Dictionary of forecasts for each series
            hierarchy_structure: Structure of the hierarchy
            
        Returns:
            Dictionary of reconciled forecasts
        """
        self.logger.info("Performing optimal reconciliation")
        
        # For simplicity, implement a basic optimal reconciliation
        # In practice, this would involve solving a constrained optimization problem
        
        reconciled_forecasts = forecasts.copy()
        
        # Get hierarchy levels
        bottom_level = hierarchy_structure['bottom_level']
        top_level = hierarchy_structure['top_level']
        
        # Calculate current inconsistency
        bottom_sum = np.zeros_like(list(forecasts.values())[0])
        for series in bottom_level:
            if series in forecasts:
                bottom_sum += forecasts[series]
        
        top_forecast = None
        for series in top_level:
            if series in forecasts:
                top_forecast = forecasts[series]
                break
        
        if top_forecast is None:
            # No top level forecast, use bottom-up
            reconciled_forecasts[top_level[0]] = bottom_sum
            return reconciled_forecasts
        
        # Calculate adjustment factor
        adjustment_factor = top_forecast / (bottom_sum + 1e-8)  # Avoid division by zero
        
        # Apply adjustment to bottom level forecasts
        for series in bottom_level:
            if series in forecasts:
                reconciled_forecasts[series] = forecasts[series] * adjustment_factor
        
        # Ensure top level consistency
        reconciled_sum = sum(reconciled_forecasts[series] for series in bottom_level)
        reconciled_forecasts[top_level[0]] = reconciled_sum
        
        return reconciled_forecasts


class ReconciliationEngine:
    """
    Engine for managing different reconciliation methods.
    """
    
    def __init__(self):
        """Initialize the reconciliation engine."""
        self.logger = logging.getLogger(__name__)
        self.methods = {
            'bottom_up': BottomUpReconciliation(),
            'top_down': TopDownReconciliation(),
            'middle_out': MiddleOutReconciliation(),
            'optimal': OptimalReconciliation()
        }
    
    def reconcile(
        self, 
        forecasts: Dict[str, np.ndarray], 
        hierarchy_structure: Dict[str, List[str]], 
        method: str = 'bottom_up'
    ) -> Dict[str, np.ndarray]:
        """
        Reconcile forecasts using the specified method.
        
        Args:
            forecasts: Dictionary of forecasts for each series
            hierarchy_structure: Structure of the hierarchy
            method: Reconciliation method to use
            
        Returns:
            Dictionary of reconciled forecasts
        """
        if method not in self.methods:
            raise ValueError(f"Unknown reconciliation method: {method}")
        
        self.logger.info(f"Using reconciliation method: {method}")
        return self.methods[method].reconcile(forecasts, hierarchy_structure)
    
    def get_available_methods(self) -> List[str]:
        """
        Get list of available reconciliation methods.
        
        Returns:
            List of method names
        """
        return list(self.methods.keys())
