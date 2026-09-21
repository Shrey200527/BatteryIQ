"""
SHAP Explainability Analysis for BatteryIQ Phase 2.

Responsibilities:
- Explain predictions from validated ML models using SHAP values
- Support global feature importance analysis
- Support local/individual prediction explanation
- Preserve exact feature names and order from ML contract
- Ensure no target leakage in SHAP analysis

This module is designed to work with actual validated models from the project.
It does not train fake models or use placeholder predictions.
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Frozen feature columns from ML contract
FROZEN_FEATURE_COLUMNS = [
    "voltage_mean",
    "voltage_min", 
    "voltage_max",
    "voltage_std",
    "current_mean",
    "current_min",
    "current_max",
    "temperature_mean",
    "temperature_max",
    "current_load_mean",
    "voltage_load_mean",
    "cycle_time",
    "voltage_change",
    "temperature_change",
    "temperature_rolling_mean",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_shap_available() -> None:
    """Check if SHAP library is installed."""
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. "
            "Install it using: pip install shap"
        )


def validate_feature_matrix(X: pd.DataFrame) -> None:
    """
    Validate that the feature matrix matches the ML contract.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix to validate
        
    Raises
    ------
    ValueError
        If features don't match the frozen contract
    """
    # Check for exact feature match
    if list(X.columns) != FROZEN_FEATURE_COLUMNS:
        missing = set(FROZEN_FEATURE_COLUMNS) - set(X.columns)
        extra = set(X.columns) - set(FROZEN_FEATURE_COLUMNS)
        
        error_msg = "Feature matrix does not match ML contract."
        if missing:
            error_msg += f" Missing features: {missing}"
        if extra:
            error_msg += f" Extra features: {extra}"
        if list(X.columns) != FROZEN_FEATURE_COLUMNS:
            error_msg += f" Feature order mismatch."
            
        raise ValueError(error_msg)
    
    # Check for target leakage
    prohibited = ['capacity', 'soh', 'capacity_change', 'capacity_retention', 
                  'capacity_rolling_mean', 'capacity_degradation', 'rul', 'rul_valid']
    found_leakage = [col for col in X.columns if any(prohibited in col.lower() for prohibited in prohibited)]
    
    if found_leakage:
        raise ValueError(
            f"Target leakage detected in feature matrix. "
            f"Prohibited features found: {found_leakage}"
        )
    
    # Check for missing values
    if X.isnull().any().any():
        raise ValueError(
            "Feature matrix contains missing values. "
            "All features must be non-null for SHAP analysis."
        )


def validate_model(model: Any) -> None:
    """
    Basic validation that the model object appears to be a valid ML model.
    
    Parameters
    ----------
    model : Any
        Model object to validate
        
    Raises
    ------
    ValueError
        If model doesn't have expected interface
    """
    if not hasattr(model, 'predict'):
        raise ValueError(
            "Model must have a 'predict' method for SHAP analysis."
        )


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------

class SHAPAnalyzer:
    """
    SHAP explainability analyzer for BatteryIQ models.
    
    This class provides interfaces for:
    - Global feature importance
    - Local prediction explanation
    - SHAP summary plots
    """
    
    def __init__(
        self,
        model: Any,
        feature_matrix: pd.DataFrame,
        feature_names: Optional[list] = None
    ):
        """
        Initialize SHAP analyzer.
        
        Parameters
        ----------
        model : Any
            Validated ML model with predict() method
        feature_matrix : pd.DataFrame
            Feature matrix matching ML contract exactly
        feature_names : list, optional
            List of feature names (uses ML contract if not provided)
        """
        validate_shap_available()
        validate_model(model)
        validate_feature_matrix(feature_matrix)
        
        self.model = model
        self.feature_matrix = feature_matrix.copy()
        
        # Use exact feature names from ML contract or provided names
        if feature_names is None:
            self.feature_names = FROZEN_FEATURE_COLUMNS
        else:
            if feature_names != FROZEN_FEATURE_COLUMNS:
                raise ValueError(
                    "Provided feature names do not match ML contract."
                )
            self.feature_names = feature_names
            
        self.explainer = None
        self.shap_values = None
        
    def fit_explainer(self, background_sample_size: int = 100) -> None:
        """
        Fit SHAP explainer using background data.
        
        Parameters
        ----------
        background_sample_size : int
            Number of samples to use for background data
        """
        # Sample background data for computational efficiency
        if len(self.feature_matrix) > background_sample_size:
            background_data = self.feature_matrix.sample(
                n=background_sample_size, 
                random_state=42
            ).values
        else:
            background_data = self.feature_matrix.values
            
        # Use KernelExplainer with a wrapper function to avoid compatibility issues
        # This provides robust SHAP values across different model types and versions
        try:
            # Create a wrapper function that works with SHAP's KernelExplainer
            def model_wrapper(x):
                return self.model.predict(x)
            
            self.explainer = shap.KernelExplainer(
                model_wrapper,
                background_data
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create SHAP explainer: {e}. "
                "Ensure the model has a working predict method."
            )
    
    def compute_shap_values(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Compute SHAP values for predictions.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix to explain (uses self.feature_matrix if None)
            
        Returns
        -------
        np.ndarray
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError(
                "Explainer not fitted. Call fit_explainer() first."
            )
        
        if X is None:
            X = self.feature_matrix
        else:
            validate_feature_matrix(X)
        
        self.shap_values = self.explainer.shap_values(X.values)
        return self.shap_values
    
    def get_global_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features ranked by importance
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )
        
        # Compute mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def explain_prediction(
        self, 
        sample_idx: int,
        X: Optional[pd.DataFrame] = None
    ) -> dict:
        """
        Explain an individual prediction.
        
        Parameters
        ----------
        sample_idx : int
            Index of the sample to explain
        X : pd.DataFrame, optional
            Feature matrix (uses self.feature_matrix if None)
            
        Returns
        -------
        dict
            Dictionary containing prediction explanation
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )
        
        if X is None:
            X = self.feature_matrix
        
        if sample_idx >= len(X):
            raise ValueError(f"sample_idx {sample_idx} out of range for {len(X)} samples")
        
        # Get actual prediction
        prediction = self.model.predict(X.iloc[[sample_idx]])[0]
        
        # Get SHAP values for this sample
        sample_shap = self.shap_values[sample_idx]
        
        # Get feature values for this sample
        sample_features = X.iloc[sample_idx].values
        
        # Create explanation dictionary
        explanation = {
            'prediction': float(prediction),
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else None,
            'feature_contributions': {}
        }
        
        for feature_name, shap_val, feature_val in zip(
            self.feature_names, sample_shap, sample_features
        ):
            explanation['feature_contributions'][feature_name] = {
                'shap_value': float(shap_val),
                'feature_value': float(feature_val)
            }
        
        return explanation
    
    def get_summary_data(self) -> tuple:
        """
        Get data for SHAP summary plot.
        
        Returns
        -------
        tuple
            (shap_values, feature_matrix) for plotting
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )
        
        return self.shap_values, self.feature_matrix.values


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def analyze_model_global_importance(
    model: Any,
    feature_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Convenience function to get global feature importance.
    
    Parameters
    ----------
    model : Any
        Validated ML model
    feature_matrix : pd.DataFrame
        Feature matrix matching ML contract
        
    Returns
    -------
    pd.DataFrame
        Feature importance rankings
    """
    analyzer = SHAPAnalyzer(model, feature_matrix)
    analyzer.fit_explainer()
    analyzer.compute_shap_values()
    return analyzer.get_global_feature_importance()


def explain_single_prediction(
    model: Any,
    feature_matrix: pd.DataFrame,
    sample_idx: int
) -> dict:
    """
    Convenience function to explain a single prediction.
    
    Parameters
    ----------
    model : Any
        Validated ML model
    feature_matrix : pd.DataFrame
        Feature matrix matching ML contract
    sample_idx : int
        Index of sample to explain
        
    Returns
    -------
    dict
        Prediction explanation
    """
    analyzer = SHAPAnalyzer(model, feature_matrix)
    analyzer.fit_explainer()
    analyzer.compute_shap_values()
    return analyzer.explain_prediction(sample_idx)