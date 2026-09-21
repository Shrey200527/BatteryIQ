"""
Explainability module for BatteryIQ Phase 2.

This module provides SHAP-based explainability for validated ML models.
"""

from .shap_analysis import (
    SHAPAnalyzer,
    analyze_model_global_importance,
    explain_single_prediction,
    validate_shap_available,
    validate_feature_matrix,
    validate_model,
    FROZEN_FEATURE_COLUMNS
)

__all__ = [
    'SHAPAnalyzer',
    'analyze_model_global_importance', 
    'explain_single_prediction',
    'validate_shap_available',
    'validate_feature_matrix',
    'validate_model',
    'FROZEN_FEATURE_COLUMNS'
]