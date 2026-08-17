"""
Evaluation metrics for BatteryIQ SOH prediction.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE and R².

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """
    return {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred)
    }