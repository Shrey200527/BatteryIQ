"""
SOH prediction models for BatteryIQ.

This module provides the baseline machine learning models
used for battery State of Health (SOH) prediction.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def get_linear_regression():
    """Return the scaled Linear Regression model."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])


def get_random_forest():
    """Return the Random Forest regression model."""
    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )


def get_svr():
    """Return the scaled SVR model."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=10,
            epsilon=0.1
        ))
    ])


def get_xgboost():
    """Return the XGBoost regression model."""
    if XGBRegressor is None:
        raise ImportError(
            "XGBoost is not installed. "
            "Install it using: pip install xgboost"
        )

    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


def get_baseline_models():
    """
    Return all baseline SOH prediction models.

    Returns
    -------
    dict
        Dictionary containing the four baseline models.
    """
    return {
        "Linear Regression": get_linear_regression(),
        "Random Forest": get_random_forest(),
        "SVR": get_svr(),
        "XGBoost": get_xgboost()
    }