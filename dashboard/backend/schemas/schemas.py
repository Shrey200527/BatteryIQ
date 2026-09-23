"""
API Schemas for BatteryIQ Dashboard Backend.

These schemas define the request/response contracts for the dashboard API,
ensuring type safety and validation while matching the existing Phase 2
inference interfaces.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Frozen Feature Contract
# ---------------------------------------------------------------------------

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
# Battery Schemas
# ---------------------------------------------------------------------------

class BatteryInfo(BaseModel):
    """Information about a single battery."""
    battery_id: str = Field(..., description="Battery identifier (e.g., B0005)")
    total_cycles: int = Field(..., description="Total number of cycles available")
    is_rul_valid: bool = Field(..., description="Whether RUL predictions are valid for this battery")


class BatteriesResponse(BaseModel):
    """Response for available batteries list."""
    batteries: List[BatteryInfo] = Field(..., description="List of available batteries")
    total: int = Field(..., description="Total number of batteries")


class BatteryDetail(BaseModel):
    """Detailed information about a specific battery."""
    battery_id: str
    total_cycles: int
    is_rul_valid: bool
    soh_range: Dict[str, float] = Field(..., description="SOH min and max values")
    available_cycles: List[int] = Field(..., description="List of available cycle numbers")


# ---------------------------------------------------------------------------
# Feature Schemas
# ---------------------------------------------------------------------------

class CycleFeatures(BaseModel):
    """Single cycle feature data matching the frozen ML contract."""
    
    voltage_mean: float
    voltage_min: float
    voltage_max: float
    voltage_std: float
    current_mean: float
    current_min: float
    current_max: float
    temperature_mean: float
    temperature_max: float
    current_load_mean: float
    voltage_load_mean: float
    cycle_time: float
    voltage_change: float
    temperature_change: float
    temperature_rolling_mean: float
    
    @validator('*')
    def validate_numeric(cls, v):
        """Ensure all values are numeric."""
        if not isinstance(v, (int, float)):
            raise ValueError("All feature values must be numeric")
        return float(v)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary matching frozen feature order."""
        return {
            "voltage_mean": self.voltage_mean,
            "voltage_min": self.voltage_min,
            "voltage_max": self.voltage_max,
            "voltage_std": self.voltage_std,
            "current_mean": self.current_mean,
            "current_min": self.current_min,
            "current_max": self.current_max,
            "temperature_mean": self.temperature_mean,
            "temperature_max": self.temperature_max,
            "current_load_mean": self.current_load_mean,
            "voltage_load_mean": self.voltage_load_mean,
            "cycle_time": self.cycle_time,
            "voltage_change": self.voltage_change,
            "temperature_change": self.temperature_change,
            "temperature_rolling_mean": self.temperature_rolling_mean,
        }


class SequenceFeatures(BaseModel):
    """Sequence of consecutive cycle features for LSTM/GRU models."""
    
    cycles: List[CycleFeatures] = Field(..., description="Exactly 10 consecutive cycles")
    
    @validator('cycles')
    def validate_sequence_length(cls, v):
        """Ensure exactly 10 cycles for LSTM/GRU models."""
        if len(v) != 10:
            raise ValueError("Sequence must contain exactly 10 cycles for LSTM/GRU models")
        return v


# ---------------------------------------------------------------------------
# Prediction Request Schemas
# ---------------------------------------------------------------------------

class SOHPredictionRequest(BaseModel):
    """Request for SOH prediction."""
    battery_id: str = Field(..., description="Battery identifier")
    cycle: int = Field(..., description="Cycle number")
    features: CycleFeatures = Field(..., description="Cycle features")


class RULPredictionRequest(BaseModel):
    """Request for RUL prediction."""
    battery_id: str = Field(..., description="Battery identifier")
    model: str = Field(..., description="Model type: xgboost, lstm, or gru")
    
    # For XGBoost: single cycle features
    # For LSTM/GRU: sequence of 10 consecutive cycles
    features: Union[CycleFeatures, SequenceFeatures] = Field(
        ..., 
        description="Single cycle for XGBoost, 10-cycle sequence for LSTM/GRU"
    )
    
    @validator('model')
    def validate_model(cls, v):
        """Ensure valid model type."""
        valid_models = ["xgboost", "lstm", "gru"]
        if v.lower() not in valid_models:
            raise ValueError(f"Model must be one of: {valid_models}")
        return v.lower()
    
    @validator('features')
    def validate_features_match_model(cls, v, values):
        """Ensure features match the requested model type."""
        model = values.get('model', 'xgboost')
        
        if model == "xgboost":
            if not isinstance(v, CycleFeatures):
                raise ValueError("XGBoost model requires single cycle features, not a sequence")
        elif model in ["lstm", "gru"]:
            if not isinstance(v, SequenceFeatures):
                raise ValueError(f"{model.upper()} model requires 10-cycle sequence, not single cycle")
        
        return v


class SHAPRequest(BaseModel):
    """Request for SHAP explanation."""
    battery_id: str = Field(..., description="Battery identifier")
    cycle: int = Field(..., description="Cycle number")
    features: CycleFeatures = Field(..., description="Cycle features")


class ConditionRequest(BaseModel):
    """Request for condition classification."""
    soh: float = Field(..., description="State of Health value (percentage)")
    
    @validator('soh')
    def validate_soh(cls, v):
        """Ensure SOH is within reasonable range."""
        if not (0 <= v <= 100):
            raise ValueError("SOH must be between 0 and 100")
        return v


# ---------------------------------------------------------------------------
# Prediction Response Schemas
# ---------------------------------------------------------------------------

class SOHPredictionResponse(BaseModel):
    """Response for SOH prediction."""
    model: str = Field(..., description="Model name used for prediction")
    target: str = Field(..., description="Prediction target")
    prediction: float = Field(..., description="Predicted SOH value")
    unit: str = Field(..., description="Unit of prediction")
    battery_id: str = Field(..., description="Battery identifier")
    cycle: int = Field(..., description="Cycle number")


class RULPredictionResponse(BaseModel):
    """Response for RUL prediction."""
    model: str = Field(..., description="Model name used for prediction")
    target: str = Field(..., description="Prediction target")
    prediction: float = Field(..., description="Predicted RUL value")
    unit: str = Field(..., description="Unit of prediction")
    battery_id: str = Field(..., description="Battery identifier")
    sequence_length: Optional[int] = Field(None, description="Sequence length for LSTM/GRU")


class SHAPResponse(BaseModel):
    """Response for SHAP explanation."""
    battery_id: str = Field(..., description="Battery identifier")
    cycle: int = Field(..., description="Cycle number")
    prediction: float = Field(..., description="Model prediction value")
    base_value: Optional[float] = Field(None, description="Base value for SHAP")
    feature_contributions: Dict[str, Dict[str, float]] = Field(
        ..., 
        description="Feature contributions to prediction"
    )


class ConditionResponse(BaseModel):
    """Response for condition classification."""
    condition: Optional[str] = Field(None, description="Battery condition: Healthy, Degraded, Critical")
    status: str = Field(..., description="Status of classification: available or thresholds_pending")
    soh: float = Field(..., description="Input SOH value")
    message: Optional[str] = Field(None, description="Additional information about the classification status")


# ---------------------------------------------------------------------------
# Error Response Schema
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict] = Field(None, description="Additional error details")


# ---------------------------------------------------------------------------
# Combined Prediction Response
# ---------------------------------------------------------------------------

class BatteryPredictionResponse(BaseModel):
    """Combined response with all predictions for a battery cycle."""
    battery_id: str
    cycle: int
    soh: Optional[SOHPredictionResponse] = None
    rul: Optional[Dict[str, RULPredictionResponse]] = None  # Multiple RUL models
    condition: Optional[ConditionResponse] = None
    shap: Optional[SHAPResponse] = None