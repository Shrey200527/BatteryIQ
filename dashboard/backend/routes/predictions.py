"""
Prediction Routes for BatteryIQ Dashboard Backend.

These routes handle ML prediction endpoints, integrating with the existing
Phase 2 inference interfaces for SOH, RUL, condition classification, and SHAP.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Optional

from dashboard.backend.schemas import (
    SOHPredictionRequest,
    SOHPredictionResponse,
    RULPredictionRequest,
    RULPredictionResponse,
    SHAPRequest,
    SHAPResponse,
    ConditionRequest,
    ConditionResponse,
    ErrorResponse,
    CycleFeatures,
    SequenceFeatures
)
from dashboard.backend.services import (
    get_battery_service,
    get_prediction_service
)

router = APIRouter(prefix="/api/predict", tags=["predictions"])


@router.post("/soh", response_model=SOHPredictionResponse)
async def predict_soh(request: SOHPredictionRequest):
    """
    Predict State of Health (SOH) for a battery cycle.
    
    Uses the validated ANN SOH model from Phase 2.
    """
    try:
        # Validate battery and cycle
        battery_service = get_battery_service()
        
        if not battery_service.validate_battery_id(request.battery_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "invalid_battery",
                    "message": f"Battery {request.battery_id} is not available"
                }
            )
        
        if not battery_service.validate_cycle(request.battery_id, request.cycle):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "invalid_cycle",
                    "message": f"Cycle {request.cycle} not found for battery {request.battery_id}"
                }
            )
        
        # Get features in the correct format
        features = request.features.to_dict()
        
        # Call existing inference interface
        prediction_service = get_prediction_service()
        result = prediction_service.predict_soh(features)
        
        return SOHPredictionResponse(
            battery_id=request.battery_id,
            cycle=request.cycle,
            **result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "prediction_error",
                "message": "SOH prediction failed",
                "details": str(e)
            }
        )


@router.post("/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """
    Predict Remaining Useful Life (RUL) for a battery.
    
    Supports three models:
    - xgboost: Single cycle input
    - lstm: 10 consecutive cycles input
    - gru: 10 consecutive cycles input
    """
    try:
        # Validate battery
        battery_service = get_battery_service()
        
        if not battery_service.validate_battery_id(request.battery_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "invalid_battery",
                    "message": f"Battery {request.battery_id} is not available"
                }
            )
        
        # Check RUL validity for this battery
        battery_info = battery_service.get_battery_info(request.battery_id)
        if not battery_info["is_rul_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "rul_not_valid",
                    "message": f"RUL predictions are not valid for battery {request.battery_id}",
                    "details": "Battery may be censored (e.g., B0007) or not have observed EOL"
                }
            )
        
        # Prepare features based on model type
        if request.model == "xgboost":
            # Single cycle features
            if not isinstance(request.features, CycleFeatures):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "invalid_features",
                        "message": "XGBoost requires single cycle features, not a sequence"
                    }
                )
            features = request.features.to_dict()
            sequence_length = None
            
        elif request.model in ["lstm", "gru"]:
            # Sequence features
            if not isinstance(request.features, SequenceFeatures):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "invalid_features",
                        "message": f"{request.model.upper()} requires 10-cycle sequence, not single cycle"
                    }
                )
            
            # Convert sequence to format expected by inference
            cycles_features = [cycle.to_dict() for cycle in request.features.cycles]
            features = cycles_features
            sequence_length = 10
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_model",
                    "message": f"Model {request.model} is not supported. Use: xgboost, lstm, or gru"
                }
            )
        
        # Call existing inference interface
        prediction_service = get_prediction_service()
        result = prediction_service.predict_rul(features, model=request.model)
        
        return RULPredictionResponse(
            battery_id=request.battery_id,
            sequence_length=sequence_length,
            **result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "prediction_error",
                "message": f"RUL prediction failed with {request.model}",
                "details": str(e)
            }
        )


@router.post("/condition", response_model=ConditionResponse)
async def classify_condition(request: ConditionRequest):
    """
    Classify battery condition based on SOH.
    
    Returns condition classification: Healthy, Degraded, or Critical.
    
    Note: This endpoint may return "thresholds_pending" status if the
    team has not yet approved numerical classification thresholds in the ML contract.
    """
    try:
        prediction_service = get_prediction_service()
        result = prediction_service.classify_condition(request.soh)
        
        return ConditionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "classification_error",
                "message": "Condition classification failed",
                "details": str(e)
            }
        )


@router.post("/explain", response_model=SHAPResponse)
async def explain_prediction(request: SHAPRequest):
    """
    Generate SHAP explanation for a prediction.
    
    Provides feature contribution analysis for model interpretability.
    """
    try:
        # Validate battery and cycle
        battery_service = get_battery_service()
        
        if not battery_service.validate_battery_id(request.battery_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "invalid_battery",
                    "message": f"Battery {request.battery_id} is not available"
                }
            )
        
        if not battery_service.validate_cycle(request.battery_id, request.cycle):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "invalid_cycle",
                    "message": f"Cycle {request.cycle} not found for battery {request.battery_id}"
                }
            )
        
        # Get features in the correct format
        features = request.features.to_dict()
        
        # Call existing SHAP interface
        prediction_service = get_prediction_service()
        result = prediction_service.explain_prediction(features)
        
        return SHAPResponse(
            battery_id=request.battery_id,
            cycle=request.cycle,
            **result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "explanation_error",
                "message": "SHAP explanation failed",
                "details": str(e)
            }
        )