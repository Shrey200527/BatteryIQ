"""
Prediction Service for BatteryIQ Dashboard Backend.

This service integrates with the existing Phase 2 inference interfaces,
providing a clean API layer that calls the validated ML models without
reimplementing any ML logic.
"""

from typing import Dict, Optional

# Add src to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.inference.predictor import (
        predict_soh,
        predict_rul,
        classify_condition,
        explain_prediction,
    )
    INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Phase 2 inference modules not available: {e}")
    INFERENCE_AVAILABLE = False


class PredictionService:
    """Service for ML predictions using existing Phase 2 inference."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.inference_available = INFERENCE_AVAILABLE
    
    def predict_soh(self, features: Dict) -> Dict:
        """
        Predict SOH using the existing ANN inference interface.
        
        Parameters
        ----------
        features : Dict
            Feature dictionary matching frozen ML contract
            
        Returns
        -------
        Dict
            Structured SOH prediction result
        """
        if not self.inference_available:
            raise RuntimeError(
                "SOH prediction unavailable: Phase 2 inference modules not loaded. "
                "Ensure tensorflow and other ML dependencies are installed."
            )
        
        try:
            result = predict_soh(features)
            
            # Extract single prediction if multiple returned
            if isinstance(result.get("predictions"), list) and len(result["predictions"]) > 0:
                prediction = result["predictions"][0]
            else:
                prediction = result.get("predictions", 0.0)
            
            return {
                "model": result.get("model", "BatteryIQ_ANN_SOH"),
                "target": result.get("target", "soh"),
                "prediction": float(prediction),
                "unit": result.get("unit", "percent")
            }
        except Exception as e:
            raise RuntimeError(f"SOH prediction failed: {str(e)}")
    
    def predict_rul(self, features: Dict, model: str = "xgboost") -> Dict:
        """
        Predict RUL using the existing RUL inference interface.
        
        Parameters
        ----------
        features : Dict
            Feature dictionary (single cycle for XGBoost, sequence for LSTM/GRU)
        model : str
            Model type: "xgboost", "lstm", or "gru"
            
        Returns
        -------
        Dict
            Structured RUL prediction result
        """
        if not self.inference_available:
            raise RuntimeError(
                f"RUL prediction unavailable: Phase 2 inference modules not loaded. "
                "Ensure tensorflow and other ML dependencies are installed."
            )
        
        try:
            result = predict_rul(features, model=model)
            
            return {
                "model": result.get("model", f"BatteryIQ_{model.upper()}_RUL"),
                "target": result.get("target", "rul"),
                "prediction": float(result.get("prediction", 0.0)),
                "unit": result.get("unit", "cycles"),
                "sequence_length": result.get("sequence_length")
            }
        except Exception as e:
            raise RuntimeError(f"RUL prediction failed with {model}: {str(e)}")
    
    def classify_condition(self, soh: float) -> Dict:
        """
        Classify battery condition using the existing classifier.
        
        Parameters
        ----------
        soh : float
            State of Health value (percentage)
            
        Returns
        -------
        Dict
            Structured condition classification result
        """
        if not self.inference_available:
            # Even without full inference, we can handle the classification blocker
            return {
                "condition": None,
                "status": "thresholds_pending",
                "soh": soh,
                "message": "Classification thresholds not yet approved by team. See ML contract."
            }
        
        try:
            condition = classify_condition(soh)
            
            return {
                "condition": condition,
                "status": "available",
                "soh": soh,
                "message": None
            }
        except NotImplementedError as e:
            # Handle the classification blocker
            return {
                "condition": None,
                "status": "thresholds_pending",
                "soh": soh,
                "message": "Classification thresholds not yet approved by team. See ML contract."
            }
        except Exception as e:
            raise RuntimeError(f"Condition classification failed: {str(e)}")
    
    def explain_prediction(self, features: Dict) -> Dict:
        """
        Generate SHAP explanation using the existing SHAP interface.
        
        Parameters
        ----------
        features : Dict
            Feature dictionary matching frozen ML contract
            
        Returns
        -------
        Dict
            Structured SHAP explanation result
        """
        if not self.inference_available:
            raise RuntimeError(
                "SHAP explanation unavailable: Phase 2 inference modules not loaded. "
                "Ensure tensorflow and other ML dependencies are installed."
            )
        
        try:
            explanation = explain_prediction(features)
            
            # Format the explanation for API response
            feature_contributions = {}
            for feature_name, contrib_data in explanation.get("feature_contributions", {}).items():
                feature_contributions[feature_name] = {
                    "shap_value": float(contrib_data.get("shap_value", 0.0)),
                    "feature_value": float(contrib_data.get("feature_value", 0.0))
                }
            
            return {
                "prediction": float(explanation.get("prediction", 0.0)),
                "base_value": float(explanation.get("base_value", 0.0)) if explanation.get("base_value") is not None else None,
                "feature_contributions": feature_contributions
            }
        except Exception as e:
            raise RuntimeError(f"SHAP explanation failed: {str(e)}")


# Global service instance
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """Get or create prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service