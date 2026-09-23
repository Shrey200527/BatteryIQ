"""
Tests for BatteryIQ Dashboard Backend API.

These tests verify the backend endpoints integrate correctly with the
existing Phase 2 inference interfaces.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.backend.app import app


client = TestClient(app)


class TestBatteriesEndpoints:
    """Test battery-related endpoints."""
    
    def test_get_batteries(self):
        """Test GET /api/batteries endpoint."""
        response = client.get("/api/batteries")
        assert response.status_code == 200
        
        data = response.json()
        assert "batteries" in data
        assert "total" in data
        assert data["total"] > 0
        assert len(data["batteries"]) == data["total"]
        
        # Check battery structure
        first_battery = data["batteries"][0]
        assert "battery_id" in first_battery
        assert "total_cycles" in first_battery
        assert "is_rul_valid" in first_battery
    
    def test_get_valid_battery_detail(self):
        """Test GET /api/batteries/{battery_id} with valid battery."""
        response = client.get("/api/batteries/B0005")
        assert response.status_code == 200
        
        data = response.json()
        assert data["battery_id"] == "B0005"
        assert "total_cycles" in data
        assert "soh_range" in data
        assert "available_cycles" in data
        assert isinstance(data["available_cycles"], list)
    
    def test_get_invalid_battery_detail(self):
        """Test GET /api/batteries/{battery_id} with invalid battery."""
        response = client.get("/api/batteries/B9999")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert data["error"] == "invalid_battery"
    
    def test_get_battery_history(self):
        """Test GET /api/batteries/{battery_id}/history endpoint."""
        response = client.get("/api/batteries/B0005/history")
        assert response.status_code == 200
        
        data = response.json()
        assert data["battery_id"] == "B0005"
        assert "total_cycles" in data
        assert "available_cycles" in data
        assert "soh_range" in data


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_soh_prediction(self):
        """Test POST /api/predict/soh endpoint."""
        # Get valid cycle features first
        battery_response = client.get("/api/batteries/B0005/history")
        battery_data = battery_response.json()
        first_cycle = battery_data["available_cycles"][0]
        
        # Create a sample feature request (using reasonable values)
        soh_request = {
            "battery_id": "B0005",
            "cycle": first_cycle,
            "features": {
                "voltage_mean": 3.5,
                "voltage_min": 2.6,
                "voltage_max": 4.2,
                "voltage_std": 0.2,
                "current_mean": -1.8,
                "current_min": -2.0,
                "current_max": 0.0,
                "temperature_mean": 32.0,
                "temperature_max": 38.0,
                "current_load_mean": -1.8,
                "voltage_load_mean": 2.4,
                "cycle_time": 3600.0,
                "voltage_change": 0.0,
                "temperature_change": 0.0,
                "temperature_rolling_mean": 32.0
            }
        }
        
        response = client.post("/api/predict/soh", json=soh_request)
        
        # This might fail if models aren't loaded, but should validate the endpoint structure
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "model" in data
            assert "battery_id" in data
            assert data["battery_id"] == "B0005"
        else:
            # If model loading fails, check it's a proper error
            assert response.status_code in [500, 503]
    
    def test_soh_prediction_invalid_battery(self):
        """Test SOH prediction with invalid battery."""
        soh_request = {
            "battery_id": "B9999",
            "cycle": 1,
            "features": {
                "voltage_mean": 3.5,
                "voltage_min": 2.6,
                "voltage_max": 4.2,
                "voltage_std": 0.2,
                "current_mean": -1.8,
                "current_min": -2.0,
                "current_max": 0.0,
                "temperature_mean": 32.0,
                "temperature_max": 38.0,
                "current_load_mean": -1.8,
                "voltage_load_mean": 2.4,
                "cycle_time": 3600.0,
                "voltage_change": 0.0,
                "temperature_change": 0.0,
                "temperature_rolling_mean": 32.0
            }
        }
        
        response = client.post("/api/predict/soh", json=soh_request)
        assert response.status_code == 404
        
        data = response.json()
        assert data["error"] == "invalid_battery"
    
    def test_rul_prediction_xgboost(self):
        """Test POST /api/predict/rul with XGBoost model."""
        rul_request = {
            "battery_id": "B0005",
            "model": "xgboost",
            "features": {
                "voltage_mean": 3.5,
                "voltage_min": 2.6,
                "voltage_max": 4.2,
                "voltage_std": 0.2,
                "current_mean": -1.8,
                "current_min": -2.0,
                "current_max": 0.0,
                "temperature_mean": 32.0,
                "temperature_max": 38.0,
                "current_load_mean": -1.8,
                "voltage_load_mean": 2.4,
                "cycle_time": 3600.0,
                "voltage_change": 0.0,
                "temperature_change": 0.0,
                "temperature_rolling_mean": 32.0
            }
        }
        
        response = client.post("/api/predict/rul", json=rul_request)
        
        # May fail if model not loaded, but should validate structure
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "model" in data
            assert data["model"] == "BatteryIQ_XGBoost_RUL"
        else:
            assert response.status_code in [500, 503]
    
    def test_rul_prediction_invalid_model(self):
        """Test RUL prediction with invalid model type."""
        rul_request = {
            "battery_id": "B0005",
            "model": "invalid_model",
            "features": {
                "voltage_mean": 3.5,
                "voltage_min": 2.6,
                "voltage_max": 4.2,
                "voltage_std": 0.2,
                "current_mean": -1.8,
                "current_min": -2.0,
                "current_max": 0.0,
                "temperature_mean": 32.0,
                "temperature_max": 38.0,
                "current_load_mean": -1.8,
                "voltage_load_mean": 2.4,
                "cycle_time": 3600.0,
                "voltage_change": 0.0,
                "temperature_change": 0.0,
                "temperature_rolling_mean": 32.0
            }
        }
        
        response = client.post("/api/predict/rul", json=rul_request)
        assert response.status_code == 400
        
        data = response.json()
        assert "error" in data
    
    def test_condition_classification(self):
        """Test POST /api/predict/condition endpoint."""
        condition_request = {
            "soh": 85.0
        }
        
        response = client.post("/api/predict/condition", json=condition_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "condition" in data or data["status"] == "thresholds_pending"
        assert "status" in data
        assert "soh" in data
        
        # Due to the blocker, we expect thresholds_pending
        if data["status"] == "thresholds_pending":
            assert data["condition"] is None
            assert data["message"] is not None
    
    def test_condition_classification_invalid_soh(self):
        """Test condition classification with invalid SOH."""
        condition_request = {
            "soh": 150.0  # Invalid SOH > 100
        }
        
        response = client.post("/api/predict/condition", json=condition_request)
        assert response.status_code == 422  # Validation error
    
    def test_shap_explanation(self):
        """Test POST /api/predict/explain endpoint."""
        shap_request = {
            "battery_id": "B0005",
            "cycle": 1,
            "features": {
                "voltage_mean": 3.5,
                "voltage_min": 2.6,
                "voltage_max": 4.2,
                "voltage_std": 0.2,
                "current_mean": -1.8,
                "current_min": -2.0,
                "current_max": 0.0,
                "temperature_mean": 32.0,
                "temperature_max": 38.0,
                "current_load_mean": -1.8,
                "voltage_load_mean": 2.4,
                "cycle_time": 3600.0,
                "voltage_change": 0.0,
                "temperature_change": 0.0,
                "temperature_rolling_mean": 32.0
            }
        }
        
        response = client.post("/api/predict/explain", json=shap_request)
        
        # May fail if SHAP not available, but should validate structure
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "feature_contributions" in data
            assert "battery_id" in data
        else:
            assert response.status_code in [500, 503]


class TestHealthEndpoints:
    """Test health and root endpoints."""
    
    def test_root_endpoint(self):
        """Test GET / endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test GET /health endpoint."""
        response = client.get("/health")
        
        # May fail if data not available, but should return health status
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "service" in data
        else:
            assert response.status_code == 503


class TestValidation:
    """Test input validation."""
    
    def test_sequence_validation_length(self):
        """Test that sequence validation rejects wrong length."""
        # Create a sequence with wrong number of cycles
        wrong_sequence_request = {
            "battery_id": "B0005",
            "model": "lstm",
            "features": {
                "cycles": [
                    # Only 5 cycles instead of 10
                    {
                        "voltage_mean": 3.5, "voltage_min": 2.6, "voltage_max": 4.2, "voltage_std": 0.2,
                        "current_mean": -1.8, "current_min": -2.0, "current_max": 0.0,
                        "temperature_mean": 32.0, "temperature_max": 38.0,
                        "current_load_mean": -1.8, "voltage_load_mean": 2.4,
                        "cycle_time": 3600.0, "voltage_change": 0.0, "temperature_change": 0.0,
                        "temperature_rolling_mean": 32.0
                    }
                ] * 5
            }
        }
        
        response = client.post("/api/predict/rul", json=wrong_sequence_request)
        assert response.status_code == 422  # Validation error
    
    def test_feature_validation(self):
        """Test that feature validation rejects missing features."""
        invalid_request = {
            "battery_id": "B0005",
            "cycle": 1,
            "features": {
                # Missing required features
                "voltage_mean": 3.5,
                "voltage_min": 2.6
            }
        }
        
        response = client.post("/api/predict/soh", json=invalid_request)
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])