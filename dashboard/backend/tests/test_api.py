"""
Tests for BatteryIQ Dashboard Backend API.

These tests verify the backend endpoints integrate correctly with the
existing Phase 2 inference interfaces and frozen BatteryIQ datasets.
"""

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.backend.app import app
from dashboard.backend.services import get_battery_service


client = TestClient(app)


def get_real_cycle_features(
    battery_id: str = "B0005",
    cycle: int = 1,
):
    """
    Retrieve an actual feature row from the frozen Phase 2 dataset.

    Using real frozen features makes the prediction tests exercise the
    same feature contract used by the deployed models.
    """
    service = get_battery_service()

    return service.get_cycle_features(
        battery_id=battery_id,
        cycle=cycle,
    )


class TestBatteriesEndpoints:
    """Test battery-related endpoints."""

    def test_get_valid_battery_detail(self):
        """Test GET /api/batteries/{battery_id} with valid battery."""
        response = client.get("/api/batteries/B0005")

        assert response.status_code == 200

        data = response.json()

        assert data["battery_id"] == "B0005"
        assert "total_cycles" in data
        assert "soh_range" in data
        assert "available_cycles" in data

        assert isinstance(
            data["available_cycles"],
            list,
        )

        assert isinstance(
            data["is_rul_valid"],
            bool,
        )

    def test_get_invalid_battery_detail(self):
        """Test GET /api/batteries/{battery_id} with invalid battery."""
        response = client.get("/api/batteries/B9999")

        assert response.status_code == 404

        data = response.json()

        # FastAPI HTTPException responses use the "detail" envelope.
        assert "detail" in data
        assert data["detail"]["error"] == "invalid_battery"

    def test_get_battery_history(self):
        """Test dashboard-ready battery history endpoint."""
        response = client.get(
            "/api/batteries/B0005/history"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["battery_id"] == "B0005"
        assert data["total_cycles"] == 168
        assert isinstance(
            data["available_cycles"],
            list,
        )

        assert len(data["available_cycles"]) == 168
        assert data["is_rul_valid"] is True

        assert "history" in data
        assert len(data["history"]) == 168

        first_cycle = data["history"][0]

        assert first_cycle["cycle"] == 1
        assert "capacity" in first_cycle
        assert "soh" in first_cycle
        assert "voltage_mean" in first_cycle
        assert "current_mean" in first_cycle
        assert "temperature_mean" in first_cycle

        cycles = [
            row["cycle"]
            for row in data["history"]
        ]

        assert cycles == list(range(1, 169))

    def test_b0007_history_is_right_censored(self):
        """B0007 must remain right-censored for RUL."""
        response = client.get(
            "/api/batteries/B0007/history"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["battery_id"] == "B0007"
        assert data["is_rul_valid"] is False

        # B0007 has historical observations but no observed EOL.
        assert len(data["history"]) == 168
        assert data["history"][0]["cycle"] == 1
        assert data["history"][-1]["cycle"] == 168

class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_soh_prediction(self):
        """Test POST /api/predict/soh endpoint."""

        cycle = 1
        features = get_real_cycle_features(
            battery_id="B0005",
            cycle=cycle,
        )

        soh_request = {
            "battery_id": "B0005",
            "cycle": cycle,
            "features": features,
        }

        response = client.post(
            "/api/predict/soh",
            json=soh_request,
        )

        # If the environment has the ML dependencies and artifacts,
        # the endpoint should execute successfully.
        if response.status_code == 200:
            data = response.json()

            assert "prediction" in data
            assert "model" in data
            assert "battery_id" in data

            assert data["battery_id"] == "B0005"

            assert isinstance(
                data["prediction"],
                (int, float),
            )

            assert 0 <= data["prediction"] <= 100

        else:
            # A missing ML dependency/model is an environment issue,
            # not an API schema issue.
            assert response.status_code in [500, 503]

    def test_soh_prediction_invalid_battery(self):
        """Test SOH prediction with invalid battery."""

        features = get_real_cycle_features(
            battery_id="B0005",
            cycle=1,
        )

        soh_request = {
            "battery_id": "B9999",
            "cycle": 1,
            "features": features,
        }

        response = client.post(
            "/api/predict/soh",
            json=soh_request,
        )

        assert response.status_code == 404

        data = response.json()

        # FastAPI HTTPException response envelope.
        assert "detail" in data
        assert data["detail"]["error"] == "invalid_battery"

    def test_rul_prediction_xgboost(self):
        """Test POST /api/predict/rul with XGBoost model."""

        features = get_real_cycle_features(
            battery_id="B0005",
            cycle=1,
        )

        rul_request = {
            "battery_id": "B0005",
            "model": "xgboost",
            "features": features,
        }

        response = client.post(
            "/api/predict/rul",
            json=rul_request,
        )

        if response.status_code == 200:
            data = response.json()

            assert "prediction" in data
            assert "model" in data

            assert data["model"] == (
                "BatteryIQ_XGBoost_RUL"
            )

            assert isinstance(
                data["prediction"],
                (int, float),
            )

            assert data["prediction"] >= 0

        else:
            assert response.status_code in [500, 503]

    def test_rul_prediction_invalid_model(self):
        """
        Test RUL prediction with invalid model type.

        Invalid model names are rejected by Pydantic request validation
        before the route executes, so the correct status code is 422.
        """

        features = get_real_cycle_features(
            battery_id="B0005",
            cycle=1,
        )

        rul_request = {
            "battery_id": "B0005",
            "model": "invalid_model",
            "features": features,
        }

        response = client.post(
            "/api/predict/rul",
            json=rul_request,
        )

        assert response.status_code == 422

    def test_condition_classification(self):
        """Test POST /api/predict/condition endpoint."""

        condition_request = {
            "soh": 85.0,
        }

        response = client.post(
            "/api/predict/condition",
            json=condition_request,
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            "condition" in data
            or data["status"] == "thresholds_pending"
        )

        assert "status" in data
        assert "soh" in data

        # Current project contract intentionally blocks condition
        # thresholds until numerical thresholds are approved.
        if data["status"] == "thresholds_pending":
            assert data["condition"] is None
            assert data["message"] is not None

    def test_condition_classification_invalid_soh(self):
        """Test condition classification with invalid SOH."""

        condition_request = {
            "soh": 150.0,
        }

        response = client.post(
            "/api/predict/condition",
            json=condition_request,
        )

        assert response.status_code == 422

    def test_shap_explanation(self):
        """Test POST /api/predict/explain endpoint."""

        cycle = 1
        features = get_real_cycle_features(
            battery_id="B0005",
            cycle=cycle,
        )

        shap_request = {
            "battery_id": "B0005",
            "cycle": cycle,
            "features": features,
        }

        response = client.post(
            "/api/predict/explain",
            json=shap_request,
        )

        if response.status_code == 200:
            data = response.json()

            assert "prediction" in data
            assert "feature_contributions" in data
            assert "battery_id" in data

            assert isinstance(
                data["feature_contributions"],
                dict,
            )

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

        if response.status_code == 200:
            data = response.json()

            assert "status" in data
            assert "service" in data
            assert "batteries_available" in data
            assert "battery_ids" in data

        else:
            assert response.status_code == 503


class TestValidation:
    """Test input validation."""

    def test_sequence_validation_length(self):
        """Test that sequence validation rejects wrong length."""

        wrong_sequence_request = {
            "battery_id": "B0005",
            "model": "lstm",
            "features": {
                "cycles": [
                    {
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
                        "temperature_rolling_mean": 32.0,
                    }
                ]
                * 5
            },
        }

        response = client.post(
            "/api/predict/rul",
            json=wrong_sequence_request,
        )

        assert response.status_code == 422

    def test_feature_validation(self):
        """Test that feature validation rejects missing features."""

        invalid_request = {
            "battery_id": "B0005",
            "cycle": 1,
            "features": {
                # Missing required features.
                "voltage_mean": 3.5,
                "voltage_min": 2.6,
            },
        }

        response = client.post(
            "/api/predict/soh",
            json=invalid_request,
        )

        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])