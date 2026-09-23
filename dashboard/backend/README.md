# BatteryIQ Dashboard Backend

REST API for BatteryIQ battery health predictions and analysis.

## Overview

The dashboard backend provides a REST API that integrates with the existing Phase 2 ML inference interfaces, enabling frontend applications to access battery health predictions without implementing ML logic.

## Architecture

```
NASA Battery Dataset
        ↓
Frozen Phase 2 Data / Features
        ↓
Existing ML Models (SOH, RUL, SHAP)
        ↓
Existing Inference Interfaces
        ↓
Dashboard Backend (This Layer)
        ↓
REST API
        ↓
Dashboard Frontend
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The backend requires the existing Phase 2 model artifacts and data files:
- `models/soh/ann/` - ANN SOH model artifacts
- `models/rul/xgboost/` - XGBoost RUL model artifacts  
- `models/rul/lstm/` - LSTM RUL model artifacts
- `models/rul/gru/` - GRU RUL model artifacts
- `data/features/` - Frozen feature datasets

## Startup

Start the backend server:

```bash
uvicorn dashboard.backend.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

## Available Endpoints

### Battery Endpoints

#### `GET /api/batteries`
Get list of all available batteries.

**Response:**
```json
{
  "batteries": [
    {
      "battery_id": "B0005",
      "total_cycles": 168,
      "is_rul_valid": true
    }
  ],
  "total": 4
}
```

#### `GET /api/batteries/{battery_id}`
Get detailed information about a specific battery.

**Response:**
```json
{
  "battery_id": "B0005",
  "total_cycles": 168,
  "is_rul_valid": true,
  "soh_range": {"min": 70.1, "max": 100.0},
  "available_cycles": [1, 2, 3, ..., 168]
}
```

#### `GET /api/batteries/{battery_id}/history`
Get historical data for a specific battery.

### Prediction Endpoints

#### `POST /api/predict/soh`
Predict State of Health (SOH) for a battery cycle.

**Request:**
```json
{
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
```

**Response:**
```json
{
  "model": "BatteryIQ_ANN_SOH",
  "target": "soh",
  "prediction": 95.5,
  "unit": "percent",
  "battery_id": "B0005",
  "cycle": 1
}
```

#### `POST /api/predict/rul`
Predict Remaining Useful Life (RUL) for a battery.

**Request (XGBoost - single cycle):**
```json
{
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
```

**Request (LSTM/GRU - 10 consecutive cycles):**
```json
{
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
        "temperature_rolling_mean": 32.0
      }
      // ... 9 more cycles
    ]
  }
}
```

**Response:**
```json
{
  "model": "BatteryIQ_XGBoost_RUL",
  "target": "rul",
  "prediction": 45.0,
  "unit": "cycles",
  "battery_id": "B0005",
  "sequence_length": null
}
```

#### `POST /api/predict/condition`
Classify battery condition based on SOH.

**Request:**
```json
{
  "soh": 85.0
}
```

**Response (when thresholds are available):**
```json
{
  "condition": "Healthy",
  "status": "available",
  "soh": 85.0,
  "message": null
}
```

**Response (when thresholds are pending):**
```json
{
  "condition": null,
  "status": "thresholds_pending",
  "soh": 85.0,
  "message": "Classification thresholds not yet approved by team. See ML contract."
}
```

#### `POST /api/predict/explain`
Generate SHAP explanation for a prediction.

**Request:**
```json
{
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
```

**Response:**
```json
{
  "battery_id": "B0005",
  "cycle": 1,
  "prediction": 95.5,
  "base_value": 85.0,
  "feature_contributions": {
    "voltage_mean": {
      "shap_value": 2.5,
      "feature_value": 3.5
    },
    "temperature_mean": {
      "shap_value": 1.2,
      "feature_value": 32.0
    }
  }
}
```

## Model Types

### SOH Models
- **ANN SOH**: Advanced Neural Network for SOH prediction

### RUL Models
- **XGBoost RUL**: Single cycle input → RUL prediction
- **LSTM RUL**: 10 consecutive cycles → RUL prediction  
- **GRU RUL**: 10 consecutive cycles → RUL prediction

## Important Notes

### Feature Contract
All prediction requests must use the exact 15 frozen features from the ML contract:
- voltage_mean, voltage_min, voltage_max, voltage_std
- current_mean, current_min, current_max
- temperature_mean, temperature_max
- current_load_mean, voltage_load_mean
- cycle_time, voltage_change, temperature_change, temperature_rolling_mean

### Sequence Requirements (LSTM/GRU)
- Exactly 10 consecutive cycles from the same battery
- Cycles must be consecutive (no gaps)
- Same battery ID for all cycles
- Same 15 features per cycle

### Battery RUL Validity
- B0005, B0006, B0018: RUL predictions available
- B0007: RUL predictions not available (right-censored)

### Condition Classification
- Currently blocked pending team-approved numerical thresholds
- API returns `status: "thresholds_pending"` when unavailable
- Frontend should handle this gracefully

### Error Handling
All endpoints return structured error responses:
```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "details": "Additional error details"
}
```

## Testing

Run the test suite:

```bash
pytest dashboard/backend/tests/test_api.py -v
```

## Project Structure

```
dashboard/backend/
├── app.py                 # Main FastAPI application
├── routes/
│   ├── batteries.py       # Battery-related endpoints
│   └── predictions.py     # Prediction endpoints
├── services/
│   ├── battery_service.py # Battery data operations
│   └── prediction_service.py # ML prediction integration
├── schemas/
│   └── schemas.py         # Pydantic schemas for validation
├── tests/
│   └── test_api.py        # API tests
└── README.md             # This file
```

## Integration with Phase 2

The backend is designed as an integration layer that:
- **Does NOT retrain models**
- **Does NOT modify ML architectures**
- **Does NOT change feature definitions**
- **Does NOT invent thresholds or confidence scores**
- **Calls existing Phase 2 inference interfaces**
- **Preserves ML contract compliance**

## Dependencies

- FastAPI: Web framework
- Pydantic: Data validation
- Uvicorn: ASGI server
- Existing Phase 2 dependencies (numpy, pandas, tensorflow, xgboost, shap, etc.)

## Support

For issues related to:
- **Backend API**: Check this README and `/docs` endpoint
- **ML Models**: See Phase 2 documentation and notebooks
- **Data**: See `docs/DATA_CONTRACT.md` and `docs/ML_CONTRACT.md`