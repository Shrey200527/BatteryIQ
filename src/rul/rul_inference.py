"""
BatteryIQ RUL Deployment Inference.

Provides inference for the three validated RUL model families:

    - XGBoost: single cycle -> RUL
    - LSTM: 10 consecutive cycles -> RUL
    - GRU: 10 consecutive cycles -> RUL

Important:
    - No training occurs here.
    - Models are loaded from deployment artifacts.
    - B0007 is right-censored and is not used as a supervised RUL target.
    - All 15 predictor features must match the frozen ML contract exactly.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

XGB_DIR = PROJECT_ROOT / "models" / "rul" / "xgboost"
LSTM_DIR = PROJECT_ROOT / "models" / "rul" / "lstm"
GRU_DIR = PROJECT_ROOT / "models" / "rul" / "gru"


XGB_MODEL_PATH = XGB_DIR / "rul_xgboost.json"

LSTM_MODEL_PATH = LSTM_DIR / "lstm_rul.keras"
LSTM_SCALER_PATH = LSTM_DIR / "scaler.joblib"

GRU_MODEL_PATH = GRU_DIR / "gru_rul.keras"
GRU_SCALER_PATH = GRU_DIR / "scaler.joblib"


# ---------------------------------------------------------------------------
# Frozen ML contract
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
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

SEQUENCE_LENGTH = 10


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_cycle_features(
    input_data: Any,
) -> pd.DataFrame:
    """
    Validate a cycle-level feature record.

    Accepted input:
        dict
        pandas.DataFrame

    Returns:
        DataFrame with exact frozen feature order.
    """

    if isinstance(input_data, dict):
        data = pd.DataFrame([input_data])

    elif isinstance(input_data, pd.DataFrame):
        data = input_data.copy()

    else:
        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    extra = [
        column
        for column in data.columns
        if column not in FEATURE_COLUMNS
    ]

    if missing:
        raise ValueError(
            f"Missing required RUL features: {missing}"
        )

    if extra:
        raise ValueError(
            f"Unexpected RUL features: {extra}"
        )

    data = data[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data.isnull().any().any():
        bad_columns = data.columns[
            data.isnull().any()
        ].tolist()

        raise ValueError(
            "RUL input contains missing or non-numeric "
            f"values in: {bad_columns}"
        )

    return data


def _validate_sequence(
    sequence: Any,
) -> np.ndarray:
    """
    Validate an LSTM/GRU sequence.

    Required shape:
        (10, 15)

    The sequence must represent 10 consecutive cycles from one battery.
    """

    if isinstance(sequence, pd.DataFrame):
        data = _validate_cycle_features(sequence)

        if len(data) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence must contain exactly "
                f"{SEQUENCE_LENGTH} cycles; received {len(data)}."
            )

        values = data.to_numpy(dtype=np.float32)

    elif isinstance(sequence, np.ndarray):
        values = np.asarray(
            sequence,
            dtype=np.float32,
        )

        if values.shape != (
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS),
        ):
            raise ValueError(
                "Sequence must have shape "
                f"({SEQUENCE_LENGTH}, {len(FEATURE_COLUMNS)}); "
                f"received {values.shape}."
            )

    else:
        raise TypeError(
            "Sequence must be a pandas DataFrame or "
            "numpy.ndarray."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Sequence contains non-finite values."
        )

    return values


# ---------------------------------------------------------------------------
# XGBoost RUL predictor
# ---------------------------------------------------------------------------

class XGBoostRULPredictor:
    """Deployment wrapper for the XGBoost RUL model."""

    def __init__(
        self,
        model_path: Path = XGB_MODEL_PATH,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"XGBoost RUL model not found: "
                f"{self.model_path}"
            )

        self.model = XGBRegressor()
        self.model.load_model(self.model_path)

    def predict(
        self,
        input_data: Any,
    ) -> np.ndarray:
        """Predict RUL from one or more cycle-level records."""

        X = _validate_cycle_features(input_data)

        predictions = self.model.predict(X)

        # RUL cannot be negative under the project contract.
        predictions = np.maximum(
            predictions,
            0.0,
        )

        return predictions.astype(float)

    def predict_one(
        self,
        input_data: Any,
    ) -> float:
        """Predict RUL for one cycle."""

        predictions = self.predict(input_data)

        if len(predictions) != 1:
            raise ValueError(
                "predict_one expects exactly one input record."
            )

        return float(predictions[0])


# ---------------------------------------------------------------------------
# LSTM RUL predictor
# ---------------------------------------------------------------------------

class LSTMRULPredictor:
    """Deployment wrapper for the LSTM RUL model."""

    def __init__(
        self,
        model_path: Path = LSTM_MODEL_PATH,
        scaler_path: Path = LSTM_SCALER_PATH,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LSTM RUL model not found: "
                f"{self.model_path}"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"LSTM RUL scaler not found: "
                f"{self.scaler_path}"
            )

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

    def predict(
        self,
        sequence: Any,
    ) -> float:
        """Predict RUL from one 10-cycle sequence."""

        X = _validate_sequence(sequence)

        X_scaled = self.scaler.transform(
            X
        )

        X_scaled = X_scaled.reshape(
            1,
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS),
        )

        prediction = self.model.predict(
            X_scaled,
            verbose=0,
        ).reshape(-1)

        rul = max(
            float(prediction[0]),
            0.0,
        )

        return rul


# ---------------------------------------------------------------------------
# GRU RUL predictor
# ---------------------------------------------------------------------------

class GRURULPredictor:
    """Deployment wrapper for the GRU RUL model."""

    def __init__(
        self,
        model_path: Path = GRU_MODEL_PATH,
        scaler_path: Path = GRU_SCALER_PATH,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GRU RUL model not found: "
                f"{self.model_path}"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"GRU RUL scaler not found: "
                f"{self.scaler_path}"
            )

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

    def predict(
        self,
        sequence: Any,
    ) -> float:
        """Predict RUL from one 10-cycle sequence."""

        X = _validate_sequence(sequence)

        X_scaled = self.scaler.transform(
            X
        )

        X_scaled = X_scaled.reshape(
            1,
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS),
        )

        prediction = self.model.predict(
            X_scaled,
            verbose=0,
        ).reshape(-1)

        rul = max(
            float(prediction[0]),
            0.0,
        )

        return rul


# ---------------------------------------------------------------------------
# Lazy-loaded project predictors
# ---------------------------------------------------------------------------

_xgb_predictor = None
_lstm_predictor = None
_gru_predictor = None


def _get_xgb_predictor():
    global _xgb_predictor

    if _xgb_predictor is None:
        _xgb_predictor = XGBoostRULPredictor()

    return _xgb_predictor


def _get_lstm_predictor():
    global _lstm_predictor

    if _lstm_predictor is None:
        _lstm_predictor = LSTMRULPredictor()

    return _lstm_predictor


def _get_gru_predictor():
    global _gru_predictor

    if _gru_predictor is None:
        _gru_predictor = GRURULPredictor()

    return _gru_predictor


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

def predict_xgboost_rul(
    input_data: Any,
) -> dict:
    """
    Predict RUL using XGBoost.

    Input:
        One cycle-level feature record.

    Returns:
        Structured RUL prediction.
    """

    predictor = _get_xgb_predictor()

    rul = predictor.predict_one(
        input_data
    )

    return {
        "model": "BatteryIQ_XGBoost_RUL",
        "target": "rul",
        "prediction": rul,
        "unit": "cycles",
    }


def predict_lstm_rul(
    sequence: Any,
) -> dict:
    """
    Predict RUL using LSTM.

    Input:
        Exactly 10 consecutive cycles from one battery.
    """

    predictor = _get_lstm_predictor()

    rul = predictor.predict(
        sequence
    )

    return {
        "model": "BatteryIQ_LSTM_RUL",
        "target": "rul",
        "prediction": rul,
        "unit": "cycles",
        "sequence_length": SEQUENCE_LENGTH,
    }


def predict_gru_rul(
    sequence: Any,
) -> dict:
    """
    Predict RUL using GRU.

    Input:
        Exactly 10 consecutive cycles from one battery.
    """

    predictor = _get_gru_predictor()

    rul = predictor.predict(
        sequence
    )

    return {
        "model": "BatteryIQ_GRU_RUL",
        "target": "rul",
        "prediction": rul,
        "unit": "cycles",
        "sequence_length": SEQUENCE_LENGTH,
    }


# ---------------------------------------------------------------------------
# Unified RUL interface
# ---------------------------------------------------------------------------

def predict_rul(
    input_data: Any,
    model: str = "xgboost",
) -> dict:
    """
    Unified RUL inference interface.

    Parameters
    ----------
    input_data:
        Cycle feature dictionary/DataFrame for XGBoost,
        or 10-cycle sequence for LSTM/GRU.

    model:
        "xgboost", "lstm", or "gru".
    """

    model = model.lower().strip()

    if model == "xgboost":
        return predict_xgboost_rul(
            input_data
        )

    if model == "lstm":
        return predict_lstm_rul(
            input_data
        )

    if model == "gru":
        return predict_gru_rul(
            input_data
        )

    raise ValueError(
        "Unsupported RUL model. "
        "Choose one of: xgboost, lstm, gru."
    )