"""
BatteryIQ RUL Deployment Model Training.

This module trains final deployment models AFTER the LOGO validation
experiments have been completed.

Models:
    1. XGBoost
    2. LSTM
    3. GRU

Important:
    - Only official valid RUL targets are used.
    - B0007 remains right-censored and is excluded from supervised RUL.
    - XGBoost uses cycle-level features.
    - LSTM/GRU use 10-cycle sequences.
    - No random row-wise split is used.
    - Deployment models are trained on all valid supervised data only
      after validation has already been performed separately.
"""

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.rul.sequence_dataset import (
    load_sequence_data,
    prepare_valid_rul_data,
    create_sequences,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
)
from src.rul.lstm_model import build_lstm_model
from src.rul.gru_model import build_gru_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUL_MODEL_ROOT = PROJECT_ROOT / "models" / "rul"

XGB_DIR = RUL_MODEL_ROOT / "xgboost"
LSTM_DIR = RUL_MODEL_ROOT / "lstm"
GRU_DIR = RUL_MODEL_ROOT / "gru"


# ---------------------------------------------------------------------------
# XGBoost deployment model
# ---------------------------------------------------------------------------

def train_xgboost_deployment_model():
    """Train and save the final XGBoost RUL deployment model."""

    data = load_sequence_data()
    valid_data = prepare_valid_rul_data(data)

    X = valid_data[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = valid_data["rul"].astype(float)

    if X.isnull().any().any():
        raise ValueError(
            "RUL deployment features contain missing values."
        )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X, y)

    XGB_DIR.mkdir(parents=True, exist_ok=True)

    model_path = XGB_DIR / "rul_xgboost.json"
    metadata_path = XGB_DIR / "metadata.json"

    model.save_model(model_path)

    metadata = {
        "model_name": "BatteryIQ_XGBoost_RUL",
        "model_type": "XGBoost",
        "target": "rul",
        "target_unit": "cycles",
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(X)),
        "training_batteries": [
            "B0005",
            "B0006",
            "B0018",
        ],
        "excluded_batteries": {
            "B0007": "right-censored; no observed EOL"
        },
        "validation_method": "Leave-One-Battery-Out",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "eol_threshold_soh": 70.0,
        "inference_input": "single cycle feature row",
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nXGBoost deployment model saved:")
    print(f"  {model_path}")
    print(f"  {metadata_path}")


# ---------------------------------------------------------------------------
# Sequence model preparation
# ---------------------------------------------------------------------------

def prepare_sequence_training_data():
    """Create valid 10-cycle RUL sequences."""

    data = load_sequence_data()
    valid_data = prepare_valid_rul_data(data)

    X, y, battery_ids = create_sequences(
        valid_data,
        sequence_length=SEQUENCE_LENGTH,
    )

    return X, y, battery_ids


def fit_sequence_scaler(X):
    """Fit a StandardScaler across all deployment-training sequences."""

    scaler = StandardScaler()

    n_samples, sequence_length, n_features = X.shape

    X_2d = X.reshape(
        -1,
        n_features,
    )

    scaler.fit(X_2d)

    return scaler


def transform_sequences(X, scaler):
    """Apply a fitted scaler while preserving 3D sequence shape."""

    n_samples, sequence_length, n_features = X.shape

    X_2d = X.reshape(
        -1,
        n_features,
    )

    X_scaled = scaler.transform(X_2d)

    return X_scaled.reshape(
        n_samples,
        sequence_length,
        n_features,
    )


# ---------------------------------------------------------------------------
# LSTM deployment model
# ---------------------------------------------------------------------------

def train_lstm_deployment_model(X, y):
    """Train and save the final LSTM RUL deployment model."""

    scaler = fit_sequence_scaler(X)

    X_scaled = transform_sequences(
        X,
        scaler,
    )

    model = build_lstm_model(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
    )

    model.fit(
        X_scaled,
        y,
        epochs=50,
        batch_size=16,
        verbose=1,
        shuffle=False,
    )

    LSTM_DIR.mkdir(parents=True, exist_ok=True)

    model_path = LSTM_DIR / "lstm_rul.keras"
    scaler_path = LSTM_DIR / "scaler.joblib"
    metadata_path = LSTM_DIR / "metadata.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": "BatteryIQ_LSTM_RUL",
        "model_type": "LSTM",
        "target": "rul",
        "target_unit": "cycles",
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": SEQUENCE_LENGTH,
        "training_sequences": int(len(X_scaled)),
        "training_batteries": sorted(
            np.unique(
                # battery IDs are handled by caller metadata
                []
            ).tolist()
        ),
        "validation_method": "Leave-One-Battery-Out",
        "epochs": 50,
        "batch_size": 16,
        "random_state": RANDOM_STATE,
        "eol_threshold_soh": 70.0,
        "inference_input": "10 consecutive cycles from one battery",
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nLSTM deployment model saved:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {metadata_path}")


# ---------------------------------------------------------------------------
# GRU deployment model
# ---------------------------------------------------------------------------

def train_gru_deployment_model(X, y):
    """Train and save the final GRU RUL deployment model."""

    scaler = fit_sequence_scaler(X)

    X_scaled = transform_sequences(
        X,
        scaler,
    )

    model = build_gru_model(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
    )

    model.fit(
        X_scaled,
        y,
        epochs=50,
        batch_size=16,
        verbose=1,
        shuffle=False,
    )

    GRU_DIR.mkdir(parents=True, exist_ok=True)

    model_path = GRU_DIR / "gru_rul.keras"
    scaler_path = GRU_DIR / "scaler.joblib"
    metadata_path = GRU_DIR / "metadata.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": "BatteryIQ_GRU_RUL",
        "model_type": "GRU",
        "target": "rul",
        "target_unit": "cycles",
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": SEQUENCE_LENGTH,
        "training_sequences": int(len(X_scaled)),
        "validation_method": "Leave-One-Battery-Out",
        "epochs": 50,
        "batch_size": 16,
        "random_state": RANDOM_STATE,
        "eol_threshold_soh": 70.0,
        "inference_input": "10 consecutive cycles from one battery",
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nGRU deployment model saved:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {metadata_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BatteryIQ — RUL Deployment Model Training")
    print("=" * 70)

    print("\nTraining final deployment models on validated supervised data.")
    print("B0007 remains excluded because it is right-censored.")

    train_xgboost_deployment_model()

    X, y, battery_ids = prepare_sequence_training_data()

    print("\nSequence training data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(
        "  Batteries:",
        sorted(np.unique(battery_ids).tolist()),
    )

    train_lstm_deployment_model(X, y)
    train_gru_deployment_model(X, y)

    print("\n" + "=" * 70)
    print("RUL deployment training complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()