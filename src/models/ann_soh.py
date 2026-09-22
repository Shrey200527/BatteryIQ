"""
ANN-based SOH prediction model for BatteryIQ.

This module contains the reusable TensorFlow/Keras ANN architecture used for
SOH prediction. Validation is performed by the notebook using
Leave-One-Battery-Out (LOGO); this module deliberately does not implement
data splitting or target generation.
"""

from __future__ import annotations

import os
import random
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

TARGET_COLUMN = "soh"
RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set Python, NumPy and TensorFlow seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_ann_soh(input_dim: int = 15, seed: int = RANDOM_SEED) -> keras.Model:
    """Build a fresh ANN SOH regressor."""
    if input_dim != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} input features, got {input_dim}."
        )

    set_random_seed(seed)

    inputs = keras.Input(shape=(input_dim,), name="battery_cycle_features")
    x = layers.Dense(64, activation="relu", name="dense_64")(inputs)
    x = layers.Dropout(0.20, name="dropout_20")(x)
    x = layers.Dense(32, activation="relu", name="dense_32")(x)
    x = layers.Dropout(0.10, name="dropout_10")(x)
    x = layers.Dense(16, activation="relu", name="dense_16")(x)
    outputs = layers.Dense(1, activation="linear", name="soh")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BatteryIQ_ANN_SOH")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_ann_soh(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 500,
    batch_size: int = 16,
    validation_split: float = 0.15,
    patience: int = 40,
    seed: int = RANDOM_SEED,
    verbose: int = 0,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Build and train a fresh ANN on already-preprocessed training data."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)

    if X_train.ndim != 2 or X_train.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"X_train must have shape (n_samples, {len(FEATURE_COLUMNS)}), "
            f"got {X_train.shape}."
        )
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")

    model = build_ann_soh(input_dim=X_train.shape[1], seed=seed)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history


def validate_feature_columns(columns: Sequence[str] | Iterable[str]) -> None:
    """Validate exact feature names and order for inference."""
    columns = list(columns)
    if columns != FEATURE_COLUMNS:
        raise ValueError(
            "ANN SOH feature contract mismatch.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Received: {columns}"
        )
