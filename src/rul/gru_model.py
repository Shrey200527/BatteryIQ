import random

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import calculate_metrics


# Reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# Model configuration
SEQUENCE_LENGTH = 10
N_FEATURES = 15

GRU_UNITS = 64
DENSE_UNITS = 32

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 16


def build_gru_model(
    sequence_length=SEQUENCE_LENGTH,
    n_features=N_FEATURES,
):
    """
    Build the GRU model for RUL prediction.

    Architecture:
        Input sequence
            ↓
        GRU(64)
            ↓
        Dense(32, ReLU)
            ↓
        Dense(1)
            ↓
        RUL
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(sequence_length, n_features)
        ),
        tf.keras.layers.GRU(GRU_UNITS),
        tf.keras.layers.Dense(
            DENSE_UNITS,
            activation="relu"
        ),
        tf.keras.layers.Dense(1),
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
    )

    return model


def fit_scaler(X_train):
    """
    Fit StandardScaler using training sequences only.

    The scaler is fitted only on training data to prevent
    information leakage from the held-out battery.
    """

    n_samples, sequence_length, n_features = X_train.shape

    X_train_flat = X_train.reshape(
        -1,
        n_features
    )

    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    return scaler


def transform_sequences(X, scaler):
    """
    Transform sequences using an already-fitted scaler.
    """

    n_samples, sequence_length, n_features = X.shape

    X_flat = X.reshape(
        -1,
        n_features
    )

    X_scaled = scaler.transform(X_flat)

    return X_scaled.reshape(
        n_samples,
        sequence_length,
        n_features
    )


def train_gru_model(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
):
    """
    Train the GRU model.
    """

    model = build_gru_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
    )

    return model


def evaluate_gru_model(model, X_test, y_test):
    """
    Evaluate the trained GRU model.

    Returns:
        metrics, predictions
    """

    predictions = model.predict(
        X_test,
        verbose=0
    ).reshape(-1)

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return metrics, predictions


if __name__ == "__main__":
    print("GRU RUL model module loaded successfully.")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Number of features: {N_FEATURES}")
    print(f"GRU units: {GRU_UNITS}")
    print(f"Dense units: {DENSE_UNITS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")