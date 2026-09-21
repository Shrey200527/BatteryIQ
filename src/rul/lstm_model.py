"""
LSTM model for battery Remaining Useful Life (RUL) prediction.

This module contains only the LSTM model definition and training helpers.
Data loading and sequence construction are handled separately.
"""

import random

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import calculate_metrics


# Reproducibility
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# Model configuration
SEQUENCE_LENGTH = 10
N_FEATURES = 15

LSTM_UNITS = 64
DENSE_UNITS = 32

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 16


def build_lstm_model(
    sequence_length=SEQUENCE_LENGTH,
    n_features=N_FEATURES,
):
    """
    Build and compile the LSTM RUL regression model.

    Parameters
    ----------
    sequence_length : int
        Number of consecutive battery cycles in each sequence.

    n_features : int
        Number of predictor features per cycle.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM model.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(sequence_length, n_features)
            ),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dense(DENSE_UNITS, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

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
    Fit a StandardScaler using training data only.

    The scaler is fitted only on the training batteries within
    a LOGO fold to prevent test-battery information leakage.

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences with shape
        (samples, sequence_length, features).

    Returns
    -------
    StandardScaler
        Fitted scaler.
    """

    scaler = StandardScaler()

    n_samples, sequence_length, n_features = X_train.shape

    X_train_2d = X_train.reshape(
        -1,
        n_features,
    )

    scaler.fit(X_train_2d)

    return scaler


def transform_sequences(X, scaler):
    """
    Transform sequence features using a previously fitted scaler.

    Parameters
    ----------
    X : np.ndarray
        Sequence data.

    scaler : StandardScaler
        Scaler fitted on training data.

    Returns
    -------
    np.ndarray
        Scaled sequence data with the original 3D shape.
    """

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


def train_lstm_model(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
):
    """
    Train an LSTM RUL regression model.

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences.

    y_train : np.ndarray
        Training RUL targets.

    epochs : int
        Number of training epochs.

    batch_size : int
        Training batch size.

    Returns
    -------
    tf.keras.Model
        Trained LSTM model.
    """

    model = build_lstm_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False,
    )

    return model


def evaluate_lstm_model(model, X_test, y_test):
    """
    Evaluate a trained LSTM model using the project's
    standard regression metrics.

    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model.

    X_test : np.ndarray
        Test sequences.

    y_test : np.ndarray
        True RUL values.

    Returns
    -------
    dict
        MAE, RMSE and R2 metrics.
    """

    predictions = model.predict(
        X_test,
        verbose=0,
    ).reshape(-1)

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    return metrics, predictions


if __name__ == "__main__":
    print("LSTM RUL model module loaded successfully.")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Number of features: {N_FEATURES}")
    print(f"LSTM units: {LSTM_UNITS}")
    print(f"Dense units: {DENSE_UNITS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")