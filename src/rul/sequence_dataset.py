"""
Sequence dataset construction for BatteryIQ RUL models.

This module converts cycle-level battery features and official RUL targets
into fixed-length sequences for LSTM/GRU models.

Sequences are constructed independently within each battery and therefore
never cross battery boundaries.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "battery_cycle_features.csv"
RUL_TARGETS_PATH = PROJECT_ROOT / "data" / "features" / "rul_targets.csv"


# Frozen predictor feature order from the ML contract.
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


# Project design choice for sequence-based RUL models.
SEQUENCE_LENGTH = 10


REQUIRED_FEATURE_COLUMNS = [
    "battery_id",
    "cycle",
    *FEATURE_COLUMNS,
]

REQUIRED_RUL_COLUMNS = [
    "battery_id",
    "cycle",
    "rul",
    "rul_valid",
]


def load_sequence_data():
    """Load cycle features and merge them with official RUL targets."""

    features = pd.read_csv(FEATURES_PATH)
    rul_targets = pd.read_csv(RUL_TARGETS_PATH)

    missing_features = [
        column
        for column in REQUIRED_FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    missing_rul = [
        column
        for column in REQUIRED_RUL_COLUMNS
        if column not in rul_targets.columns
    ]

    if missing_rul:
        raise ValueError(
            f"Missing RUL columns: {missing_rul}"
        )

    data = features.merge(
        rul_targets,
        on=["battery_id", "cycle"],
        how="inner",
        validate="one_to_one",
    )

    return data


def prepare_valid_rul_data(data):
    """
    Keep only rows with valid official RUL targets.

    Batteries without an observed EOL, such as B0007, therefore do not
    contribute supervised RUL samples.
    """

    valid_data = data.loc[data["rul_valid"] == True].copy()

    if valid_data.empty:
        raise ValueError("No valid RUL observations found.")

    valid_data["rul"] = pd.to_numeric(
        valid_data["rul"],
        errors="coerce",
    )

    if valid_data["rul"].isna().any():
        raise ValueError(
            "Valid RUL rows contain missing RUL values."
        )

    if (valid_data["rul"] < 0).any():
        raise ValueError(
            "RUL values cannot be negative."
        )

    valid_data = valid_data.sort_values(
        ["battery_id", "cycle"]
    ).reset_index(drop=True)

    return valid_data


def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
    """
    Create fixed-length sequences independently for each battery.

    The target for each sequence is the RUL corresponding to the final
    cycle in that sequence.

    Parameters
    ----------
    data : pandas.DataFrame
        Valid RUL cycle-level data.

    sequence_length : int
        Number of consecutive cycles in each input sequence.

    Returns
    -------
    X : numpy.ndarray
        Shape: (n_sequences, sequence_length, n_features)

    y : numpy.ndarray
        Shape: (n_sequences,)

    battery_ids : numpy.ndarray
        Battery ID corresponding to each sequence.
    """

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    sequences = []
    targets = []
    battery_ids = []

    for battery_id, battery_data in data.groupby(
        "battery_id",
        sort=False,
    ):
        battery_data = battery_data.sort_values(
            "cycle"
        ).reset_index(drop=True)

        if len(battery_data) < sequence_length:
            continue

        feature_values = battery_data[
            FEATURE_COLUMNS
        ].to_numpy(dtype=float)

        rul_values = battery_data[
            "rul"
        ].to_numpy(dtype=float)

        if not np.isfinite(feature_values).all():
            raise ValueError(
                f"Missing or non-finite predictor values found "
                f"in {battery_id}."
            )

        for end_index in range(
            sequence_length - 1,
            len(battery_data),
        ):
            start_index = (
                end_index - sequence_length + 1
            )

            # Explicitly verify that the sequence contains
            # consecutive battery cycles.
            cycle_numbers = battery_data.loc[
                start_index:end_index,
                "cycle",
            ].to_numpy()

            if not np.all(
                np.diff(cycle_numbers) == 1
            ):
                raise ValueError(
                    f"Non-consecutive cycles found in "
                    f"{battery_id}: "
                    f"{cycle_numbers.tolist()}"
                )

            sequence = feature_values[
                start_index:end_index + 1
            ]

            target = rul_values[end_index]

            sequences.append(sequence)
            targets.append(target)
            battery_ids.append(battery_id)

    if not sequences:
        raise ValueError(
            "No sequences could be created. "
            "Check sequence_length and valid RUL data."
        )

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    )

    battery_ids = np.asarray(
        battery_ids
    )

    return X, y, battery_ids


def validate_sequences(
    X,
    y,
    battery_ids,
    sequence_length=SEQUENCE_LENGTH,
):
    """Validate the generated sequence dataset."""

    if X.ndim != 3:
        raise ValueError(
            f"X must be 3-dimensional, "
            f"got shape {X.shape}."
        )

    if X.shape[1] != sequence_length:
        raise ValueError(
            f"Expected sequence length "
            f"{sequence_length}, got {X.shape[1]}."
        )

    if X.shape[2] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features, "
            f"got {X.shape[2]}."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number "
            "of sequences."
        )

    if len(X) != len(battery_ids):
        raise ValueError(
            "X and battery_ids must contain the same "
            "number of sequences."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains non-finite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains non-finite values."
        )

    if (y < 0).any():
        raise ValueError(
            "RUL targets cannot be negative."
        )


def main():
    """Load, construct, and validate the sequence dataset."""

    data = load_sequence_data()

    valid_data = prepare_valid_rul_data(
        data
    )

    X, y, battery_ids = create_sequences(
        valid_data,
        sequence_length=SEQUENCE_LENGTH,
    )

    validate_sequences(
        X,
        y,
        battery_ids,
        sequence_length=SEQUENCE_LENGTH,
    )

    print(
        "Sequence dataset created successfully."
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    print(
        f"Number of features: {X.shape[2]}"
    )

    print(
        f"Number of sequences: {X.shape[0]}"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print("\nSequences by battery:")

    print(
        pd.Series(battery_ids)
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()