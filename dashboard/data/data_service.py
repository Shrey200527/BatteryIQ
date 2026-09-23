
from pathlib import Path

import pandas as pd


# Project root:
# BatteryIQ/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "features"

TARGETS_FILE = FEATURES_DIR / "battery_cycle_targets.csv"
FEATURES_FILE = FEATURES_DIR / "battery_cycle_features.csv"


# Load the datasets once when this module is imported.
targets_df = pd.read_csv(TARGETS_FILE)
features_df = pd.read_csv(FEATURES_FILE)


def get_batteries():
    """Return the list of available battery IDs."""
    return sorted(targets_df["battery_id"].unique().tolist())

def battery_exists(battery_id):
    """Check if a battery ID exists in the targets dataset."""
    return battery_id in targets_df["battery_id"].unique()

def get_battery_history(battery_id):
    """Return dashboard-ready cycle history for one battery.

    Combines approved target data (capacity, SOH) with
    approved cycle-level feature data (voltage, current,
    temperature) using battery_id + cycle as the join key.
    """

    target_data = targets_df[
        targets_df["battery_id"] == battery_id
    ][
        ["battery_id", "cycle", "capacity", "soh"]
    ]

    feature_columns = [
        "battery_id",
        "cycle",
        "voltage_mean",
        "current_mean",
        "temperature_mean",
    ]

    feature_data = features_df[
        features_df["battery_id"] == battery_id
    ][feature_columns]

    history = target_data.merge(
        feature_data,
        on=["battery_id", "cycle"],
        how="inner",
        validate="one_to_one",
    ).sort_values("cycle")

    return history.to_dict(orient="records")


def get_soh_history(battery_id):
    """Return SOH history for one battery."""
    battery_data = targets_df[
        targets_df["battery_id"] == battery_id
    ].sort_values("cycle")

    return battery_data[
        ["battery_id", "cycle", "soh"]
    ].to_dict(orient="records")


def get_capacity_history(battery_id):
    """Return capacity history for one battery."""
    battery_data = targets_df[
        targets_df["battery_id"] == battery_id
    ].sort_values("cycle")

    return battery_data[
        ["battery_id", "cycle", "capacity"]
    ].to_dict(orient="records")


def get_latest_features(battery_id):
    """Return the latest feature row for one battery."""
    battery_data = features_df[
        features_df["battery_id"] == battery_id
    ].sort_values("cycle")

    if battery_data.empty:
        return None

    return battery_data.iloc[-1].to_dict()

