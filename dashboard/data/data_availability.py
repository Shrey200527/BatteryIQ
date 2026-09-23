from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_available_data_sources():
    """
    Report which BatteryIQ datasets exist.
    """

    feature_dir = PROJECT_ROOT / "data" / "features"

    sources = {
        "battery_cycle_features": (
            feature_dir / "battery_cycle_features.csv"
        ).exists(),

        "battery_cycle_targets": (
            feature_dir / "battery_cycle_targets.csv"
        ).exists(),

        "rul_targets": (
            feature_dir / "rul_targets.csv"
        ).exists(),

        "raw_measurements": False,
    }

    return sources


def is_measurement_data_available():
    """
    Check if raw measurement curves exist.
    """

    sources = get_available_data_sources()

    return sources["raw_measurements"]
