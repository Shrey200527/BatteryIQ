import pandas as pd


REQUIRED_TARGET_COLUMNS = {
    "battery_id",
    "cycle",
    "capacity",
    "soh",
}

REQUIRED_FEATURE_COLUMNS = {
    "battery_id",
    "cycle",
}


def validate_target_data(df):
    """
    Validate the battery cycle target dataset.
    """

    errors = []

    # 1. Check required columns
    missing_columns = REQUIRED_TARGET_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing target columns: {sorted(missing_columns)}"
        )

    if errors:
        return errors

    # 2. Check missing values
    missing_values = df[
        ["battery_id", "cycle", "capacity", "soh"]
    ].isnull().sum()

    for column, count in missing_values.items():
        if count > 0:
            errors.append(
                f"{column} contains {count} missing values"
            )

    # 3. Check duplicate battery + cycle pairs
    duplicates = df.duplicated(
        subset=["battery_id", "cycle"]
    ).sum()

    if duplicates > 0:
        errors.append(
            f"Found {duplicates} duplicate battery_id + cycle pairs"
        )

    # 4. Check cycle ordering within each battery
    for battery_id, battery_data in df.groupby("battery_id"):
        cycles = battery_data["cycle"].tolist()

        if cycles != sorted(cycles):
            errors.append(
                f"Cycles are not ordered for battery {battery_id}"
            )

    return errors


def validate_feature_target_alignment(features_df, targets_df):
    """
    Check whether feature and target datasets contain
    matching battery_id + cycle pairs.
    """

    errors = []

    feature_keys = set(
        zip(
            features_df["battery_id"],
            features_df["cycle"],
        )
    )

    target_keys = set(
        zip(
            targets_df["battery_id"],
            targets_df["cycle"],
        )
    )

    missing_targets = feature_keys - target_keys
    missing_features = target_keys - feature_keys

    if missing_targets:
        errors.append(
            f"{len(missing_targets)} feature rows have no matching target"
        )

    if missing_features:
        errors.append(
            f"{len(missing_features)} target rows have no matching feature"
        )

    return errors


def validate_all(features_df, targets_df):
    """
    Run all dashboard data validation checks.
    """

    errors = []

    errors.extend(
        validate_target_data(targets_df)
    )

    errors.extend(
        validate_feature_target_alignment(
            features_df,
            targets_df,
        )
    )

    return errors
