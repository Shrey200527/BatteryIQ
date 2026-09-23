from .data_service import targets_df


def get_battery_metadata():
    """
    Return metadata for all batteries.
    """

    metadata = []

    for battery_id, battery_data in targets_df.groupby("battery_id"):
        battery_data = battery_data.sort_values("cycle")

        metadata.append({
            "battery_id": battery_id,
            "number_of_cycles": int(battery_data["cycle"].max()),
            "available_observations": int(len(battery_data)),
            "eol_status": (
                "Not observed"
                if battery_id == "B0007"
                else "Observed"
            ),
            "rul_supervision_status": (
                "Right-censored"
                if battery_id == "B0007"
                else "Supervised"
            ),
        })

    return metadata


def get_battery_metadata_by_id(battery_id):
    """
    Return metadata for one battery.
    """

    metadata = get_battery_metadata()

    for battery in metadata:
        if battery["battery_id"] == battery_id:
            return battery

    return None