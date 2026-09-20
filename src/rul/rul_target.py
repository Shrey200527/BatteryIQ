"""
RUL Target Construction for BatteryIQ Phase 2.

Responsibilities:
- Detect battery End-of-Life (EOL): first cycle where SOH <= 70%
- Calculate Remaining Useful Life (RUL) = EOL cycle - current cycle
- Handle B0007 (never reaches EOL) as censored / rul_valid=False
- Generate data/features/rul_targets.csv

EOL condition : SOH <= 70%
RUL formula   : RUL = EOL cycle - current cycle  (0 at EOL, None beyond)
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "battery_cycle_targets.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rul_targets.csv"
)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_target_data() -> pd.DataFrame:
    """Load the frozen Phase 1 target dataset (battery_cycle_targets.csv)."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("Dataset loaded successfully.")
    print("  Shape   :", df.shape)
    print("  Columns :", list(df.columns))
    print("  Batteries:", sorted(df["battery_id"].unique()))

    return df


# ---------------------------------------------------------------------------
# Validate input
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame) -> None:
    """Ensure required columns exist in the input dataframe."""

    required = {"battery_id", "cycle", "soh"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")


# ---------------------------------------------------------------------------
# Construct RUL targets
# ---------------------------------------------------------------------------

def construct_rul_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the RUL target table from the SOH dataset.

    For each battery:
      - Find the first cycle where SOH <= 70  → EOL cycle
      - For cycles <= EOL : RUL = EOL cycle - current cycle, rul_valid = True
      - For cycles > EOL  : RUL = None,                      rul_valid = False
      - If EOL never reached (B0007): all rows get RUL = None, rul_valid = False

    Returns a DataFrame with columns: battery_id, cycle, rul, rul_valid
    """

    result_rows = []

    for battery_id, group in df.groupby("battery_id"):

        group = group.sort_values("cycle").reset_index(drop=True)

        eol_records = group[group["soh"] <= 70]

        if eol_records.empty:
            # ── B0007-type: EOL never observed → censored ──────────────────
            print(
                f"  {battery_id}: EOL not observed "
                "(SOH never reached 70%). Marking all rows as censored."
            )
            for _, row in group.iterrows():
                result_rows.append({
                    "battery_id": battery_id,
                    "cycle":      int(row["cycle"]),
                    "rul":        None,
                    "rul_valid":  False,
                })

        else:
            # ── Normal battery: EOL at first qualifying cycle ───────────────
            eol_cycle = int(eol_records["cycle"].iloc[0])
            print(f"  {battery_id}: EOL detected at cycle {eol_cycle}")

            for _, row in group.iterrows():
                current_cycle = int(row["cycle"])

                if current_cycle <= eol_cycle:
                    rul = eol_cycle - current_cycle          # 0 at EOL
                    result_rows.append({
                        "battery_id": battery_id,
                        "cycle":      current_cycle,
                        "rul":        rul,
                        "rul_valid":  True,
                    })
                else:
                    # Cycles recorded after EOL → not usable for RUL modelling
                    result_rows.append({
                        "battery_id": battery_id,
                        "cycle":      current_cycle,
                        "rul":        None,
                        "rul_valid":  False,
                    })

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Validate output
# ---------------------------------------------------------------------------

def validate_rul_targets(rul_df: pd.DataFrame) -> None:
    """
    Run correctness checks on the generated RUL dataset.

    Checks:
      □ Required columns present
      □ No duplicate battery_id + cycle combinations
      □ No negative RUL values
      □ RUL == 0 exists for every battery that has valid RUL
      □ RUL decreases monotonically within each valid battery
      □ B0007 is marked as censored (rul_valid=False for all rows)
    """

    required = {"battery_id", "cycle", "rul", "rul_valid"}
    if not required.issubset(rul_df.columns):
        raise ValueError("Required output columns are missing.")

    # Duplicate check
    duplicates = rul_df.duplicated(subset=["battery_id", "cycle"])
    if duplicates.any():
        raise ValueError(
            f"Duplicate battery_id-cycle combinations found:\n"
            f"{rul_df[duplicates]}"
        )

    valid = rul_df[rul_df["rul_valid"] == True].copy()

    # No negative RUL
    if (valid["rul"] < 0).any():
        raise ValueError("Negative RUL values found.")

    # RUL == 0 at EOL for each valid battery
    for bid in valid["battery_id"].unique():
        bdata = valid[valid["battery_id"] == bid]
        if not (bdata["rul"] == 0).any():
            raise ValueError(f"No RUL = 0 found for battery {bid}.")

    # Monotone decrease check
    for bid in valid["battery_id"].unique():
        bdata = valid[valid["battery_id"] == bid].sort_values("cycle")
        if not bdata["rul"].is_monotonic_decreasing:
            raise ValueError(
                f"RUL is not monotonically decreasing for battery {bid}."
            )

    # B0007 must be entirely censored
    b0007 = rul_df[rul_df["battery_id"] == "B0007"]
    if not b0007.empty and b0007["rul_valid"].any():
        raise ValueError("B0007 should have rul_valid=False for all rows.")

    print("\nValidation passed:")
    print(f"  Total rows          : {len(rul_df)}")
    print(f"  Valid RUL rows      : {len(valid)}")
    print(f"  Censored rows       : {len(rul_df) - len(valid)}")
    print(f"  Batteries (valid)   : {sorted(valid['battery_id'].unique())}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_rul_targets(rul_df: pd.DataFrame) -> None:
    """Save the RUL target dataset to data/features/rul_targets.csv."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rul_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nRUL targets saved to: {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  BatteryIQ Phase 2 — RUL Target Construction")
    print("=" * 55)

    df = load_target_data()
    validate_columns(df)

    print("\nConstructing RUL targets...")
    rul_df = construct_rul_targets(df)

    print("\nRunning validation checks...")
    validate_rul_targets(rul_df)

    save_rul_targets(rul_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
