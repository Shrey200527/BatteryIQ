"""
XGBoost RUL Regression — BatteryIQ Phase 2.

Responsibilities:
- Load frozen Phase 1 features (battery_cycle_features.csv)
- Merge features with RUL targets (rul_targets.csv) on battery_id + cycle
- Filter to valid RUL rows only (rul_valid=True)
- Train XGBoost with Leave-One-Battery-Out (LOGO) cross-validation
- Report per-fold and overall MAE, RMSE, R²

No random row-wise split is used.  The same battery never appears
in both train and test within a fold.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = (
    PROJECT_ROOT / "data" / "features" / "battery_cycle_features.csv"
)

RUL_FILE = (
    PROJECT_ROOT / "data" / "features" / "rul_targets.csv"
)

# Feature columns sourced from Phase 1 frozen feature set
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


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load frozen feature and RUL target CSVs."""

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}"
        )
    if not RUL_FILE.exists():
        raise FileNotFoundError(
            f"RUL target file not found: {RUL_FILE}. "
            "Run rul_target.py first."
        )

    features_df = pd.read_csv(FEATURES_FILE)
    rul_df = pd.read_csv(RUL_FILE)

    print("Feature dataset shape :", features_df.shape)
    print("RUL dataset shape     :", rul_df.shape)

    return features_df, rul_df


# ---------------------------------------------------------------------------
# Prepare merged dataset
# ---------------------------------------------------------------------------

def prepare_dataset(
    features_df: pd.DataFrame,
    rul_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge features and RUL targets on [battery_id, cycle].
    Keep only rows where rul_valid=True.
    """

    merged_df = features_df.merge(
        rul_df,
        on=["battery_id", "cycle"],
        how="inner",
    )

    merged_df = merged_df[merged_df["rul_valid"] == True].copy()

    if merged_df.empty:
        raise ValueError(
            "No valid RUL records available after merging. "
            "Ensure rul_targets.csv was generated correctly."
        )

    print("\nMerged dataset shape  :", merged_df.shape)
    print(
        "Batteries included    :",
        sorted(merged_df["battery_id"].unique()),
    )

    return merged_df


# ---------------------------------------------------------------------------
# Feature / target extraction
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract feature matrix X, target y, and group labels.

    Returns:
        X      : DataFrame of feature columns
        y      : Series of RUL values
        groups : Series of battery_id (used for LOGO splitting)
    """

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy().apply(pd.to_numeric, errors="coerce")
    y = df["rul"].copy()
    groups = df["battery_id"].copy()

    if X.isnull().any().any():
        raise ValueError(
            "Missing or non-numeric values found in feature columns."
        )

    return X, y, groups


# ---------------------------------------------------------------------------
# LOGO training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """
    Train XGBoost with Leave-One-Battery-Out cross-validation.

    Each fold holds out one battery entirely; the model trains on
    all remaining batteries. Reports MAE, RMSE, R² per fold and overall.

    Returns:
        fold_results_df : DataFrame with per-fold metrics
    """

    logo = LeaveOneGroupOut()

    all_actual:      list[float] = []
    all_predictions: list[float] = []
    fold_results:    list[dict]  = []

    for fold_number, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_battery = groups.iloc[test_idx].unique()[0]

        print(f"\nFold {fold_number} — Test battery: {test_battery}")
        print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        all_actual.extend(y_test.tolist())
        all_predictions.extend(predictions.tolist())

        fold_mae  = mean_absolute_error(y_test, predictions)
        fold_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        fold_r2   = r2_score(y_test, predictions)

        fold_results.append({
            "fold":         fold_number,
            "test_battery": test_battery,
            "n_test":       len(y_test),
            "mae":          round(fold_mae,  4),
            "rmse":         round(fold_rmse, 4),
            "r2":           round(fold_r2,   4),
        })

        print(f"  MAE : {fold_mae:.4f}")
        print(f"  RMSE: {fold_rmse:.4f}")
        print(f"  R²  : {fold_r2:.4f}")

    # Overall metrics (pooled predictions)
    overall_mae  = mean_absolute_error(all_actual, all_predictions)
    overall_rmse = float(np.sqrt(mean_squared_error(all_actual, all_predictions)))
    overall_r2   = r2_score(all_actual, all_predictions)

    print("\n" + "=" * 55)
    print("  OVERALL XGBoost RUL Results (LOGO)")
    print("=" * 55)
    print(f"  MAE : {overall_mae:.4f}")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  R²  : {overall_r2:.4f}")

    results_df = pd.DataFrame(fold_results)
    print("\nFold-wise results:")
    print(results_df.to_string(index=False))

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  BatteryIQ Phase 2 — XGBoost RUL Baseline")
    print("=" * 55)

    features_df, rul_df = load_datasets()
    merged_df = prepare_dataset(features_df, rul_df)
    X, y, groups = select_features(merged_df)

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Target vector shape  : {y.shape}")
    print(f"Unique batteries     : {sorted(groups.unique())}")

    train_and_evaluate(X, y, groups)


if __name__ == "__main__":
    main()
