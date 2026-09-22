"""
BatteryIQ Unified Inference Interface.

This module provides the project-level inference contract:

    predict_soh(input_data)
    predict_rul(input_data)
    classify_condition(soh)
    explain_prediction(input_data)

Current deployment status
-------------------------
SOH:
    Available through the validated ANN deployment artifact.

RUL:
    Pending deployment-model artifacts from the RUL model owners.
    No model is retrained during inference.

Condition classification:
    Blocked until team-approved numerical thresholds are added
    to the ML contract.

SHAP:
    Available through the existing SHAPAnalyzer implementation.
    The ANN model is wrapped so SHAP receives the same preprocessing
    used during model inference.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.classification.condition_classifier import (
    classify_condition as _classify_condition,
)
from src.explainability.shap_analysis import (
    FROZEN_FEATURE_COLUMNS,
    SHAPAnalyzer,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANN_MODEL_PATH = PROJECT_ROOT / "models" / "soh" / "ann" / "ann_soh.keras"
ANN_SCALER_PATH = PROJECT_ROOT / "models" / "soh" / "ann" / "scaler.joblib"
ANN_TARGET_SCALER_PATH = (
    PROJECT_ROOT / "models" / "soh" / "ann" / "target_scaler.joblib"
)


# ---------------------------------------------------------------------------
# Frozen feature contract
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = FROZEN_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# ANN SOH predictor
# ---------------------------------------------------------------------------

class SOHPredictor:
    """
    Deployment wrapper for the validated BatteryIQ ANN SOH model.

    The model, input scaler, and target scaler are loaded from
    committed deployment artifacts.

    No training occurs inside this class.
    """

    def __init__(
        self,
        model_path: Path = ANN_MODEL_PATH,
        scaler_path: Path = ANN_SCALER_PATH,
        target_scaler_path: Path = ANN_TARGET_SCALER_PATH,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.target_scaler_path = Path(target_scaler_path)

        self._validate_artifacts()

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )

        self.scaler = joblib.load(self.scaler_path)
        self.target_scaler = joblib.load(self.target_scaler_path)

    def _validate_artifacts(self) -> None:
        """Ensure all required ANN deployment artifacts exist."""

        required = {
            "ANN model": self.model_path,
            "feature scaler": self.scaler_path,
            "target scaler": self.target_scaler_path,
        }

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Required ANN deployment artifacts are missing:\n"
                + "\n".join(missing)
            )

    @staticmethod
    def _prepare_input(input_data: Any) -> pd.DataFrame:
        """
        Validate and normalize a single-cycle inference input.

        Accepted input:
            - pandas.DataFrame
            - dictionary containing all 15 feature values

        The feature order is always forced to the frozen ML contract.
        """

        if isinstance(input_data, dict):
            data = pd.DataFrame([input_data])

        elif isinstance(input_data, pd.DataFrame):
            data = input_data.copy()

        else:
            raise TypeError(
                "input_data must be a dictionary or pandas DataFrame."
            )

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in data.columns
        ]

        extra = [
            column
            for column in data.columns
            if column not in FEATURE_COLUMNS
        ]

        if missing:
            raise ValueError(
                f"Missing required predictor features: {missing}"
            )

        if extra:
            raise ValueError(
                f"Unexpected features supplied: {extra}"
            )

        data = data[FEATURE_COLUMNS].copy()

        for column in FEATURE_COLUMNS:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        if data.isnull().any().any():
            bad_columns = data.columns[
                data.isnull().any()
            ].tolist()

            raise ValueError(
                f"Input contains missing or non-numeric values "
                f"in: {bad_columns}"
            )

        return data

    def predict(self, input_data: Any) -> np.ndarray:
        """
        Predict SOH percentage for one or more cycle records.
        """

        X = self._prepare_input(input_data)

        X_scaled = self.scaler.transform(X)

        prediction_scaled = self.model.predict(
            X_scaled,
            verbose=0,
        ).reshape(-1, 1)

        prediction = self.target_scaler.inverse_transform(
            prediction_scaled
        ).reshape(-1)

        return prediction

    def predict_one(self, input_data: Any) -> float:
        """Predict SOH for exactly one cycle."""

        predictions = self.predict(input_data)

        if len(predictions) != 1:
            raise ValueError(
                "predict_one expects exactly one input record."
            )

        return float(predictions[0])


# ---------------------------------------------------------------------------
# Project-level inference contract
# ---------------------------------------------------------------------------

_soh_predictor = None


def _get_soh_predictor() -> SOHPredictor:
    """Lazy-load the SOH deployment model."""

    global _soh_predictor

    if _soh_predictor is None:
        _soh_predictor = SOHPredictor()

    return _soh_predictor


def predict_soh(input_data: Any) -> dict:
    """
    Predict battery State of Health.

    Returns a structured inference result.
    """

    predictor = _get_soh_predictor()

    soh_predictions = predictor.predict(input_data)

    return {
        "model": "BatteryIQ_ANN_SOH",
        "target": "soh",
        "predictions": [
            float(value)
            for value in soh_predictions
        ],
        "unit": "percent",
    }


def predict_rul(input_data: Any) -> dict:
    """
    Predict Remaining Useful Life.

    RUL deployment artifacts have not yet been committed by the
    RUL model owners. Inference must not retrain a model or fabricate
    a prediction.

    Raises
    ------
    RuntimeError
        Until a validated RUL deployment artifact and loader are available.
    """

    raise RuntimeError(
        "RUL inference is not yet available. "
        "Validated deployment artifacts and a model-loading interface "
        "for the RUL model are required. No RUL model is retrained "
        "during inference."
    )


def classify_condition(soh: float) -> str:
    """
    Classify battery condition using the project classifier.
    """

    return _classify_condition(soh)


def explain_prediction(input_data: Any) -> dict:
    """
    Generate a SHAP explanation for an ANN SOH prediction.

    The SHAP model wrapper performs the same feature scaling and
    target inverse transformation as normal ANN inference.
    """

    predictor = _get_soh_predictor()

    X = predictor._prepare_input(input_data)

    def prediction_wrapper(values):
        values_df = pd.DataFrame(
            values,
            columns=FEATURE_COLUMNS,
        )

        scaled_values = predictor.scaler.transform(
            values_df
        )

        scaled_predictions = predictor.model.predict(
            scaled_values,
            verbose=0,
        ).reshape(-1, 1)

        predictions = predictor.target_scaler.inverse_transform(
            scaled_predictions
        ).reshape(-1)

        return predictions

    class SHAPModelWrapper:
        def predict(self, values):
            return prediction_wrapper(values)

    wrapped_model = SHAPModelWrapper()

    background_path = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "battery_cycle_features.csv"
    )

    if not background_path.exists():
        raise FileNotFoundError(
            f"SHAP background dataset not found: {background_path}"
        )

    background = pd.read_csv(background_path)

    background = background[FEATURE_COLUMNS].copy()

    analyzer = SHAPAnalyzer(
        model=wrapped_model,
        feature_matrix=background,
    )

    analyzer.fit_explainer(
        background_sample_size=100
    )

    analyzer.compute_shap_values(X)

    explanation = analyzer.explain_prediction(
        sample_idx=0,
        X=X,
    )

    return explanation