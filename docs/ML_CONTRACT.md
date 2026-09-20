# BatteryIQ — ML Contract v1

**Project:** BatteryIQ — AI-Powered Predictive Battery Health Analyzer **Contract Version:** v1.0 **Status:** **FROZEN** **Purpose:** Define the common machine-learning interfaces, targets, validation rules, leakage policy, model conventions, and integration requirements for Phase 2.

---

# 1. Purpose

This document defines the machine-learning contract for BatteryIQ.

The purpose of this contract is to ensure that all team members develop their ML components against the same:

- Dataset
- Battery IDs
- Cycle definition
- Feature definitions
- **SOH** definition
- **EOL** definition
- **RUL** definition
- Validation methodology
- Leakage rules
- Model interfaces
- Classification interface
- Explainability interface
- Inference interface

Once this contract is frozen, individual members must not independently redefine these concepts.

Any change to this contract requires team agreement and must be committed to `main` before dependent work continues.

---

# 2. Frozen Dataset

## 2.1 Primary Dataset

The project uses:

****NASA** Ames PCoE Li-ion Battery Aging Dataset**

The initial BatteryIQ development set contains:

- **B0005**
- **B0006**
- **B0007**
- **B0018**

No additional battery dataset is part of the Phase-2 training/validation pipeline unless explicitly approved by the team.

---

# 3. Frozen Data Artifacts

The following datasets are frozen inputs for Phase 2.

## 3.1 Processed Cycle-Level Dataset

```text data/processed/battery_discharge_cycles.csv```

Current structure:

battery_id cycle time voltage current temperature current_load voltage_load capacity

Current verified shape:

**185721** rows × 9 columns

The processed dataset must be treated as read-only.

3.2 Feature Dataset data/features/battery_cycle_features.csv

Current structure:

battery_id cycle voltage_mean voltage_min voltage_max voltage_std current_mean current_min current_max temperature_mean temperature_max current_load_mean voltage_load_mean cycle_time voltage_change temperature_change temperature_rolling_mean

Current verified shape:

**636** rows × 17 columns

The feature dataset contains predictor features only.

Capacity and capacity-derived target information must not be included in this predictor dataset.

3.3 Target Dataset data/features/battery_cycle_targets.csv

Current structure:

battery_id cycle capacity soh

Current verified shape:

**636** rows × 4 columns

This dataset contains the official capacity and **SOH** targets corresponding to the feature dataset.

## ML Unit

The initial ML unit is:

One row = one battery cycle.

The unique identifier for a cycle-level ML record is:

battery_id + cycle

Every feature-target merge must use:

battery_id cycle

as the join keys.

A cycle from one battery must never be treated as a cycle from another battery.

## Battery Coverage

The current cycle counts are:

Battery	Cycles
**B0005**	**168**
**B0006**	**168**
**B0007**	**168**
**B0018**	**132**
Total	**636**

The Phase-2 models must preserve the battery_id information because battery-level validation depends on it.

## SOH Definition

The official State of Health definition is:

**SOH** = (Current Available Capacity / Rated Capacity) × **100**

The rated capacity is:

Rated Capacity = 2.0 Ah

Therefore:

**SOH** (%) = (capacity / 2.0) × **100**

The **SOH** target must be generated from the official capacity value.

No team member may introduce an alternative **SOH** formula.

## SOH Prediction Contract

7.1 Input

**SOH** models receive the frozen predictor feature set from:

data/features/battery_cycle_features.csv

The predictors must not contain:

capacity soh capacity_change capacity_retention capacity_rolling_mean capacity_degradation

or any equivalent feature that directly or indirectly reveals the target capacity/**SOH**.

7.2 Target target = soh

The target is measured in percentage points.

Example:

92.5 78.3 70.0 7.3 Required **SOH** Models

Phase 2 must support:

Existing baseline models: ### Linear Regression ### Random Forest **SVR** XGBoost Advanced model: **ANN**

The existing baseline methodology must not be silently changed when adding **ANN**.

## RUL Definition

The official Remaining Useful Life definition is:

**RUL** = number of remaining battery cycles until **SOH** reaches 70%.

The End-of-Life threshold is:

**EOL** = **SOH** ≤ 70%

For batteries where **EOL** is observed:

**RUL** = EOL_cycle - current_cycle

Therefore, at the first **EOL** cycle:

**RUL** = 0 ## EOL Availability

The current verified **EOL** status is:

Battery	First **EOL** Cycle	**EOL** Observed
**B0005**	**125**	Yes
**B0006**	**109**	Yes
**B0007**	—	No
**B0018**	97	Yes

**B0007** has a minimum observed **SOH** slightly above 70%.

Therefore:

**B0007** must **NOT** be assigned an artificial **EOL** at its final observed cycle.

**B0007** is treated as right-censored for **RUL** development.

No team member may convert **B0007**'s last observed cycle into an **EOL** label merely because it is the final available cycle.

## RUL Target Contract

The validated **RUL** target will be produced by the **RUL** target owner.

Expected output:

data/features/rul_targets.csv

Minimum required columns:

battery_id cycle rul rul_valid

Where:

battery_id identifies the battery. cycle identifies the battery cycle. rul is the remaining number of cycles until **EOL** where **EOL** is observed. rul_valid indicates whether a valid **EOL**-based **RUL** target exists.

For **B0007**:

rul_valid = False

unless the team later defines an approved censoring methodology.

No artificial **RUL** value may be assigned to **B0007**.

## Validation Strategy

11.1 Primary Validation

The primary validation strategy is:

Leave-One-Battery-Out (**LOGO**) validation.

The model must be evaluated on a battery that was not used during training.

Example:

Training: **B0005** **B0006** **B0007**

Testing: **B0018**

The held-out battery is then rotated so that each battery is evaluated as an unseen battery.

## Forbidden Validation Strategies

The following are not acceptable as the primary validation methodology:

Random row train/test split

or:

Random cycle-level train/test split

because cycles from the same battery are temporally and statistically related.

A model must not receive training rows from the same battery whose later rows are used for testing when the experiment is intended to measure unseen-battery generalization.

## Sequence-Model Validation

For **LSTM**/**GRU** models:

Sequences must never cross battery boundaries. A sequence must contain observations from only one battery. Test batteries must remain unseen during training. Sequence construction must not allow future test information to enter training sequences. Random sequence splitting is prohibited if it causes battery-level leakage.

The battery-level split must be established before sequence-based evaluation, or the implementation must guarantee equivalent strict battery isolation.

## Leakage Policy

Data leakage is strictly prohibited.

The following must not be used as **SOH** predictor features:

capacity soh capacity_change capacity_retention capacity_rolling_mean capacity_degradation

Any feature mathematically derived from the target or future target information is also prohibited.

## Future Information / Temporal Leakage

A model must not use future observations to predict an earlier cycle.

Examples of prohibited behavior:

Using future-cycle capacity to construct a current-cycle feature. Using information from the held-out battery during training. Using future observations from the test sequence during preprocessing.

Any rolling or sequence-based transformation must respect temporal order.

## Feature Contract

The currently frozen predictor feature order is:

voltage_mean voltage_min voltage_max voltage_std current_mean current_min current_max temperature_mean temperature_max current_load_mean voltage_load_mean cycle_time voltage_change temperature_change temperature_rolling_mean

The identifier columns:

battery_id cycle

must be preserved for grouping, joining, tracing, and validation.

They must not automatically be treated as numerical predictors.

Any future feature addition requires agreement and an update to this contract.

## Feature Order

Model training and inference must use a deterministic feature order.

The model artifact must record the exact feature list used during training.

Inference must verify that incoming data contains the expected features.

A model must not silently accept an arbitrary reordered feature vector.

## Missing Values

Before model inference:

NaN / missing predictor values are not permitted.

The preprocessing behavior for missing values must be deterministic and documented.

Training and inference must use compatible preprocessing.

No member may silently drop rows during inference.

## Model Evaluation Metrics

19.1 Regression

**SOH** and **RUL** regression models must report:

**MAE** **RMSE** R²

These metrics must be generated programmatically.

Metrics must not be manually entered into notebooks, reports, or comparison tables.

## Model Comparison

Model comparisons must use the same:

Dataset Feature definition Target definition Validation strategy Evaluation metrics

for all compared models.

A model must not be declared superior solely because it was evaluated under a different split or preprocessing pipeline.

## Per-Battery Evaluation

In addition to aggregate metrics, **SOH** model evaluation should report performance for each held-out battery where applicable.

The evaluation should preserve:

battery_id **MAE** **RMSE** R²

This allows the team to examine cross-battery generalization.

## ANN SOH Contract

The **ANN** is an advanced **SOH** regression model.

The intended architecture is:

Input ↓ Dense ↓ ReLU ↓ Dropout ↓ Dense ↓ ReLU ↓ Dense ↓ **SOH** Output

The exact hidden-layer sizes and training hyperparameters may be tuned experimentally.

However:

Input features must follow the frozen feature contract. Target must remain **SOH**. Validation must remain **LOGO**. Leakage rules remain unchanged. Metrics must remain **MAE**, **RMSE**, and R². ## RUL Model Contract

Phase 2 will contain:

**RUL** XGBoost

Input:

Cycle-level battery features

Target:

**RUL**

Validation:

**LOGO** **LSTM** / **GRU**

Input:

Sequential cycle-level feature data

Output:

**RUL**

Constraints:

sequences cannot cross batteries; validation must preserve battery-level separation; no future information may enter the input sequence; **RUL** targets must follow the official **RUL** contract. ## Condition Classification Contract

Battery condition classification must expose a deterministic interface:

classify_condition(soh)

The output classes are:

Healthy Degraded Critical Important

The project specification defines these three classes but does not currently define numerical **SOH** boundaries for them.

Therefore:

No member may invent or independently choose the numerical classification thresholds.

The exact numerical boundaries must be agreed upon by the team and added to this contract before the classifier is treated as a frozen project component.

Until then, the classification interface is defined, but the numerical threshold policy is not frozen.

## SHAP Explainability Contract

BatteryIQ will use **SHAP** for model explainability.

The **SHAP** implementation must explain predictions from an actual validated project model.

It must not use:

fake models, placeholder predictions, unrelated datasets, demonstration-only feature matrices.

**SHAP** explanations must use:

the same feature definitions the same feature order the same preprocessing the same trained model

used by the prediction pipeline.

The explainability layer should support:

Global explanation

Which features contribute most strongly across the evaluation data.

Local explanation

Which features contributed to an individual battery prediction.

The **SHAP** implementation must identify the model and feature set being explained.

## Model Artifact Contract

Validated models must be saved as reusable artifacts.

Example structure:

models/
└── soh/
    └── ann/
    ├── model.keras
    └── metadata.json

The exact serialization format may depend on the model framework.

Every validated model artifact should have associated metadata containing, where applicable:

model name model version target feature list feature order validation strategy training batteries test batteries evaluation metrics rated capacity **EOL** threshold ## Model Metadata Example

Example:

{
    *model*: *ANN*,
    *target*: *soh*,
    *rated_capacity_ah*: 2.0,
    *eol_soh_percent*: 70.0,
    *validation*: *Leave-One-Battery-Out*,
    *features*: [
    *voltage_mean*,
    *voltage_min*,
    *voltage_max*,
    *voltage_std*,
    *current_mean*,
    *current_min*,
    *current_max*,
    *temperature_mean*,
    *temperature_max*,
    *current_load_mean*,
    *voltage_load_mean*,
    *cycle_time*,
    *voltage_change*,
    *temperature_change*,
    *temperature_rolling_mean*
    ]
}

The actual metrics and artifact-specific values must be generated from the experiment.

## Inference Contract

The final ML system will expose a common inference layer.

Primary interface:

predict_soh(input_data) predict_rul(input_data) classify_condition(soh) explain_prediction(input_data)

The intended high-level flow is:

### Input Battery Data

       ↓
### Feature Validation
       ↓
Feature Extraction / Preprocessing
       ↓
**SOH** Model
       ↓
**SOH** Prediction
       ↓
### Condition Classification
       ↓
**RUL** Model
       ↓
**RUL** Prediction
       ↓
**SHAP** Explanation
       ↓
Dashboard
## Inference Layer Responsibilities

The inference layer is responsible for:

Loading validated model artifacts. Validating input features. Preserving feature order. Applying required preprocessing. Generating predictions. Calling the condition classifier. Calling the explainability layer where required. Returning structured prediction results.

The inference layer must not retrain models.

The dashboard must also not retrain models.

## Expected Inference Output

The integrated inference system should ultimately provide a structured result conceptually containing:

**SOH** **RUL** ### Battery Condition Prediction / Model Information Feature Contributions / Explanation

Example conceptual structure:

{
    *soh*: ...,
    *rul*: ...,
    *condition*: ...,
    *explanation*: ...
}

The exact Python object/schema may be refined during integration, but the semantic fields must remain consistent with this contract.

## RUL and Condition Consistency

The final system must maintain logical consistency between:

**SOH** **EOL** threshold **RUL** Condition

For example:

A predicted **SOH** below the **EOL** threshold must not be represented as a healthy battery if the approved condition thresholds define otherwise. **RUL** must not become negative unless a clearly documented post-**EOL** inference policy is explicitly approved. **B0007**'s historical right-censoring must not be changed into an observed **EOL** label. ## Repository Ownership

Phase-2 ownership is:

Member	Responsibility	Branch
Member 1 — Vidya	**RUL** target + XGBoost	feature/rul-xgboost
Member 2 — Renuka	**LSTM** + **GRU** **RUL**	feature/rul-sequence-models
Member 3 — Shrushti	Condition classification + **SHAP**	feature/condition-shap
Member 4 — Shreya	**ANN** **SOH** + ML integration + inference	feature/advanced-soh-integration
## File Ownership
Member 1
src/rul/rul_target.py
src/rul/rul_xgboost.py
notebooks/06_rul_xgboost.ipynb
data/features/rul_targets.csv
Member 2
src/rul/sequence_dataset.py
src/rul/lstm_model.py
src/rul/gru_model.py
notebooks/07_rul_sequence_models.ipynb
Member 3
src/classification/condition_classifier.py
src/explainability/shap_analysis.py
notebooks/09_condition_classification.ipynb
notebooks/10_shap_analysis.ipynb
Member 4
src/models/ann_soh.py
src/inference/predictor.py
notebooks/08_ann_soh.ipynb
models/soh/ann/

Members must not modify another member's owned implementation without coordination.

## Frozen Files

The following are treated as frozen inputs during Phase 2:

data/raw/* data/processed/battery_discharge_cycles.csv data/features/battery_cycle_features.csv data/features/battery_cycle_targets.csv src/models/soh_models.py src/evaluation/metrics.py docs/DATA_CONTRACT.md docs/ML_CONTRACT.md

Changes to frozen datasets or contracts require explicit team agreement.

## Git Rules

All Phase-2 development must follow the team Git workflow.

Rules:

Do not develop directly on main. Each member works on their assigned branch. Pull the latest main before creating the Phase-2 branch. All branches must originate from the same frozen contract commit. Push work to the member's feature branch. Create a Pull Request into main. Another team member should review the PR. Do not force-push main. Do not commit .venv. Do not commit generated caches such as __pycache__. Keep commits focused and related to the assigned component. ## Notebook Rules

Every Phase-2 notebook must:

run top-to-bottom; use repository-relative paths; use the project environment; use the frozen datasets; generate metrics programmatically; clearly document the experiment; avoid hidden/manual state; avoid manually entered results; preserve the project's validation methodology.

A notebook must not depend on variables that only exist because an earlier notebook happened to be executed.

## No Invented Data

Team members must not fabricate:

resistance values; capacity values; **SOH** values; **RUL** labels; model metrics; prediction confidence; classification labels; **SHAP** values.

If a required quantity cannot be reliably derived from the available data, this limitation must be documented rather than replaced with invented values.

## No Silent Methodology Changes

The following cannot be changed independently by an individual member:

**SOH** formula Rated capacity **EOL** threshold **RUL** definition Battery IDs Primary validation strategy Leakage policy Frozen feature definitions Classification semantics

Any proposed change must be discussed and incorporated into a new contract version.

## Versioning

This document is:

ML Contract v1.0

If a fundamental ML definition changes, increment the contract version.

Examples:

v1.1

for a compatible clarification.

v2.0

for a breaking change to targets, validation, interfaces, or data contracts.

## Phase-2 Integration Architecture

The intended architecture is:

    **FROZEN** **FOUNDATION**
    │
    ▼
    Feature + Target Data
    │
    ┌───────────────┼────────────────┐
    │               │                │
    ▼               ▼                ▼
    **RUL** Target       **ANN** **SOH**       Condition + **SHAP**
    + XGBoost
    │               │                │
    │               │                │
    └───────────────┼────────────────┘
    │
    ▼
    ML **INTEGRATION**
    │
    ▼
    **INFERENCE**
    │
    ▼
    **DASHBOARD**

**RUL** sequence models consume the validated **RUL** target and follow the same battery-level validation principles.

## Phase-2 Completion Criteria

Phase 2 is considered technically integrated when:

**SOH** **ANN** is implemented and evaluated. Existing **SOH** baselines remain reproducible. **RUL** target construction is validated. XGBoost **RUL** is evaluated. **LSTM** and **GRU** **RUL** models are evaluated. Condition classification has an approved threshold policy. **SHAP** explanations are generated from a validated model. Model artifacts can be loaded without retraining. The inference layer can connect the validated components. Feature order is validated. No leakage is detected. **LOGO** validation is preserved. **B0007** remains correctly treated as right-censored for **RUL**. The final pipeline produces consistent **SOH**/**RUL**/condition outputs. ## Final Non-Negotiable Rules

The following rules override convenience during development:

Do not modify frozen datasets to make a model work. Do not use capacity or **SOH**-derived information as **SOH** predictors. Do not use random row splitting as the primary validation strategy. Do not allow the same battery to leak across train and test in unseen-battery experiments. Do not create sequences that cross battery boundaries. Do not invent **EOL** for **B0007**. Do not invent **RUL** values for censored observations. Do not invent resistance measurements. Do not manually enter model metrics. Do not fabricate **SHAP** explanations. Do not independently define classification thresholds. Do not retrain models inside the dashboard. Do not change another member's owned component without coordination. Do not bypass the GitHub PR workflow. Do not silently change any frozen ML definition. ## Contract Status ML_CONTRACT.md Version: 1.0 Status: **FROZEN** **FOR** **PHASE** 2