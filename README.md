# BatteryIQ

## AI-Powered Predictive Battery Health Analyzer

BatteryIQ is an AI/ML-based battery health analysis system designed to understand battery degradation and predict **State of Health (SOH)** from battery operating characteristics.

The project currently focuses on building a reliable end-to-end pipeline from raw battery aging data to degradation analysis, feature engineering, and machine-learning-based SOH prediction.

---

## Project Objectives

- Analyze battery aging and degradation behavior
- Extract meaningful battery health features
- Estimate battery State of Health (SOH)
- Establish machine-learning baselines for SOH prediction
- Evaluate models using battery-level validation
- Provide a foundation for future Remaining Useful Life (RUL) prediction
- Support future explainable AI and real-time battery monitoring

---

## Current Progress

### ✅ Completed

- Dataset understanding and quality analysis
- Raw NASA battery data preprocessing
- Discharge-cycle extraction
- Standardized cycle-level dataset creation
- Battery degradation analysis
- Capacity, voltage, current and temperature analysis
- Battery discharge curve analysis
- Cycle-level feature engineering
- SOH target generation
- Target leakage prevention
- Feature and target validation
- Leave-One-Battery-Out model validation
- Machine learning baseline comparison

### 🚧 Planned

- Advanced SOH prediction models
- Remaining Useful Life (RUL) prediction
- Explainable AI
- Battery condition classification
- Real-time IoT-based battery monitoring
- Interactive dashboard

---

## Dataset

BatteryIQ uses the **NASA Ames PCoE Li-ion Battery Aging Dataset**.

Dataset source:

https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

The current analysis uses four batteries:

- `B0005`
- `B0006`
- `B0007`
- `B0018`

The raw dataset is stored locally under:

```text
data/raw/
```

## Project Pipeline

```text
NASA Battery Data
       │
       ▼
Data Understanding
       │
       ▼
Data Preprocessing
       │
       ▼
Standardized Discharge Cycles
       │
       ▼
Battery Degradation Analysis
       │
       ▼
Feature Engineering
       │
       ▼
SOH Target Generation
       │
       ▼
ML Baseline Models
       │
       ▼
Battery-Level Validation
       │
       ▼
Model Comparison
```

## Dataset Statistics

The current processed dataset contains:

- 4 batteries
- 636 discharge cycles
- 185,721 measurement records

The standardized discharge dataset contains:
battery_id
cycle
time
voltage
current
temperature
current_load
voltage_load
capacity
The cycle-level feature dataset contains 636 battery-cycle records.

## Feature Engineering

Cycle-level features include:

- Voltage mean, minimum, maximum and standard deviation
- Current mean, minimum and maximum
- Temperature mean and maximum
- Mean current load
- Mean load voltage
- Discharge duration
- Voltage change
- Temperature change
- Temperature rolling mean

Target-derived variables are separated from the ML feature matrix to prevent target leakage.

## State of Health

SOH is currently defined using discharge capacity relative to the rated battery capacity:
SOH (%) = (Measured Capacity / Rated Capacity) × 100

The current baseline uses a rated capacity of:
2.0 Ah

The resulting SOH values are stored separately from the predictive feature matrix.

## Machine Learning Baseline

The current baseline evaluates four regression models:

1. Linear Regression
2. Random Forest
3. Support Vector Regression (SVR)
4. XGBoost
Models are evaluated using Leave-One-Battery-Out validation, where an entire battery is held out for testing.

Evaluation metrics:

- MAE
- RMSE
- R²
  
Current baseline results

| Model                 |        MAE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: |
| **Linear Regression** | **0.7711** | **0.8438** | **0.9884** |
| XGBoost               |     1.5532 |     1.9509 |     0.9554 |
| Random Forest         |     1.8377 |     2.3197 |     0.9378 |
| SVR                   |     4.2675 |     5.2275 |     0.5973 |

Current best baseline: Linear Regression

## Project Structure

```text
BatteryIQ/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_battery_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_ml_baseline.ipynb
│
├── src/
│   ├── models/
│   │   └── soh_models.py
│   │
│   └── evaluation/
│       └── metrics.py
│
├── models/
├── dashboard/
├── reports/
│
├── requirements.txt
└── README.md
```

## Reproducibility

Create and activate the project environment:

```text
python -m venv .venv
```

Activate on Git Bash:

```text
source .venv/Scripts/activate
```

Install dependencies:

```text
pip install -r requirements.txt
```

The notebooks can then be executed sequentially:

```text
01 → 02 → 03 → 04 → 05
```

## Team Workflow

The project is developed using Git-based feature branches and pull requests.

Major project stages are maintained as separate contributions and merged into the main branch after verification.

## Future Scope

The current implementation establishes the foundation for:

- Advanced deep-learning-based SOH prediction
- Remaining Useful Life (RUL) estimation
- Battery condition classification
- Explainable AI
- IoT-based battery monitoring
- Interactive battery health dashboard

## Project Status

Current status: ML baseline pipeline completed.

BatteryIQ currently provides a complete pipeline from raw NASA battery aging data through preprocessing, degradation analysis, feature engineering, SOH formulation, and battery-level machine-learning evaluation.
