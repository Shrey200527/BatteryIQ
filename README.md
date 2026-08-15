# BatteryIQ

## AI-Powered Predictive Battery Health Analyzer

BatteryIQ is an AI/ML-based battery diagnostic system designed to estimate battery State of Health (SOH), Remaining Useful Life (RUL), and degradation condition using battery operating characteristics.

## Project Goals

- Battery data analysis
- Feature engineering
- SOH prediction
- RUL prediction
- Battery condition classification
- Explainable AI
- IoT-based real-time battery monitoring
- Interactive dashboard

## Project Structure

- `data/` — datasets and processed data
- `notebooks/` — experimentation and analysis
- `src/` — reusable Python modules
- `models/` — trained ML models
- `dashboard/` — Streamlit application
- `reports/` — project documentation

## Dataset Setup

BatteryIQ uses the NASA Ames PCoE Li-ion Battery Aging Dataset.
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip?utm_source=chatgpt.com
Required battery files:

- B0005.mat
- B0006.mat
- B0007.mat
- B0018.mat

Download the official dataset from NASA and place these files in:

data/raw/

The raw dataset is excluded from Git tracking through `.gitignore`.