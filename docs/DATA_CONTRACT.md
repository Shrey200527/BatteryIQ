# BatteryIQ Data Contract v1

## Dataset
NASA Ames PCoE Li-ion Battery Aging Dataset

## Batteries
- B0005
- B0006
- B0007
- B0018

## ML Unit
One row = one battery cycle.

## Processed Dataset
data/processed/battery_discharge_cycles.csv

## Feature Dataset
data/features/battery_cycle_features.csv

## Target Dataset
data/features/battery_cycle_targets.csv

## SOH Definition
SOH = (Current Available Capacity / Rated Capacity) × 100

Rated Capacity = 2 Ah

## EOL Definition
EOL = SOH ≤ 70%

## RUL Definition
RUL = number of remaining cycles until SOH reaches 70%.

## Validation
Primary validation:
Leave-One-Battery-Out (LOGO)

Random row splitting is not permitted.

## Leakage Policy
Capacity and capacity-derived variables must not be used
as SOH predictor features.

## EOL Availability
B0005 → EOL observed
B0006 → EOL observed
B0018 → EOL observed
B0007 → EOL not observed

B0007 must be treated as right-censored for RUL development.

## Raw Data
data/raw/ is read-only.