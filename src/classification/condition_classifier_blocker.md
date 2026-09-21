# Condition Classification Implementation Blocker

**Member 3 — Shrushti**  
**Branch:** `feature/condition-shap`  
**Date:** 2026-09-22

---

## BLOCKED COMPONENT

**Battery Condition Classification**

**File:** `src/classification/condition_classifier.py`

**Notebook:** `notebooks/09_condition_classification.ipynb`

---

## REASON FOR BLOCKER

The ML contract (`docs/ML_CONTRACT.md`) explicitly states that numerical SOH thresholds for the three condition classes are **NOT DEFINED**:

> "The project specification defines these three classes but does not currently define numerical **SOH** boundaries for them. Therefore: No member may invent or independently choose the numerical classification thresholds. The exact numerical boundaries must be agreed upon by the team and added to this contract before the classifier is treated as a frozen project component."

---

## CLASSIFICATION INTERFACE (DEFINED)

The ML contract **does** define the classification interface:

```python
classify_condition(soh)
```

**Output Classes:**
- Healthy
- Degraded  
- Critical

---

## CLASSIFICATION THRESHOLDS (NOT DEFINED)

The ML contract **does NOT** define the numerical SOH boundaries for:

- **Healthy** threshold (e.g., SOH ≥ ?)
- **Degraded** threshold (e.g., ? > SOH ≥ ?)
- **Critical** threshold (e.g., SOH < ?)

---

## ML CONTRACT EXCERPT

From `docs/ML_CONTRACT.md`, Section 22 (Condition Classification Contract):

> "Battery condition classification must expose a deterministic interface:
> classify_condition(soh)
> 
> The output classes are:
> Healthy
> Degraded
> Critical
> 
> **Important**
> 
> The project specification defines these three classes but does not currently define numerical **SOH** boundaries for them.
> 
> Therefore:
> 
> No member may invent or independently choose the numerical classification thresholds.
> 
> The exact numerical boundaries must be agreed upon by the team and added to this contract before the classifier is treated as a frozen project component."

---

## REQUIRED TEAM ACTION

Before the condition classification component can be implemented, the team must:

1. **Agree on numerical SOH thresholds** for the three condition classes
2. **Add the approved thresholds** to `docs/ML_CONTRACT.md` in the Condition Classification Contract section
3. **Commit the updated ML contract** to the main branch
4. **Notify Member 3** that the thresholds are approved

---

## PROPOSED THRESHOLD DISCUSSION POINTS

The team should consider:

- What SOH range constitutes "Healthy" battery performance?
- What SOH range constitutes "Degraded" but still operational performance?
- What SOH range constitutes "Critical" performance requiring immediate attention?
- Should thresholds be absolute (e.g., 80%, 70%) or relative (e.g., percentage of rated capacity)?
- Should thresholds align with the existing EOL definition (SOH ≤ 70%)?
- Are there industry standards or domain expertise to guide threshold selection?

---

## IMPLEMENTATION STATUS

**BLOCKED** - Pending team-approved numerical thresholds

**Cannot Proceed Until:**
- ML contract is updated with approved classification thresholds
- Team consensus is documented in `docs/ML_CONTRACT.md`

---

## COMPLETED MEMBER 3 WORK

While condition classification is blocked, the following Member 3 components **ARE COMPLETED**:

✅ **SHAP Explainability Module** (`src/explainability/shap_analysis.py`)
- Global feature importance analysis
- Individual prediction explanation  
- SHAP summary plot generation
- Feature leakage prevention
- ML contract compliance

✅ **SHAP Analysis Notebook** (`notebooks/10_shap_analysis.ipynb`)
- Explains validated RUL XGBoost model
- Uses frozen feature set from ML contract
- Preserves exact feature order
- No target leakage
- Tested and verified

✅ **Dependencies** (`requirements.txt`)
- Added SHAP library (v0.46.0)

---

## NEXT STEPS FOR TEAM

1. **Discuss and agree** on classification thresholds
2. **Update** `docs/ML_CONTRACT.md` with approved thresholds
3. **Notify** Member 3 when thresholds are approved
4. **Member 3** will then implement:
   - `src/classification/condition_classifier.py`
   - `notebooks/09_condition_classification.ipynb`

---

## MEMBER 3 READINESS

When the team provides approved thresholds, Member 3 is ready to:

1. Implement `classify_condition(soh)` function with approved thresholds
2. Add boundary handling and invalid SOH handling per ML contract
3. Create comprehensive classification notebook with:
   - Threshold testing
   - Boundary case validation
   - Condition distribution analysis
   - Integration with SHAP explanations
4. Test classifier against all three condition classes
5. Verify classifier outputs match ML contract requirements

---

## CONTACT

**Member 3:** Shrushti  
**Branch:** `feature/condition-shap`  
**Blocker Status:** Active - awaiting team decision on classification thresholds