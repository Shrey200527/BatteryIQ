"""
Battery Condition Classification for BatteryIQ Phase 2.

**BLOCKER: This implementation is awaiting team-approved numerical thresholds.**

See: src/classification/condition_classifier_blocker.md

The ML contract (docs/ML_CONTRACT.md) defines the classification interface:
    classify_condition(soh)

Output classes:
    - Healthy
    - Degraded
    - Critical

However, the numerical SOH thresholds for these classes are NOT defined
in the ML contract. Per the contract:

"No member may invent or independently choose the numerical classification
thresholds. The exact numerical boundaries must be agreed upon by the team
and added to this contract before the classifier is treated as a frozen
project component."

This file is a placeholder. The actual implementation will be completed
once the team provides approved thresholds via ML contract update.
"""

def classify_condition(soh: float) -> str:
    """
    Classify battery condition based on State of Health (SOH).
    
    **BLOCKER: Implementation pending team-approved numerical thresholds.**
    
    Parameters
    ----------
    soh : float
        State of Health value (percentage)
        
    Returns
    -------
    str
        One of: "Healthy", "Degraded", "Critical"
        
    Raises
    ------
    NotImplementedError
        Until team approves classification thresholds in ML contract
    """
    raise NotImplementedError(
        "Condition classification is blocked pending team-approved "
        "numerical thresholds in docs/ML_CONTRACT.md. "
        "See src/classification/condition_classifier_blocker.md for details."
    )


# Future implementation structure (when thresholds are approved):
#
# def classify_condition(soh: float) -> str:
#     """
#     Classify battery condition based on State of Health (SOH).
#     
#     Uses team-approved thresholds from ML contract.
#     
#     Parameters
#     ----------
#     soh : float
#         State of Health value (percentage)
#         
#     Returns
#     -------
#     str
#         One of: "Healthy", "Degraded", "Critical"
#     """
#     # Placeholder for approved thresholds
#     HEALTHY_THRESHOLD = None    # To be defined by team
#     DEGRADED_THRESHOLD = None   # To be defined by team
#     
#     if soh >= HEALTHY_THRESHOLD:
#         return "Healthy"
#     elif soh >= DEGRADED_THRESHOLD:
#         return "Degraded"
#     else:
#         return "Critical"