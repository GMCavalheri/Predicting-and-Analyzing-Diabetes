"""Named feature sets used to train and compare models.

FULL_FEATURES reproduces the original project's feature set, including the
two columns that turn out to leak the target (see the README's "Methodology
& the Data Leakage Finding"). REALISTIC_FEATURES drops those and is the set
actually used for the deployed model. MINIMAL_FEATURES additionally drops
lab results, approximating a non-invasive screening scenario using only
demographics, lifestyle, and vitals -- useful to show how much predictive
power comes from labs a patient wouldn't have without already seeing a
doctor.
"""

from __future__ import annotations

from diabetes_prediction.config import COL_CATEGORICAL, COL_NUMERIC, LEAKAGE_COLUMNS

FULL_FEATURES: list[str] = COL_CATEGORICAL + COL_NUMERIC

REALISTIC_FEATURES: list[str] = [c for c in FULL_FEATURES if c not in LEAKAGE_COLUMNS]

_LAB_COLUMNS = [
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
]

MINIMAL_FEATURES: list[str] = [c for c in REALISTIC_FEATURES if c not in _LAB_COLUMNS]

FEATURE_SETS: dict[str, list[str]] = {
    "full": FULL_FEATURES,
    "realistic": REALISTIC_FEATURES,
    "minimal": MINIMAL_FEATURES,
}
