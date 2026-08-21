"""Paths and column definitions shared across the pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Deriving this from __file__ would break under a real (non-editable)
# install -- e.g. in Docker, `pip install .` puts this file under
# site-packages/, nowhere near the repo. Every entrypoint (train.py,
# dashboard/app.py, pytest, the Docker WORKDIR) is run from the repo root,
# so that's the reliable anchor instead; PROJECT_ROOT overrides it directly
# if that assumption ever doesn't hold.
ROOT_DIR = Path(os.environ.get("PROJECT_ROOT") or Path.cwd())

DATA_PATH = Path(os.environ.get("DATA_PATH") or ROOT_DIR / "data" / "raw" / "diabetes_dataset.csv")
MODEL_DIR = Path(os.environ.get("MODEL_DIR") or ROOT_DIR / "models")
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR") or ROOT_DIR / "reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "diagnosed_diabetes"

COL_CATEGORICAL = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
    "diabetes_stage",
]

COL_NUMERIC = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    "diabetes_risk_score",
]

# Columns that leak the target. diabetes_stage maps to diagnosed_diabetes
# with zero exceptions for 3 of its 5 values (see data/README.md and the
# README's "Methodology & the Data Leakage Finding" section for the
# investigation that identified these).
LEAKAGE_COLUMNS = ["diabetes_risk_score", "diabetes_stage"]

RANDOM_STATE = 42
