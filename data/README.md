# Dataset

`raw/diabetes_dataset.csv` — 100,000 patient records, 30 predictive columns plus the target
`diagnosed_diabetes` (0 = not diagnosed, 1 = diagnosed). Class balance: ~60% positive, ~40% negative.

## Columns

| Column | Type | Notes |
|---|---|---|
| `age` | int | |
| `gender` | category | |
| `ethnicity` | category | |
| `education_level` | category | |
| `income_level` | category | |
| `employment_status` | category | |
| `smoking_status` | category | |
| `alcohol_consumption_per_week` | int | |
| `physical_activity_minutes_per_week` | int | |
| `diet_score` | float | |
| `sleep_hours_per_day` | float | |
| `screen_time_hours_per_day` | float | |
| `family_history_diabetes` | int (0/1) | |
| `hypertension_history` | int (0/1) | |
| `cardiovascular_history` | int (0/1) | |
| `bmi` | float | |
| `waist_to_hip_ratio` | float | |
| `systolic_bp` | int | |
| `diastolic_bp` | int | |
| `heart_rate` | int | |
| `cholesterol_total` | int | |
| `hdl_cholesterol` | int | |
| `ldl_cholesterol` | int | |
| `triglycerides` | int | |
| `glucose_fasting` | int | Clinical diagnostic input (fasting glucose ≥ 126 mg/dL is a diabetes threshold) |
| `glucose_postprandial` | int | Clinical diagnostic input |
| `insulin_level` | float | |
| `hba1c` | float | Clinical diagnostic input (HbA1c ≥ 6.5% is a diabetes threshold) |
| `diabetes_risk_score` | float | **Excluded from the "realistic" feature set** — a composite score that correlates strongly with the target (mean 32.3 for diagnosed vs. 27.1 for non-diagnosed patients), most likely derived downstream of the diagnosis rather than a genuine independent predictor |
| `diabetes_stage` | category | **Excluded from every model** — direct label leakage. Values `No Diabetes` and `Pre-Diabetes` map to `diagnosed_diabetes == 0` with zero exceptions in the data, and `Type 2` maps to `diagnosed_diabetes == 1` with zero exceptions; this column effectively *is* the target under a different name |
| `diagnosed_diabetes` | int (0/1) | **Target** |

See the main [README](../README.md#methodology--the-data-leakage-finding) for the full leakage
investigation and how it shaped the three feature sets used for training
(`src/diabetes_prediction/features.py`).

## Source

Recovered from the original version of this project
([GMCavalheri/Predicting-and-Analyzing-Diabetes](https://github.com/GMCavalheri/Predicting-and-Analyzing-Diabetes)).
Committed directly to the repo (14MB) for reviewer convenience — no separate download step required.
