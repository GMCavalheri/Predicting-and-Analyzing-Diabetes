"""Streamlit dashboard: EDA, model performance, feature importance, and an
interactive single-patient prediction form.

Run: streamlit run dashboard/app.py
Requires reports/metrics.json, reports/figures/, and models/ to already
exist -- run `python -m diabetes_prediction.train` first if they don't.
Never retrains on request; it only loads pre-trained artifacts.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from diabetes_prediction.config import FIGURES_DIR, METRICS_PATH, TARGET
from diabetes_prediction.data_loading import load_dataset
from diabetes_prediction.features import REALISTIC_FEATURES
from diabetes_prediction.predict import load_model, predict_single

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_data
def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_resource
def get_prediction_model():
    return load_model()


st.title("🩺 Predicting Diabetes Diagnosis")
st.caption(
    "Trained on 100,000 patient records. Includes a documented data-leakage "
    "finding from the original version of this project -- see the "
    "Model Performance tab."
)

metrics = get_metrics()
if not metrics:
    st.error(
        "No trained models found. Run `python -m diabetes_prediction.train` first "
        "(see README)."
    )
    st.stop()

tab_overview, tab_performance, tab_importance, tab_try = st.tabs(
    ["Overview", "Model Performance", "Feature Importance", "Try It Yourself"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    df = get_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Patients", f"{len(df):,}")
    col2.metric("Predictive columns", 30)
    col3.metric("Diagnosed with diabetes", f"{df[TARGET].mean() * 100:.1f}%")

    st.subheader("Class balance")
    st.bar_chart(df[TARGET].value_counts().rename({0: "Not diagnosed", 1: "Diagnosed"}))

    st.subheader("Distributions")
    dist_col = st.selectbox(
        "Column",
        ["age", "bmi", "hba1c", "glucose_fasting", "glucose_postprandial"],
    )
    binned = pd.cut(df[dist_col], bins=20)
    counts = binned.value_counts().sort_index()
    counts.index = [f"{interval.left:.1f}" for interval in counts.index]
    st.bar_chart(counts)

# ------------------------------------------------------------- Performance
with tab_performance:
    st.subheader("Model comparison across feature sets")
    results_df = pd.DataFrame(
        [
            {
                "Feature set": m["feature_set"],
                "Model": m["model"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "ROC-AUC": m["roc_auc"],
            }
            for m in metrics.values()
        ]
    ).sort_values(["Feature set", "Model"])
    st.dataframe(
        results_df.style.format(
            {c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("The data leakage finding")
    st.markdown(
        """
The original project's feature set includes `diabetes_stage` and
`diabetes_risk_score` -- columns that turn out to leak the target.
`diabetes_stage` maps to `diagnosed_diabetes` with **zero exceptions** for
3 of its 5 values, and it's the single most important feature (58%
importance) in a random forest trained on the full feature set. Comparing
accuracy across feature sets makes the effect visible directly:

- **Full** (includes leakage columns): **~99.8-99.97%** accuracy
- **Realistic** (leakage columns removed): **~88.8-92.0%** accuracy -- honest,
  and still strong, since `hba1c`/glucose are legitimate clinical inputs
- **Minimal** (labs removed too): **~60-63%** accuracy, barely above the
  ~60% majority-class baseline -- demographics and lifestyle alone carry
  little signal

The deployed model (**Try It Yourself** tab) uses the **realistic**
feature set.
        """
    )

    st.subheader("Diagnostic plots")
    run_key = st.selectbox("Run", sorted(metrics.keys()))
    plot_col1, plot_col2 = st.columns(2)
    cm_path = FIGURES_DIR / f"{run_key}_confusion_matrix.png"
    roc_path = FIGURES_DIR / f"{run_key}_roc_curve.png"
    if cm_path.exists():
        plot_col1.image(str(cm_path), caption="Confusion matrix")
    if roc_path.exists():
        plot_col2.image(str(roc_path), caption="ROC curve")

# -------------------------------------------------------------- Importance
with tab_importance:
    st.subheader("Feature importance -- realistic__random_forest")
    top_features = metrics.get("realistic__random_forest", {}).get("top_features", [])
    if top_features:
        imp_df = pd.DataFrame(top_features, columns=["Feature", "Importance"]).set_index(
            "Feature"
        )
        st.bar_chart(imp_df)
    else:
        st.info("No importance data available for realistic__random_forest.")

# -------------------------------------------------------------------- Try
with tab_try:
    st.subheader("Predict a diagnosis")
    st.caption(
        "Uses the realistic-features random forest model "
        f"({len(REALISTIC_FEATURES)} inputs, no leakage columns)."
    )

    model = get_prediction_model()

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        patient = {
            "gender": c1.selectbox("Gender", ["Female", "Male", "Other"]),
            "ethnicity": c1.selectbox(
                "Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"]
            ),
            "education_level": c1.selectbox(
                "Education level", ["No formal", "Highschool", "Graduate", "Postgraduate"]
            ),
            "income_level": c1.selectbox(
                "Income level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
            ),
            "employment_status": c1.selectbox(
                "Employment status", ["Employed", "Unemployed", "Retired", "Student"]
            ),
            "smoking_status": c1.selectbox("Smoking status", ["Never", "Former", "Current"]),
            "age": c1.slider("Age", 18, 90, 50),
            "family_history_diabetes": int(c1.checkbox("Family history of diabetes")),
            "hypertension_history": int(c1.checkbox("History of hypertension")),
            "cardiovascular_history": int(c1.checkbox("History of cardiovascular disease")),
            "alcohol_consumption_per_week": c2.slider("Alcohol drinks/week", 0, 10, 2),
            "physical_activity_minutes_per_week": c2.slider(
                "Physical activity (min/week)", 0, 850, 100
            ),
            "diet_score": c2.slider("Diet score (0-10)", 0.0, 10.0, 6.0),
            "sleep_hours_per_day": c2.slider("Sleep (hours/day)", 3.0, 10.0, 7.0),
            "screen_time_hours_per_day": c2.slider("Screen time (hours/day)", 0.5, 17.0, 6.0),
            "bmi": c2.slider("BMI", 15.0, 40.0, 25.6),
            "waist_to_hip_ratio": c2.slider("Waist-to-hip ratio", 0.6, 1.1, 0.86),
            "systolic_bp": c2.slider("Systolic BP", 90, 180, 116),
            "diastolic_bp": c2.slider("Diastolic BP", 50, 110, 75),
            "heart_rate": c3.slider("Heart rate", 40, 105, 70),
            "cholesterol_total": c3.slider("Total cholesterol", 100, 320, 186),
            "hdl_cholesterol": c3.slider("HDL cholesterol", 20, 100, 54),
            "ldl_cholesterol": c3.slider("LDL cholesterol", 50, 265, 102),
            "triglycerides": c3.slider("Triglycerides", 30, 345, 121),
            "glucose_fasting": c3.slider("Fasting glucose (mg/dL)", 60, 175, 111),
            "glucose_postprandial": c3.slider("Postprandial glucose (mg/dL)", 70, 290, 160),
            "insulin_level": c3.slider("Insulin level", 2.0, 33.0, 8.8),
            "hba1c": c3.slider("HbA1c (%)", 4.0, 10.0, 6.5),
        }

        submitted = st.form_submit_button("Predict")

    if submitted:
        result = predict_single(patient, model=model)
        if result["prediction"] == 1:
            st.error(f"**Diabetes likely** -- {result['probability'] * 100:.1f}% probability")
        else:
            st.success(f"**Diabetes unlikely** -- {result['probability'] * 100:.1f}% probability")
