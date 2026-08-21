"""Prediction contract used by the dashboard.

`load_model` loads a persisted pipeline (preprocessing + classifier bundled
via `preprocessing.build_pipeline`), so `predict_single` accepts a raw
patient dict with human-readable categorical values (e.g. `gender="Male"`)
and needs no separate encoder to manage.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from diabetes_prediction.config import MODEL_DIR
from diabetes_prediction.features import REALISTIC_FEATURES

DEFAULT_MODEL_PATH = MODEL_DIR / "realistic__random_forest.joblib"


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m diabetes_prediction.train` first."
        )
    return joblib.load(path)


def predict_single(
    patient: dict,
    model=None,
    feature_columns: list[str] = REALISTIC_FEATURES,
) -> dict:
    """Predict a diagnosis for a single patient.

    `patient` must contain every column in `feature_columns`; raises
    ValueError listing what's missing rather than letting pandas/sklearn
    fail with a less specific error deep in the pipeline.
    """
    missing = [c for c in feature_columns if c not in patient]
    if missing:
        raise ValueError(f"Missing required patient fields: {missing}")

    model = model or load_model()
    X = pd.DataFrame([{c: patient[c] for c in feature_columns}])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])
    return {"prediction": prediction, "probability": probability}
