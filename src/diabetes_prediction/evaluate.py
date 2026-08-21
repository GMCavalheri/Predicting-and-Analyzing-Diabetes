"""Metrics, cross-validation, feature importance, and diagnostic plots.

Everything here operates on a fitted `Pipeline` (preprocessing + model, see
`preprocessing.build_pipeline`) and raw feature dataframes -- there's no
separately-encoded X to keep track of.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from diabetes_prediction.config import RANDOM_STATE


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute standard classification metrics on a held-out test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def cross_validate_model(pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """Stratified k-fold cross-validation on the training split.

    `cross_validate` clones and refits the whole pipeline (preprocessing
    included) once per fold -- scoring both metrics from that single fit,
    unlike calling `cross_val_score` once per metric, which would refit
    (and, for the slower gradient-boosting model, roughly double the
    training time) redundantly. `n_jobs=-1` fits folds in parallel.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=["accuracy", "f1"], n_jobs=-1)
    return {
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_mean": scores["test_f1"].mean(),
        "cv_f1_std": scores["test_f1"].std(),
    }


def get_feature_importance(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Feature importance: native for tree models, permutation otherwise.

    Native `feature_importances_` isn't available on `LogisticRegression`,
    so permutation importance (which works on any fitted estimator/pipeline)
    is used as a fallback -- this keeps all three models' importance charts
    directly comparable, in the original units of the input columns.
    """
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        raw_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in raw_names]
        importances = model.feature_importances_
    else:
        result = permutation_importance(
            pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
        )
        names = list(X_test.columns)
        importances = result.importances_mean
    return dict(sorted(zip(names, importances.tolist()), key=lambda kv: kv[1], reverse=True))


def save_confusion_matrix_plot(y_test: pd.Series, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_roc_curve_plot(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series, path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
