"""Training pipeline entrypoint.

Trains and compares the candidate models (model.get_models) against all
three named feature sets (features.FEATURE_SETS), on a correctly-directed
80/20 train/test split (preprocessing.train_test_split_data). Writes
per-run metrics to reports/metrics.json, diagnostic plots to
reports/figures/, and persists each trained pipeline to
models/<feature_set>__<model_name>.joblib.

The "full" runs intentionally reproduce the original leakage bug (all
columns, including diabetes_stage/diabetes_risk_score) so the metrics.json
output can show, side by side, how much of the original "100% accuracy"
claim was the split bug vs. genuine leakage vs. real signal. See the
README's "Methodology & the Data Leakage Finding".

Usage:
    python -m diabetes_prediction.train
    python -m diabetes_prediction.train --feature-set realistic --model random_forest
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
from sklearn.base import clone

from diabetes_prediction.config import FIGURES_DIR, METRICS_PATH, MODEL_DIR
from diabetes_prediction.data_loading import load_dataset
from diabetes_prediction.evaluate import (
    cross_validate_model,
    evaluate_model,
    get_feature_importance,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
)
from diabetes_prediction.features import FEATURE_SETS
from diabetes_prediction.model import get_models
from diabetes_prediction.preprocessing import (
    build_pipeline,
    split_features_target,
    train_test_split_data,
)


def run_pipeline(
    feature_sets: list[str] | None = None,
    model_names: list[str] | None = None,
    cv: int = 5,
) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    feature_sets = feature_sets or list(FEATURE_SETS)
    candidate_models = get_models()
    model_names = model_names or list(candidate_models)

    results: dict[str, dict] = {}

    for feature_set_name in feature_sets:
        columns = FEATURE_SETS[feature_set_name]
        X, y = split_features_target(df, columns)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        for model_name in model_names:
            run_key = f"{feature_set_name}__{model_name}"
            print(f"Training {run_key} ...")
            start = time.time()

            pipeline = build_pipeline(clone(candidate_models[model_name]), columns)
            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(pipeline, X_test, y_test)
            cv_metrics = cross_validate_model(pipeline, X_train, y_train, cv=cv)
            importance = get_feature_importance(pipeline, X_test, y_test)

            model_path = MODEL_DIR / f"{run_key}.joblib"
            joblib.dump(pipeline, model_path)

            y_pred = pipeline.predict(X_test)
            save_confusion_matrix_plot(
                y_test, y_pred, FIGURES_DIR / f"{run_key}_confusion_matrix.png", run_key
            )
            save_roc_curve_plot(
                pipeline, X_test, y_test, FIGURES_DIR / f"{run_key}_roc_curve.png", run_key
            )

            results[run_key] = {
                "feature_set": feature_set_name,
                "model": model_name,
                "n_features": len(columns),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "training_seconds": round(time.time() - start, 2),
                **metrics,
                **cv_metrics,
                "top_features": list(importance.items())[:10],
                "model_path": str(model_path.relative_to(MODEL_DIR.parent)),
            }
            m = metrics
            print(
                f"  accuracy={m['accuracy']:.4f} f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f} "
                f"({time.time() - start:.1f}s)"
            )

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote metrics for {len(results)} runs to {METRICS_PATH}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set", choices=list(FEATURE_SETS), action="append", dest="feature_sets"
    )
    parser.add_argument(
        "--model", choices=list(get_models()), action="append", dest="model_names"
    )
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    run_pipeline(feature_sets=args.feature_sets, model_names=args.model_names, cv=args.cv)


if __name__ == "__main__":
    main()
