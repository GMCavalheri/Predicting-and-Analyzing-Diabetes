"""End-to-end smoke test: does the whole training pipeline actually run?

Runs on the small `sample_df` fixture rather than the full 100k-row CSV, and
redirects model/figure/metrics output to a tmp directory so it never
touches the real generated artifacts in models/ or reports/.
"""

from __future__ import annotations

import json

import pandas as pd

from diabetes_prediction import train


def test_run_pipeline_end_to_end(sample_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(train, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(train, "load_dataset", lambda: sample_df)

    results = train.run_pipeline(
        feature_sets=["realistic"], model_names=["logistic_regression"], cv=2
    )

    run_key = "realistic__logistic_regression"
    assert run_key in results
    assert 0.0 <= results[run_key]["accuracy"] <= 1.0
    assert (tmp_path / "models" / f"{run_key}.joblib").exists()

    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        assert run_key in json.load(f)
