import pandas as pd
import pytest

from diabetes_prediction.features import REALISTIC_FEATURES
from diabetes_prediction.model import get_models
from diabetes_prediction.preprocessing import (
    build_pipeline,
    split_features_target,
    train_test_split_data,
)


def test_get_models_returns_expected_keys() -> None:
    models = get_models()
    assert set(models) == {"logistic_regression", "random_forest", "gradient_boosting"}


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "random_forest", "gradient_boosting"]
)
def test_each_model_fits_and_predicts(sample_df: pd.DataFrame, model_name: str) -> None:
    X, y = split_features_target(sample_df, REALISTIC_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    pipeline = build_pipeline(get_models()[model_name], REALISTIC_FEATURES)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert proba.shape == (len(X_test), 2)
