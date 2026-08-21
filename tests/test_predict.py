import pandas as pd
import pytest

from diabetes_prediction.features import REALISTIC_FEATURES
from diabetes_prediction.model import get_models
from diabetes_prediction.predict import predict_single
from diabetes_prediction.preprocessing import (
    build_pipeline,
    split_features_target,
    train_test_split_data,
)


@pytest.fixture
def fitted_pipeline(sample_df: pd.DataFrame):
    X, y = split_features_target(sample_df, REALISTIC_FEATURES)
    X_train, _, y_train, _ = train_test_split_data(X, y)
    pipeline = build_pipeline(get_models()["logistic_regression"], REALISTIC_FEATURES)
    pipeline.fit(X_train, y_train)
    return pipeline


def test_predict_single_returns_well_formed_output(
    sample_df: pd.DataFrame, fitted_pipeline
) -> None:
    patient = sample_df.iloc[0][REALISTIC_FEATURES].to_dict()
    result = predict_single(patient, model=fitted_pipeline)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_single_raises_on_missing_fields(fitted_pipeline) -> None:
    incomplete_patient = {"age": 45}
    with pytest.raises(ValueError, match="Missing required patient fields"):
        predict_single(incomplete_patient, model=fitted_pipeline)
