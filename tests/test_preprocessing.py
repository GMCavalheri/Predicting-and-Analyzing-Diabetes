import numpy as np
import pandas as pd
import pytest

from diabetes_prediction.features import REALISTIC_FEATURES
from diabetes_prediction.model import get_models
from diabetes_prediction.preprocessing import (
    build_pipeline,
    split_features_target,
    train_test_split_data,
)


def test_split_features_target(sample_df: pd.DataFrame) -> None:
    X, y = split_features_target(sample_df, REALISTIC_FEATURES)
    assert list(X.columns) == REALISTIC_FEATURES
    assert y.name == "diagnosed_diabetes"
    assert len(X) == len(y) == len(sample_df)


def test_train_test_split_proportions(sample_df: pd.DataFrame) -> None:
    X, y = split_features_target(sample_df, REALISTIC_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.25)
    assert len(X_test) == pytest.approx(len(X) * 0.25, abs=2)
    assert len(X_train) + len(X_test) == len(X)


def test_train_test_split_no_row_overlap(sample_df: pd.DataFrame) -> None:
    X, y = split_features_target(sample_df, REALISTIC_FEATURES)
    X_train, X_test, _, _ = train_test_split_data(X, y)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_train_test_split_rejects_invalid_test_size(sample_df: pd.DataFrame) -> None:
    X, y = split_features_target(sample_df, REALISTIC_FEATURES)
    with pytest.raises(ValueError):
        train_test_split_data(X, y, test_size=1.5)


def test_pipeline_encodes_to_numeric_and_predicts(sample_df: pd.DataFrame) -> None:
    """Regression guard for the original notebook's split-direction bug: a
    correctly-directed 80/20 split should leave the model with a real
    training set, able to fit and predict without error."""
    X, y = split_features_target(sample_df, REALISTIC_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    pipeline = build_pipeline(get_models()["logistic_regression"], REALISTIC_FEATURES)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
