"""Encoding and train/test splitting for the diabetes dataset.

The original notebook encoded categorical columns on the full dataframe
before splitting (leaking test-set categories into the encoder) and
inverted the train_test_split call: it computed `test_size = 1 - train_size`
with `train_size = 0.005`, so the model trained on 0.5% of the data and was
"tested" on the other 99.5% -- backwards from normal practice, and the
likely reason the original README could claim "100% accuracy". Both issues
are fixed here.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from diabetes_prediction.config import RANDOM_STATE, TARGET


def split_features_target(
    df: pd.DataFrame, feature_columns: list[str], target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into predictive features X and target y."""
    return df[feature_columns].copy(), df[target].copy()


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test sets, stratified on the target.

    `test_size` is the fraction of data held out for testing (e.g. 0.2 means
    20% test / 80% train) -- standard scikit-learn semantics, unlike the
    original notebook's inverted split (see module docstring).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """Ordinal-encode categorical columns, fitting only on the training split.

    Only encodes columns from `categorical_columns` that are actually present
    in X_train, so this works unchanged across feature sets that drop some
    categorical columns (e.g. `diabetes_stage` in the "realistic" set).
    """
    cols = [c for c in categorical_columns if c in X_train.columns]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy()
    X_test = X_test.copy()
    if cols:
        X_train[cols] = encoder.fit_transform(X_train[cols])
        X_test[cols] = encoder.transform(X_test[cols])
    return X_train, X_test, encoder
