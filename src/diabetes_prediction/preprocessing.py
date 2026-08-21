"""Preprocessing pipeline and train/test splitting for the diabetes dataset.

The original notebook encoded categorical columns on the full dataframe
before splitting (leaking test-set categories into the encoder) and
inverted the train_test_split call: it computed `test_size = 1 - train_size`
with `train_size = 0.005`, so the model trained on 0.5% of the data and was
"tested" on the other 99.5% -- backwards from normal practice, and the
likely reason the original README could claim "100% accuracy". Both issues
are fixed here.

Encoding is wrapped in a scikit-learn Pipeline together with the classifier
(see `build_pipeline` and `model.py`) rather than applied as a separate
manual step. That way `cross_val_score`/`GridSearchCV`-style tooling always
refits the encoder on the training fold only, and the single persisted
`.joblib` file is self-contained -- the dashboard can hand it a raw patient
dict (e.g. `gender="Male"`) with no separate encoder to load and keep in
sync.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from diabetes_prediction.config import COL_CATEGORICAL, RANDOM_STATE, TARGET


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


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Ordinal-encode categorical columns and standard-scale numeric ones.

    Only affects `feature_columns` that are actually present, so this works
    unchanged across feature sets that drop some columns (e.g.
    `diabetes_stage` is absent from the "realistic" set). Numeric columns
    span very different scales (e.g. `age` in tens vs. `triglycerides` in
    hundreds); without scaling, LogisticRegression's lbfgs solver fails to
    converge within a reasonable iteration budget (measured: ~34s/fit and a
    ConvergenceWarning instead of ~1s/fit once scaled). Tree models are
    scale-invariant, so this is free for them.
    """
    categorical = [c for c in COL_CATEGORICAL if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ]
    )


def build_pipeline(model, feature_columns: list[str]) -> Pipeline:
    """Bundle preprocessing and a classifier into a single fit/predict unit."""
    return Pipeline([("preprocess", build_preprocessor(feature_columns)), ("model", model)])
