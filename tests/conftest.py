"""Shared fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from diabetes_prediction.config import DATA_PATH
from diabetes_prediction.data_loading import load_dataset


@pytest.fixture(scope="session")
def full_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}; see data/README.md")
    return load_dataset()


@pytest.fixture
def sample_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """A small, class-balanced sample for fast tests (both classes present,
    enough rows for cross-validation folds to be meaningful)."""
    per_class = [
        group.sample(min(len(group), 150), random_state=0)
        for _, group in full_df.groupby("diagnosed_diabetes")
    ]
    return pd.concat(per_class).reset_index(drop=True)
