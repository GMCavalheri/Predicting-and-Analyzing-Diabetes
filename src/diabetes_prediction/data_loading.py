"""Loading and schema validation for the diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from diabetes_prediction.config import COL_CATEGORICAL, COL_NUMERIC, DATA_PATH, TARGET

EXPECTED_COLUMNS = set(COL_CATEGORICAL) | set(COL_NUMERIC) | {TARGET}


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the patient dataset and validate it has the expected columns.

    Raises FileNotFoundError if the CSV is missing and ValueError if any
    expected column is absent, so schema drift fails fast instead of
    surfacing as a confusing KeyError deep in the pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected the CSV at "
            "data/raw/diabetes_dataset.csv (see data/README.md)."
        )
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df
