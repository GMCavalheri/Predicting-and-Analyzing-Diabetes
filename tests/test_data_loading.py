import pandas as pd
import pytest

from diabetes_prediction.data_loading import EXPECTED_COLUMNS, load_dataset


def test_load_dataset_returns_expected_columns(full_df: pd.DataFrame) -> None:
    assert EXPECTED_COLUMNS.issubset(set(full_df.columns))


def test_load_dataset_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_load_dataset_missing_columns_raises(tmp_path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_dataset(bad_csv)
