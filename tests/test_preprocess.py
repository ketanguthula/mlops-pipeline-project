import pandas as pd
import pytest

from src.preprocess import split_features_target


def test_split_features_target_returns_correct_shapes():
    df = pd.DataFrame({
        "age": [25, 30],
        "salary": [50000, 60000],
        "target": [1, 0]
    })

    X, y = split_features_target(df, "target")

    assert X.shape == (2, 2)
    assert y.shape == (2,)


def test_target_column_removed_from_features():
    df = pd.DataFrame({
        "feature": [1, 2],
        "target": [0, 1]
    })

    X, _ = split_features_target(df, "target")

    assert "target" not in X.columns


def test_invalid_target_column_raises_error():
    df = pd.DataFrame({
        "feature": [1, 2]
    })

    with pytest.raises(ValueError):
        split_features_target(df, "missing_target")


def test_original_dataframe_not_modified():
    df = pd.DataFrame({
        "feature": [1, 2],
        "target": [0, 1]
    })

    original_columns = df.columns.tolist()

    split_features_target(df, "target")

    assert df.columns.tolist() == original_columns