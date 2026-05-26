import pandas as pd


DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"


def test_expected_columns_present():
    df = pd.read_csv(DATA_PATH)

    expected_columns = {
        "Age",
        "Attrition",
        "BusinessTravel",
        "Department",
        "DistanceFromHome",
        "Education",
        "JobRole",
        "MonthlyIncome",
    }

    assert expected_columns.issubset(set(df.columns))


def test_target_contains_expected_values():
    df = pd.read_csv(DATA_PATH)

    assert set(df["Attrition"].unique()).issubset({"Yes", "No"})


def test_numeric_features_within_expected_ranges():
    df = pd.read_csv(DATA_PATH)

    assert df["Age"].between(18, 70).all()
    assert df["DistanceFromHome"].between(0, 50).all()
    assert df["MonthlyIncome"].gt(0).all() 