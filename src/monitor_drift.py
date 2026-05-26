import sys
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def create_production_data(df: pd.DataFrame, random_state: int = 42):
    production_df = df.sample(frac=0.3, random_state=random_state).copy()

    if "MonthlyIncome" in production_df.columns:
        production_df["MonthlyIncome"] = production_df["MonthlyIncome"] * 1.25

    if "Age" in production_df.columns:
        production_df["Age"] = production_df["Age"] + 3

    return production_df


def extract_drift_summary(snapshot_dict):
    drift_share = 0.0
    drifted_features = []

    for metric in snapshot_dict.get("metrics", []):
        value = metric.get("value", {})

        if isinstance(value, dict):
            if "share" in value:
                drift_share = value["share"]

            if "drifted_columns" in value:
                drifted_features = value["drifted_columns"]

            if "number_of_drifted_columns" in value and "number_of_columns" in value:
                drift_share = (
                    value["number_of_drifted_columns"] / value["number_of_columns"]
                )

    return drift_share, drifted_features


def monitor_drift(config_path="configs/config.yaml", drift_threshold=0.30):
    config = load_config(config_path)

    data_path = config["data"]["raw_path"]
    target_column = config["data"]["target_column"]
    random_state = config["data"]["random_state"]

    df = pd.read_csv(data_path)

    features_df = df.drop(columns=[target_column])

    reference_df = features_df.sample(frac=0.7, random_state=random_state).copy()
    production_df = create_production_data(features_df, random_state=random_state)

    report = Report([DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_df,
        current_data=production_df,
    )

    Path("reports").mkdir(exist_ok=True)
    snapshot.save_html("reports/data_drift_report.html")

    snapshot_dict = snapshot.dict()
    drift_share, drifted_features = extract_drift_summary(snapshot_dict)

    print(f"Drift share: {drift_share:.2f}")
    print(f"Drifted features: {drifted_features}")
    print("Drift report saved to reports/data_drift_report.html")

    if drift_share > drift_threshold:
        print("Drift threshold exceeded.")
        sys.exit(1)

    print("Drift is within acceptable limits.")


if __name__ == "__main__":
    monitor_drift()