import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import split_features_target, build_preprocessor


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_model(config):
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():

        df = pd.read_csv(config["data"]["raw_path"])

        X, y = split_features_target(df, config["data"]["target_column"])

        y = y.map({"Yes": 1, "No": 0})

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            stratify=y,
        )

        preprocessor = build_preprocessor(X_train)

        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=config["model"]["random_state"],
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
        }

        mlflow.log_params(config["model"])

        mlflow.log_param(
            "data_path",
            config["data"]["raw_path"]
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        print("Evaluation metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return pipeline, metrics


if __name__ == "__main__":
    config = load_config()
    train_model(config)