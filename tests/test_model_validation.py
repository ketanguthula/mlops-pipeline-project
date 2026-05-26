import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import split_features_target, build_preprocessor


DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"


def train_small_model():
    df = pd.read_csv(DATA_PATH)

    X, y = split_features_target(df, "Attrition")
    y = y.map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


def test_model_predictions_have_correct_shape():
    model, X_test, y_test = train_small_model()

    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape


def test_model_meets_minimum_f1_threshold():
    model, X_test, y_test = train_small_model()

    predictions = model.predict(X_test)
    score = f1_score(y_test, predictions)

    assert score >= 0.20