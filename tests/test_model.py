"""Model Validation Tests: train a model on a small sample and verify predictions and thresholds"""


from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from src.evaluate import compute_metrics
from src.preprocessing import encode_categorical_columns, handle_missing_values

DATA_PATH = "data/raw/heart.csv"
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
COLUMNS_TO_ENCODE = ["cp", "restecg", "slope", "thal", "ca"]


@pytest.fixture(scope="module")
def trained_model_and_test_set():
    df = pd.read_csv(DATA_PATH).sample(n=300, random_state=42).reset_index(drop=True)
    df = handle_missing_values(df, numeric_columns= NUMERIC_COLUMNS, categorical_columns=CATEGORICAL_COLUMNS)
    df= encode_categorical_columns(df, categorical_columns=COLUMNS_TO_ENCODE)

    X= df.drop(columns=['target'])
    y= df['target']
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model= LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def test_predictions_have_correct_type_and_shape(trained_model_and_test_set: tuple[LogisticRegression, Any, Any]):
    model, X_test, y_test= trained_model_and_test_set
    predictions= model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X_test),)
    assert set(predictions).issubset({0, 1})

def test_model_meets_min_threshold(trained_model_and_test_set):
    model, X_test, y_test= trained_model_and_test_set
    predictions= model.predict(X_test)
    metrics= compute_metrics(y_test, predictions)

    assert metrics['accuracy'] >= 0.6
