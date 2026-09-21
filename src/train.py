"""Train a heart disease prdiction model, logging the run to MLflow."""

import argparse
import mlflow
import pandas as pd
import sys
import mlflow.sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.evaluate import compute_metrics, meets_thresholds
from src.preprocessing import handle_missing_values, encode_categorical_columns

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier
}

def load_config(config_path):
    """ Loads the configuration from a YAML file. """
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_version(dvc_pointer_path):
    with open(dvc_pointer_path) as f:
        pointer = yaml.safe_load(f)
    return pointer["outs"][0]["md5"]


def build_model(model_config):
    model_type = model_config["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    params = model_config.get("params", {})
    return MODEL_REGISTRY[model_type](**params)

def main(config_path):
    config = load_config(config_path)
    data_cfg = config["data"]

    df = pd.read_csv(data_cfg["raw_path"])
    data_version = get_data_version(data_cfg["dvc_pointer_path"])

    df = handle_missing_values(
        df,
        numeric_columns=data_cfg["numeric_columns"],
        categorical_columns=data_cfg["categorical_columns"],
    )

    df = encode_categorical_columns(df, categorical_columns=data_cfg["columns_to_encode"])

    target_column = data_cfg["target_column"]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"], stratify=y)

    model = build_model(config["model"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_proba)

    thresholds = config.get("thresholds", {})

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_param('model_type', config["model"]["type"])
        mlflow.log_param('data_version', data_version)
        mlflow.log_params(config["model"].get('params', {}))
        mlflow.log_param('test_size', data_cfg["test_size"])
        mlflow.log_param('random_state', data_cfg["random_state"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])


        print('Metrics:', metrics)

        if thresholds and not meets_thresholds(metrics, thresholds):
            print("Model did not meet the specified thresholds.")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a heart disease prediction model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
