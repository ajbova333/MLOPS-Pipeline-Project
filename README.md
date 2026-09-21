# Heart Disease Prediction — MLOps Pipeline

An end-to-end MLOps pipeline built around a heart disease classification model. The
model itself is intentionally simple (logistic regression / random forest) — the focus
of this project is the infrastructure around it: version-controlled data, experiment
tracking, automated testing, CI/CD, and drift monitoring.

## Dataset

[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
from Kaggle (a mirror of the UCI Cleveland Heart Disease dataset), ~1,025 rows and 14
columns. Binary classification task: predict presence of heart disease (`target`) from
a mix of numeric (age, cholesterol, blood pressure, etc.) and categorical-coded
(chest pain type, thalassemia, etc.) clinical features.

The source dataset has no missing values, so `src/prepare_data.py` was run once to
inject synthetic missing values into a subset of columns, satisfying the project's data
requirements while keeping the pipeline's missing-value handling meaningful.

## Project Structure

```
heart-prediction/
├── src/                    # Pipeline source code
│   ├── prepare_data.py     # One-time: inject synthetic missing values into raw data
│   ├── preprocessing.py    # Missing-value imputation, categorical encoding
│   ├── train.py            # Trains a model from a config, logs to MLflow
│   ├── evaluate.py         # Metric computation and threshold checking
│   ├── compare_experiments.py  # Queries MLflow to find the best experiment run
│   └── monitor_drift.py    # Evidently drift detection between train and prod data
├── configs/                # YAML training configs (one per experiment)
├── tests/                  # pytest suite: unit, data validation, model validation
├── data/raw/                # DVC-tracked dataset (not committed to git)
├── dvc-storage/            # DVC remote storage (committed, in-repo)
├── reports/                # Generated drift reports (HTML)
├── .github/workflows/      # CI/CD pipeline
└── MONITORING.md            # Drift analysis writeup
```

## Setup

```bash
git clone https://github.com/ajbova333/MLOPS-Pipeline-Project.git
cd MLOPS-Pipeline-Project/heart-prediction
pip install -r requirements.txt
dvc pull
```

`dvc pull` fetches `data/raw/heart.csv` from the DVC remote (stored inside this repo
under `dvc-storage/`), so no external credentials or cloud storage access are needed.

## Running Training

```bash
python -m src.train --config configs/train_config.yaml
```

Each run logs its hyperparameters, data version, and metrics to MLflow, and saves the
trained model as an MLflow artifact. Several other configs are included, representing
different model types and hyperparameters:

```bash
python -m src.train --config configs/train_config_lr_high_reg.yaml
python -m src.train --config configs/train_config_lr_low_reg.yaml
python -m src.train --config configs/train_config_rf_shallow.yaml
python -m src.train --config configs/train_config_rf_deep.yaml
```

Training exits with a non-zero status if the resulting model doesn't meet the minimum
performance thresholds defined in the config.

View results in the MLflow UI:

```bash
mlflow ui
```

Or compare all runs from the command line:

```bash
python -m src.compare_experiments --experiment heart_disease_prediction
```

## Running Tests

```bash
pytest tests/ -v
```

14 tests across three tiers: unit tests for preprocessing (`test_preprocessing.py`),
data validation tests against the real dataset (`test_data_validation.py`), and model
validation tests that train on a sample and check predictions/performance
(`test_model.py`).

## Drift Monitoring

```bash
python -m src.monitor_drift
```

Compares the training data against a simulated production dataset (with deliberately
shifted features), prints a drift summary, and saves an HTML report to `reports/`. Exits
non-zero if overall drift exceeds a configurable threshold (`--threshold`, default 30%).
See [MONITORING.md](MONITORING.md) for the full analysis.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request to
`main`: a `test` job installs dependencies, pulls data via DVC, and runs the full test
suite; a `train` job (gated on `test` passing) does the same and then runs training,
failing the pipeline if the model misses its performance thresholds.
