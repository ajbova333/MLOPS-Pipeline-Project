"""Detect data drift between training dataset and a simulated production dataset using Evidently."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH= "data/raw/heart.csv"
REPORT_PATH= "reports/data_drift_report.html"
DRIFT_SHARE_THRESHOLD= 0.3
RANDOM_SEED= 12345


def load_reference(path=REFERENCE_PATH):
    """Load the reference dataset from the specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference dataset not found at {path}.")
    return pd.read_csv(path)

def simulate_production_data(reference_df, seed=RANDOM_SEED):
   """Simulate a production dataset with drift in a few features: an older patient population, higher cholesterol levels, and more missing values."""
   rng = np.random.default_rng(seed)
   prod_df = reference_df.copy()

   # Simulate older patient population by increasing age by 5 years on average
   prod_df['age'] = prod_df['age'] + rng.normal(loc=5, scale=2, size=len(prod_df)).astype(int)

   # Simulate higher cholesterol levels by adding a random increase
   prod_df['chol'] = prod_df['chol'] + rng.normal(loc=20, scale=10, size=len(prod_df)).astype(int)

   # Introduce additional missing values in 'thalach' and 'oldpeak'
   for col in ['thalach', 'oldpeak']:
       mask = rng.random(len(prod_df)) < 0.1  # 10% missing
       prod_df.loc[mask, col] = np.nan

   return prod_df

def run_drift_report(reference_df, production_df):
    report= Report(metrics=[DataDriftPreset()])
    return report.run(reference_data=reference_df, current_data=production_df)

def summarize_drift(result):
    data= result.dict()['metrics']
    overall_share= data[0]['value']['share']

    drifted_columns = []
    for metric in data[1:]:
        if "ValueDrift" not in metric["metric_name"]:
            continue
        column = metric["config"]["column"]
        threshold = metric["config"]["threshold"]
        p_value = metric["value"]
        if p_value < threshold:
            drifted_columns.append(column)
    return overall_share, drifted_columns

def main(threshold):
    reference_df= load_reference()
    production_df= simulate_production_data(reference_df)

    result= run_drift_report(reference_df, production_df)

    os.makedirs('reports', exist_ok=True)
    result.save_html(REPORT_PATH)

    drift_share, drifted_columns= summarize_drift(result)

    print(f' Overall drift share: {drift_share:.2f}')
    print(f'Drifted features ({len(drifted_columns)}): {drifted_columns}')
    print(f'Drift report saved to {REPORT_PATH}')

    if drift_share > threshold:
        print(f'Warning: Overall drift share {drift_share:.2f} exceeds threshold {threshold:.2f}.')
        sys.exit(1)

    print('Drift within acceptable threshold.')

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DRIFT_SHARE_THRESHOLD)
    args= parser.parse_args()
    main(threshold=args.threshold)

