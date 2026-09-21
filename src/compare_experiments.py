""" Query MLflow experiments and compare their metrics. """

import argparse

import mlflow

PRIMARY_METRIC= "accuracy"

def get_runs(experiment_name):
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics." + PRIMARY_METRIC + " DESC"])
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")
    return runs


def main(experiment_name):
      runs= get_runs(experiment_name)

      metric_cols= [ c for c in runs.columns if c.startswith("metrics.")]
      param_cols= [ c for c in runs.columns if c.startswith("params.")]

      print(f'Found {len(runs)} runs for experiment "{experiment_name}".')
      print(runs[["run_id", "status"] + metric_cols + param_cols].to_string(index=False))

      best_run= runs.iloc[0]
      print(f'\nBest run based on {PRIMARY_METRIC}: {best_run["run_id"]}')
      print(f" {PRIMARY_METRIC}: {best_run['metrics.' + PRIMARY_METRIC]}")
      for col in param_cols:
          print(f" {col}: {best_run[col]}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="Compare MLflow experiments.")
    parser.add_argument("--experiment", default="heart_disease_prediction")
    args= parser.parse_args()

    main(args.experiment)