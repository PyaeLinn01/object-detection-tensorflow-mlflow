import os
import sys

import mlflow
from get_or_create_mlflow_experiment import get_experiment_id

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


def get_last_run_id(exp_name):
    exp = get_experiment_id(exp_name)
    client = mlflow.MlflowClient()
    runs = client.search_runs(experiment_ids=exp)
    if len(runs) == 0:
        return None
    last_run = runs[0]
    return last_run.info.run_id


if __name__ == "__main__":
    run_id = get_last_run_id("mario_wario")
    os.environ[sys.argv[1]] = run_id
    print(f"Saving {run_id} to environment variable {sys.argv[1]}")
