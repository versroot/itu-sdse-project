import datetime
import json
import time

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient
import pandas as pd

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
artifact_path = "model"
model_name = "lead_model"
experiment_name = current_date


def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)


def main():
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]

    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids, order_by=["metrics.f1_score DESC"], max_results=1
    ).iloc[0]
    experiment_best

    with open("./artifacts/model_results.json", "r") as f:
        model_results = json.load(f)
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name

    client = MlflowClient()
    prod_model = [
        model
        for model in client.search_model_versions(f"name='{model_name}'")
        if dict(model)["current_stage"] == "Production"
    ]
    prod_model_exists = len(prod_model) > 0

    if prod_model_exists:
        # prod_model_version = dict(prod_model[0])['version'] not in use in the codebase
        prod_model_run_id = dict(prod_model[0])["run_id"]

    train_model_score = experiment_best["metrics.f1_score"]
    model_details = {}
    model_status = {}
    run_id = None

    if prod_model_exists:
        data, details = mlflow.get_run(prod_model_run_id)
        prod_model_score = data[1]["metrics.f1_score"]

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

        if train_model_score > prod_model_score:
            run_id = experiment_best["run_id"]
    else:
        run_id = experiment_best["run_id"]

    if run_id is not None:
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        wait_until_ready(model_details.name, model_details.version)
        model_details = dict(model_details)

    return {
        "best_model": best_model,
        "experiment_best": experiment_best,
        "model_status": model_status,
        "run_id": run_id,
        "model_details": model_details,
    }


if __name__ == "__main__":
    main()
