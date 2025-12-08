import time

from mlflow.tracking import MlflowClient

model_version = 1
model_name = "lead_model"
client = MlflowClient()


def wait_for_deployment(model_name, model_version, stage="Staging"):
    status = False
    while not status:
        model_version_details = dict(client.get_model_version(name=model_name, version=model_version))
        if model_version_details["current_stage"] == stage:
            status = True
            break
        else:
            time.sleep(2)
    return status


def deploy_model(model_name=model_name, model_version=model_version):
    """
    Deploy a model by transitioning it to Staging stage.
    This function should be called explicitly, not executed at import time.
    """
    model_version_details = dict(client.get_model_version(name=model_name, version=model_version))
    model_status = True
    if model_version_details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True,
        )
        model_status = wait_for_deployment(model_name, model_version, "Staging")
    return model_status


if __name__ == "__main__":
    # only execute deployment when run as a script, not at testing
    deploy_model()
