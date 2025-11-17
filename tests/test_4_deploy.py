import sys
import types

#Fake Mlflow client so deploy.py doesn't hit a real registry

class DummyModelVersion:
    def __init__(self, stage="Staging"):
        self.current_stage = stage


class DummyMlflowClient:
    def get_model_version(self, name, version):
        # Pretend the model exists and is already in "Staging"
        return DummyModelVersion(stage="Staging")

    def transition_model_version_stage(
        self,
        name,
        version,
        stage,
        archive_existing_versions=True,
    ):
        return None


# Create a fake mlflow.tracking module
mlflow_mod = types.ModuleType("mlflow")
tracking_mod = types.ModuleType("mlflow.tracking")
tracking_mod.MlflowClient = DummyMlflowClient

# Wire them into sys.modules so imports see our fake versions
mlflow_mod.tracking = tracking_mod
sys.modules["mlflow"] = mlflow_mod
sys.modules["mlflow.tracking"] = tracking_mod


# Now import deploy; it will use DummyMlflowClient instead of real one
from mlops_project import deploy


def test_deploy_imports_and_has_client():
    # If we got here, import didn't crash.
    # Basic sanity check: module has a MlflowClient instance.
    assert hasattr(deploy, "client")
