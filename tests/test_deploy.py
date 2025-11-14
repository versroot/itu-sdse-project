import types

from mlops_project import deploy


def test_wait_for_deployment_retries_until_stage_reached(monkeypatch):
    class DummyClient:
        def __init__(self):
            # first two calls -> "None", third call -> "Staging"
            self.stages = ["None", "None", "Staging"]
            self.calls = 0

        def get_model_version(self, name, version):
            idx = min(self.calls, len(self.stages) - 1)
            stage = self.stages[idx]
            self.calls += 1
            return {"current_stage": stage}

    dummy_client = DummyClient()

    # Replace the global client in deploy.py with our dummy
    deploy.client = dummy_client

    # Inject a fake `time` object into deploy, allowing attribute creation
    monkeypatch.setattr(
        deploy,
        "time",
        types.SimpleNamespace(sleep=lambda _secs: None),
        raising=False,
    )

    status = deploy.wait_for_deployment(
        model_name="lead_model", model_version=1, stage="Staging"
    )

    assert status is True
    assert dummy_client.calls >= 2
