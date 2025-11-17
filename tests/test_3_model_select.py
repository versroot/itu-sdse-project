

def test_wait_until_ready_stops_when_model_is_ready(monkeypatch):
    """
    Test that wait_until_ready:
    - calls MlflowClient().get_model_version repeatedly
    - stops early once status == READY
    - does not actually sleep during the test
    """
    # Import inside the test so monkeypatches on the module can be applied afterwards if needed
    from mlops_project import model_select

    # --- set up a dummy client that returns statuses PENDING -> READY ---

    class DummyModelVersion:
        def __init__(self, status):
            self.status = status

    class DummyClient:
        def __init__(self):
            # first call -> "PENDING", second call -> "READY"
            self._statuses = ["PENDING", "READY"]
            self.calls = 0

        def get_model_version(self, name, version):
            idx = min(self.calls, len(self._statuses) - 1)
            status = self._statuses[idx]
            self.calls += 1
            return DummyModelVersion(status=status)

    dummy_client = DummyClient()

    # Patch MlflowClient in the model_select module to return our dummy client
    monkeypatch.setattr(model_select, "MlflowClient", lambda: dummy_client)

    # --- patch ModelVersionStatus so that from_string(...) returns a comparable object ---

    class DummyStatus:
        READY = "READY"

        @classmethod
        def from_string(cls, s):
            # just return the raw string like "PENDING" / "READY"
            return s

    monkeypatch.setattr(model_select, "ModelVersionStatus", DummyStatus)

    # --- avoid real sleeping in tests ---

    monkeypatch.setattr(model_select.time, "sleep", lambda _: None)

    # --- act: call the function under test ---

    model_select.wait_until_ready(model_name="some-model", model_version="1")

    # --- assert: client.get_model_version should have been called twice:
    # 1st: PENDING, 2nd: READY, then loop exits
    assert dummy_client.calls == 2
