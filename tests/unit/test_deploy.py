from unittest.mock import patch

from mlops_project.deploy import wait_for_deployment


@patch("mlops_project.deploy.client")
@patch("mlops_project.deploy.time.sleep")
def test_wait_for_deployment_reaches_stage(mock_sleep, mock_client):
    """
    Simulate the model being 'Not ready' first, then reaching the target stage.
    We only care that we poll MLflow until the desired stage is seen.
    """

    # First call: not in Staging; second call: in Staging
    mock_client.get_model_version.side_effect = [
        {"current_stage": "None"},
        {"current_stage": "Staging"},
    ]

    wait_for_deployment("lead_model", 1, stage="Staging")
    assert mock_client.get_model_version.call_count == 2
