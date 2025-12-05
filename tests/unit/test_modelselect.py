from unittest.mock import MagicMock, patch

from mlops_project.model_select import wait_until_ready


@patch("mlops_project.model_select.MlflowClient")
@patch("mlops_project.model_select.time.sleep")
def test_wait_until_ready_polls_until_ready(mock_sleep, mock_mlflow_client):
    """
    Checks that wait_until_ready keeps polling until the model version is READY.
    We don't care how many times exactly; just that we call get_model_version.
    """

    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance

    mock_client_instance.get_model_version.return_value = MagicMock(
        status="READY"
    )

    wait_until_ready("lead_model", 1)

    assert mock_client_instance.get_model_version.called
