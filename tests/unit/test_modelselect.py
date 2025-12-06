from unittest.mock import patch, MagicMock
import mlops_project.model_select as model_select


@patch("mlops_project.model_select.time.sleep")  # avoid real waiting
@patch("mlops_project.model_select.MlflowClient")  # mock out MLflow
def test_wait_until_ready_reaches_ready(mock_client_class, mock_sleep):
    """
    Tests that wait_until_ready stops polling when MLflow returns
    a status of READY.
    """
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate two polls: first NOT ready, second READY
    mock_client.get_model_version.side_effect = [
        MagicMock(status="PENDING"),
        MagicMock(status="READY"),
    ]

    model_select.wait_until_ready("lead_model", 1)

    # The client should be called twice
    assert mock_client.get_model_version.call_count == 2
