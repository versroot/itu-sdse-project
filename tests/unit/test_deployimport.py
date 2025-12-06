from unittest.mock import patch, MagicMock


@patch("mlops_project.deploy.client")
@patch("mlops_project.deploy.wait_for_deployment", return_value=True)
def test_deploy_module_import(mock_wait, mock_client):
    """
    This test ensures that importing mlops_project.deploy does not
    trigger real MLflow operations due to top-level code.
    """
    mock_client.get_model_version.return_value = {"current_stage": "Staging"}

    import mlops_project.deploy

    mock_client.get_model_version.assert_called()
