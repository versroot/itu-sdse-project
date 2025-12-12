from unittest.mock import patch, MagicMock


@patch("mlops_project.model_select.wait_until_ready")
@patch("mlops_project.model_select.mlflow")
@patch("mlops_project.model_select.pd")
@patch("mlops_project.model_select.open", create=True)
@patch("mlops_project.model_select.MlflowClient")
def test_model_select_imports_safely(
    mock_client_class,
    mock_open,
    mock_pd,
    mock_mlflow,
    mock_wait
):
    """
    Smoke test: ensures that importing mlops_project.model_select
    does not crash when external dependencies are mocked.
    """

    mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id=123)

    mock_df = MagicMock()
    mock_df.iloc.__getitem__.return_value = {
        "metrics.f1_score": 0.9,
        "run_id": "12345",
    }
    mock_mlflow.search_runs.return_value = mock_df

    mock_open.return_value.__enter__.return_value.read.return_value = (
        '{"modelA":{"weighted avg":{"f1-score":0.7}}}'
    )

    mock_pd.DataFrame.return_value = MagicMock()

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.search_model_versions.return_value = []

    # The test passes if this import does not raise
    import mlops_project.model_select


    mock_mlflow.get_experiment_by_name.assert_called()
    mock_mlflow.search_runs.assert_called()
    mock_open.assert_called()
