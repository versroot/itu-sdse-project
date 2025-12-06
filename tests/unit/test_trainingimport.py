from unittest.mock import patch, MagicMock


@patch("mlops_project.training.pd")
@patch("mlops_project.training.RandomizedSearchCV")
@patch("mlops_project.training.XGBRFClassifier")
@patch("mlops_project.training.LogisticRegression")
@patch("mlops_project.training.train_test_split")
@patch("mlops_project.training.joblib.dump")
@patch("mlops_project.training.mlflow")
@patch("mlops_project.training.open", create=True)
def test_training_import_safe(
    mock_open,
    mock_mlflow,
    mock_joblib_dump,
    mock_train_test_split,
    mock_logreg,
    mock_xgbrf,
    mock_random_search,
    mock_pd,
):
    """
    Ensure that importing mlops_project.training does NOT run heavy training,
    access real files, or crash. All expensive operations are mocked.
    """

    fake_df = MagicMock()
    # Behaviors used in training pipeline
    fake_df.drop.return_value = fake_df
    fake_df.astype.return_value = fake_df
    fake_df.__getitem__.return_value = fake_df
    fake_df.dtypes = MagicMock()
    fake_df.columns = ["col1", "col2"]
    fake_df.to_csv.return_value = None
    mock_pd.read_csv.return_value = fake_df

    mock_train_test_split.return_value = (fake_df, fake_df, [1, 0], [1, 0])

    mock_search_instance = MagicMock()
    mock_search_instance.fit.return_value = None
    mock_search_instance.predict.return_value = [1, 0]
    mock_search_instance.best_estimator_ = MagicMock()
    mock_search_instance.best_params_ = {"param": 123}

    mock_random_search.return_value = mock_search_instance

    mock_xgbrf.return_value = MagicMock()
    mock_logreg.return_value = MagicMock()
    mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id=123)
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    mock_open.return_value.__enter__.return_value.write.return_value = None

    import mlops_project.training

    mock_pd.read_csv.assert_called_once()
    mock_random_search.assert_called()
    mock_joblib_dump.assert_called()
    mock_mlflow.start_run.assert_called()
