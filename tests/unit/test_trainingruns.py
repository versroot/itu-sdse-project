from unittest.mock import patch, MagicMock


@patch("mlops_project.training.pd.read_csv")
@patch("mlops_project.training.RandomizedSearchCV")
@patch("mlops_project.training.XGBRFClassifier")
@patch("mlops_project.training.LogisticRegression")
@patch("mlops_project.training.train_test_split")
@patch("mlops_project.training.joblib.dump")
@patch("mlops_project.training.mlflow")
def test_training_runs_like_script(
    mock_mlflow,
    mock_joblib,
    mock_train_test_split,
    mock_logreg,
    mock_xgbrf,
    mock_random_search,
    mock_read_csv,
):
    """
    Unit-level smoke test.

    Ensures that importing mlops_project.training executes the module
    top-to-bottom without crashing when all heavy dependencies
    (MLflow, sklearn, file I/O) are mocked.
    """

    import pandas as pd
    fake_df = pd.DataFrame({
        "customer_group": ["A", "B"],
        "onboarding": ["X", "Y"],
        "bin_source": ["signup", "signup"],
        "source": ["signup", "signup"],
        "lead_indicator": [1, 0],
        "feature_num": [1.0, 2.0],
        "date_part": ["2024-01-01", "2024-01-02"],
        "lead_id": ["id1", "id2"],
        "customer_code": ["c1", "c2"],
    })

    mock_read_csv.return_value = fake_df
    mock_train_test_split.return_value = (fake_df, fake_df, [1, 0], [1, 0])
    fake_search = MagicMock()
    fake_search.fit.return_value = None
    fake_search.predict.return_value = [1, 0]
    fake_search.best_estimator_ = MagicMock()
    fake_search.best_params_ = {"param": "value"}
    mock_random_search.return_value = fake_search
    mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id=123)
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Test passes if this import does not raise
    import mlops_project.training
