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
    Smoke test: importing mlops_project.training should not crash.
    Heavy computation must NOT run at import time.
    """

    # If this import raises, the test fails
    import mlops_project.training
