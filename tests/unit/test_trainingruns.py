import subprocess
import sys
from unittest.mock import patch
import pandas as pd


@patch("mlops_project.training.pd.read_csv")
@patch("mlops_project.training.RandomizedSearchCV")
@patch("mlops_project.training.XGBRFClassifier")
@patch("mlops_project.training.LogisticRegression")
@patch("mlops_project.training.train_test_split")
@patch("mlops_project.training.joblib.dump")
@patch("mlops_project.training.mlflow")
def test_training_runs_as_script(
    mock_mlflow,
    mock_joblib,
    mock_train_test_split,
    mock_logreg,
    mock_xgbrf,
    mock_random_search,
    mock_read_csv,
    tmp_path,
    monkeypatch,
):
    """
    Runs `python -m mlops_project.training` with heavy operations mocked.
    Ensures the script completes without crashing (like CI would check).
    """

    # ---- Fake dataframe returned from read_csv ----
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

    monkeypatch.chdir(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "mlops_project.training"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
