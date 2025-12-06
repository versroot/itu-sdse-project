import subprocess
import sys
from unittest.mock import patch, MagicMock
import pandas as pd


@patch("mlops_project.preprocessing.pd.read_csv")
@patch("mlops_project.training.pd.read_csv")
@patch("mlops_project.model_select.mlflow")
@patch("mlops_project.deploy.client")
def test_pipeline_fast(
    mock_client,
    mock_mlflow,
    mock_read_training,
    mock_read_preprocessing,
    tmp_path,
    monkeypatch
):
    """
    FAST integration test: pipeline completes end-to-end with mocks.
    No real training, no real MLflow, no real artifacts needed.
    """
    monkeypatch.chdir(tmp_path)

    # --- Fake CSV data (used by both preprocessing and training)
    df = pd.DataFrame({
        "date_part": ["2024-01-05", "2024-01-06"],
        "source": ["signup", "signup"],
        "lead_indicator": [1, 0],
        "lead_id": ["1", "2"],
        "customer_code": ["c1", "c2"],
        "customer_group": ["A", "B"],
        "onboarding": ["X", "Y"],
        "bin_source": ["signup", "signup"],
        "feature_num": [1.0, 2.0],
    })

    mock_read_preprocessing.return_value = df.copy()
    mock_read_training.return_value = df.copy()

    # --- Fake MLflow behaviors
    mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id=123)
    mock_mlflow.search_runs.return_value = MagicMock(
        iloc={0: {"metrics.f1_score": 0.9, "run_id": "123"}}
    )
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_client.get_model_version.return_value = {"current_stage": "Staging"}

    p1 = subprocess.run(
        [sys.executable, "-m", "mlops_project.preprocessing"],
        capture_output=True,
        text=True
    )
    assert p1.returncode == 0

    p2 = subprocess.run(
        [sys.executable, "-m", "mlops_project.training"],
        capture_output=True,
        text=True
    )
    assert p2.returncode == 0

    p3 = subprocess.run(
        [sys.executable, "-m", "mlops_project.model_select"],
        capture_output=True,
        text=True
    )
    assert p3.returncode == 0

    p4 = subprocess.run(
        [sys.executable, "-m", "mlops_project.deploy"],
        capture_output=True,
        text=True
    )
    assert p4.returncode == 0
