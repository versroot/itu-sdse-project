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
    Fast integration test for the full pipeline.

    This test runs preprocessing, training, model selection, and deployment
    end-to-end using mocks to avoid heavy computation and external services.
    """
    monkeypatch.chdir(tmp_path)

    # Synthetic dataset used across pipeline steps
    df = pd.DataFrame({
        "date_part": ["2024-01-05"] * 20,
        "source": ["signup"] * 20,
        "lead_indicator": [0, 1] * 10,
        "lead_id": [str(i) for i in range(20)],
        "customer_code": ["c"] * 20,
        "customer_group": ["A", "B"] * 10,
        "onboarding": ["X", "Y"] * 10,
        "bin_source": ["signup"] * 20,
        "feature_num": list(range(20)),
    })

    mock_read_preprocessing.return_value = df.copy()
    mock_read_training.return_value = df.copy()

    mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id=123)
    mock_mlflow.search_runs.return_value = MagicMock(
        iloc={0: {"metrics.f1_score": 0.9, "run_id": "123"}}
    )
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_client.get_model_version.return_value = {"current_stage": "Staging"}

    # ------------------------------------------------------------------
    # FIX APPLIED:
    # Training runs in a subprocess and cannot see mocks from this test.
    # The training script expects a real file at:
    #   ./artifacts/train_data_gold.csv
    # To prevent FileNotFoundError, the file is created explicitly here.
    # ------------------------------------------------------------------
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    train_gold = artifacts_dir / "train_data_gold.csv"
    df.to_csv(train_gold, index=False)

    p1 = subprocess.run(
        [sys.executable, "-m", "mlops_project.preprocessing"],
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0

    p2 = subprocess.run(
        [sys.executable, "-m", "mlops_project.training"],
        capture_output=True,
        text=True,
    )
    assert p2.returncode == 0

    p3 = subprocess.run(
        [sys.executable, "-m", "mlops_project.model_select"],
        capture_output=True,
        text=True,
    )
    assert p3.returncode == 0

    p4 = subprocess.run(
        [sys.executable, "-m", "mlops_project.deploy"],
        capture_output=True,
        text=True,
    )
    assert p4.returncode == 0

