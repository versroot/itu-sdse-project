import subprocess
import sys
from unittest.mock import patch


@patch("mlops_project.preprocessing.os.system")
@patch("mlops_project.preprocessing.pd.read_csv")
def test_preprocessing_runs_as_script(mock_read_csv, mock_system, tmp_path, monkeypatch):
    """
    Execute preprocessing.py as a script (black-box test) to ensure it does not crash.
    CSV read is mocked so no actual data/raw/raw_data.csv is required.
    """
    import pandas as pd
    fake_df = pd.DataFrame({
        "date_part": ["2024-01-05", "2024-01-07"],
        "source": ["signup", "signup"],
        "lead_indicator": [1, 1],
        "lead_id": ["a", "b"],
        "customer_group": ["x", "y"],
        "onboarding": ["a", "b"],
        "customer_code": ["c", "d"],
        "some_numeric": [1.0, 2.0],
    })
    mock_read_csv.return_value = fake_df

    monkeypatch.chdir(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "mlops_project.preprocessing"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
