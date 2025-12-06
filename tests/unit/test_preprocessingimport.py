from unittest.mock import patch, MagicMock
import builtins


@patch("mlops_project.preprocessing.os.system")
@patch("mlops_project.preprocessing.joblib.dump")
@patch("mlops_project.preprocessing.MinMaxScaler")
@patch("mlops_project.preprocessing.pd")
@patch("mlops_project.preprocessing.open", create=True)
def test_preprocessing_import_safe(
    mock_open,
    mock_pd,
    mock_scaler,
    mock_joblib,
    mock_system,
):
    """
    Ensure that importing preprocessing.py does NOT crash.
    All external calls (file I/O, DVC pull, CSV reads, scaler, etc) are mocked.
    """

    fake_df = MagicMock()
    fake_df.__getitem__.return_value = MagicMock()
    fake_df.drop.return_value = fake_df
    fake_df.dropna.return_value = fake_df
    fake_df.__getattr__.return_value = fake_df
    fake_df.dtypes = MagicMock()
    fake_df.loc = MagicMock(return_value=fake_df)
    fake_df.apply.return_value = fake_df
    fake_df.value_counts.return_value = MagicMock()
    fake_df.to_csv.return_value = None

    mock_pd.read_csv.return_value = fake_df
    mock_pd.to_datetime.return_value = fake_df

    fake_scaler = MagicMock()
    fake_scaler.fit.return_value = None
    fake_scaler.transform.return_value = fake_df
    mock_scaler.return_value = fake_scaler

    mock_open.return_value.__enter__.return_value.write.return_value = None

    import mlops_project.preprocessing

    mock_pd.read_csv.assert_called_once()
    mock_scaler.assert_called_once()
    mock_joblib.assert_called_once()
