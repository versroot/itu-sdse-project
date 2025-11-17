import numpy as np
import pytest

from mlops_project import deploy, model_select, preprocessing, training

# =============================================================================
# Fixtures: shared test data for all stages
# =============================================================================

@pytest.fixture
def raw_data():
    """
    Synthetic raw dataset used as input for preprocessing.
    Adjust this to match your actual data format (numpy, pandas, etc.).
    """
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, size=50)
    return X, y


@pytest.fixture
def preprocessed_data(raw_data):
    """
    Data after preprocessing step.
    This assumes your preprocessing module exposes preprocess(X, y).
    """
    X, y = raw_data
    X_proc, y_proc = preprocessing.preprocess(X, y)
    return X_proc, y_proc


# =============================================================================
# Block 1. Preprocessing 
# =============================================================================

def test_preprocessing_shapes(raw_data, preprocessed_data):
    """Check that preprocessing does not change the number of samples."""
    X_raw, y_raw = raw_data
    X_proc, y_proc = preprocessed_data

    assert X_proc.shape[0] == X_raw.shape[0]
    assert y_proc.shape == y_raw.shape


def test_preprocessing_no_nans(preprocessed_data):
    """Check that no NaN values appear after preprocessing."""
    X_proc, y_proc = preprocessed_data

    assert not np.isnan(X_proc).any()
    assert not np.isnan(y_proc).any()


# =============================================================================
# Block 2. Training 
# =============================================================================

def test_training_returns_fitted_model(preprocessed_data):
    """
    Ensure that train_model returns a fitted model that has
    both fit() and predict() methods working correctly.
    """
    X, y = preprocessed_data

    model = training.train_model(X, y)

    # Model must support basic ML interface
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    # Predict should produce correct output shape
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


# =============================================================================
# Block 3. Model selection 
# =============================================================================

def test_model_select_returns_best_model(preprocessed_data):
    """
    Ensure the model selection module returns a best model
    along with evaluation metrics.
    """
    X, y = preprocessed_data

    best_model, metrics = model_select.select_best_model(X, y)

    assert best_model is not None
    assert hasattr(best_model, "predict")
    assert isinstance(metrics, dict)

    # Expect at least one performance metric
    assert any(m in metrics for m in ("score", "accuracy", "f1"))


# =============================================================================
# Block 4. Deployment
# =============================================================================

def test_deploy_saves_model(tmp_path, preprocessed_data):
    """
    Ensure the deploy module correctly saves the trained/best model.
    """
    X, y = preprocessed_data
    model, _ = model_select.select_best_model(X, y)

    model_path = tmp_path / "model.joblib"
    deploy.deploy_model(model, model_path)

    assert model_path.exists()


def test_loaded_model_can_predict(tmp_path, preprocessed_data):
    """
    Ensure that the deployed model can be loaded back and used for inference.
    """
    X, y = preprocessed_data
    model, _ = model_select.select_best_model(X, y)

    model_path = tmp_path / "model.joblib"
    deploy.deploy_model(model, model_path)

    loaded_model = deploy.load_model(model_path)

    y_pred = loaded_model.predict(X)
    assert y_pred.shape == y.shape
