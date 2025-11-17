import numpy as np
import pytest

# =============================================================================
# Fixtures: synthetic data
# =============================================================================

@pytest.fixture
def raw_data():
    """
    Generate a tiny synthetic dataset.
    """
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)
    return X, y


@pytest.fixture
def preprocessed_data(raw_data):
    """
    Pretend preprocessing happened.
    For this lightweight test we just reuse the raw data.
    """
    X, y = raw_data
    return X, y


# =============================================================================
# Dummy model (no heavy ML)
# =============================================================================

class DummyModel:
    """
    A very simple model with fit() and predict() to mimic an ML model.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, X, y):
        # In real code you'd train here. We just mark it as fitted.
        assert X.shape[0] == len(y)
        self._fitted = True
        return self

    def predict(self, X):
        # In real code you'd use learned parameters.
        # Here we just output zeros, but require that the model was "fitted".
        assert self._fitted, "Model must be fitted before predicting"
        return np.zeros(X.shape[0], dtype=int)


# =============================================================================
# Tests
# =============================================================================

def test_preprocessing_shapes(preprocessed_data):
    """
    Check that 'preprocessed' data has consistent shapes.
    """
    X, y = preprocessed_data

    assert X.shape[0] == len(y)
    assert X.ndim == 2
    assert y.ndim == 1


def test_training_returns_model(preprocessed_data):
    """
    Simulate training: DummyModel.fit returns a model that can predict.
    """
    X, y = preprocessed_data

    model = DummyModel().fit(X, y)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    # Predictions are 0/1 labels in this toy example
    assert set(np.unique(preds)).issubset({0, 1})


def test_end_to_end_pipeline(preprocessed_data, tmp_path):
    """
    Very lightweight 'pipeline' test:

    1. Train DummyModel on preprocessed data
    2. 'Deploy' it by saving a dummy file
    3. 'Load' a new DummyModel and run predictions
    """
    X, y = preprocessed_data

    # 1. "Train"
    model = DummyModel().fit(X, y)

    # 2. "Deploy" (just write a small file)
    model_path = tmp_path / "model.txt"
    model_path.write_text("dummy serialized model")
    assert model_path.exists()

    # 3. "Load" model (for this simple test, we just create a new DummyModel)
    loaded_model = DummyModel().fit(X, y)
    preds = loaded_model.predict(X)

    assert preds.shape[0] == X.shape[0]
