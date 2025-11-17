import sys
import types

import pandas as pd

# Lightweight stand-ins for XGBoost & RandomizedSearchCV

class DummyModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        # trivial deterministic predictions
        return [0] * len(X)

    # training.py calls save_model on the XGBoost model
    def save_model(self, path):
        with open(path, "w") as f:
            f.write("dummy model")


class DummyRandomizedSearchCV:
    def __init__(self, estimator, param_distributions=None, **kwargs):
        # In real life this would tune hyperparameters;
        # here we just expose a simple best_estimator_.
        self.best_estimator_ = DummyModel()

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)


# Stub external libs BEFORE importing training

# Fake sklearn.model_selection.RandomizedSearchCV
sklearn_pkg = types.ModuleType("sklearn")
sklearn_pkg.__path__ = []  # mark as package
sys.modules.setdefault("sklearn", sklearn_pkg)

sklearn_ms = types.ModuleType("sklearn.model_selection")
sklearn_ms.RandomizedSearchCV = DummyRandomizedSearchCV
sys.modules["sklearn.model_selection"] = sklearn_ms

# Fake xgboost.XGBRFClassifier
xgb_mod = types.ModuleType("xgboost")
xgb_mod.XGBRFClassifier = DummyModel
sys.modules["xgboost"] = xgb_mod


# Now import training; it will see the dummy classes above
from mlops_project import training


def test_create_dummy_cols_basic():
    df = pd.DataFrame({"group": ["a", "b", "a"]})
    out = training.create_dummy_cols(df, "group")

    # original column should be dropped
    assert "group" not in out.columns
    # at least one dummy column should exist
    assert any(col.startswith("group_") for col in out.columns)
    # row count is preserved
    assert len(out) == len(df)
