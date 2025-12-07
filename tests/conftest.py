
"""
Pytest configuration for CI.

The CI image has a broken scikit-learn where
`sklearn.model_selection.check_cv` is missing.
This file adds a small compatibility function before
`sklearn.linear_model` is imported anywhere.
"""

import importlib
import matplotlib
matplotlib.use('Agg')

def _ensure_sklearn_check_cv():
    try:
        ms = importlib.import_module("sklearn.model_selection")
    except Exception:
        # If sklearn is completely unavailable, let it fail normally.
        # Locally you should have a proper sklearn installation.
        return

    # If check_cv is already there (local dev), do nothing.
    if hasattr(ms, "check_cv"):
        return

    # Minimal compatible implementation – just returns cv unchanged.
    def check_cv(cv=5, y=None, classifier=False):
        return cv

    ms.check_cv = check_cv  # type: ignore[attr-defined]

    # Now import sklearn.linear_model once so that it sees the patched symbol.
    try:
        importlib.import_module("sklearn.linear_model")
    except Exception:
        # If this still fails, tests will fail in the normal way.
        pass


_ensure_sklearn_check_cv()
