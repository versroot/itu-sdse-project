"""
Unit tests for preprocessing functions using small synthetic datasets.

These tests validate individual preprocessing functions (impute_missing_values,
describe_numeric_col) in isolation using minimal artificial data. They do not
test the full preprocessing pipeline or use real production data.

For end-to-end testing with real data, see test_preprocessing_integration.py
"""

import logging

import numpy as np
import pandas as pd
import pytest

from mlops_project.preprocessing import describe_numeric_col, impute_missing_values

# Shared logger so every test emits consistent INFO lines.
logger = logging.getLogger("tests.preprocessing")


def _log_imputation_stats(label, series):
    """Emit key stats for a pandas Series before/after imputation."""
    logger.info(
        "%s -> n=%d missing=%d mean=%.3f std=%.3f values=%s",
        label,
        len(series),
        series.isna().sum(),
        series.mean(skipna=True),
        series.std(skipna=True),
        series.tolist(),
    )


def test_impute_missing_values_with_mean(caplog):
    """Ensure numeric imputation by mean removes NaNs and centers correctly."""
    caplog.set_level(logging.INFO, logger="tests.preprocessing")

    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    _log_imputation_stats("input", series)

    result = impute_missing_values(series, method="mean")
    _log_imputation_stats("output", result)

    assert result.isna().sum() == 0
    assert result.mean() == pytest.approx(3.0, abs=0.1)


def test_impute_missing_values_with_median(caplog):
    """Verify median strategy handles NaNs and preserves distribution."""
    caplog.set_level(logging.INFO, logger="tests.preprocessing")

    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    _log_imputation_stats("input", series)

    result = impute_missing_values(series, method="median")
    _log_imputation_stats("output", result)

    assert result.isna().sum() == 0
    assert result.median() == pytest.approx(3.0, abs=0.1)


def test_impute_missing_values_categorical(caplog):
    """Mode-based fill for categorical data should remove missing items."""
    caplog.set_level(logging.INFO, logger="tests.preprocessing")

    series = pd.Series(["A", "B", None, "A", "B"])
    logger.info("categorical input -> %s", series.tolist())

    result = impute_missing_values(series, method="mean")
    logger.info("categorical output -> %s", result.tolist())

    assert result.isna().sum() == 0
    assert "A" in result.values or "B" in result.values


def test_describe_numeric_col(caplog):
    """`describe_numeric_col` must report basic stats accurately."""
    caplog.set_level(logging.INFO, logger="tests.preprocessing")

    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    logger.info("describe input -> %s", series.tolist())

    result = describe_numeric_col(series)
    logger.info("describe output -> %s", result.to_dict())

    assert "Count" in result.index
    assert "Mean" in result.index
    assert "Min" in result.index
    assert "Max" in result.index
    assert result["Count"] == 5
    assert result["Mean"] == pytest.approx(3.0)
    assert result["Min"] == 1.0
    assert result["Max"] == 5.0
