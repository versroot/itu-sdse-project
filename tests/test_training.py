import pandas as pd
import pytest

from mlops_project.training import create_dummy_cols


def test_create_dummy_cols():
    """Test creation of dummy variables from categorical column."""
    df = pd.DataFrame({"category": ["A", "B", "A", "C"], "other_col": [1, 2, 3, 4]})

    result = create_dummy_cols(df, "category")

    # Original column should be removed
    assert "category" not in result.columns

    # Dummy columns should be created
    assert "category_B" in result.columns or "category_C" in result.columns

    # Other columns should remain
    assert "other_col" in result.columns

    # Should have correct number of rows
    assert len(result) == 4
