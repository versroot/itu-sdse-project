import pandas as pd

from mlops_project.training import create_dummy_cols


def test_create_dummy_cols_drops_original_column():
    df = pd.DataFrame({"source": ["signup", "fb", "li"]})

    result = create_dummy_cols(df, "source")

    # original column must be gone
    assert "source" not in result.columns
    # at least one dummy column must be present
    assert any(col.startswith("source_") for col in result.columns)


def test_create_dummy_cols_uses_drop_first():
    df = pd.DataFrame({"customer_group": [1, 2, 3, 1]})

    result = create_dummy_cols(df, "customer_group")

    # there were 3 unique values -> with drop_first=True we expect 2 dummy columns
    dummy_cols = [c for c in result.columns if c.startswith("customer_group_")]
    assert len(dummy_cols) == 2
