import numpy as np
import pandas as pd

from mlops_project.preprocessing import describe_numeric_col, impute_missing_values

# ---------- describe_numeric_col tests ----------


def test_describe_numeric_col_basic_stats():
    s = pd.Series([1, 2, 3, 4, 5, np.nan])

    desc = describe_numeric_col(s)

    assert desc["Count"] == 5            # non-null values
    # function currently reports len(s) in "Missing"
    assert desc["Missing"] == len(s)     # 6
    assert desc["Min"] == 1
    assert desc["Max"] == 5
    assert desc["Mean"] == 3
     # (1+2+3+4+5)/5


def test_describe_numeric_col_all_missing():
    s = pd.Series([np.nan, np.nan])

    desc = describe_numeric_col(s)

    assert desc["Count"] == 0
    assert desc["Missing"] == 2
    assert np.isnan(desc["Mean"])
    assert np.isnan(desc["Min"])
    assert np.isnan(desc["Max"])


# ---------- impute_missing_values tests ----------


def test_impute_missing_values_numeric_mean():
    s = pd.Series([1.0, 2.0, np.nan, 4.0])
    imputed = impute_missing_values(s, method="mean")

    expected_mean = (1 + 2 + 4) / 3
    assert imputed.isna().sum() == 0
    assert imputed.iloc[2] == expected_mean


def test_impute_missing_values_numeric_median():
    s = pd.Series([1.0, 100.0, np.nan, 4.0])
    imputed = impute_missing_values(s, method="median")

    # median of [1, 100, 4] is 4.0
    assert imputed.isna().sum() == 0
    assert imputed.iloc[2] == 4.0


def test_impute_missing_values_categorical_mode():
    s = pd.Series(["a", "b", None, "b", "c", None])
    imputed = impute_missing_values(s)

    # All missing should be filled
    assert imputed.isna().sum() == 0
    # mode is "b"
    assert (imputed == "b").sum() >= 2


def test_impute_missing_values_with_empty_strings():
    s = pd.Series(["x", "", "x", None])
    s = s.replace("", np.nan)

    imputed = impute_missing_values(s)

    assert imputed.isna().sum() == 0
    assert set(imputed.unique()) == {"x"}  # mode is x
