import datetime
import json
import os
import subprocess
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

max_date = "2024-01-31"
min_date = "2024-01-01"

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
artifacts_dir = os.path.join(root_folder, "artifacts")
raw_data_path = os.path.join(root_folder, "data", "raw", "raw_data.csv")

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: "%.3f" % x)


def describe_numeric_col(x):
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


def impute_missing_values(x, method="mean"):
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x


def load_and_filter_data(min_d, max_d):
    data = pd.read_csv(raw_data_path)
    max_dt = pd.to_datetime(datetime.datetime.now().date()).date() if not max_d else pd.to_datetime(max_d).date()
    min_dt = pd.to_datetime(min_d).date()
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_dt) & (data["date_part"] <= max_dt)]
    return data, min_dt, max_dt


def drop_unneeded_columns(df):
    cols = [
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
        "domain",
        "country",
        "visited_learn_more_before_booking",
        "visited_faq",
    ]
    return df.drop(cols, axis=1)


def preprocess():
    os.makedirs(artifacts_dir, exist_ok=True)

    data, min_dt, max_dt = load_and_filter_data(min_date, max_date)

    with open(os.path.join(artifacts_dir, "date_limits.json"), "w") as f:
        json.dump({"min_date": str(min_dt), "max_date": str(max_dt)}, f)

    data = drop_unneeded_columns(data)

    for col in ["lead_indicator", "lead_id", "customer_code"]:
        data[col] = data[col].replace("", np.nan)

    data = data.dropna(axis=0, subset=["lead_indicator", "lead_id"])
    data = data[data.source == "signup"]

    cat_cols = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in cat_cols:
        data[col] = data[col].astype("object")

    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    cont_vars = cont_vars.apply(lambda x: x.clip(lower=(x.mean() - 2 * x.std()), upper=(x.mean() + 2 * x.std())))
    cont_vars.apply(describe_numeric_col).T.to_csv(os.path.join(artifacts_dir, "outlier_summary.csv"))

    cat_vars.mode(numeric_only=False, dropna=True).to_csv(os.path.join(artifacts_dir, "cat_missing_impute.csv"))

    cont_vars = cont_vars.apply(impute_missing_values)
    cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"
    cat_vars = cat_vars.apply(impute_missing_values)

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(value=scaler, filename=os.path.join(artifacts_dir, "scaler.pkl"))

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns).reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)

    with open(os.path.join(artifacts_dir, "columns_drift.json"), "w+") as f:
        json.dump(list(data.columns), f)

    data.to_csv(os.path.join(artifacts_dir, "training_data.csv"), index=False)

    data["bin_source"] = data["source"]
    values_list = ["li", "organic", "signup", "fb"]
    data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"
    mapping = {"li": "socials", "fb": "socials", "organic": "group1", "signup": "group1"}
    data["bin_source"] = data["source"].map(mapping)
    data.to_csv(os.path.join(artifacts_dir, "train_data_gold.csv"), index=False)


def main():
    # fist let's try updating the dvc file and pulling the data, maybe it will work
    subprocess.run(["dvc", "update", "data/raw/raw_data.csv.dvc"], check=True, cwd=root_folder)
    subprocess.run(["dvc", "pull"], check=True, cwd=root_folder)
    preprocess()


if __name__ == "__main__":
    main()
