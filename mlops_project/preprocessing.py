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

root_folder= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.makedirs("artifacts", exist_ok=True)

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

#fist let's try updating the dvc file and pulling the data, maybe it will work
subprocess.run(["dvc", "update", "data/raw/raw_data.csv.dvc"], check=True, cwd=root_folder)
subprocess.run(["dvc", "pull"], check=True, cwd=root_folder)

data = pd.read_csv("data/raw/raw_data.csv")

if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()

data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
with open("./artifacts/date_limits.json", "w") as f:
    json.dump(date_limits, f)

data = data.drop(
    [
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
    ],
    axis=1,
)

data = data.drop(
    ["domain", "country", "visited_learn_more_before_booking", "visited_faq"], axis=1
)


data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)
data["customer_code"].replace("", np.nan, inplace=True)

data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])

data = data[data.source == "signup"]
result = data.lead_indicator.value_counts(normalize=True)

vars = [
    "lead_id",
    "lead_indicator",
    "customer_group",
    "onboarding",
    "source",
    "customer_code",
]

for col in vars:
    data[col] = data[col].astype("object")

cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
cat_vars = data.loc[:, (data.dtypes == "object")]

cont_vars = cont_vars.apply(
    lambda x: x.clip(lower=(x.mean() - 2 * x.std()), upper=(x.mean() + 2 * x.std()))
)
outlier_summary = cont_vars.apply(describe_numeric_col).T
outlier_summary.to_csv("./artifacts/outlier_summary.csv")

cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")

cont_vars = cont_vars.apply(impute_missing_values)
cont_vars.apply(describe_numeric_col).T

cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"
cat_vars = cat_vars.apply(impute_missing_values)
cat_vars.apply(
    lambda x: pd.Series([x.count(), x.isnull().sum()], index=["Count", "Missing"])
).T


scaler_path = "./artifacts/scaler.pkl"

scaler = MinMaxScaler()
scaler.fit(cont_vars)

joblib.dump(value=scaler, filename=scaler_path)

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)


data_columns = list(data.columns)
with open("./artifacts/columns_drift.json", "w+") as f:
    json.dump(data_columns, f)

data.to_csv("./artifacts/training_data.csv", index=False)


data["bin_source"] = data["source"]
values_list = ["li", "organic", "signup", "fb"]
data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"
mapping = {"li": "socials", "fb": "socials", "organic": "group1", "signup": "group1"}

data["bin_source"] = data["source"].map(mapping)

data.to_csv("./artifacts/train_data_gold.csv", index=False)
