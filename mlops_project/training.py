import datetime
import json
import os

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRFClassifier

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(experiment_name)


def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


data = pd.read_csv(data_gold_path)

data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)

for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")

y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)
y_train

model = XGBRFClassifier(random_state=42)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"],
}

model_grid = RandomizedSearchCV(
    model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10
)

model_grid.fit(X_train, y_train)

best_model_xgboost_params = model_grid.best_params_

y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred_test)

conf_matrix = confusion_matrix(y_train, y_pred_train)

xgboost_model = model_grid.best_estimator_
xgboost_model_path = "./artifacts/lead_model_xgboost.json"
xgboost_model.save_model(xgboost_model_path)

model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}


class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()
    lr_model_path = "./artifacts/lead_model_lr.pkl"

    params = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01],
    }
    model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")

    joblib.dump(value=best_model, filename=lr_model_path)

    mlflow.pyfunc.log_model("model", python_model=lr_wrapper(model))


model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

best_model_lr_params = model_grid.best_params_

conf_matrix = confusion_matrix(y_test, y_pred_test)

conf_matrix = confusion_matrix(y_train, y_pred_train)

model_results[lr_model_path] = model_classification_report

column_list_path = "./artifacts/columns_list.json"
with open(column_list_path, "w+") as columns_file:
    columns = {"column_names": list(X_train.columns)}
    json.dump(columns, columns_file)

model_results_path = "./artifacts/model_results.json"
with open(model_results_path, "w+") as results_file:
    json.dump(model_results, results_file)
