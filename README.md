# The project is part of Data Science in Production: MLOps and Software Engineering [BSDSPMS1KU] at IT University of Copenhagen

<a href="https://github.com/lasselundstenjensen/itu-sdse-project">
    The original project description
</a>

# Team: Git Gut

# Team members:

Alexandru Jizdan, <br>
Kateryna Tkachuk, <br>
Mykyta Taranov, <br>
Vivien Ivett Pribula

## Project structure

```
├── .github/
│ └── workflows/
│ ├── ci.yml <- CI pipeline: build Docker image, lint with ruff, run tests
│ └── train-model.yml <- Training pipeline: runs Dagger workflow, uploads model artifacts
│
├── bin/
│ └── dagger <- Dagger CLI binary
│
├── ci/
│ ├── main.go <- Dagger pipeline
│ ├── go.mod <- Go module dependencies for Dagger
│ └── go.sum <- Dependency checksums
│
├── data/
│ └── raw/
│ │ └── raw_data.csv.dvc <- DVC metadata for data versioning
│
├── mlops_project/ <- Core ML pipeline modules
│ ├── **init**.py
│ ├── preprocessing.py <- Data loading, cleaning, feature engineering, scaling
│ ├── training.py <- Model training loop (LogReg, RF, XGBoost) with MLflow tracking
│ ├── model_select.py <- Selects best model from experiments, registers in MLflow
│ └── deploy.py <- Transitions model to Staging in MLflow registry
│
├── model/ <- Trained model artifacts (exported by pipeline)
│ ├── model.pkl <- Serialized best model
│ ├── scaler.pkl <- Fitted MinMaxScaler
│ └── columns_list.json <- Feature names for inference
│
├── notebooks/
│ └──0.01_Data_exploration.ipynb <- Exploratory data analysis
│ 
│
├── tests/ <- Test suite (pytest)
│ ├── unit/ <- Unit tests
│ │ ├── test_deploy.py <- Deployment function tests
│ │ ├── test_deployimport.py <- Import smoke test
│ │ ├── test_modelselect.py <- Model selection function tests
│ │ ├── test_modelselectimport.py <- Import smoke test
│ │ ├── test_preprocessing_unit.py <- Preprocessing function tests
│ │ ├── test_preprocessingimport.py <- Import smoke test
│ │ ├── test_training.py <- Training utility function tests
│ │ ├── test_trainingimport.py <- Import smoke test
│ │ └── test_trainingruns.py <- Module execution smoke test
│ └── integration/ <- Integration tests
│   ├── test_pipeline.py <- End-to-end pipeline test
│   ├── test_pipelinefast.py <- Fast pipeline test with mocks
│   └── test_preprocessingmock.py <- Preprocessing script execution test
│
├── Dockerfile <- Production container image
├── pyproject.toml <- Project metadata, dependencies, tool configs (ruff, pytest)
├── uv.lock <- Locked dependency versions
├── start_container.sh <- Helper script to launch interactive Docker container
└── README.md <- Project structure and instruction how to run the code
```

# How to run the code and generate the model artifact

## GitHub Actions Workflow

The training workflow runs automatically after the CI pipeline completes successfully, or can be triggered manually.

### Automatic Trigger
The training workflow (`train-model.yml`) is automatically triggered when the CI workflow completes successfully.

## Local generation
### Prerequisites

- Python 3.10
- `uv` package manager (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Steps
1. Install dependencies from `pyproject.toml`
uv sync

2. Update data from DVC
dvc update daata/raw/raw_data.csv.dvc

3. Set MLflow tracking
export MLFLOW_TRACKING_URI="file:./mlruns"

4. Generate artifacts with preprocessing
uv run python -m mlops_project.preprocessing

5. Run training, model_selection and deployment
uv run python -m mlops_project.training
uv run python -m mlops_project.model_select
uv run python -m mlops_project.deploy

6. Package final model artifacts
mkdir -p model
cp artifacts/lead_model_lr.pkl model/model.pkl
cp artifacts/columns_list.json model/columns_list.json
cp artifacts/scaler.pkl model/scaler.pkl

7. Verfy artifacts
ls -lh model/

# Refferences:

This repository is a fork from [Lasse Lund Sten Jensen's original project repo](https://github.com/lasselundstenjensen/itu-sdse-project).

---

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Note: We wanted to implement DVC with GoogleDrive connection, but dues to an authentication issue we werew not able to use this approach - it only worked for the repo owner. We decided to fall back to DVC hanling from GitHub.

```

```
