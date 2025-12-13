# The project is part of Data Science in Production: MLOps and Software Engineering [BSDSPMS1KU] at IT University of Copenhagen

<a href="https://github.com/lasselundstenjensen/itu-sdse-project">
    The original project description />
</a>

# Team: Git Gut

# Team members:

Alexandru Jizdan,
Kateryna Tkachuk,
Mykyta Taranov,
Vivien Ivett Pribula

## Project structure

├── .github/
│ └── workflows/  
│ ├── ci.yml <- CI pipeline: build Docker image, lint with ruff, run tests
│ └── train-model.yml <- Training pipeline
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
│ ├── 0.01_Data_exploration.ipynb <- Exploratory data analysis
│ └── original_files/ <- Legacy notebook files (archived)
│
├── tests/ <- Test suite (pytest)
│ ├── test_pipeline.py <-
│ ├── test_preprocessing_unit.py <-
│ ├── test_training.py <-
│ ├── unit/ <-
│ │ ├── test_deploy.py
│ │ ├── test_modelselect.py
│ │ ├── test_preprocessingmock.py
│ │ └── test_trainingruns.py
│ └── integration/ <-
│ └── test_pipelinefast.py
│
├── Dockerfile <- Production container image
├── pyproject.toml <- Project metadata, dependencies, tool configs (ruff, pytest)
├── uv.lock <- Locked dependency versions
├── start_container.sh <- Helper script to launch interactive Docker container
└── README.md <- Project structure and instruction how to run the code

# How to run the code and generate the model artifact

TBD, need workflow_dispatch

- automatically, with Pull request
- in github actions [for contributors]
- locally

# Refferences:

This repository is a fork from [Lasse Lund Sten Jensen's original project repo](https://github.com/lasselundstenjensen/itu-sdse-project).

---

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Note: We wanted to implement DVC with GoogleDrive connection, but dues to an authentication issue we werew not able to use this approach - it only worked for the repo owner. We decided to fall back to DVC hanling from GitHub.
