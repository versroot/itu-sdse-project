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
need to be cleaned


```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mlops_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── source   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

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
