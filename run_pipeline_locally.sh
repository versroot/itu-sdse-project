#!/bin/bash
# Script to run the complete ML pipeline locally
# This script automates all steps from the README.md local generation section

set -e  # Exit on any error

echo "=========================================="
echo "Starting ML Pipeline - Local Generation"
echo "=========================================="


echo ""
echo "Step 1: Installing dependencies..."
uv sync

echo ""
echo "Step 2: Updating data from DVC..."
dvc update data/raw/raw_data.csv.dvc

echo ""
echo "Step 3: Setting MLflow tracking URI..."
export MLFLOW_TRACKING_URI="file:./mlruns"

echo ""
echo "Disabling MLflow autolog for local runs..."
export DISABLE_MLFLOW_AUTOLOG="true"

echo ""
echo "Step 4: Running preprocessing..."
uv run python -m mlops_project.preprocessing

echo ""
echo "Step 5: Running training..."
uv run python -m mlops_project.training

echo ""
echo "Step 5: Running model selection..."
uv run python -m mlops_project.model_select

echo ""
echo "Step 5: Running deployment..."
uv run python -m mlops_project.deploy

echo ""
echo "Step 6: Packaging final model artifacts..."
mkdir -p model
cp artifacts/lead_model_lr.pkl model/model.pkl
cp artifacts/columns_list.json model/columns_list.json
cp artifacts/scaler.pkl model/scaler.pkl

echo ""
echo "Step 7: Verifying artifacts..."
ls -lh model/

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="