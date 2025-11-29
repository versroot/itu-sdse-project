package main

import (
    "context"
    "fmt"
    "log"

    "dagger.io/dagger"
)

func main() {
    ctx := context.Background()

    // Connect to Dagger engine
    client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
    if err != nil {
        panic(err)
    }
    defer client.Close()

    // Mount the *entire project root* so Python scripts & DVC can see everything
    project := client.Host().Directory(".")

    // Base Python image with all repo mounted
    container := client.Container().
        From("python:3.10").
        WithMountedDirectory("/src", project).
        WithWorkdir("/src")

    // Install dependencies (including DVC)
    container = container.WithExec([]string{
        "pip", "install", "-r", "requirements.txt",
    })

    // --- STEP 1: DVC pull data ---
    fmt.Println("Pulling data via DVC...")
    container = container.WithExec([]string{
        "dvc", "pull",
    })

    // --- STEP 2: Preprocessing ---
    fmt.Println("Running preprocessing...")
    container = container.WithExec([]string{
        "python", "mlops_project/preprocessing.py",
    })

    // --- STEP 3: Training ---
    fmt.Println("Running training...")
    container = container.WithExec([]string{
        "python", "mlops_project/training.py",
    })

    // --- STEP 4: Model selection ---
    fmt.Println("Running model selection...")
    container = container.WithExec([]string{
        "python", "mlops_project/model_select.py",
    })

    // --- STEP 5: Deployment ---
    fmt.Println("Running deployment...")
    container = container.WithExec([]string{
        "python", "mlops_project/deploy.py",
    })

    // --- STEP 6: Export model artifacts ---
    fmt.Println("Exporting model artifacts...")

    // Logistic Regression
    _, err = container.File("artifacts/lead_model_lr.pkl").Export(ctx,
        "artifacts/lead_model_lr.pkl")
    if err != nil {
        panic(err)
    }

    // XGBoost
    _, err = container.File("artifacts/lead_model_xgboost.json").Export(ctx,
        "artifacts/lead_model_xgboost.json")
    if err != nil {
        panic(err)
    }

    fmt.Println("Pipeline completed successfully ✓")
}
