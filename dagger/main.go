package main

import (
    "context"
    "fmt"
    "dagger.io/dagger"
)

func main() {
    ctx := context.Background()

    client, err := dagger.Connect(ctx)
    if err != nil {
        panic(err)
    }
    defer client.Close()

    container := client.Container().
        From("python:3.10").
        WithMountedDirectory("/src", client.Host().Directory(".")).
        WithWorkdir("/src")

    container = container.WithExec([]string{
        "pip", "install", "-r", "requirements.txt",
    })

    fmt.Println("Running preprocessing...")
    container = container.WithExec([]string{
        "python", "mlops_project/preprocessing.py",
    })

    fmt.Println("Running training...")
    container = container.WithExec([]string{
        "python", "mlops_project/training.py",
    })
    
    fmt.Println("Selecting best model...")
    container = container.WithExec([]string{
        "python", "mlops_project/model_select.py",
    })
    
    fmt.Println("Deploying model...")
    container = container.WithExec([]string{
        "python", "mlops_project/deploy.py",
    })

    // Example: export Logistic Regression
    _, err = container.File("artifacts/lead_model_lr.pkl").Export(ctx, "artifacts/lead_model_lr.pkl")
    if err != nil {
        panic(err)
    }

    // Example: export XGBoost model
    _, err = container.File("artifacts/lead_model_xgboost.json").Export(ctx, "artifacts/lead_model_xgboost.json")
    if err != nil {
        panic(err)
    }

    fmt.Println("Pipeline complete!")
}
