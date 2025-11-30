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

    // Mount the entire project into /src
    container := client.Container().
        From("python:3.10").
        WithMountedDirectory("/src", client.Host().Directory(".")).
        WithWorkdir("/src")

    // Install ALL dependencies including DVC
    container = container.WithExec([]string{
        "pip", "install", "--upgrade", "pip",
    })

    container = container.WithExec([]string{
        "pip", "install", "-r", "requirements.txt",
    })

    container = container.WithExec([]string{
        "pip", "install", "dvc", "dvc-gdrive",
    })

    // Pull data from DVC
    fmt.Println("Pulling data from DVC...")
    container = container.WithExec([]string{
        "dvc", "pull",
    })

    // Run preprocessing
    fmt.Println("Running preprocessing...")
    container = container.WithExec([]string{
        "python", "mlops_project/preprocessing.py",
    })

    // Run training
    fmt.Println("Running training...")
    container = container.WithExec([]string{
        "python", "mlops_project/training.py",
    })

    // Run model selection
    fmt.Println("Running model selection...")
    container = container.WithExec([]string{
        "python", "mlops_project/model_select.py",
    })

    // Run deployment
    fmt.Println("Running deploy...")
    container = container.WithExec([]string{
        "python", "mlops_project/deploy.py",
    })

    // Export model artifacts
    fmt.Println("Exporting artifacts...")

    _, err = container.File("artifacts/lead_model_lr.pkl").Export(ctx, "artifacts/lead_model_lr.pkl")
    if err != nil {
        panic(err)
    }

    _, err = container.File("artifacts/lead_model_xgboost.json").Export(ctx, "artifacts/lead_model_xgboost.json")
    if err != nil {
        panic(err)
    }

    _, err = container.File("artifacts/model_results.json").Export(ctx, "artifacts/model_results.json")
    if err != nil {
        panic(err)
    }

    fmt.Println("Pipeline completed successfully!")
}
