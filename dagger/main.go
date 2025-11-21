package main

import (
    "context"
    "github.com/dagger/dagger/core"
)

func main() {
    ctx := context.Background()

    client, err := dagger.Connect(ctx)
    if err != nil {
        panic(err)
    }
    defer client.Close()

    // 1. Build Python environment
    builder := client.Container().
        From("python:3.10").
        WithMountedDirectory("/src", client.Host().Directory(".")).
        WithWorkdir("/src")

    // 2. Install deps
    builder = builder.WithExec([]string{"pip", "install", "-r", "requirements.txt"})

    // 3. Run training
    builder = builder.WithExec([]string{"python", "mlops_project/training.py"})

    // 4. Save model artifact
    _, err = builder.File("models/model.joblib").Export(ctx, "model")
    if err != nil {
        panic(err)
    }
}
