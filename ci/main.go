// dagger pipeline
// preprocessing -> training -> model_select -> deploy -> package model -> export model
// these steps mirror the mlops_project module structure in mlops_project/modelling/
// so essentially calling each module's main entry point in sequence
// finally exporting the model artifacts to ./model in repo root 

package main

import (
	"context"
	"log"
	"os"

	"dagger.io/dagger"
)

func main() {

	// initialization of Dagger client
	ctx := context.Background()

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr)) 
	
	// error handling
	if err != nil {
		log.Fatalf("failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// repo root as build context
	src := client.Host().Directory("..")

	// build Dockerfile in repo root
	cont := src.DockerBuild().WithWorkdir("/app")

	// sync deps (cheap if already done in Dockerfile)
	cont = cont.WithExec([]string{"bash", "-lc", "uv sync"})
    
    // !actual pipeline steps

	// 1) preprocessing
	cont = mustRunStep(ctx, cont, "preprocessing",
		"uv run python -m mlops_project.preprocessing")

	// 2) training
	cont = mustRunStep(ctx, cont, "training",
		"uv run python -m mlops_project.training")

	// 3) model_select
	cont = mustRunStep(ctx, cont, "model_select",
		"uv run python -m mlops_project.model_select")

	// 4) deploy
	cont = mustRunStep(ctx, cont, "deploy",
		"uv run python -m mlops_project.deploy")

	// 5) package model into /app/model
	cont = mustRunStep(ctx, cont, "package_model",
		"mkdir -p model && "+
			"cp artifacts/lead_model_lr.pkl model/model.pkl && "+ 
			"cp artifacts/columns_list.json model/columns_list.json && "+
			"cp artifacts/scaler.pkl model/scaler.pkl",
	)

	// 6) export /app/model -> ../model (repo root)
	_, err = cont.Directory("/app/model").Export(ctx, "../model")
	if err != nil {
		log.Fatalf("failed to export model artifact: %v", err)
	}

	log.Println("Dagger pipeline finished successfully. Model exported to ./model in repo root.")
}

func mustRunStep(ctx context.Context, c *dagger.Container, name, cmd string) *dagger.Container {
	log.Printf("\n=== Running step: %s ===\n", name)
	next := c.WithExec([]string{"bash", "-lc", cmd})
	_, err := next.Sync(ctx)
	if err != nil {
		log.Fatalf("step %s failed: %v", name, err)
	}
	return next
}
