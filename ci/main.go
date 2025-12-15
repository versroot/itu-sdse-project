// dagger pipeline
// preprocessing -> training -> model_select -> deploy -> package model -> export model
// these steps mirror the mlops_project module structure in mlops_project/modeling/
// so essentially calling each module's main entry point in sequence
// finally exporting the model artifacts to ./model in repo root

package main // declares that this is Go executable (not a library)

import (
	"context"       // to create root context for Dagger
	"log"           // for logging
	"os"            // for error output
	"path/filepath" // for filepath.Join

	"dagger.io/dagger" // Dagger SDK for container
)

func main() {

	// initialization of Dagger client
	ctx := context.Background() // root context

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))

	// error handling
	if err != nil {
		log.Fatalf("failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// Get absolute path to repo root (where Dockerfile is located)
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("failed to get working directory: %v", err)
	}

	var repoRoot string
	// Check if we're in ci/ directory or repo root
	if filepath.Base(wd) == "ci" {
		// We're in ci/, go up one level
		repoRoot = filepath.Dir(wd)
	} else {
		// We're already at repo root
		repoRoot = wd
	}

	// check Dockerfile exists at repo root
	dockerfilePath := filepath.Join(repoRoot, "Dockerfile")
	if _, err := os.Stat(dockerfilePath); err != nil {
		log.Fatalf("Dockerfile not found at %s: %v", dockerfilePath, err)
	}

	// Convert to absolute path (required by Dagger)
	repoRootAbs, err := filepath.Abs(repoRoot)
	if err != nil {
		log.Fatalf("failed to get absolute path: %v", err)
	}

	// repo root as build context (directory) - use absolute path for local pipeline
	src := client.Host().Directory(repoRootAbs, dagger.HostDirectoryOpts{
		Exclude: []string{".venv", "venv", "__pycache__", "mlruns", "artifacts"},
	})
	// create container from Dockerfile in repo root, set workdir to /app
	cont := src.DockerBuild().WithWorkdir("/app")

	// sync dependencies with UV
	cont = cont.WithExec([]string{"bash", "-lc", "uv sync"}) // -lc load shell and run command; uv sync installs dependencies from pyproject.toml

	// actual pipeline steps
	// helper function mustRunStep runs each step and handles errors
	// implementation of mustRunStep is below main()

	// 1) preprocessing
	cont = mustRunStep(ctx, cont, "preprocessing",
		"uv run python -m mlops_project.preprocessing") // uv run automatically: activates venv, runs python with all packages

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
	modelExportPath := filepath.Join(repoRootAbs, "model")
	_, err = cont.Directory("/app/model").Export(ctx, modelExportPath)
	if err != nil {
		log.Fatalf("failed to export model artifact: %v", err)
	}

	log.Println("Dagger pipeline finished successfully. Model exported to ./model in repo root.")
}

// helper function to run a pipeline step and handle errors
func mustRunStep(ctx context.Context, c *dagger.Container, name, cmd string) *dagger.Container {
	log.Printf("\n=== Running step: %s ===\n", name)
	next := c.WithExec([]string{"bash", "-lc", cmd})
	_, err := next.Sync(ctx)
	if err != nil {
		log.Fatalf("step %s failed: %v", name, err)
	}
	return next
}
