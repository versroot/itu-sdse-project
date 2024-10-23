# ITU SDSE'24 - Project

## Project architecture

![Project architecture](./docs/project-architecture.png)

## Inputs

- Python monolith
- (Sterile) data
- GitHub action (to test model inference)

## Outputs

- Your GitHub repository (including all history)
  - A README.md file that describes the project
  - GitHub automation workflow
  - Dagger workflow (in Go)
- Model artefact produced by GitHub workflow

## How will we assess

#### Versioning

- Use of Git (semantic commit messages, branches, branch longevity, commit frequency/size)
- Management of data
- Use of pull requests (OPTIONAL)
- Use of code reviews (OPTIONAL)

#### Programming

- Decomposition of Python notebook
- Presence of tests (OPTIONAL)

#### Workflow automation

- Structure of Dagger workflow
- Orchestration of Dagger workflow through GitHub workflow

#### Documentation

- Description of project structure
- How to run the code and generate the model artefact
