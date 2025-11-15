# Use a slim Python image for speed but still Debian-based for apt packages
FROM python:3.10-slim

# System deps:
# - build-essential & gcc: build native wheels if needed
# - curl: install uv
# - git: some installers / deps expect it
# - libgomp1: OpenMP runtime for xgboost on Linux
# - tini: proper PID 1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc git curl ca-certificates libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Workdir
WORKDIR /app

# Copy everything at once
COPY . .

# Install project dependencies using uv (incl. dev deps for lint/tests)
RUN uv sync --dev

# Make uv-managed .venv visible as default Python
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:${PATH}"

# Good container hygiene (tini as entrypoint)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
