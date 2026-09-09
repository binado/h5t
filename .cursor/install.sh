#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the h5t project.
# Installs the uv package manager (if missing) and syncs the locked
# dependency set into .venv. Safe to run repeatedly.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# The uv installer drops binaries in ~/.local/bin; make sure this shell sees them.
export PATH="$HOME/.local/bin:$PATH"

# Resolve and install the pinned dependency graph (matches CI's `uv sync --locked`).
uv sync --locked
