#!/usr/bin/env bash
#
# install.sh
#
# Creates the conda environment for this repository.
# Run from the repository root after cloning:
#
#   bash install.sh
#
# Creates or updates the conda environment as ./taxane_conda_env from
# environment.yaml, or from the file given as the first argument.
#
# Requirements:
#   - conda (Miniconda or Anaconda) must be installed and on PATH
#   - Run from the project root directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Environment location: <repository>/taxane_conda_env unless TAXANE_ENV_PREFIX
# names another directory.
ENV_PREFIX="${TAXANE_ENV_PREFIX:-$SCRIPT_DIR/taxane_conda_env}"

ENV_FILE="${1:-$SCRIPT_DIR/environment.yaml}"
case "$ENV_FILE" in
    /*) : ;;                          # absolute path, use as-is
    *)  ENV_FILE="$SCRIPT_DIR/$ENV_FILE" ;;
esac

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "[install] $*"; }
error() { echo "[install] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Locate conda
# ---------------------------------------------------------------------------

if ! command -v conda &> /dev/null; then
    error "conda not found on PATH. Please install Miniconda or Anaconda first."
fi

# Source conda's shell integration so that 'conda activate' works in this
# script (it requires shell functions, not just the conda binary).
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

info "Using conda at: $CONDA_BASE"
info "Environment prefix: $ENV_PREFIX"
info "Environment file:   $ENV_FILE"

# numpy only as a wheel (old numpy sdists do not build on Python 3.12), and
# wheels preferred for everything else. PIP_CACHE_DIR can point at shared storage.
export PIP_ONLY_BINARY="numpy"
export PIP_PREFER_BINARY="1"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRIPT_DIR/.pip_cache}"
info "pip cache: $PIP_CACHE_DIR"

# ---------------------------------------------------------------------------
# 2. Create or update the conda environment
# ---------------------------------------------------------------------------

if [ -d "$ENV_PREFIX" ]; then
    info "Environment already exists — updating..."
    conda env update --prefix "$ENV_PREFIX" --file "$ENV_FILE" --prune
else
    info "Creating environment..."
    conda env create --prefix "$ENV_PREFIX" --file "$ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

info ""
info "Setup complete. To get started:"
info "  conda activate $ENV_PREFIX"