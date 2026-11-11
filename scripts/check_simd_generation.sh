#!/usr/bin/env bash
# Lightweight SIMD reproducibility check: build and test the simd package.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
CGO_ENABLED=1 go test ./internal/simd/...