#!/usr/bin/env bash
# Verify that fuzz-critical packages contain at least one Fuzz target.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PKGS=(
  "internal/simd"
  "internal/vector"
  "internal/engine"
  "internal/api"
  "internal/gguf"
  "internal/sampler"
  "internal/metrics"
)

missing=0
for pkg in "${PKGS[@]}"; do
    if grep -R "func Fuzz" "${pkg}" >/dev/null 2>&1; then
        echo "fuzz-coverage: OK   ${pkg}"
    else
        echo "fuzz-coverage: MISS ${pkg}"
        missing=1
    fi
done

if [ "${missing}" -ne 0 ]; then
    echo "fuzz-coverage: one or more packages lack Fuzz targets"
    exit 1
fi
echo "fuzz-coverage: all fuzz-critical packages covered"