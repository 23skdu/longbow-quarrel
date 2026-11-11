#!/usr/bin/env bash
# ==============================================================================
# Longbow-Quarrel pre-commit verification hook (scripts/verify.sh)
# ==============================================================================
# Installed by `make makehooks` into .git/hooks/pre-commit. Runs the fast
# quality gates that should block a commit: formatting, vet, lint, security,
# and the short unit test suite.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export GOMEMLIMIT="${QUARREL_TEST_MEMLIMIT:-2GiB}"
export GOMAXPROCS="${QUARREL_TEST_PARALLEL:-${GOMAXPROCS:-4}}"

GO="${GO:-go}"
if [ -x /home/linuxbrew/.linuxbrew/bin/go ]; then GO=/home/linuxbrew/.linuxbrew/bin/go; fi

fail() { echo "verify: FAIL: $*" >&2; exit 1; }

echo "verify: checking formatting..."
UNFORMATTED="$(gofmt -l .)"
if [ -n "${UNFORMATTED}" ]; then
    echo "${UNFORMATTED}"
    fail "unformatted Go files (run 'make fmt')"
fi

echo "verify: running go vet..."
$GO vet ./... || fail "go vet"

if command -v golangci-lint >/dev/null 2>&1; then
    echo "verify: running golangci-lint..."
    golangci-lint run ./... || fail "golangci-lint"
fi

if command -v gosec >/dev/null 2>&1; then
    echo "verify: running gosec..."
    gosec -quiet -exclude=G115 ./... || fail "gosec"
fi

echo "verify: running short unit tests..."
$GO test -short ./... || fail "unit tests"

echo "verify: running fuzz smoke (seed corpus)..."
$GO test -short -run='^Fuzz' ./... || fail "fuzz smoke"

echo "verify: OK"
exit 0