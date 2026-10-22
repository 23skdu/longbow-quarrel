#!/usr/bin/env bash
# ==============================================================================
# Longbow-Quarrel Consolidated Test Suite Runner
# ==============================================================================
# Executes CPU tests, CUDA tests (if GPU/nvcc present), SIMD/VNNI tests,
# VLM multimodal tests, Grammar/CFG tests, Race detector, and Coverage validation.
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Memory limit for test processes to prevent OOM
# Default: 2GB. Override with QUARREL_TEST_MEMLIMIT env var.
export GOMEMLIMIT="${QUARREL_TEST_MEMLIMIT:-2GiB}"

# Bound CPU concurrency to prevent pinning all host cores and exhausting RAM across parallel package tests.
# Default: 4. Override with QUARREL_TEST_PARALLEL or GOMAXPROCS.
export GOMAXPROCS="${QUARREL_TEST_PARALLEL:-${GOMAXPROCS:-4}}"

# Styling
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PASSED_STEPS=0
FAILED_STEPS=0
SKIPPED_STEPS=0

log_header() {
    echo -e "\n${BOLD}${BLUE}====================================================================${NC}"
    echo -e "${BOLD}${BLUE}>>> $1${NC}"
    echo -e "${BOLD}${BLUE}====================================================================${NC}"
}

log_pass() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    ((PASSED_STEPS++)) || true
}

log_fail() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    ((FAILED_STEPS++)) || true
}

log_skip() {
    echo -e "${YELLOW}- SKIP:${NC} $1"
    ((SKIPPED_STEPS++)) || true
}

echo -e "${BOLD}Starting Longbow-Quarrel Consolidated Test Suite...${NC}"
echo -e "Host: $(uname -s) $(uname -m), Go: $(go version)"

# ------------------------------------------------------------------------------
# 1. Standard Unit Tests (CPU)
# ------------------------------------------------------------------------------
log_header "1. Running Core Package Unit Tests (CPU)"
if go test -timeout 120s -count=1 ./internal/gguf/... ./internal/tokenizer/... ./internal/metrics/... ./internal/config/... ./internal/api/...; then
    log_pass "Core Package Unit Tests"
else
    log_fail "Core Package Unit Tests"
fi

# ------------------------------------------------------------------------------
# 2. SIMD & VNNI / AMX Tests
# ------------------------------------------------------------------------------
log_header "2. Running SIMD, AVX-512 VNNI, & AMX Acceleration Tests"
if go test -v -timeout 60s -count=1 ./internal/simd/...; then
    log_pass "SIMD & VNNI Tests"
else
    log_fail "SIMD & VNNI Tests"
fi

# ------------------------------------------------------------------------------
# 3. Vision-Language Model (VLM) Tests
# ------------------------------------------------------------------------------
log_header "3. Running Multiplatform Vision-Language (VLM) Pipeline Tests"
if go test -v -timeout 60s -count=1 ./internal/vlm/...; then
    log_pass "VLM Pipeline Tests"
else
    log_fail "VLM Pipeline Tests"
fi

# ------------------------------------------------------------------------------
# 4. Grammar & CFG / Regex Constrained Sampling Tests
# ------------------------------------------------------------------------------
log_header "4. Running Grammar-Constrained PDA / Regex Sampling Tests"
if go test -v -timeout 60s -count=1 ./internal/sampler/...; then
    log_pass "Grammar PDA & Sampler Tests"
else
    log_fail "Grammar PDA & Sampler Tests"
fi

# ------------------------------------------------------------------------------
# 5. Speculative Decoding & Paged KV Cache Tests
# ------------------------------------------------------------------------------
log_header "5. Running Speculative Decoding & Paged Attention Cache Tests"
if go test -v -timeout 90s -count=1 ./internal/engine -run 'Speculative|PagedKV|Batch'; then
    log_pass "Speculative Decoding & Paged Cache Tests"
else
    log_fail "Speculative Decoding & Paged Cache Tests"
fi

# ------------------------------------------------------------------------------
# 6. CUDA Acceleration Tests (if hardware/nvcc present)
# ------------------------------------------------------------------------------
log_header "6. Checking CUDA Hardware & Running CUDA Kernel Tests"
HAS_CUDA=0
if command -v nvcc &> /dev/null && command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_CUDA=1
    fi
fi

if [ "${HAS_CUDA}" -eq 1 ]; then
    echo "CUDA hardware and nvcc compiler detected."
    # Ensure libcuda_kernels.a is compiled
    if [ ! -f "internal/device/libcuda_kernels.a" ]; then
        echo "Compiling internal/device/libcuda_kernels.a..."
        nvcc -c -O3 -Xcompiler -fPIC internal/device/cuda_kernels.cu -o internal/device/cuda_kernels.o
        ar rcs internal/device/libcuda_kernels.a internal/device/cuda_kernels.o
        rm -f internal/device/cuda_kernels.o
    fi

    if go vet -tags cuda ./... && go build -tags cuda ./cmd/quarrel; then
        log_pass "CUDA Compile & Vet Gate"
    else
        log_fail "CUDA Compile & Vet Gate"
    fi

    if go test -v -tags cuda -timeout 90s -count=1 ./internal/device -run 'CUDA|Flash|Dequant|Graph'; then
        log_pass "CUDA Zero-Dequant, Flash Attention, & Graph Capture Tests"
    else
        log_fail "CUDA Zero-Dequant, Flash Attention, & Graph Capture Tests"
    fi

    if go test -tags cuda -timeout 300s -count=1 ./internal/engine/...; then
        log_pass "CUDA Engine Tests"
    else
        log_fail "CUDA Engine Tests"
    fi
else
    log_skip "CUDA tests skipped (no NVIDIA GPU or nvcc compiler detected)"
fi

# ------------------------------------------------------------------------------
# 7. Race Detector Validation
# ------------------------------------------------------------------------------
# engine/api hold the actual concurrent code (batch scheduler, seq manager,
# resource monitor); the earlier gate only covered leaf packages.
log_header "7. Running Race Detector on Concurrency-Critical Components"
if go test -race -p "${GOMAXPROCS}" -timeout 300s -count=1 ./internal/engine/... ./internal/api/... ./internal/metrics/... ./internal/sampler/... ./internal/tokenizer/...; then
    log_pass "Race Detector Validation"
else
    log_fail "Race Detector Validation"
fi

# ------------------------------------------------------------------------------
# 8. Code Coverage Gate (Threshold >= 80% on Core Packages)
# ------------------------------------------------------------------------------
# Both the 80% threshold and the ratchet floors are computed from a single
# coverage profile by scripts/coverage_gate.sh, which is the same script the
# CI coverage job uses. The previous implementation ran the test suite once
# per package (twice more in total) and parsed "go test -cover" output with
# a grep that silently took only the first match.
log_header "8. Validating Code Coverage Gates (>= 80% + ratchet floors)"
COVERAGE_PROFILE="${TMPDIR:-/tmp}/quarrel-coverage.out"
if go test -p "${GOMAXPROCS}" -timeout 300s -coverprofile="${COVERAGE_PROFILE}" -covermode=atomic ./... >/dev/null 2>&1; then
    if "${SCRIPT_DIR}/coverage_gate.sh" "${COVERAGE_PROFILE}"; then
        log_pass "Coverage Gates (>= 80% + ratchet floors)"
    else
        log_fail "Coverage Gates (>= 80% + ratchet floors)"
    fi
else
    log_fail "Coverage Gates (could not produce a coverage profile)"
fi
rm -f "${COVERAGE_PROFILE}"

# ------------------------------------------------------------------------------
# 9. Duplicate declaration check
# ------------------------------------------------------------------------------
# A duplicate package-level declaration only breaks the build when two files
# that declare the same name land in the same package, which with build-tag
# selected files can hide in a configuration nobody builds locally.
log_header "9. Checking for duplicate package-level declarations"
if "${SCRIPT_DIR}/dupcheck.sh"; then
    log_pass "Duplicate Declaration Check"
else
    log_fail "Duplicate Declaration Check"
fi

# ------------------------------------------------------------------------------
# 9d. Dead code
# ------------------------------------------------------------------------------
# A function that no code path reaches is a subsystem a reader would
# reasonably believe is live. The analysis found 250 of them in the 2026-09
# audit; the gate keeps the number from growing again.
log_header "9d. Checking for unreachable code"
if "${SCRIPT_DIR}/deadcode.sh"; then
    log_pass "Dead Code Check"
else
    log_fail "Dead Code Check"
fi

# ------------------------------------------------------------------------------
# 9e. Tests that cannot fail
# ------------------------------------------------------------------------------
# A test with no assertion passes whatever the code under test does.
log_header "9e. Checking for tests that cannot fail"
if go run ./scripts/vacuous_tests; then
    log_pass "Vacuous Test Check"
else
    log_fail "Vacuous Test Check"
fi

# ------------------------------------------------------------------------------
# 9b. Metal type check
# ------------------------------------------------------------------------------
# The Metal backend is a `darwin && metal` build, so it is normally only
# compiled on a macOS runner. Its compile errors are Go type errors, though,
# so they can be found from here: the check strips only the `darwin` token
# from the build tags and builds with cgo against the plain-C bridge.
log_header "9b. Type-checking the Metal backend"
if "${SCRIPT_DIR}/metalcheck.sh"; then
    log_pass "Metal Backend Type Check"
else
    log_fail "Metal Backend Type Check"
fi

# ------------------------------------------------------------------------------
# 9c. ARM64 / NEON build and tests
# ------------------------------------------------------------------------------
# The NEON TurboQuant kernel did not compile, and its absolute value mask
# was built as an integer literal converted to a float, so the bit pattern
# was 0x4F000000 instead of 0x7FFFFFFF: the kernel rounded half to even
# while the CPU reference and the GPU round half away from zero. Nothing
# caught it because nothing built the arm64 configuration.
#
# The job runs under QEMU, so it needs docker. It is skipped when docker
# is not available; the CI job is the gate.
log_header "9c. Building and testing the arm64 / NEON configuration"
if command -v docker >/dev/null 2>&1; then
    DOCKER_CACHE_ARGS=()
    if [ -d "${HOME}/go/pkg/mod" ]; then
        DOCKER_CACHE_ARGS+=(-v "${HOME}/go/pkg/mod:/go/pkg/mod")
    fi
    DOCKER_BUILD_CACHE="${HOME}/.cache/go-build-arm64"
    mkdir -p "${DOCKER_BUILD_CACHE}"
    DOCKER_CACHE_ARGS+=(-v "${DOCKER_BUILD_CACHE}:/root/.cache/go-build")
    if docker run --rm --platform linux/arm64 \
            --cpus="${QUARREL_DOCKER_CPUS:-4}" --memory="${QUARREL_DOCKER_MEM:-4g}" \
            "${DOCKER_CACHE_ARGS[@]}" \
            -v "${REPO_ROOT}:/src" -w /src -e CGO_ENABLED=1 \
            -e GOMAXPROCS=4 -e GOMEMLIMIT=2GiB \
            golang:1.27.3 go test -p 4 -timeout 180s -count=1 ./internal/simd/... >/dev/null 2>&1; then
        log_pass "ARM64 / NEON Build and Tests"
    else
        log_fail "ARM64 / NEON Build and Tests"
    fi
else
    log_skip "ARM64 / NEON Build and Tests (docker is not available)"
fi

# ------------------------------------------------------------------------------
# 10. Zero-Allocation Hotpath Regression Gate
# ------------------------------------------------------------------------------
log_header "10. Validating Zero-Allocation Hotpath Invariants"
if "${REPO_ROOT}/scripts/bench_gate.sh" check-zero-alloc "${REPO_ROOT}/internal/simd"; then
    log_pass "Zero-Allocation Hotpath Invariants (0 allocs/op)"
else
    log_fail "Zero-Allocation Hotpath Invariants (Regressions detected)"
fi

# ------------------------------------------------------------------------------
# 11. Workflow lint
# ------------------------------------------------------------------------------
# actionlint checks the workflow files for the mistakes it finds: a `needs:`
# entry that names a job which does not exist, an expression that references
# an undefined context, and - when shellcheck is installed - the shell
# problems in the `run:` blocks. The GitHub runners have shellcheck, so the
# check covers more there than it does here; the SC2086 warnings it found
# were masked for months because the pipeline was disabled.
log_header "11. Linting the workflow files"
if (cd "${REPO_ROOT}" && go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.7 \
        -config-file .github/actionlint.yaml .github/workflows/*.yml); then
    log_pass "Workflow Lint (actionlint)"
else
    log_fail "Workflow Lint (actionlint)"
fi

# ------------------------------------------------------------------------------
# 12. Documentation drift check
# ------------------------------------------------------------------------------
# The Helm chart, the Grafana dashboard and the reference documentation all
# describe the code. They had all drifted: the chart passed flags the
# command no longer had and scraped the wrong port for the metrics, the
# dashboard queried 27 of the 133 declared metrics, and the API reference
# described a set of endpoints that never existed.
log_header "12. Checking the documentation against the code"
if (cd "${REPO_ROOT}" && go run ./scripts/docsync); then
    log_pass "Documentation Drift Check"
else
    log_fail "Documentation Drift Check"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
log_header "Test Suite Execution Summary"
echo -e "Passed:  ${GREEN}${PASSED_STEPS}${NC}"
echo -e "Failed:  ${RED}${FAILED_STEPS}${NC}"
echo -e "Skipped: ${YELLOW}${SKIPPED_STEPS}${NC}"

if [ "${FAILED_STEPS}" -gt 0 ]; then
    echo -e "\n${RED}${BOLD}Consolidated Test Suite FAILED with ${FAILED_STEPS} error(s).${NC}"
    exit 1
else
    echo -e "\n${GREEN}${BOLD}Consolidated Test Suite PASSED all checks!${NC}"
    exit 0
fi
