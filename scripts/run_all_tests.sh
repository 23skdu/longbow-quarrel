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
if go test -timeout 120s ./internal/gguf/... ./internal/tokenizer/... ./internal/metrics/... ./internal/config/... ./internal/api/...; then
    log_pass "Core Package Unit Tests"
else
    log_fail "Core Package Unit Tests"
fi

# ------------------------------------------------------------------------------
# 2. SIMD & VNNI / AMX Tests
# ------------------------------------------------------------------------------
log_header "2. Running SIMD, AVX-512 VNNI, & AMX Acceleration Tests"
if go test -v -timeout 60s ./internal/simd/...; then
    log_pass "SIMD & VNNI Tests"
else
    log_fail "SIMD & VNNI Tests"
fi

# ------------------------------------------------------------------------------
# 3. Vision-Language Model (VLM) Tests
# ------------------------------------------------------------------------------
log_header "3. Running Multiplatform Vision-Language (VLM) Pipeline Tests"
if go test -v -timeout 60s ./internal/vlm/...; then
    log_pass "VLM Pipeline Tests"
else
    log_fail "VLM Pipeline Tests"
fi

# ------------------------------------------------------------------------------
# 4. Grammar & CFG / Regex Constrained Sampling Tests
# ------------------------------------------------------------------------------
log_header "4. Running Grammar-Constrained PDA / Regex Sampling Tests"
if go test -v -timeout 60s ./internal/sampler/...; then
    log_pass "Grammar PDA & Sampler Tests"
else
    log_fail "Grammar PDA & Sampler Tests"
fi

# ------------------------------------------------------------------------------
# 5. Speculative Decoding & Paged KV Cache Tests
# ------------------------------------------------------------------------------
log_header "5. Running Speculative Decoding & Paged Attention Cache Tests"
if go test -v -timeout 90s ./internal/engine -run 'Speculative|PagedKV|Batch'; then
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

    if go test -v -tags cuda -timeout 90s ./internal/device -run 'CUDA|Flash|Dequant'; then
        log_pass "CUDA Zero-Dequant & Flash Attention Tests"
    else
        log_fail "CUDA Zero-Dequant & Flash Attention Tests"
    fi
else
    log_skip "CUDA tests skipped (no NVIDIA GPU or nvcc compiler detected)"
fi

# ------------------------------------------------------------------------------
# 7. Race Detector Validation
# ------------------------------------------------------------------------------
log_header "7. Running Race Detector on Core Concurrency Components"
if go test -race -timeout 60s ./internal/metrics/... ./internal/sampler/... ./internal/tokenizer/...; then
    log_pass "Race Detector Validation"
else
    log_fail "Race Detector Validation"
fi

# ------------------------------------------------------------------------------
# 8. Code Coverage Gate (Threshold >= 80% on Core Packages)
# ------------------------------------------------------------------------------
log_header "8. Validating Code Coverage Thresholds (>= 80%)"
COVERAGE_PKGS=(
    "./internal/sampler/..."
    "./internal/simd/..."
    "./internal/tokenizer/..."
    "./internal/metrics/..."
    "./internal/vlm/..."
    "./internal/gguf/..."
    "./internal/api/..."
)

COVERAGE_FAILED=0
for pkg in "${COVERAGE_PKGS[@]}"; do
    COV_OUT=$(go test -cover "${pkg}" 2>&1 | grep -o 'coverage: [0-9.]*%' | awk '{print $2}' | tr -d '%')
    if [ -n "${COV_OUT}" ]; then
        IS_SUFFICIENT=$(awk -v cov="${COV_OUT}" 'BEGIN { if (cov >= 80.0) print "1"; else print "0" }')
        if [ "${IS_SUFFICIENT}" -eq 1 ]; then
            echo -e "${GREEN}✓ ${pkg}:${NC} ${COV_OUT}% (>= 80%)"
        else
            echo -e "${RED}✗ ${pkg}:${NC} ${COV_OUT}% (< 80%)"
            COVERAGE_FAILED=1
        fi
    fi
done

if [ "${COVERAGE_FAILED}" -eq 0 ]; then
    log_pass "Core Package Coverage Gate (>= 80%)"
else
    log_fail "Core Package Coverage Gate (< 80%)"
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
