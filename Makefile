# =============================================================================
# Longbow-Quarrel Makefile
# =============================================================================
# Build targets for all supported architectures:
#   - CPU (linux/amd64, linux/arm64): Pure Go with SIMD fallbacks
#   - Metal (darwin/arm64, darwin/amd64): Apple Metal GPU acceleration
#   - NVIDIA (linux/amd64): CUDA/cuBLAS acceleration
#   - TPU (linux/amd64): Google TPU/XLA acceleration
# =============================================================================

.PHONY: all clean test build cpu metal nvidia tpu install check deps help

# Default target
all: cpu

# =============================================================================
# Variables
# =============================================================================
BINARY_NAME ?= quarrel
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GO := go
GOARCH ?= $(shell go env GOARCH)
GOOS ?= $(shell go env GOOS)

# Build flags
LDFLAGS ?= -s -w -extldflags=-static
CGO_FLAGS ?=

# Output directories
BIN_DIR ?= bin
DIST_DIR ?= dist

# CUDA/TPU specific
CUDA_PATH ?= /usr/local/cuda
TPU_PATH ?= /usr/local/tpu

# SIMD options (auto-detect unless specified)
SIMD_LEVEL ?= auto  # auto, avx512, avx2, scalar

# =============================================================================
# Build Targets
# =============================================================================

# Build for CPU (default, cross-platform)
build: cpu

# CPU builds (pure Go, no GPU)
cpu: cpu-linux-amd64 cpu-linux-arm64

cpu-linux-amd64:
	@echo "Building $(BINARY_NAME) for linux/amd64 (CPU)..."
	$(GO) build -ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-linux-amd64 ./cmd/quarrel

cpu-linux-arm64:
	@echo "Building $(BINARY_NAME) for linux/arm64 (CPU)..."
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 $(GO) build -ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-linux-arm64 ./cmd/quarrel

cpu-darwin-amd64:
	@echo "Building $(BINARY_NAME) for darwin/amd64 (CPU)..."
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 $(GO) build -ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-destdir=$(BIN_DIR) -o $(BIN_DIR)/$(BINARY_NAME)-darwin-amd64 ./cmd/quarrel

cpu-darwin-arm64:
	@echo "Building $(BINARY_NAME) for darwin/arm64 (CPU)..."
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 $(GO) build -ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-darwin-arm64 ./cmd/quarrel

# Metal builds (macOS with GPU)
metal: metal-darwin-arm64 metal-darwin-amd64

metal-darwin-arm64:
	@echo "Building $(BINARY_NAME) for darwin/arm64 (Metal)..."
	CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 $(GO) build -tags metal \
		-ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-darwin-arm64-metal ./cmd/quarrel

metal-darwin-amd64:
	@echo "Building $(BINARY_NAME) for darwin/amd64 (Metal)..."
	CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 $(GO) build -tags metal \
		-ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-darwin-amd64-metal ./cmd/quarrel

# NVIDIA CUDA builds (linux/amd64)
nvidia: nvidia-cuda

internal/device/libcuda_kernels.a: internal/device/cuda_kernels.cu
	@echo "Compiling CUDA kernels..."
	nvcc -c -O3 -Xcompiler -fPIC internal/device/cuda_kernels.cu -o internal/device/cuda_kernels.o
	ar rcs internal/device/libcuda_kernels.a internal/device/cuda_kernels.o
	rm -f internal/device/cuda_kernels.o

CUDA_LDFLAGS ?= -s -w

nvidia-cuda: internal/device/libcuda_kernels.a
	@echo "Building $(BINARY_NAME) for linux/amd64 (CUDA)..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 $(GO) build -tags cuda \
		-ldflags "$(CUDA_LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-linux-amd64-cuda ./cmd/quarrel

# TPU builds (linux/amd64)
tpu: tpu-xla

tpu-xla:
	@echo "Building $(BINARY_NAME) for linux/amd64 (TPU)..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 $(GO) build -tags tpu \
		-ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-linux-amd64-tpu ./cmd/quarrel

# =============================================================================
# SIMD-specific builds (x86 with AVX512/AVX2 fallbacks)
# =============================================================================

# AVX-512 build (will auto-detect and fallback to AVX2 or scalar at runtime)
simd-avx512:
	@echo "Building $(BINARY_NAME) with AVX-512 (auto-fallback to AVX2/scalar)..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 $(GO) build -tags cgo \
		-ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-linux-amd64-simd ./cmd/quarrel

# Force AVX2 only (no AVX512)
simd-avx2:
	@echo "Building $(BINARY_NAME) with AVX2..."
	DISABLE_AVX512=1 CGO_ENABLED=1 GOOS=linux GOARCH=amd64 $(GO) build -tags cgo \
		-ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-linux-amd64-avx2 ./cmd/quarrel

# Pure scalar (no SIMD)
simd-scalar:
	@echo "Building $(BINARY_NAME) with scalar fallbacks only..."
	CGO_ENABLED=0 $(GO) build -ldflags "$(LDFLAGS) -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o $(BIN_DIR)/$(BINARY_NAME)-linux-amd64-scalar ./cmd/quarrel

# =============================================================================
# Docker builds
# =============================================================================

docker-cpu:
	@echo "Building Docker image for CPU..."
	docker build -f Dockerfile.cpu -t $(BINARY_NAME):cpu .

docker-metal:
	@echo "Building Docker image for Metal..."
	docker build -f Dockerfile.metal -t $(BINARY_NAME):metal .

docker-nvidia:
	@echo "Building Docker image for NVIDIA CUDA..."
	docker build -f Dockerfile.nvidia -t $(BINARY_NAME):nvidia .

docker-tpu:
	@echo "Building Docker image for TPU..."
	docker build -f Dockerfile.tpu -t $(BINARY_NAME):tpu .

docker-all: docker-cpu docker-nvidia docker-tpu

# =============================================================================
# Development
# =============================================================================

# Install dependencies
deps:
	$(GO) mod download
	$(GO) mod tidy

# Run tests
test:
	$(GO) test -v -race -cover ./...

# Run tests with specific build tags
test-cuda:
	$(GO) test -v -tags cuda ./...

test-metal:
	$(GO) test -v -tags metal ./...

test-tpu:
	$(GO) test -v -tags tpu ./...

# SIMD tests (AVX-512/AVX-2)
test-simd:
	CGO_ENABLED=1 $(GO) test -v -tags cgo ./internal/simd/...

test-simd-quick:
	CGO_ENABLED=1 $(GO) test -v -tags cgo -run "QuickCheck" ./internal/simd/...

test-simd-fuzz:
	CGO_ENABLED=1 $(GO) test -v -tags cgo -fuzz -fuzztime=10s ./internal/simd/...

# SIMD benchmarks
bench-simd:
	CGO_ENABLED=1 $(GO) test -v -bench=AVX512 -benchmem ./internal/simd/...

bench-simd-all:
	CGO_ENABLED=1 $(GO) test -bench=. -benchmem ./internal/simd/...

# Go vet
check:
	$(GO) vet ./...

# Go vet with security
check-sec:
	gosec ./...

# Run linter
lint:
	golangci-lint run --timeout 5m

# Format code
fmt:
	$(GO) fmt ./...
	gofmt -s -w .

# =============================================================================
# CI/CD Targets
# =============================================================================

# Build all binaries for release
release: release-linux release-darwin release-docker

release-linux: cpu-linux-amd64 cpu-linux-arm64 nvidia-cuda tpu-xla

release-darwin: cpu-darwin-amd64 cpu-darwin-arm64 metal-darwin-arm64

release-docker: docker-all

# Build with race detector
race:
	CGO_ENABLED=1 $(GO) build -race -o $(BIN_DIR)/$(BINARY_NAME)-race ./cmd/quarrel

# Benchmark
bench:
	$(GO) test -bench=. -benchmem ./...

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(DIST_DIR)
	$(GO) clean

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Longbow-Quarrel Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all          Build all CPU binaries (default)"
	@echo "  cpu          Build CPU versions (linux/amd64, linux/arm64)"
	@echo "  cpu-*        Build for specific CPU platform"
	@echo "  metal       Build Metal GPU versions (darwin)"
	@echo "  nvidia      Build NVIDIA CUDA version"
	@echo "  tpu         Build Google TPU version"
	@echo "  simd-*      Build with specific SIMD level"
	@echo "  docker-*    Build Docker images"
	@echo "  test        Run tests"
	@echo "  check       Run go vet"
	@echo "  check-sec   Run security check (gosec)"
	@echo "  lint        Run linter"
	@echo "  release     Build all release binaries"
	@echo "  race        Build with race detector"
	@echo "  bench       Run benchmarks"
	@echo "  clean       Clean build artifacts"
	@echo "  help        Show this help"
	@echo ""
	@echo "Build examples:"
	@echo "  make cpu                    # Build CPU versions"
	@echo "  make nvidia               # Build CUDA version"
	@echo "  make tpu                  # Build TPU version"
	@echo "  make simd-avx512          # Build with AVX-512"
	@echo "  make docker-nvidia         # Build CUDA Docker image"
	@echo ""
	@echo "Environment variables:"
	@echo "  BINARY_NAME    Output binary name (default: quarrel)"
	@echo "  VERSION      Version string"
	@echo "  LDFLAGS      Additional ldflags"
	@echo "  SIMD_LEVEL   SIMD level: auto, avx512, avx2, scalar"

# =============================================================================
# Verify build environment
# =============================================================================

verify-env:
	@echo "Build Environment:"
	@echo "  GOOS:       $(GOOS)"
	@echo "  GOARCH:     $(GOARCH)"
	@echo "  CGO_ENABLED: $(shell go env CGO_ENABLED)"
	@echo "  SIMD support:"
	@echo "    AVX-512:  $$([ -r /proc/cpuinfo ] && grep -q avx512f /proc/cpuinfo && echo 'available' || echo 'N/A')"
	@echo "    AVX2:     $$([ -r /proc/cpuinfo ] && grep -q avx2 /proc/cpuinfo && echo 'available' || echo 'N/A')"
	@echo "    NEON:     $(shell echo $(GOARCH))"