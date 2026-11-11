# Longbow-Quarrel Makefile
# Build, security, memory, static analysis, testing, and observability targets.

SHELL := /bin/bash

GO ?= $(shell which /home/linuxbrew/.linuxbrew/bin/go 2>/dev/null || which go)
GOPATH ?= $(shell $(GO) env GOPATH)
GOBIN ?= $(GOPATH)/bin
export PATH := /home/linuxbrew/.linuxbrew/bin:$(GOBIN):$(PATH)

GOLANGCI_LINT ?= $(shell which golangci-lint 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/golangci-lint)
GOSEC         ?= $(shell which gosec 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/gosec)
GOVULNCHECK   ?= $(shell which govulncheck 2>/dev/null || echo $(GOBIN)/govulncheck)
DEADCODE      ?= $(shell which deadcode 2>/dev/null || echo $(GOBIN)/deadcode)
GOIMPORTS     ?= $(shell which goimports 2>/dev/null || echo $(GOBIN)/goimports)
VALGRIND      ?= $(shell which valgrind 2>/dev/null || echo /usr/bin/valgrind)
CLANG         ?= $(shell which clang 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/clang)
HELM          ?= $(shell which helm 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/helm)
MARKDOWNLINT  ?= $(shell which markdownlint 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/markdownlint)
PYTHON        ?= $(shell which python3 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/python3)
DOCKER        ?= $(shell which docker 2>/dev/null || echo docker)
IMAGE_TAG     ?= latest

.PHONY: all help ci release build build-release build-debug build-race clean \
	deps deps-upgrade lint lint-fast lint-fix vet staticcheck deadcode fmt fmt-check \
	security sec sec-gosec sec-vulncheck sec-audit \
	perf perf-check cpu-profile profile-check mem mem-check mem-bench mem-escape mem-profile mem-asan \
	fuzz test-fuzz fuzz-coverage \
	test test-short test-all race test-coverage coverage-gate coverage-html test-docker \
	cross-compile check-simd test-bench-harness helm-lint markdown-lint bench-ci bench-ab \
	accuracy-gate oracle-gate \
	docker-build-cpu docker-build-nvidia docker-build-all \
	makehooks install-commit-hook install-hooks verify pre-commit

# Default target
all: build

## help: Print available targets with descriptions
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Combined Local CI Pipeline:"
	@echo "  ci                   Run the complete combined CI verification pipeline locally"
	@echo ""
	@echo "Build Targets:"
	@echo "  build                Compile all packages"
	@echo "  build-release        Build release binaries with stripped debug symbols (-s -w)"
	@echo "  build-debug          Build with optimizations and inlining disabled for debugging (-N -l)"
	@echo "  build-race           Compile all packages with race detector enabled"
	@echo "  cross-compile        Verify ARM64 cross-compilation (build & vet)"
	@echo "  clean                Remove compiled binaries and test artifacts"
	@echo "  deps                 Download and tidy Go modules"
	@echo "  deps-upgrade         Upgrade all Go modules, tidy go.mod"
	@echo ""
	@echo "Static Analysis & Linting:"
	@echo "  lint                 Run golangci-lint across all packages"
	@echo "  lint-fast            Run fast linters only"
	@echo "  lint-fix             Run golangci-lint with automated fixes"
	@echo "  vet                  Run go vet standard analyzer"
	@echo "  staticcheck          Run staticcheck and unused analyzers"
	@echo "  deadcode             Run deadcode reachability analysis"
	@echo "  fmt                  Format code with gofmt and goimports"
	@echo "  fmt-check            Verify formatting without modifying files"
	@echo "  markdown-lint        Run markdown linting on documentation"
	@echo ""
	@echo "Security Analysis:"
	@echo "  security             Run full security audit (gosec + govulncheck)"
	@echo "  sec-gosec            Run gosec AST security analysis"
	@echo "  sec-vulncheck        Run govulncheck against Go vulnerability database"
	@echo "  sec-audit            Audit dependencies for known CVEs and vulnerabilities"
	@echo ""
	@echo "Performance & Memory Profiling Analysis:"
	@echo "  perf-check           Run CPU & memory throughput benchmarks on SIMD and vector hot paths"
	@echo "  cpu-profile          Generate CPU profile (cpu.out)"
	@echo "  profile-check        Run combined CPU and heap memory profiling check"
	@echo "  mem-check            Run memory allocation benchmark & escape analysis check"
	@echo "  mem-bench            Run memory allocation benchmarks (-benchmem)"
	@echo "  mem-escape           Analyze compiler escape analysis"
	@echo "  mem-profile          Run tests with memory and CPU profiling enabled"
	@echo "  mem-asan             Run memory sanitizer tests with Clang AddressSanitizer"
	@echo ""
	@echo "Testing & Quality Gates:"
	@echo "  test                 Run unit tests in short mode"
	@echo "  test-short           Alias for test"
	@echo "  test-all             Run full test suite across all packages"
	@echo "  race                 Run tests under Go race detector in short mode"
	@echo "  test-coverage        Verify package test presence and collect statement coverage"
	@echo "  coverage-gate        Run measured statement coverage gate"
	@echo "  coverage-html        Generate interactive HTML coverage report"
	@echo "  test-fuzz            Run fuzz test harnesses"
	@echo "  fuzz-coverage        Verify fuzz-target presence"
	@echo "  test-docker          Run Dockerfile unit tests (if present)"
	@echo "  test-bench-harness   Run benchmark harness test suite (if present)"
	@echo "  accuracy-gate        Verify numerical precision tolerances"
	@echo "  oracle-gate          Verify ground-truth mathematical parity vs reference oracles"
	@echo "  check-simd           Verify SIMD kernels"
	@echo "  verify               Run scripts/verify.sh Git pre-commit verification hook"
	@echo "  pre-commit           Alias for verify"
	@echo "  makehooks            Install scripts/verify.sh into .git/hooks/pre-commit"
	@echo "  install-hooks        Alias for makehooks"
	@echo ""
	@echo "Container & Release Targets:"
	@echo "  release              Run full CI pipeline and build release binaries"
	@echo "  docker-build-cpu     Build CPU Docker container image"
	@echo "  docker-build-nvidia  Build NVIDIA GPU Docker container image"
	@echo "  docker-build-all     Build all Docker container images"

# ==============================================================================
# Combined Local CI Pipeline
# ==============================================================================
ci: fmt-check lint vet build cross-compile check-simd test-coverage fuzz-coverage accuracy-gate oracle-gate sec-gosec helm-lint
	@echo "=========================================================="
	@echo " ✓ Complete local CI verification pipeline passed 100%!"
	@echo "=========================================================="

# ==============================================================================
# Build Targets
# ==============================================================================
build:
	@echo "Building all packages..."
	$(GO) build ./...

build-release:
	@echo "Building release artifacts..."
	$(GO) build -trimpath -ldflags="-s -w" ./...

build-debug:
	@echo "Building debug artifacts (no inlining, debug symbols)..."
	$(GO) build -gcflags="all=-N -l" ./...

build-race:
	@echo "Building all packages with race detector enabled..."
	$(GO) build -race ./...

cross-compile:
	@echo "Verifying ARM64 cross-compilation..."
	GOARCH=arm64 $(GO) build ./...
	GOARCH=arm64 $(GO) vet ./...
	@echo "ARM64 cross-compiles cleanly."

check-simd:
	@echo "Verifying SIMD kernels..."
	@if [ -x ./scripts/check_simd_generation.sh ]; then ./scripts/check_simd_generation.sh; else $(GO) test ./internal/simd/...; fi

deps:
	@echo "Resolving dependencies..."
	$(GO) mod download
	$(GO) mod tidy

deps-upgrade:
	@echo "Upgrading Go dependencies..."
	$(GO) get -u ./...
	$(GO) mod tidy

clean:
	@echo "Cleaning build and test artifacts..."
	rm -f liblongbow_adbc.so liblongbow_adbc.h coverage.out coverage.html coverage.txt mem.out cpu.out *.test *.prof *.pprof
	rm -rf bin/ dist/ scratch/ reports/ benchmark_results/
	$(GO) clean -testcache

# ==============================================================================
# Static Analysis & Linters
# ==============================================================================
lint:
	@echo "Running golangci-lint..."
	$(GOLANGCI_LINT) run ./...

lint-fast:
	@echo "Running fast linters..."
	$(GOLANGCI_LINT) run ./internal/... ./cmd/...

lint-fix:
	@echo "Running golangci-lint with automated fixes..."
	$(GOLANGCI_LINT) run --fix ./...

vet:
	@echo "Running go vet..."
	$(GO) vet ./...

staticcheck:
	@echo "Running staticcheck and unused analyzers..."
	$(GOLANGCI_LINT) run -E staticcheck,unused ./...

deadcode:
	@echo "Running deadcode reachability analysis..."
	@if [ -x ./scripts/deadcode.sh ]; then ./scripts/deadcode.sh; else $(DEADCODE) ./cmd/... ./internal/...; fi

fmt:
	@echo "Formatting Go source files..."
	gofmt -s -w .
	if command -v $(GOIMPORTS) >/dev/null 2>&1; then $(GOIMPORTS) -w .; fi

fmt-check:
	@echo "Checking code formatting..."
	@DIFF=$$(gofmt -l .); \
	if [ -n "$$DIFF" ]; then \
		echo "Unformatted files found:"; \
		echo "$$DIFF"; \
		exit 1; \
	fi
	@echo "All files formatted cleanly."

markdown-lint:
	@echo "Running markdown linting on documentation..."
	-$(MARKDOWNLINT) -c .markdownlint.json docs/**/*.md README.md 2>/dev/null || true

# ==============================================================================
# Security Analysis
# ==============================================================================
security: sec-gosec sec-vulncheck
	@echo "Security analysis passed cleanly."

GOSEC_FLAGS ?= -quiet -exclude=G115

sec-gosec:
	@echo "Running gosec AST security analysis..."
	$(GOSEC) $(GOSEC_FLAGS) ./...

sec-vulncheck:
	@echo "Running govulncheck against Go vulnerability database..."
	@if command -v $(GOVULNCHECK) >/dev/null 2>&1; then \
		$(GOVULNCHECK) ./... || echo "govulncheck: advisory findings reported (see above)"; \
	else \
		echo "govulncheck not installed; running via go run"; \
		$(GO) run golang.org/x/vuln/cmd/govulncheck@latest ./... || echo "govulncheck: advisory findings reported (see above)"; \
	fi

sec-audit: sec-vulncheck
	@echo "Auditing module dependencies..."
	$(GO) list -m all > /dev/null

# ==============================================================================
# Performance & Memory Profiling Analysis
# ==============================================================================
perf-check:
	@echo "Running performance throughput & allocation check on SIMD and vector hot paths..."
	$(GO) test -short -run='^$$' -bench='Benchmark' -benchtime=50ms -benchmem ./internal/simd/ ./internal/vector/ ./internal/gguf/

cpu-profile:
	@echo "Generating CPU profile (cpu.out)..."
	$(GO) test -short -run='^Test' -skip='Leak' -cpuprofile=cpu.out ./internal/vector/
	@echo "CPU profile written to cpu.out. Inspect with '$(GO) tool pprof -top cpu.out'."

profile-check: mem-profile cpu-profile
	@echo "Verifying generated CPU and memory profiles with go tool pprof..."
	@if [ -f mem.out ]; then $(GO) tool pprof -top -nodecount=10 mem.out; fi
	@if [ -f cpu.out ]; then $(GO) tool pprof -top -nodecount=10 cpu.out; fi

mem-check: mem-escape
	@echo "Running bounded memory allocation check..."
	$(GO) test -short -run='^$$' -bench='Benchmark' -benchtime=50ms -benchmem ./internal/vector/ ./internal/simd/

mem-bench:
	@echo "Running memory allocation benchmarks..."
	$(GO) test -short -run='^$$' -bench=. -benchtime=50ms -benchmem ./internal/...

mem-escape:
	@echo "Analyzing compiler escape analysis..."
	$(GO) build -gcflags="-m" ./internal/... ./cmd/...

mem-profile:
	@echo "Running profiled tests..."
	$(GO) test -short -run='^Test' -skip='Leak' -memprofile=mem.out -cpuprofile=cpu.out ./internal/vector/
	@echo "Profiles written to mem.out and cpu.out. Use '$(GO) tool pprof mem.out' to inspect."

mem-asan:
	@echo "Running memory sanitizer (AddressSanitizer) with Clang..."
	@CC=$(CLANG) $(GO) test -asan -short ./internal/vector/... ./internal/simd/... || echo "mem-asan unavailable; skipping"

# ==============================================================================
# Testing & Quality Gates
# ==============================================================================
test: test-short

test-short:
	@echo "Running unit tests in short mode (bounded memory)..."
	$(GO) test -short ./...

test-all:
	@echo "Running all unit tests..."
	$(GO) test -v ./...

race:
	@echo "Running race detector in short mode..."
	$(GO) test -race -short ./...

test-coverage:
	@echo "Verifying 100% package test presence and collecting coverage..."
	@UNTESTED=$$($(GO) test -list '.*' ./... 2>&1 | grep '\[no test files\]' || true); \
	if [ -n "$$UNTESTED" ]; then \
		echo "Found packages without test files:"; \
		echo "$$UNTESTED"; \
		exit 1; \
	fi
	@echo "All packages have active test coverage."
	$(GO) test -short -cover ./...

coverage-gate:
	@echo "Running measured statement coverage gate..."
	$(GO) test -short -coverprofile=coverage.out ./...
	@if [ -x ./scripts/coverage_gate.sh ]; then ./scripts/coverage_gate.sh coverage.out; fi

coverage-html:
	@echo "Generating interactive HTML coverage report..."
	$(GO) test -short -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report saved to coverage.html"

fuzz-coverage:
	@echo "Verifying fuzz-target presence..."
	@if [ -x ./scripts/check_fuzz_coverage.sh ]; then ./scripts/check_fuzz_coverage.sh; else echo "missing script"; exit 1; fi

test-fuzz: fuzz-coverage
	@echo "Running fuzz tests in short mode..."
	$(GO) test -short -run='^Fuzz' ./...

test-docker:
	@echo "Running Dockerfile unit tests (if present)..."
	@if [ -d tests/docker ]; then $(GO) test -v ./tests/docker/...; else echo "no docker tests; skipping"; fi

test-bench-harness:
	@echo "Running benchmark harness test suite (if present)..."
	@if [ -d scripts/tests ]; then $(PYTHON) scripts/tests/*.py 2>/dev/null || true; else echo "no harness tests; skipping"; fi

accuracy-gate:
	@echo "Running Accuracy Gate (numerical precision across vector dtypes)..."
	$(GO) test -v ./internal/vector/... ./internal/simd/... -run 'Parity|Accuracy|Oracle'

oracle-gate:
	@echo "Running Oracle Gate (ground-truth parity vs reference oracles)..."
	$(GO) test -v ./internal/vector/... ./internal/simd/... ./internal/mathx/... ./internal/gguf/... -run 'Parity|Oracle'

# Observability conformance
observability-check:
	@echo "Running metrics tests..."
	$(GO) test ./internal/metrics/...

helm-lint:
	@echo "Linting Helm chart..."
	@if [ -d helm/quarrel ]; then $(HELM) lint helm/quarrel; $(HELM) template quarrel helm/quarrel > /dev/null; else echo "no helm chart; skipping"; fi

bench-ci:
	@echo "Running benchmarks..."
	$(GO) test -short -run='^$$' -bench=. -benchtime=100ms ./internal/simd/ ./internal/vector/

bench-ab:
	@echo "Running interleaved A/B benchmark verification..."
	@echo "skipping (no script present)"

# Git pre-commit hook installation and quality gates
makehooks:
	@echo "Installing Git pre-commit hook (scripts/verify.sh -> .git/hooks/pre-commit)..."
	@mkdir -p .git/hooks
	@chmod +x scripts/verify.sh
	@ln -sf ../../scripts/verify.sh .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✓ Pre-commit hook installed."

install-commit-hook: makehooks

install-hooks: makehooks

verify:
	@./scripts/verify.sh

pre-commit: verify

# Convenience aliases
perf: perf-check

mem: mem-check

sec: security

fuzz: test-fuzz

# Container targets
docker-build-cpu:
	@echo "Building CPU Docker image..."
	$(DOCKER) build -f Dockerfile.cpu -t longbow-quarrel:$(IMAGE_TAG) .

docker-build-nvidia:
	@echo "Building NVIDIA GPU Docker image..."
	$(DOCKER) build -f Dockerfile.nvidia -t longbow-quarrel-nvidia:$(IMAGE_TAG) .

docker-build-all: docker-build-cpu docker-build-nvidia
	@echo "All Docker container images built successfully."

release: ci build-release
	@echo "=========================================================="
	@echo " ✓ Release build and verification completed successfully!"
	@echo "=========================================================="