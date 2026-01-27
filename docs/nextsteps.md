# Production Readiness & Refactoring Plan

This document outlines a 10-step plan to refactor `longbow-quarrel` from a research/prototype codebase into a production-ready inference engine. The goal is to remove debug artifacts, standardize interfaces, and ensure stability and observability.

## 1. Codebase Audit & Cleanup Inventory

- **Objective**: Identifying all "smoke test" code, debug flags, hardcoded paths, and temporary commands.
- **Actions**:
  - Scan for `//go:build` tags that separate debug code.
  - List all `cmd/` binaries and classify them as "keep", "refactor", or "delete".
  - Identify `panic()` calls in library code that need error returns.
  - Locate all `"TODO"` and `"FIXME"` comments related to hacky workarounds.

## 2. Removal of Temporary Artifacts

- **Objective**: Delete code that was only for initial bring-up.
- **Actions**:
  - Delete `cmd/smoke_test`, `cmd/metal_benchmark`, `cmd/quarrel_metal_bench`.
  - Remove `DebugDequant`, `LastLogits`, and other debug fields from the `Engine` struct unless strictly needed for observability.
  - Remove strictly temporary test files in `internal/device/` that duplicate proper unit tests.

## 3. Structured Logging Implementation

- **Objective**: Replace ad-hoc `fmt.Printf` and `log.Println` with a unified structured logger (e.g., `log/slog` or `zap`).
- **Actions**:
  - Define a global logger interface/singleton with levels (DEBUG, INFO, WARN, ERROR).
  - Replace "printf debugging" in hot paths (like `engine.go` loops) with Trace/Debug level logs that are disabled by default.
  - Ensure kernels do not print to stdout/stderr directly.

## 4. Configuration Unification

- **Objective**: specific configuration structs into a validated configuration module.
- **Actions**:
  - Create `internal/config` package.
  - Define a strict `Config` struct that covers Model, Device, and Engine settings.
  - Implement validation logic (e.g., check for valid paths, positive dimensions) before Engine startup.
  - Update `NewEngine` to accept this strongly-typed config.

## 5. Test Suite Consolidation

- **Objective**: Organize tests into Unit, Integration, and Benchmark suites.
- **Actions**:
  - Move valid device tests from `internal/device` to `internal/engine` tests where appropriate.
  - Create a dedicated `test/` directory for integration tests that require model files.
  - Ensure tests use `internal/config` for setup.
  - Remove reliance on hardcoded absolute paths in tests; use environment variables or flags.

## 6. Error Handling & Safety

- **Objective**: Make the engine robust against invalid state or inputs.
- **Actions**:
  - Audit `internal/device` and `internal/gguf` for `panic()`. Replace with `error` returns.
  - Ensure all C/Metal resources (`void*`, buffers) are correctly freed in `Defer` or `Close` methods.
  - Add timeout safety to Metal command buffer executions.

## 7. Observability & Metrics

- **Objective**: Replace one-off timing printouts with standard metrics.
- **Actions**:
  - Expand `internal/metrics` to cover all critical paths (token gen time, prompt eval time, cache usage).
  - Instrument `Engine.Infer` with proper tracing spans.
  - Expose metrics via a standard endpoint (if running as a server) or hook.

## 8. API Stabilization

- **Objective**: Define clear public interfaces for `Engine` and `Device`.
- **Actions**:
  - Make internal helpers private (lowercamelCase).
  - Document all public Exported methods with GoDoc.
  - Finalize the `KVCache` interface to allow future swappable implementations.

## 9. Comprehensive Benchmarking Suite

- **Objective**: Create a reproducible benchmark command.
- **Actions**:
  - Create `cmd/bench` as the single source of truth for performance.
  - Support flags for prompt size, batch size, and quantization type.
  - Output results in a machine-readable format (JSON/CSV) for regression tracking.

## 10. CI/CD & Documentation Finalization

- **Objective**: Ensure the repository is ready for public use and contribution.
- **Actions**:
  - Create a `Makefile` with targets: `build`, `test`, `bench`, `lint`.
  - Update `README.md` and `docs/usage.md` to reflect the new API and commands.
  - Set up a basic CI workflow (GitHub Actions) to run `make test` on non-Metal inputs (mocked) or skip Metal tests gracefully.
