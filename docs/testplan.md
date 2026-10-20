# Longbow-Quarrel Performance A/B Testing Plan & Quality CI/CD Gates

> **Document Status**: Production Test Plan  
> **Target Hardware Envelope**: Intel Core i7-12650H (Alder Lake: 16 vCPUs, AVX2 + FMA + AVX_VNNI, no AVX-512) & NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, Ada Lovelace compute capability 8.9, CUDA 12.4).  
> **Companion Documents**: [docs/performance.md](performance.md), [docs/roadmap.md](roadmap.md), [scripts/bench_gate.sh](../scripts/bench_gate.sh).

---

## 1. Executive Summary & Goals

This test plan defines the formal, statistically sound methodology for A/B testing performance optimizations, SIMD vectorization routines, and CUDA kernel enhancements in Longbow-Quarrel. It explicitly addresses the specific architectural constraints of consumer and edge hardware lacking AVX-512 but equipped with **AVX2, FMA, AVX-VNNI (256-bit VEX)**, and modern NVIDIA GPUs (RTX 40-series).

### Core Objectives
1. **Prevent Regressions**: Ensure no commit degrades Time-to-First-Token (TTFT), inter-token decode latency, or tokens/sec by $\ge 5\%$.
2. **Zero-Allocation Hotpaths**: Enforce strict zero heap allocations ($0\text{ allocs/op}$) in CPU and GPU token decode loops.
3. **Statistical Integrity**: Eliminate thermal throttling, scheduler migration, and background system noise via thread pinning, clock stabilization, and sample size requirements ($N \ge 20, CV < 3\%$).
4. **CI/CD Quality Gates**: Integrate regression detection into automated test runs (`scripts/bench_gate.sh` and `scripts/run_all_tests.sh`).

---

## 2. Target Hardware & Architectural Envelope

| Component | Hardware Specification | Capabilities & Features | Constraints |
|:---|:---|:---|:---|
| **Host CPU** | Intel Core i7-12650H (10 Cores: 6 P-Cores, 4 E-Cores / 16 Threads) | AVX, AVX2, FMA3, F16C, BMI1/2, **AVX_VNNI** | **No AVX-512**. Heterogeneous P/E core architecture requires thread pinning. |
| **Host RAM** | 24 GB DDR5 | L1: 48KB/core, L2: 1.25MB/core, L3: 24MB shared | GOMEMLIMIT=2GiB enforced in CI/CD to prevent OOM. |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU | 8 GB GDDR6, 24 SMs (3072 CUDA Cores), 96 4th-Gen Tensor Cores, FP8/BF16/FP16 | Max VRAM ~7.5 GB usable; PCIe 4.0 x8 interface. |
| **Compiler / Toolchain** | GCC 13+ / Clang 17+, NVCC 12.4, Go 1.27+ | CGo, `#pragma GCC target("avx2,fma,f16c,avxvnni")`, `-O3` | Strict CGo boundary overhead (~5μs per call) during decode. |

---

## 3. Key Performance Indicators (KPIs) & Metrics

All A/B comparisons evaluate the following five orthogonal vectors:

```
+-----------------------------------------------------------------------------------+
|                           Longbow-Quarrel KPI Matrix                              |
+--------------------------+----------------------------+---------------------------+
| Latency & Speed          | Memory & Allocations       | Compute & Accuracy        |
+--------------------------+----------------------------+---------------------------+
| 1. TTFT (ms)             | 4. Heap Allocs (allocs/op) | 7. SIMD Throughput (M/s)  |
| 2. ITL (ms/tok)          | 5. Heap Growth (B/op)      | 8. GPU SM Occupancy (%)   |
| 3. Throughput (tok/s)    | 6. Peak RSS (MB)           | 9. Max Diff vs Ref (ULP)  |
+--------------------------+----------------------------+---------------------------+
```

1. **Time To First Token (TTFT)**: Elapsed time from prompt submission to generation of token #1 (prefill latency in milliseconds).
2. **Inter-Token Latency (ITL)**: Time per decode step $t_{i} - t_{i-1}$ across autoregressive generation. Measured as P50, P95, and P99.
3. **Generation Throughput**: Sustained tokens per second $\text{tok/s} = \frac{N_{\text{tokens}}}{\sum \text{ITL}}$.
4. **Per-Token Heap Allocations**: Number of allocations during steady-state decode. Target: **0 allocs/token**.
5. **Peak Resident Set Size (RSS)**: Operating system memory consumed by process via `/proc/self/statm`.
6. **Numerical Accuracy (Parity)**: Maximum absolute difference against FP32 ground truth ($\max |y_{\text{candidate}} - y_{\text{ref}}| \le 1.0 \times 10^{-5}$).

---

## 4. Control of Confounding Variables

To ensure benchmark reproducibility on desktop/laptop hardware, tests must isolate system noise:

### 4.1 CPU Pinning & Thread Isolation (P-Cores vs E-Cores)
Alder Lake CPUs contain performance cores (P-cores 0-11 hyperthreaded) and efficiency cores (E-cores 12-15). Running benchmarks across E-cores introduces up to $45\%$ artificial latency variance.
- **Harness Rule**: Pin benchmarking harnesses to physical P-cores using `taskset`:
  ```bash
  # Run benchmark strictly on first 6 physical Performance Cores
  taskset -c 0,2,4,6,8,10 go test -run '^$' -bench . -benchmem ./internal/simd/
  ```
- **GOMAXPROCS Pinning**: Set `GOMAXPROCS=6` matching the number of pinned physical cores.

### 4.2 GPU Clock Stabilization
Laptop GPUs dynamically throttle clock speeds based on thermal headroom and power profiles (causing up to $30\%$ run-to-run drift).
- **Harness Rule**: If running as root or with NVML privileges, lock GPU clocks to nominal base:
  ```bash
  sudo nvidia-smi -pm 1                  # Enable persistence mode
  sudo nvidia-smi -lgc 1470,1470         # Lock graphics clock to 1470 MHz
  ```
- If running unprivileged, verify GPU temperature is $< 65^\circ\text{C}$ before commencing benchmarks using `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader`.

### 4.3 Garbage Collection & Go Runtime Stabilization
- Set `GODEBUG=gctrace=0` and `GOMEMLIMIT=2GiB`.
- Enforce explicit runtime GC sweeps before benchmark timer start (`runtime.GC()`).

---

## 5. Statistical Methodology & Acceptance Thresholds

### 5.1 Sample Size and Warmup
- **Warmup Cycles**: $N_{\text{warmup}} = 5$ iterations discarded to populate kernel code caches, CUDA driver module tables, and mmap page faults.
- **Active Iterations**: $N_{\text{eval}} \ge 20$ iterations per candidate configuration.
- **Outlier Filtering**: Apply Interquartile Range (IQR) filtering: values outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ are reported and logged.

### 5.2 Coefficient of Variation ($CV$)
The benchmark suite calculates the coefficient of variation:
$$CV = \frac{\sigma}{\mu} \times 100\%$$
- If $CV > 3.0\%$ in isolated runs ($CV > 5.0\%$ in CI environments), the benchmark is marked **UNSTABLE** and re-run.

### 5.3 A/B Decision Matrix & Regression Rejection Gates

A candidate change (Branch B) is accepted against baseline (Branch A) if and only if:
1. **Throughput Gate**: Throughput ratio $\frac{\text{tok/s}_B}{\text{tok/s}_A} \ge 0.95$ (no more than 5% regression under noisy CI, $\ge 1.00$ in controlled local runs).
2. **Allocation Gate**: Decode allocations must satisfy $\Delta \text{allocs} \le 0$. Any increase in `allocs/op` on decode loops fails immediately.
3. **Parity Gate**: Max relative error across logits or activations $\epsilon_{\text{rel}} \le 10^{-4}$.
4. **Latency Gate**: P99 inter-token latency does not exceed $1.10 \times \text{Baseline P99}$.

---

## 6. A/B Test Scenarios for Localhost Hardware

### Scenario 1: CPU Quantized Matrix-Vector Multiplication (AVX2 vs AVX_VNNI)
- **Goal**: Compare Q4_K and Q8_0 dot product throughput between:
  1. Scalar Go loop (baseline).
  2. Pure AVX2 integer dot product (`vpmaddubsw` + `vpmaddwd`).
  3. AVX-VNNI integer dot product (`vpdpbusd` 256-bit VEX).
- **Benchmark Command**:
  ```bash
  taskset -c 0,2,4,6,8,10 go test -run '^$' -bench 'BenchmarkVecDotQ|BenchmarkMatVecMulQ' -benchmem -count=5 ./internal/simd/ > simd_candidate.txt
  scripts/bench_gate.sh compare simd_baseline.txt simd_candidate.txt 1.2
  ```
- **Success Criteria**:
  - Q8_0 dot product throughput $\ge 2,500\text{ M elem/s}$.
  - Q4_K dot product throughput $\ge 1,500\text{ M elem/s}$.
  - Zero heap allocations ($0\text{ B/op}, 0\text{ allocs/op}$).

### Scenario 2: CPU Layer Execution & Worker Pool
- **Goal**: Measure memory allocation and latency impact of persistent `WorkerPool` and `CPULayerScratch` vs dynamic goroutine spawning.
- **Benchmark Command**:
  ```bash
  go test -run '^$' -bench 'BenchmarkWorkerPool|BenchmarkEndToEndInferenceStep' -benchmem -count=5 ./internal/simd/
  ```
- **Success Criteria**:
  - $0\text{ allocs/op}$ in `BenchmarkWorkerPool_ParallelFor`.
  - Latency reduction $\ge 10\times$ vs `BenchmarkWorkerPool_RawGoroutines`.

### Scenario 3: GPU Attention (Standard vs Split-K FlashDecoding)
- **Goal**: Validate that Split-K FlashDecoding eliminates SM starvation at sequence lengths $\ge 2048$ on the RTX 4060 (24 SMs).
- **Test Command**:
  ```bash
  go test -v -tags cuda -run 'TestCUDA_AttentionPagedSplitK' ./internal/device
  ```
- **Success Criteria**:
  - Kernel executes across $S \ge 8$ splits.
  - Maximum absolute difference vs reference attention $\le 1.0 \times 10^{-6}$.
  - Execution time on $N=4096$ context drops by $\ge 4\times$ over single-block sequential reduction.

### Scenario 4: CUDA Graph Capture & Replay
- **Goal**: Eliminate host CGo kernel launch latency (~5μs per kernel across 30 kernels = 150μs) by replaying captured graphs.
- **Test Command**:
  ```bash
  go test -v -tags cuda -run 'TestCUDA_GraphManager_' ./internal/device
  ```
- **Success Criteria**:
  - Graph capture completes without CUDA runtime errors.
  - Graph replay produces bit-exact outputs matching stream execution.
  - Dispatch overhead drops to $< 20\mu$s per decode step.

---

## 7. CI/CD Integration & Automated Regression Gates

The test runner `scripts/run_all_tests.sh` executes the regression gates
in sequence. The list and what each one catches is in
[roadmap.md](roadmap.md#the-gates); the four that need more than a
command are described here.

### Gate 8: Code Coverage Gates
Applies the >= 80% threshold to the core packages and a ratchet floor to
the two that are still below it. Both are derived from a single coverage
profile, so the test suite is not run again per package:
```bash
scripts/coverage_gate.sh coverage.out
```

### Gate 9: Duplicate Declaration Check
Fails when two files in the same package declare the same top-level name.
Such a duplicate only breaks the build when build tags put both files in the
same package, which is how the darwin+metal build was broken:
```bash
scripts/dupcheck.sh
```

### Gate 10: SIMD AVX2 Performance & Zero-Alloc Gate
Ensures all SIMD dot product and activation kernels execute with zero heap allocations and meet baseline latency floors:
```bash
scripts/bench_gate.sh check-zero-alloc ./internal/simd
```

> The CUDA precision and graph checks are part of Gate 6, which runs them
> when a CUDA device is present. They are not a separate gate.

### Gate 9b: Metal Backend Type Check

The Metal backend is a `darwin && metal` build, so it is normally only
compiled on a macOS runner. Its compile errors are Go type errors, though,
and none of them needs the Metal framework — so they can be found from
Linux by stripping only the `darwin` token from the build tags and
building with cgo against the plain-C bridge:
```bash
scripts/metalcheck.sh
```

### Gate 11: Workflow Lint

`actionlint` over the workflow files, with `shellcheck` over the `run:`
blocks:
```bash
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.7 \
  -config-file .github/actionlint.yaml .github/workflows/*.yml
```

### Gate 12: Documentation Drift Check
Fails when the documentation, the Helm chart or the Grafana dashboard stops
describing the code:
```bash
go run ./scripts/docsync
```

It checks six things:

1. every flag the Helm chart passes is a flag the command has, and every
   value the chart sets matches the default the code uses;
2. every metric the Grafana dashboard queries is declared, in a query that
   matches the metric's type, and every declared metric is on the
   dashboard;
3. every flag the reference documentation mentions exists;
4. every endpoint the API reference lists is one the server registers, and
   every route the server registers is in the reference;
5. the pinned versions in `dependencies.md` are the ones the module files
   hold, and the number of metrics the documents claim is the real one;
6. every document is listed in the documentation index.

The three things it protects against had all drifted: the chart passed
flags the command no longer had and scraped the wrong port for the
metrics, the dashboard queried 27 of the 133 declared metrics, and the API
reference described a set of endpoints that never existed.

The check runs in the lint job, and `go test ./scripts/docsync/` covers
it as well.

---

## 8. Summary of Execution Commands

| Action | Command Line |
|:---|:---|
| **Run Complete CI/CD Suite** | `./scripts/run_all_tests.sh` |
| **Record Baseline Benchmark** | `./scripts/bench_gate.sh measure ./internal/simd > baseline.txt` |
| **Record Candidate Benchmark** | `./scripts/bench_gate.sh measure ./internal/simd > candidate.txt` |
| **Compare A/B Regression** | `./scripts/bench_gate.sh compare baseline.txt candidate.txt 1.2` |
| **Verify Zero-Allocation Gate** | `./scripts/bench_gate.sh check-zero-alloc ./internal/simd` |
| **Run CUDA Unit Tests** | `go test -v -tags cuda -count=1 ./internal/device` |
