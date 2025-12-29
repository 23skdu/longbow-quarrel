# Benchmarking Longbow-Quarrel

This document describes the benchmark and validation scripts available for testing the longbow-quarrel inference engine.

## Available Scripts

### 1. benchmark_compare.sh

**Purpose**: Compare performance between longbow-quarrel and llama.cpp reference implementation.

**Location**: `scripts/benchmark_compare.sh`

**Usage**:

```bash
./scripts/benchmark_compare.sh [OPTIONS]
```

**Options**:

- `-m, --model PATH` - Path to GGUF model file (default: uses smollm2 from ~/.ollama)
- `-p, --prompt TEXT` - Prompt to use for generation (default: "The capital of France is")
- `-n, --tokens NUM` - Number of tokens to generate (default: 32)
- `--profile` - Enable CPU profiling (saves to cpu.pprof)

**Example**:

```bash
# Basic comparison with default settings
./scripts/benchmark_compare.sh

# Custom prompt and token count
./scripts/benchmark_compare.sh -p "Once upon a time" -n 64

# Enable profiling for performance analysis
./scripts/benchmark_compare.sh --profile
```

**Output**:

- Throughput comparison (tokens/second) between llama.cpp and longbow-quarrel
- Performance ratio showing quarrel's performance relative to llama.cpp
- Sample generated text from quarrel
- Optional: CPU profile saved to `cpu.pprof` (use with `go tool pprof`)

### 2. validity_check.py

**Purpose**: Validate output correctness by comparing longbow-quarrel generations with `llama.cpp` (via `llama-completion`) on factual questions.

**Location**: `scripts/validity_check.py`

**How It Works**:

1. Loads benchmark questions from [docs/generic_questions.md](file:///Users/rsd/REPOS/longbow-quarrel/docs/generic_questions.md)
2. Runs each question through both quarrel and `llama-completion`
3. Calculates Jaccard similarity between outputs
4. Classifies results as MATCH, PARTIAL, or MISMATCH.

> [!NOTE]
> The script defaults to comparing the first 20 questions for a quick sanity check but can be expanded to the full 100-question set.

## Performance Targets

Based on testing with `SmolLM2-135M` on Apple Silicon:

**Throughput**:

- Target: > 100 tokens/second
- Current (FP16): ~300+ tokens/second (Metal Backend)
- Current (Q4_K/Q6_K): High-performance quantized inference validated for correctness.

**Correctness**:

- Target: > 80% match rate with llama.cpp
- Similarity: > 0.70 average Jaccard similarity validated across FP16 and K-Quants.

## Profiling

For detailed performance analysis:

```bash
# Run with profiling enabled
./scripts/benchmark_compare.sh --profile

# Analyze CPU profile
go tool pprof -http=:8080 cpu.pprof
```

This will open an interactive web interface showing:

- CPU hotspots
- Function call graphs
- Flame graphs for visualization
- Detailed execution statistics

## Continuous Integration

These benchmarks should be run:

- Before merging performance-related changes
- After major refactoring
- Before each release candidate
- When validating correctness fixes

The validity check should show no regressions in correctness, and benchmark_compare should maintain or improve throughput relative to previous versions.
