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

**Purpose**: Validate output correctness by comparing longbow-quarrel generations with llama.cpp reference on factual questions.

**Location**: `scripts/validity_check.py`

**Dependencies**:

```bash
# Python 3.6+ (uses dataclasses and pathlib)
# No additional pip packages required
```

**Usage**:

```bash
./scripts/validity_check.py
```

**Configuration**:
Edit the script to configure paths:

```python
quarrel_bin = "./quarrel"              # Path to quarrel binary
llama_bin = "/opt/homebrew/bin/llama-cli"  # Path to llama.cpp CLI
model_path = "path/to/model.gguf"      # GGUF model file
```

**How It Works**:

1. Loads benchmark questions from [docs/generic_questions.md](file:///Users/rsd/REPOS/longbow-quarrel/docs/generic_questions.md)
2. Runs each question through both quarrel and llama.cpp
3. Calculates Jaccard similarity between outputs
4. Classifies results as:
   - **Match** (similarity >= 70%) - Outputs are very similar
   - **Partial** (similarity >= 40%) - Outputs have some overlap
   - **Mismatch** (similarity < 40%) - Outputs differ significantly

**Output**:

```
======================================================================
Longbow-Quarrel Validity Check
======================================================================

Loaded 100 questions

[1/10] What is the capital of France?
  Similarity: 0.85 (match)

[2/10] Who wrote Romeo and Juliet?
  Similarity: 0.72 (match)

...

======================================================================
Summary
======================================================================
Match:    8/10 (80.0%)
Partial:  2/10 (20.0%)
Mismatch: 0/10 (0.0%)

Average Similarity: 0.763
```

## Benchmark Questions

The benchmark questions are curated factual questions organized by category:

**Location**: [docs/generic_questions.md](file:///Users/rsd/REPOS/longbow-quarrel/docs/generic_questions.md)

**Categories** (100 questions total):

1. Geography (10 questions)
2. History (10 questions)
3. Science (10 questions)
4. Literature (10 questions)
5. Mathematics (10 questions)
6. Technology (10 questions)
7. Sports (10 questions)
8. Arts (10 questions)
9. General Knowledge (10 questions)
10. Mixed Topics (10 questions)

These questions are designed to test:

- Factual accuracy
- Common knowledge recall
- Consistent response formatting
- Model behavior across different domains

## Performance Targets

Based on testing with smollm2-135M on Apple Silicon:

**Throughput**:

- Target: > 100 tokens/second
- Current: ~298 tokens/second (116% of llama.cpp reference)

**Correctness**:

- Target: > 80% match rate with llama.cpp
- Similarity: > 0.70 average Jaccard similarity

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
