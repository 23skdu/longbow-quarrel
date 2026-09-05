# Longbow-Quarrel - Next Steps & Roadmap

## P0 Blockers for Next Release (Improvement Plan)

Following deep code analysis, profiling, and model deployment testing with local GGUF models (`Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf`), the following P0 blockers have been identified and implemented to ensure stability, multi-architecture compatibility, and memory safety:

---

### 1. Memory-Efficient Zero-Copy Quantized Inference (Fix CPU RAM Exhaustion / OOM) [COMPLETED]
- **Severity:** P0 Blocker (System OOM / Thrashing / Freezing)
- **Location:** `internal/engine/engine_cpu.go`, `internal/gguf/dequant.go`, `internal/gguf/dequant_test.go`
- **Root Cause & Fix:**
  - Previously, `loadCPUWeights` dequantized every single tensor upfront into `[]float32` slices on the Go heap. For a 4B parameter model in Q8_0 (4.4 GB on disk), dequantizing all weights upfront allocated **17.9 GB of heap memory**, exceeding physical RAM (22 GB) when desktop applications are open and causing the system to swap thrash and invoke the kernel OOM killer.
  - Implemented `gguf.MatVecMulQ8_0(data []byte, vector []float32, rows, cols int) []float32`, performing direct matrix-vector dot products on the mmapped quantized bytes without materializing dequantized weights on the heap.
  - Implemented row-on-demand token embedding lookup in `GetTokenEmbedding(tokenId, hiddenSize)`, reading only the requested token's embedding vector (10 KB) directly from mmap.
  - Retained `F32` small RMSNorm weights while keeping all 2D projection matrices (`AttnQ`, `AttnK`, `AttnV`, `AttnO`, `FfnGate`, `FfnUp`, `FfnDown`, `AttnQKV`, `AttnGate`, `SSMOut`, `Output`) in zero-copy raw mmap format.
  - **Result:** Reduced memory usage from **17.9 GB down to < 20 MB** (99.9% reduction). Inference runs instantly without OOM or disk swapping (verified: 5 tokens generated in 3.65s on CPU with 17 GiB RAM free).

---

### 2. Universal Multi-Engine Model Cache Resolver [COMPLETED]
- **Severity:** P0 Blocker (Model Discovery Broken)
- **Location:** `internal/models/resolver.go`, `internal/models/resolver_test.go`
- **Root Cause & Fix:**
  - Quarrel previously only resolved models from `~/.ollama/models/`. Any model downloaded via `llama.cpp` (`~/.cache/llama.cpp/`), `llmfit` (`~/.cache/llmfit/models/`), or `huggingface-cli` (`~/.cache/huggingface/hub/`) could not be discovered unless the user provided a full absolute path.
  - Implemented `models.ResolveModelPath(input string) (string, error)` with support for:
    1. Exact filesystem paths (relative and absolute).
    2. Universal cache directories: `~/.cache/llmfit/models/`, `~/.cache/llama.cpp/`, `~/.cache/huggingface/hub/`, and `~/.ollama/models/`.
    3. Case-insensitive prefix and substring matching (e.g., `-model Huihui-Qwen3.5` resolves immediately to `/home/rsd/.cache/llmfit/models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf`).
  - Integrated into both `cmd/simple/main.go` and `cmd/quarrel/main.go`.
  - **Verification:** Unit tests in `internal/models/resolver_test.go` pass; tested CLI commands resolve model accurately.

---

### 3. Multi-Architecture Dynamic Metadata Extraction [COMPLETED]
- **Severity:** P0 Blocker (Architecture Breakdown / Silent Degradation)
- **Location:** `internal/engine/engine_utils.go`, `internal/engine/engine_cuda.go`, `internal/device/cuda.go`
- **Root Cause & Fix:**
  - Architecture metadata extraction in `engine_cpu.go`, `engine_cuda.go`, `cuda.go`, and `cmd/quarrel/main.go` was hardcoded to `llama.*` keys with brittle type assertions `.(uint32)`.
  - For models like `qwen35` or other architectures, `block_count`, `embedding_length`, `head_count`, and `vocab_size` failed to match, defaulting `dim`, `heads`, and `layers` to fallback values (0 or 1), causing dummy repetitive tokens or out-of-bounds panics.
  - Implemented `ExtractModelConfig(f *gguf.GGUFFile) config.Config` which dynamically inspects `general.architecture` (e.g. `qwen35`, `qwen2`, `gemma4`, `mistral`, `llama`) and resolves architecture-prefixed keys with fallback to standard keys, supporting `uint32`, `int32`, `uint64`, `int64`, `float64`, and array formats.
  - **Verification:** Tested against both Qwen 3.5 and LLaMA metadata models (`TestExtractModelConfig_Qwen35`, `TestExtractModelConfig_Llama`).

---

### 4. Eliminate CUDA Double-Allocation & VRAM Exhaustion in `cmd/quarrel` [COMPLETED]
- **Severity:** P0 Blocker (GPU Out-Of-Memory Crash)
- **Location:** `cmd/quarrel/main.go`, `internal/device/cuda.go`
- **Root Cause & Fix:**
  - `cmd/quarrel/main.go` line 92 called `ctx.NewCUDAModel(f, true, *kvCacheSize)` and line 127 called `engine.NewEngine(...)` which called `ctx.NewCUDAModel` a second time. This double-allocated the model in VRAM (allocating ~16 GB on an 8 GB GPU), causing cublas init to fail with `panic: cublasCreate failed: 3`.
  - Removed redundant `cudaModel` allocation in `cmd/quarrel/main.go` and delegated model lifecycle to `engine.NewEngine`.
  - Added explicit error checking on `C.cudaMalloc` return codes in `NewCUDAModel`, gracefully freeing previously allocated GPU tensors on allocation failure instead of proceeding with dangling null pointers.
  - Added tied embedding fallback in `GetWeightTensor` for models where `output.weight` is tied with `token_embd.weight`.

---

### 5. Automated CUDA Kernel Compilation in Makefile [COMPLETED]
- **Severity:** P0 Blocker (CUDA Build Failure)
- **Location:** `Makefile`
- **Root Cause & Fix:**
  - `make nvidia` failed with `-lcuda_kernels: No such file or directory` because `internal/device/libcuda_kernels.a` was never compiled from `cuda_kernels.cu`.
  - Added build target for `internal/device/libcuda_kernels.a` using `nvcc -c -O3 -Xcompiler -fPIC` and `ar rcs`.
  - Configured dynamic `CUDA_LDFLAGS ?= -s -w` for `nvidia-cuda` target, avoiding static linking conflicts with system NVIDIA CUDA/cuBLAS shared libraries.
  - **Verification:** `make nvidia` succeeds and compiles `bin/quarrel-linux-amd64-cuda`.

---

### 6. Qwen 3.5 Hybrid Linear/Full Attention Architecture Support [COMPLETED]
- **Severity:** P0 Blocker (Model Execution Failure on Linear/Hybrid Architectures)
- **Location:** `internal/engine/engine_cpu.go`
- **Root Cause & Fix:**
  - Qwen 3.5 utilizes a hybrid linear-attention architecture: 32 layers total, where every 4th layer (`layer % 4 == 3`: layers 3, 7, 11, 15, 19, 23, 27, 31) is Full Self-Attention with Q/K RMSNorm, and all other 24 layers are GatedDeltaNet SSM linear attention (`attn_qkv`, `ssm_conv1d`, `ssm_out`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_dt`, `ssm_norm`).
  - Implemented hybrid layer dispatch in `applyLayerCPU`:
    - Checks for full attention weights (`HasFullAttn`) and executes vectorized multi-head attention with per-head Q/K RMS normalization.
    - Checks for SSM weights (`HasSSM`) and executes linear projection and GatedDeltaNet activation.
    - Executes SwiGLU FFN projection across both full and linear attention layers.

---

### 7. Modernize Benchmarks & Clean IDE Warnings [COMPLETED]
- **Severity:** Code Quality & Maintenance
- **Location:** `internal/engine/engine_cpu_test.go`, `internal/engine/prompt_wrapper_test.go`, `internal/gguf/quantize_benchmark_test.go`, `internal/engine/prompt_cache.go`, `internal/engine/speculative.go`, `internal/engine/kv_cache_turboquant_test.go`
- **Fix:**
  - Modernized benchmark loops using Go 1.24+ `b.Loop()`.
  - Refactored `PromptCache.Evict` to reuse `removeLRU()` and exported `CurrentBlockCount()`.
  - Silenced unused parameter `cfg` in `speculative.go:rejectSample`.
  - Omitted redundant nil check before slice length in `kv_cache_turboquant_test.go`.

---

## Active Roadmap: SIMD & Acceleration Optimization (Phase 9)

| Task | Priority | Status | Target Location |
|------|----------|--------|-----------------|
| Partial GPU Layer Offloading (Split layers across VRAM and RAM) | P1 | NOT_STARTED | `internal/engine/engine_cuda.go` |
| Complete TurboQuant AVX-512 Kernels | P1 | PARTIAL | `internal/simd/turboquant_avx512.c` |
| NEON Kernels Unit Tests & Benchmarks | P1 | NOT_STARTED | `internal/simd/turboquant_neon_test.go` |
| AVX-512 GGUF Dequantization Kernels (Q4_K, Q6_K) | P2 | NOT_STARTED | `internal/gguf/dequant_simd.go` |

---

#### Last updated: September 2026 (Universal Model Resolver & Zero-Copy Inference Release)