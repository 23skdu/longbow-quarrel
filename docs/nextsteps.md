# Longbow-Quarrel - Next Steps & Roadmap

## P0 Blockers for This Release (Performance, Accuracy & Coherence Improvement Plan)

Following deep code analysis, profiling, and model deployment testing across local GGUF models, the following 10 P0 blockers have been established to fundamentally elevate Quarrel's inference speed, contextual accuracy, and semantic coherence:

---

### 1. Sequential Prompt Prefill & Context Awareness in `CPUEngine`
- **Impact Area:** Coherence, Accuracy, Memory
- **Severity:** P0 Blocker (Prompt Amnesia)
- **Location:** `internal/engine/engine_cpu.go`
- **Description:** Currently, `CPUEngine.forward(tokens)` only processes `tokens[len(tokens)-1]` and discards all preceding prompt tokens $0 \dots N-2$. The model generates responses without conditioning on the prompt text.
- **Target Implementation:**
  - Implement full sequential prompt prefill in `CPUEngine.forward` to process all prompt tokens through the transformer stack.
  - Track sequence position `pos` across prefill and incremental generation steps.
  - Return output logits conditioned on the complete prompt context.

---

### 2. Rotary Position Embeddings (RoPE) in CPU Layer Pipeline
- **Impact Area:** Accuracy & Semantic Quality
- **Severity:** P0 Blocker (Missing Positional Geometry)
- **Location:** `internal/engine/cpu_weights.go`, `internal/simd/`
- **Description:** `ApplyLayerCPU` performs Q, K, V projections and RMS normalization, but completely omits rotary position embeddings ($Q_{\text{rot}}, K_{\text{rot}}$). Without RoPE on CPU, tokens lose all positional relationships, degrading grammar and coherence.
- **Target Implementation:**
  - Call `simd.Rope` on $Q$ and $K$ heads at token position `pos` with frequency base `cfg.RopeTheta` inside `ApplyLayerCPU`.
  - Ensure correct head dimension and rotary frequency calculation across Llama, Qwen, Mistral, and Gemma.

---

### 3. Layer-Level KV Cache Accumulation for Multi-Token CPU Attention
- **Impact Area:** Coherence & Context Retention
- **Severity:** P0 Blocker (No Attention Memory)
- **Location:** `internal/engine/cpu_weights.go`, `internal/engine/engine_cpu.go`
- **Description:** `ApplyLayerCPU` computes self-attention solely on the current token query $Q$ with current key/value $K_t, V_t$. Prior keys and values are neither stored nor retrieved, preventing the model from attending to preceding text.
- **Target Implementation:**
  - Add sequence-aware per-layer Key/Value cache management (`LayerKVCache`) in CPU execution.
  - Store $K_t, V_t$ at each position and layer; compute scaled dot-product attention across all cached keys $K_{0:t}$ and values $V_{0:t}$.
  - Support sequence resetting and rollback upon generation completion.

---

### 4. Multi-Model Dynamic EOS & Stop Token Detection
- **Impact Area:** Coherence & Termination Stability
- **Severity:** P0 Blocker (Runaway Generation Loop)
- **Location:** `internal/tokenizer/tokenizer.go`, `internal/engine/engine_utils.go`, `internal/engine/engine_cpu.go`, `internal/engine/engine_cuda.go`
- **Description:** Inference loops currently hardcode `token == 2` as the only end-of-sequence condition. Qwen models (EOS=151643, 151645), Gemma models (EOS=1, 107), and LLaMA 3 models (EOS=128001, 128009) never terminate naturally, causing endless repetitive rambling until hitting `MaxTokens`.
- **Target Implementation:**
  - Extract `tokenizer.ggml.eos_token_id` and architecture-specific EOS tokens in `ExtractModelConfig` and `tokenizer.Tokenizer`.
  - Expose `Tokenizer.IsEOS(tokenID int) bool` and `Tokenizer.GetEOSTokenIDs() []int`.
  - Check dynamic EOS tokens in both CPU and CUDA generation loops.

---

### 5. Advanced Sampling: Presence, Frequency, and Min-P Penalties
- **Impact Area:** Accuracy, Creativity & Vocabulary Control
- **Severity:** P0 Blocker (Topic Looping & Repetition)
- **Location:** `internal/engine/sampler.go`, `internal/engine/sampler_config.go`
- **Description:** `SamplerConfig` currently lacks presence and frequency penalties (OpenAI/llama.cpp standards). Repetition penalty alone fails to prevent semantic repetition and word loops.
- **Target Implementation:**
  - Add `PresencePenalty float64` and `FrequencyPenalty float64` to `SamplerConfig`.
  - Track token occurrence frequencies across `history` and apply linear penalties to logits before probability transformation.
  - Enforce Min-P filtering relative to top candidate probability.

---

### 6. Empty Logit Slice Guards, Finite Clamping, and Softmax Stability
- **Impact Area:** Stability & Numerical Precision
- **Severity:** P0 Blocker (Panic & NaN Ingestion)
- **Location:** `internal/engine/sampler.go`, `internal/simd/simd.go`
- **Description:** `applyTemperatureAndSoftmax` indexes `logits[0]` unconditionally without checking `len(logits) == 0`. Unclamped extreme logits (>1e4 or <-1e4) cause `math.Exp` overflow/underflow to $\pm\infty$ or NaN.
- **Target Implementation:**
  - Guard empty logit slices across all sampler helper functions.
  - Implement numerical clamping $[-60.0, 60.0]$ prior to exponentiation.
  - Provide fallback to greedy argmax with warning logging if all probabilities are zero or NaN.

---

### 7. Radix-Tree `PromptCache` Integration in `CPUEngine` & `cudaEngine`
- **Impact Area:** Performance (Time-to-First-Token / TTFT)
- **Severity:** P0 Blocker (Redundant Prompt Computations)
- **Location:** `internal/engine/engine_cpu.go`, `internal/engine/engine_cuda.go`
- **Description:** `PromptCache` is implemented in `internal/engine/prompt_cache.go` with LRU eviction and prefix matching, but `CPUEngine` and `cudaEngine` pass `nil` into `BatchManager.Step`, disabling prefix caching entirely.
- **Target Implementation:**
  - Instantiate `PromptCache` on `CPUEngine` and `cudaEngine`.
  - Pass `e.PromptCache` into `BatchManager.Step`.
  - Reuse cached prompt prefix blocks across multi-turn conversations and shared prompts.

---

### 8. Multi-Threaded Parallel Matrix-Vector Multiplication for Large Matrices
- **Impact Area:** Performance (CPU Generation Throughput)
- **Severity:** P0 Blocker (Single-Core CPU Bottleneck)
- **Location:** `internal/engine/cpu_weights.go`, `internal/gguf/dequant.go`
- **Description:** Single-vector matrix multiplication runs sequentially on a single CPU thread, severely restricting token throughput for large models (4B-8B) with large hidden dimensions.
- **Target Implementation:**
  - Partition matrix rows across worker goroutines using `runtime.NumCPU()` when row count $M \ge 256$.
  - Parallelize zero-copy kernels `MatVecMulQ8_0`, `MatVecMulQ4_K`, and `MatVecMulQ6_K`.
  - Instrument with throughput latency metrics.

---

### 9. Vectorized FP32 Embedding Dot-Products and Fused Multiply-Add
- **Impact Area:** Performance & SIMD Utilization
- **Severity:** P0 Blocker (Unvectorized Attention Hotpaths)
- **Location:** `internal/engine/cpu_weights.go`, `internal/simd/`
- **Description:** `vecDot` and `vecFMA` in `cpu_weights.go` use basic scalar unrolled Go loops rather than utilizing AVX2 / AVX-512 vector extensions.
- **Target Implementation:**
  - Replace `vecDot` and `vecFMA` with vectorized routines `simd.VecDotF32` and `simd.VecFMAF32`.
  - Provide 8-lane AVX2 and 16-lane AVX-512 implementations with auto-fallback to portable scalar implementations.

---

### 10. Complete CLI Sampling Parameter Ingestion & Flag Aliases
- **Impact Area:** Usability, Accuracy, Coherence
- **Severity:** P0 Blocker (Ignored User Flags)
- **Location:** `cmd/simple/main.go`, `cmd/quarrel/main.go`
- **Description:** `cmd/simple/main.go` declares `--temp` and `--topk` flags but hardcodes `Temperature: 0.8, TopK: 40` when instantiating `SamplerConfig`. In `cmd/quarrel/main.go`, `--gpu-layers` is assigned to a blank identifier `_`.
- **Target Implementation:**
  - Bind all CLI sampling flags (`--temp`, `--topk`, `--topp`, `--rep-penalty`, `--presence-penalty`, `--frequency-penalty`, `--minp`, `--seed`) into `SamplerConfig`.
  - Properly handle `--gpu-layers` and `-ngl` aliases.
  - Support streaming output token callbacks in `cmd/simple`.

---

## Completed in v0.2.0 (Archived Milestones)

All 11 milestone tasks from Phase 9 and v0.2.0 have been implemented, validated with unit/fuzz tests, security-audited, and deployed:
1. **Zero-Copy Quantized Inference (Fix CPU RAM Exhaustion / OOM)** (`internal/engine/cpu_weights.go`, `internal/gguf/dequant.go`)
2. **Universal Multi-Engine Model Cache Resolver** (`internal/models/resolver.go`)
3. **Multi-Architecture Dynamic Metadata Extraction** (`internal/engine/engine_utils.go`)
4. **CUDA Double-Allocation Elimination** (`cmd/quarrel/main.go`, `internal/device/cuda.go`)
5. **Automated CUDA Kernel Compilation in Makefile** (`Makefile`)
6. **Qwen 3.5 Hybrid Linear/Full Attention Architecture Support** (`internal/engine/cpu_weights.go`)
7. **Modernized Benchmarks & IDE Warning Cleanup** (`internal/engine/`, `internal/gguf/`)
8. **Partial GPU Layer Offloading (Hybrid VRAM / RAM)** (`internal/engine/engine_cuda.go`, `internal/engine/cpu_weights.go`)
9. **TurboQuant AVX-512 & AVX2 Kernels** (`internal/simd/turboquant_avx512.c`, `internal/simd/turboquant_avx2.c`)
10. **NEON Kernels Unit Tests & Benchmarks** (`internal/simd/turboquant_neon.c`)
11. **AVX-512 / SIMD GGUF Dequantization Kernels (Q4_K, Q6_K)** (`internal/gguf/dequant_simd.go`)

---

#### Last updated: September 2026 (v0.2.0+ P0 Performance, Accuracy & Coherence Plan)