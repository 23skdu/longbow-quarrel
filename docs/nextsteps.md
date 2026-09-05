# Longbow-Quarrel - Next Steps & Roadmap

## v0.3.0 Improvement Plan

### Quality & Correctness Fixes
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P0 | BF16 tensor dequantization | `internal/gguf/dequant.go` | Add `GGMLTypeBF16` handling in `decodeTensorData` — currently silently zeros weights for Gemma 3, Mistral Nemo |
| P0 | F16 IEEE 754 bit-conversion | `internal/gguf/dequant.go` | Rewrite `DequantizeF16` using proper IEEE 754 half-to-float (currently uses integer division) |
| P0 | Top-P sampling correctness | `internal/engine/engine_cpu.go` | Fix `applyTopPCPU` to sort by CDF and zero tokens outside nucleus |
| P0 | Chat template rendering | `internal/engine/prompt_wrapper.go` | Read and apply `tokenizer.chat_template` Jinja2 string from GGUF KV metadata |
| P0 | CPU LoRA adapter loading | `internal/engine/engine_cpu.go`, `internal/engine/lora.go` | Implement merge-on-load strategy for CPU path |
| P0 | PromptCache population | `internal/engine/engine_cpu.go`, `internal/engine/engine_cuda.go` | Call `Insert` after `CompleteSequence` in CPU and CUDA batch loops |

### Performance & Scalability
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P0 | Zero-copy MatVec for Q2_K/Q3_K/Q4_0/Q5_0/Q5_K | `internal/gguf/dequant_simd.go`, `internal/engine/cpu_weights.go` | Add parallel kernels; eliminate fallback dequantization |
| P0 | Parallel MatMul | `internal/simd/simd.go` | Rewrite with outer-row parallelism and SIMD dot products |
| P0 | Speculative decoding ForwardDraft | `internal/engine/engine_cpu.go` | Implement real multi-token forward pass returning logit vectors |
| P0 | Sliding-window KV cache wiring | `internal/engine/engine_cpu.go`, `internal/engine/kv_cache_sliding_window.go` | Detect `sliding_window` in config and use `SlidingWindowKVCache` + attention sink |

### Architecture & Extensibility
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P1 | Remote worker engine completion | `internal/engine/remote.go` | Implement `Infer`, `InferWithCallback`, `ForwardShard` for multi-node gRPC |
| P1 | NCCL integration | `internal/device/cuda.go` | Replace stubs with actual NCCL library linking for multi-GPU allreduce |
| P1 | TPU/XLA backend validation | `cmd/quarrel/main_tpu.go` | Test and validate Google TPU inference path |
| P1 | Grammar-constrained sampling expansion | `internal/sampler/` | Add support for regex and CFG grammars beyond JSON |

### Developer Experience & CI
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P2 | Consolidate test scripts | `scripts/` | Merge redundant test/benchmark scripts into unified runner |
| P2 | Add coverage gates | `.github/workflows/ci.yml` | Enforce minimum coverage threshold on PRs |
| P2 | Structured gosec output | `.github/workflows/ci.yml` | JSON-formatted gosec results for automated triage |
| P2 | Release automation | `Makefile` | Single `make release` target with changelog, Docker push, Helm bump |

### Observability & Operations
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P2 | Per-model metric labels | `internal/metrics/` | Add model name and quantization type as Prometheus labels |
| P2 | Distributed tracing for batch inference | `internal/telemetry/` | Span lifecycle covering prefill, decode, and cache lookup |
| P2 | Memory pressure alerting | `internal/monitoring/` | Proactive alert when GPU VRAM or CPU RSS exceeds threshold |

---

## P0 Blockers for This Release (v0.3.0 — Correctness, Completeness & Throughput)

Following deep code analysis and verification of all v0.2.0 P0 items, the following 10 P0 blockers target the gaps discovered. These are ranked by user-visible impact.

---

### 1. Zero-Copy MatVec for Q2_K, Q3_K, Q4_0, Q5_0, Q5_K — Missing Parallel Kernels
- **Impact Area:** Performance & Memory (CPU inference throughput for non-Q8/Q4_K/Q6_K quants)
- **Severity:** P0 Blocker (Partial Inference Correctness)
- **Location:** `internal/engine/cpu_weights.go` (`MatVec`), `internal/gguf/dequant_simd.go`
- **Description:** `CPUWeights.MatVec` dispatches zero-copy parallel kernels for Q8_0, Q4_K, and Q6_K, but **falls back to `decodeTensorData` + `simd.MatVecMul`** (full dequantization into RAM) for Q2_K, Q3_K, Q4_0, Q5_0, and Q5_K. This causes OOM on large Q2/Q3 models and eliminates any throughput gain over the naive path.
- **Target Implementation:**
  - Add `MatVecMulQ2_K`, `MatVecMulQ3_K`, `MatVecMulQ4_0`, `MatVecMulQ5_K` to `internal/gguf/dequant_simd.go` following the same parallel-worker pattern as `MatVecMulQ4_K`.
  - Dispatch all new kernels from `cpu_weights.go:MatVec` using the raw tensor path, eliminating fallback dequantization.
  - Add benchmarks and fuzz tests for each new kernel.

---

### 2. PromptCache `Insert` Not Called in `CPUEngine` or `cudaEngine` Batch Loops
- **Impact Area:** Performance (TTFT for shared/repeated prompts)
- **Severity:** P0 Blocker (Prefix Cache Never Populated)
- **Location:** `internal/engine/engine_cpu.go` (`runBatchLoop`), `internal/engine/engine_cuda.go` (`runBatchLoop`)
- **Description:** Both `CPUEngine.runBatchLoop` and `cudaEngine.runBatchLoop` correctly call `BatchManager.Step(..., e.PromptCache)` to **look up** prefix hits, but **never call `e.PromptCache.Insert`** after completing a sequence. The radix-tree cache builds correctly only in the Metal/generic engine (`engine.go:1261`). CPU and CUDA engines never populate the cache, so TTFT gains are zero across all runs.
- **Target Implementation:**
  - After `CompleteSequence`, call `e.PromptCache.Insert(seq.Tokens[:promptLen], blocks)` in both CPU and CUDA batch loops, guarded by a `GetBlockTable` call.
  - Add an integration test verifying second-call TTFT reduction.

---

### 3. BF16 (`GGMLTypeBF16`) Tensor Dequantization & MatVec — Completely Absent
- **Impact Area:** Accuracy & Compatibility (newer Gemma, Mistral, Qwen models)
- **Severity:** P0 Blocker (Silent Weight Corruption)
- **Location:** `internal/gguf/dequant.go`, `internal/engine/cpu_weights.go` (`decodeTensorData`)
- **Description:** `decodeTensorData` handles F32, F16, Q8_0, Q4_K, Q6_K, Q5_0, Q2_K, Q3_K, Q5_K, and IQ4_XS. **BF16 (`GGMLTypeBF16`) is completely absent**. Gemma 3 (2B/9B), Mistral Nemo, and many modern GGUF files use BF16 norms and embeddings. When encountered, these fall through to `make([]float32, numElements)` returning an all-zero tensor — silently zeroing weights and destroying all inference output.
- **Target Implementation:**
  - Add `DequantizeBF16(data []byte, numElements int) []float32` to `dequant.go` with correct bfloat16 → float32 bit manipulation.
  - Dispatch from `decodeTensorData` and `DequantizeBlock`.
  - Add `MatVecMulBF16` with parallel workers.
  - Add fuzz test against known BF16 reference values.

---

### 4. `PromptWrapper.Wrap` — Static Template String Substitution Instead of Jinja2/GGUF Chat Template
- **Impact Area:** Coherence & Instruction-Following (chat models)
- **Severity:** P0 Blocker (Incorrect Prompt Format)
- **Location:** `internal/engine/prompt_wrapper.go`, `internal/tokenizer/tokenizer.go`
- **Description:** `PromptWrapper` applies a hardcoded Go template string (`{{ .System }}{{ .User }}: {{ .Input }}{{ .Response }}:`). Modern instruction-tuned models embed a `tokenizer.chat_template` Jinja2 string in the GGUF KV metadata. Quarrel does not read or apply this field. As a result, Llama 3, Qwen 3, Gemma 3, and Mistral chat models receive malformed prompts without `<|im_start|>`, `<|begin_of_text|>`, `[INST]` delimiters, producing incoherent outputs.
- **Target Implementation:**
  - Read `tokenizer.chat_template` from GGUF KV in `tokenizer.go` / `engine_utils.go` and expose it on `Tokenizer`.
  - Implement a lightweight Jinja2-subset renderer (covering `{% if %}`, `{% for %}`, `{{ var }}`, `{% set %}`) in `prompt_wrapper.go` sufficient to cover Llama 3, Qwen, Gemma, Mistral templates.
  - Auto-detect and apply the embedded template when available; fall back to existing simple wrap.
  - Add tests with real template strings extracted from known GGUF files.

---

### 5. Speculative Decoding `ForwardDraft` (CPUEngine) Returns Stub Embeddings — Not Real Logits
- **Impact Area:** Performance (tokens/sec for speculative decoding paths)
- **Severity:** P0 Blocker (Speculative Acceptance Rate = 0)
- **Location:** `internal/engine/engine_cpu.go` (`ForwardDraft`)
- **Description:** `CPUEngine.ForwardDraft` returns `draftCount=4` copies of the last token's embedding vector — **not logit distributions**. `SpeculativeManager.rejectSample` samples from target logits and compares against draft token probability; if draft logits are embeddings, the acceptance ratio is undefined and tokens are rejected 100% of the time, yielding no speedup over serial generation.
- **Target Implementation:**
  - Implement `ForwardDraft` as a multi-token forward pass returning one logit vector per draft token position using the `forward()` method and a shared KV cache.
  - Support `draftK` 1–8 tokens with configurable depth.
  - Add an integration test measuring speculative acceptance rate (target ≥ 50% with matched draft/target model).

---

### 6. CPU LoRA Adapter Loading — `LoadAdapter` Returns `nil` Without Applying Weights
- **Impact Area:** Accuracy (fine-tuned model correctness)
- **Severity:** P0 Blocker (LoRA Ignored on CPU)
- **Location:** `internal/engine/engine_cpu.go` (`LoadAdapter`), `internal/engine/lora.go`
- **Description:** `CPUEngine.LoadAdapter` returns `nil` immediately with comment "Not supported on CPU for now". The CUDA engine has a working `LoRAManager.LoadAdapter`. Fine-tuned models using LoRA (e.g., instruction-tuned Llama derivatives) produce base-model outputs on CPU inference paths.
- **Target Implementation:**
  - Implement `LoRAManager.LoadAdapter` → `ApplyLoRA` for the CPU path, merging LoRA delta weights (`A × B × α/r`) into the `CPUWeights` matrices at load time (merge-on-load strategy for zero inference-time overhead).
  - Support `.safetensors` and GGUF sidecar LoRA formats.
  - Add tests verifying that a loaded LoRA changes output logits.

---

### 7. `applyTopPCPU` Logic Is Incorrect — Truncates Probabilities Not Tokens
- **Impact Area:** Accuracy & Coherence (sampling quality)
- **Severity:** P0 Blocker (Broken Top-P in Legacy CPU Path)
- **Location:** `internal/engine/engine_cpu.go` (`applyTopPCPU`)
- **Description:** `applyTopPCPU` (used in `softmaxCPU` legacy path) computes a probability-weighted rescaling on the original logit values rather than zeroing out tokens below the nucleus threshold. The correct Top-P algorithm should sort candidates by probability, accumulate a CDF, and zero (`-Inf`) all tokens outside the nucleus. The current implementation neither sorts nor correctly zeroes tokens, distorting the distribution for all CPU-path callers that use this function.
- **Target Implementation:**
  - Fix `applyTopPCPU` to correctly sort tokens by softmax probability, accumulate the CDF, and set `logits[i] = -Inf` for all tokens not in the top-P nucleus.
  - Verify the fix produces identical results to `applyTopP` in `sampler.go`.
  - Add a property-based fuzz test verifying nucleus sum ≥ p after filtering.

---

### 8. `DequantizeF16` Float16 Bit-Conversion Is Wrong — Uses Integer Division
- **Impact Area:** Accuracy (F16 model weights corruption)
- **Severity:** P0 Blocker (Silently Incorrect F16 Decoding)
- **Location:** `internal/gguf/dequant.go` (`DequantizeF16`)
- **Description:** The current implementation reads the 16-bit float as a `uint16` and divides by `32767.0` — treating it as a **fixed-point integer**, not an IEEE 754 half-precision float. This produces completely wrong values for any F16 tensor (embeddings, norms, attention weights in F16 GGUF files). The correct approach uses the IEEE 754 bit pattern: sign, 5-bit exponent (bias 15→127), 10-bit mantissa.
- **Target Implementation:**
  - Rewrite `DequantizeF16` using proper IEEE 754 half-to-float conversion (bit manipulation: `sign | ((exp + 112) << 23) | (mantissa << 13)`), handling subnormals, infinities, and NaN.
  - Cross-validate against Go's `math/bits` or a reference implementation with known F16 bit patterns.
  - Add table-driven tests with boundary values (0.0, -0.0, 1.0, -1.0, max F16, min positive F16, NaN, ±Inf).

---

### 9. `MatMul` in `simd.go` Is O(N³) Scalar — Never Parallelized or SIMD-Accelerated
- **Impact Area:** Performance (batch prefill throughput, training/eval forward pass)
- **Severity:** P0 Blocker (Prefill Speed Bottleneck for Multi-Token Batches)
- **Location:** `internal/simd/simd.go` (`MatMul`)
- **Description:** `simd.MatMul` uses a plain triple-nested scalar loop. It is called during multi-token batch prefill and attention score computation for longer sequences. With `MatVecMul` parallelized since P0-8, `MatMul` is now the dominant bottleneck for any prefill batch with more than one token. It is neither SIMD-vectorized nor multi-threaded.
- **Target Implementation:**
  - Rewrite `simd.MatMul` with outer-row parallelism using `runtime.NumCPU()` goroutines (same pattern as `MatVecMul`).
  - Inner loop should use `VecDotF32` for SIMD-accelerated dot products.
  - Threshold parallelism at `rowsA * colsB >= 1024` to avoid goroutine overhead for small matrices.
  - Add benchmark showing ≥4× speedup for 256×256 × 256×256 multiply.

---

### 10. Sliding-Window KV Cache Not Wired into `CPUEngine` Attention — Attention Sink Absent
- **Impact Area:** Accuracy & Context Length (Mistral, Gemma, Falcon models with sliding window)
- **Severity:** P0 Blocker (Indefinite Context Corruption for Long Sequences)
- **Location:** `internal/engine/kv_cache_sliding_window.go`, `internal/engine/engine_cpu.go` (`ApplyLayerCPUKV`)
- **Description:** `SlidingWindowKVCache` is implemented and has metrics instrumentation, but `CPUEngine.ForwardBatch` always allocates a standard `CPUKVCache` (`NewCPUKVCache`). Models with `attention.sliding_window` set in their GGUF (Mistral 7B, Gemma 2) will accumulate unbounded KV caches in the CPU path, eventually causing OOM and ignoring the architectural sliding-window constraint. Additionally, **attention sink tokens** (position 0 kept in all windows) are not preserved.
- **Target Implementation:**
  - In `NewCPUEngine`, detect `cfg.SlidingWindowSize > 0` and use `SlidingWindowKVCache` for the per-sequence cache.
  - Implement attention sink: always retain position 0 token's K/V across window evictions.
  - Ensure `ApplyLayerCPUKV` receives window-bounded K/V views via a `CacheView` interface.
  - Add stress test verifying memory stability for 10,000-token sequences with `windowSize=512`.

---

#### Last updated: September 2026 (v0.3.0 P0 Correctness, Completeness & Throughput Plan)