# Performance Optimization Plan: Closing the Gap

## Current Task: Phase 1 - Robust Baselines & Regression

- [x] Planning Phase 1 implementation
- [x] Implement `cmd/smoke_test/regression_suite.go`
  - [x] Define baseline prompts and expected outputs
  - [x] Implement logit difference / perplexity check
  - [x] Support multi-model batch testing
  - [x] **Add multi-token coherence tests** (expanded to detect tokenization corruption)
- [x] Establish new performance baselines
  - [x] Benchmark Mistral 7B (FP16/Quant)
  - [x] Benchmark Granite 4B
  - [x] Benchmark TinyLlama 1.1B
  - [ ] Benchmark Nemotron-3-Nano (deferred - MOE model)
  - [ ] Benchmark GPT-OSS (deferred - MOE model, requires separate testing)
- [x] Update `docs/performance.md` with new baselines

## Critical Issue Identified

**Status:** Tokenization/Generation corruption RESOLVED ✅

**Root Cause:** Missing GPU synchronization in NaN/Inf clamping path
- **File:** `internal/engine.go:1545` (missing `e.Cx.Synchronize()` after async Metal kernel)
- **Impact:** Async RMSNorm write was read before Metal kernel completed, causing stale/uninitialized data in subsequent operations

**Fix Applied:** Added synchronization after pool return
**Verification:**
- ✅ Multi-token coherence: All 3 tests now COHERENT  
- ✅ KV Cache wrapping: Verified (CachePos reaches 44 with window size 32)
- ✅ No more mixed alphabet gibberish output

**Next Required Actions:**
1. ~~Debug tokenizer decoding (mixed alphabets suggests token table corruption)~~ ✅ RESOLVED
2. ~~Investigate KV cache corruption between generation steps~~ ✅ RESOLVED (no corruption - sync bug was the issue)
3. ~~Compare token-by-token output with llama.cpp to isolate issue~~ ✅ RESOLVED (sync bug was the issue, not algorithmic)

**Analysis:** The issue was NOT tokenizer corruption or KV cache algorithm error. The high MSE failures in baseline tests were caused by missing GPU synchronization that led to zero logits and gibberish output. After adding the sync call, all coherence tests pass.

**Goal:** Close the performance gap (currently ~4x-8x vs Llama.cpp) while maintaining strict coherence and correctness.
**Targets:** `mistral:latest`, `nemotron-3-nano:latest`, `tinyllama:latest`, `granite4:latest`.

## Phase 0: MOE & Nemotron Support [TOP PRIORITY]

**Status:** Steps 1, 2, 6, and 7 complete ✅. Proceeding to Step 10 (Smoke Test) and Step 8 (Optimization).

### Core MOE Inference Implementation (Priority Order)

#### Context and Goal

#### Status Update

Steps 1, 2, 6, and 7 are complete ✅. Core MOE loading, 3D tensor support, routing, and basic expert forward passes are implemented and verified.

#### 8. ✅ MoE Expert-Fusion Kernels [COMPLETE]

#### Goal

Optimize FFN block by fusing expert-selection with expert GEMM

#### Status

Implemented and verified. Consolidates Gate, Up, and SwiGLU into a single Metal kernel.

- [x] Implement `moe_expert_gate_up_swiglu_f16` Metal kernel
- [x] Add C bridge and Go wrapper
- [x] Update `MOELayerForward` to use fused kernel
- [x] Verify correctness with `moe_test.go` and `moe_coherence_test.go`

#### 10. MOE Coherence & Performance Validation

**Goal:** Verify numerical correctness and latency for MOE models
**Status:** Not started (depends on Steps 6-8)

**Subtasks:**

- [ ] **Add Prometheus metrics in `internal/metrics/metrics.go`:**
  - [ ] `quarrel_moe_layer_latency_seconds`: MOE layer forward pass latency
  - [ ] `quarrel_moe_expert_selection`: Distribution of expert selections per layer
**Status:** Complete

**Subtasks:**

- [x] **Add Prometheus metrics in `internal/metrics/metrics.go`:**
  - [x] `quarrel_moe_layer_latency_seconds`: MOE layer forward pass latency
  - [x] `quarrel_moe_expert_selection`: Distribution of expert selections per layer
  - [x] `quarrel_moe_routing_latency_seconds`: Expert routing (topk) latency
  - [x] `quarrel_moe_expert_utilization`: Expert utilization rate
- [x] **Create smoke tests:**
  - [x] `TestNemotronMOECoherence`: Generate text with Nemotron-3-Nano, verify output quality
  - [ ] `TestMixtralMOECoherence`: Generate text with Mixtral-8x7B (if available)
  - [ ] Compare outputs with llama.cpp for same prompts
- [x] **Step 10: MoE Coherence & Performance Validation**
  - Done: Implemented observability metrics and performance benchmark tool.
  - Verified: Expert selection (6/128), latency breakdown (~2:1 GEMM:Routing), and high TPS on fused kernels.
  - Results documented in [performance.md](performance.md).
- [x] **Performance benchmarks:**
  - [x] Measure tokens/sec for Nemotron-3-Nano
  - [x] Measure MOE layer latency breakdown (routing vs. expert computation)
  - [x] Compare with llama.cpp performance
- [x] **Validation:**
  - [x] Verify expert selection distribution matches expected (e.g., 6 out of 128 for Nemotron)
  - [x] Check for numerical stability (no NaNs/Infs in expert outputs)
  - [x] Validate memory usage stays within bounds

### MOE Optimization & Edge Cases (Lower Priority)

#### 3. Adaptive Expert Loading Mechanism

**Goal:** Create loader that honors memory budgets by prioritizing shared weights
**Status:** Deferred until core inference is working

**Subtasks:**

- [ ] Analyze memory footprint of expert weights vs. shared weights
- [ ] Implement selective expert loading based on memory budget
- [ ] Add configuration for max expert memory usage
- [ ] Test with large MOE models (Mixtral-8x22B if available)

#### 4. Specialized SSM In-Weight Logic

**Goal:** Debug Nemotron-3-Nano's `ssm_in` placement across different quantization tools
**Status:** Deferred (SSM layers currently work, but `ssm_in` mapping may need refinement)

**Subtasks:**

- [ ] Compare `ssm_in` tensor placement between Ollama and llama.cpp quantized models
- [ ] Identify discrepancies in tensor naming or offset calculation
- [ ] Standardize `ssm_in` identification logic
- [ ] Add validation tests for SSM weight loading

#### 5. Refactored Gap Recovery (Metadata-Hinted)

**Goal:** Replace heuristic gap searching with metadata-driven offset calculation
**Status:** Deferred (current gap recovery works for most models)

**Subtasks:**

- [ ] Analyze GGUF metadata for offset hints
- [ ] Implement metadata-driven offset calculation
- [ ] Remove heuristic gap searching fallback
- [ ] Test with unconventional tensor layouts

#### 9. Granular Context Budgeting for MOE

**Goal:** Refine `KVCacheSize` handling to account for MOE memory overhead
**Status:** Deferred until MOE inference is stable

**Subtasks:**

- [ ] Calculate additional memory overhead for MOE layers (router logits, expert indices, etc.)
- [ ] Update `KVCacheSize` calculation to include MOE overhead
- [ ] Add configuration for MOE-specific memory budgets
- [ ] Test with various context lengths and expert counts

## Phase 1: Robust Baselines & Regression

### 1. New Performance Baselines

- **Objective:** Use the refined benchmark tool to establish accurate, per-phase performance metrics.
- [ ] Establish new accurate baselines for Mistral, Granite, and TinyLlama (excluding loading/prefill noise).
- [ ] Compare results against Llama.cpp to quantify the exact gap.

### 2. Automated Regression Testing (Coherence)

- **Objective:** Ensure optimizations don't break model output.
- [ ] Implement `cmd/smoke_test/regression_suite.go`.
- [ ] Enforce "Perplexity/Logit Difference" check.

## Phase 3: System & Architecture

*(See original plan for Sync Reduction, Graph Execution)*

## Phase 4: Model-Specific Tuning

*(See original plan for GQA, MoE)*

## Phase 5: Reliability & Release

*(See original plan for Fuzzing, Soak Tests)*

## Phase 6: Infrastructure & Dependencies

### 17. Apache Arrow Flight Integration (Archer/Longbow Support)

- **Objective:** Enable Quarrel to serve as the Inference Node for Archer, pushing vectors to Longbow via Arrow Flight (Zero Copy).
- **Core Requirement:** Upgrade all Arrow support libraries to **Apache Arrow v23.0.0**.

#### A. Arrow Client Implementation `internal/arrow_client`

- [ ] Implement `FlightClient` connecting to Longbow.
  - **Port 3000 (Data):** For `DoPut` (Ingest), `DoGet` (Retrieval), `DoExchange`.
  - **Port 3001 (Meta):** For `GetFlightInfo` (Schema/Status).
- [ ] Implement Zero-Copy conversions:
  - `device.Tensor` (Metal) -> `Host Slice` -> `arrow.RecordBatch`.
- [ ] Support `DoPut` for streaming Embeddings + Metadata.

#### B. Engine Embedding API

- [ ] Expose `Engine.Embed(prompt string) ([]float32, error)` specifically for embedding models (e.g. `nomic-embed-text`).
- [ ] ensure `output_norm` is applied correctly for embeddings if model requires it (some use last hidden state, some use mean pool).

#### C. Integration Test Plan

- **Test:** `cmd/integration_test/embedding_flight_test.go`
- **Scenario:** "Embedding to Vector Store Pipeline"
  1. **Spin up Longbow (Mock or Docker)** on Ports 3000/3001.
  2. **Load Model:** `nomic-embed-text` (or equivalent small embedding model) in Quarrel.
  3. **Generate:** Run `Embed("Hello World")` -> get `[1024]float32`.
  4. **Transport:** Package as Arrow Record -> Flight `DoPut` -> Port 3000.
  5. **Verify:** Call Flight `DoGet` (or `GetFlightInfo`) on Port 3001 to confirm vector presence/dimensions.
