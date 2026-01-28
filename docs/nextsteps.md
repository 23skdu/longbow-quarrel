# Performance Optimization Plan: Closing the Gap

**Goal:** Close the performance gap (currently ~4x-8x vs Llama.cpp) while maintaining strict coherence and correctness.
**Targets:** `mistral:latest`, `nemotron-3-nano:latest`, `tinyllama:latest`, `granite4:latest`.

## Phase 0: Specialized MOE & Nemotron Support [TOP PRIORITY]

### 10-Part Plan: Robust MOE & SSM In-Weight Support

1. **GGUF MOE Metadata Parsing**: Implement support for `llama.expert_count`, `llama.expert_used_count`, and `llama.expert_shared_count` to robustly detect MOE architectures.
2. **Shared vs. Expert-Specific Weight Containers**: Update `ModelWeights` to differentiate between shared (common) and expert-specific tensors, avoiding monolithic loading.
3. **Adaptive Expert Loading Mechanism**: Create a loader that honors memory budgets by prioritizing shared weights and strictly filtering expert weights for MOE models.
4. **Specialized SSM In-Weight Logic**: Debug Nemotron-3-Nano's `ssm_in` placement across different quantization tools (Ollama, llama.cpp) to standardize its identification.
5. **Refactored Gap Recovery (Metadata-Hinted)**: Replace heuristic gap searching with metadata-driven offset calculation to reliably load unconventional tensor layouts.
6. **3D Tensor & Batch Dispatching**: Support multi-dimensional expert tensors (e.g. `[8 2048 512]`) and implement Metal kernels that can process batches of Experts.
7. **Expert Routing (Gate/Router) Kernels**: Implement Top-K expert selection logic (Softmax + TopK) in Metal to handle the router phase of MOE layers.
8. **MoE Expert-Fusion Kernels**: Optimize the FFN block by fusing expert-selection with expert GEMM to reduce memory bandwidth usage.
9. **Granular Context Budgeting for MOE**: Refine `KVCacheSize` handling to account for the additional memory overhead of Experts when calculating prefill budgets.
10. **MOE Coherence & Performance Validation**: Add dedicated smoke tests for Nemotron-3-Nano and Mixtral-8x7B to verify both numerical correctness and latency.

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
