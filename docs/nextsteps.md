# Performance Optimization Plan: Closing the Gap

**Goal:** Close the performance gap (currently ~4x-8x vs Llama.cpp) while maintaining strict coherence and correctness.
**Targets:** `mistral:latest`, `nemotron-3-nano:latest`, `tinyllama:latest`, `granite4:latest`.

## Phase 1: Robust Baselines & Regression

### 1. New Performance Baselines

- **Objective:** Use the refined benchmark tool to establish accurate, per-phase performance metrics.
- [ ] Establish new accurate baselines for Mistral, Granite, and TinyLlama (excluding loading/prefill noise).
- [ ] Compare results against Llama.cpp to quantify the exact gap.

### 2. Automated Regression Testing (Coherence)

- **Objective:** Ensure optimizations don't break model output.
- [ ] Implement `cmd/smoke_test/regression_suite.go`.
- [ ] Enforce "Perplexity/Logit Difference" check.

## Phase 2: Kernel Micro-Optimization

*(See original plan for details on GEMM, Flash Attention, Fusion)*

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
