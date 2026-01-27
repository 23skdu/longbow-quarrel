# Performance Optimization Plan: Closing the Gap

**Goal:** Close the performance gap (currently ~4x-8x vs Llama.cpp) while maintaining strict coherence and correctness.
**Targets:** `mistral:latest`, `nemotron-3-nano:latest`, `tinyllama:latest`, `granite4:latest`.

## Phase 0: Mamba/SSM Architecture Support [TOP PRIORITY]

**Objective:** Enable full support for Nemotron-3-Nano (Hybrid Mamba/Transformer) which is currently loading but producing incorrect output due to missing SSM layers.

### 1. Research & Architecture Design

- [ ] Analyze Mamba/SSM data flow (State Space Model parameters: A, B, C, D, dt, z).
- [ ] Map GGUF tensor names (e.g., `ssm_a`, `ssm_conv1d`) to the correct Mamba algorithm steps.
- [ ] Design `MambaLayer` struct in Go to handle hybrid architecture (interleaved or blocks).

### 2. Tensor Loading (GGUF)

- [ ] Extend `engine.go` weight loading to capture SSM-specific tensors.
- [ ] Handle `ssm_in`, `ssm_out`, `ssm_conv1d`, `ssm_a`, `ssm_d`, `ssm_dt` weights.
- [ ] Add "Hybrid" model architecture flag to `EngineConfig`.

### 3. Metal Kernels: Causal Convolution (Conv1d)

- [ ] Write `conv1d_f16` Metal kernel for causal 1D convolution.
- [ ] Support flexible kernel size (usually 4 for Mamba).
- [ ] Add unit test: Compare `conv1d` output against Python/Torch reference.

### 4. Metal Kernels: Selective Scan (SSM Core)

- [ ] **Complex Task:** Implement the parallel or sequential selective scan in Metal (MSL).
- [ ] Handle `scan_f16` operation with time-variant parameters (A, B, C, dt).
- [ ] Optimize for threadgroup memory usage (crucial for SSM scan performance).

### 5. Layer Implementation (Go Engine)

- [ ] Implement `ComputeMambaLayer` in `engine.go`.
- [ ] Integrate Input Projection -> Conv1d -> SSM Scan -> Output Projection flow.
- [ ] Ensure correct residual connection handling (often different in Mamba vs Transformer).

### 6. Hybrid Engine Integration

- [ ] Modify `Infer` loop to switch between `ComputeAttentionLayer` and `ComputeMambaLayer`.
- [ ] Handle state passing (Mamba has internal state, unlike stateless Transformer attention).

### 7. Unit Testing & Validation

- [ ] Creating `cmd/smoke_test/mamba_test.go`.
- [ ] **Unit Test:** Run single Mamba layer with fixed weights/input and verify against pre-computed golden values (from `mamba_ssm` Python lib).
- [ ] **Integration Test:** Verify Nemotron-3-Nano perplexity on small text compared to `transformers`.

### 8. Fuzz Testing (SSM Stability)

- [ ] Create fuzzer for `Scan` kernel: Randomize inputs and `dt` to check for numeric instability (NaNs/Infs).
- [ ] Verify handling of extreme float16 values in the recursive scan.

### 9. Prometheus Metrics (SSM Specific)

- [ ] Add metrics for `ssm_scan_latency` and `ssm_conv_latency`.
- [ ] Track `ssm_state_size` memory usage.
- [ ] Alert on `ssm_divergence` (NaN detection in state).

### 10. Final Verification

- [ ] Run full Nemotron-3-Nano benchmark.
- [ ] Confirm throughput matches or exceeds `llama.cpp` (Mamba should be faster than Attention).

---

## Phase 1: Critical Fixes & Foundation

### 11. Fix Nemotron Loading [DONE]

- **Objective:** Remedy "token embedding weights not loaded" error.
- [x] Debug GGUF tensor names for Nemotron.
- [x] Implement fallback for tied embeddings (`token_embd` <-> `output`).
- [x] Cap context length to prevent OOM.

### 12. Robust Profiling & Baselines

- **Objective:** Identify exact bottlenecks.
- [x] Create `scripts/profile_metal.sh`.
- [x] Establish baseline reports (Mistral, Granite, TinyLlama).
- [ ] Refine benchmark to exclude GGUF load time.

### 13. Automated Regression Testing (Coherence)

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
