# Longbow-Quarrel - Project Status

## Completed Features (v0.1.0)

All major features from the original 10-part plan have been implemented:

| Feature | Status | Location |
| -------- | -------- | ---------- |
| Metal GPU Backend | ✅ Complete | `internal/device/metal.go` |
| CUDA GPU Backend | ✅ Complete | `internal/device/cuda.go` |
| GGUF Model Loading | ✅ Complete | `internal/gguf/` |
| Sliding Window Attention | ✅ Complete | Mistral 4096 tokens |
| Gemma4 Hybrid Attention | ✅ Complete | 5 sliding + 1 full per 6 layers |
| OpenAI API Endpoints | ✅ Complete | `/v1/chat/completions`, `/v1/completions` |
| Benchmark Tool | ✅ Complete | `cmd/benchmark` |
| Output Validation | ✅ Complete | `compareTokenSequences()` |
| cuDNN Flash Attention | ✅ Complete | `internal/device/cudnn.go` |
| Fused Kernels | ✅ Complete | `cudaFusedAttention`, `cudaFusedRoPE` |

## Phase 1-4: COMPLETE ✅

All roadmap items from Phase 1-4 are complete including:
- Arrow integration (zero-copy, Flight streaming, metrics)
- Continuous batching and PagedAttention
- Speculative decoding with rejection sampling
- Multi-LoRA, grammar sampling, VLM support
- Quantization (Q5_K, Q2_K, Q1_K)
- Kubernetes probes and memory management
- Tensor parallelism framework

---

## Phase 7: Code Quality & Remediation

### Known Issues & Remediation Steps

#### Issue 1: ForwardDraft stub in CUDA engine
- **Location:** `internal/engine/engine_cuda.go:908-910`
- **Problem:** Returns nil, nil - no actual draft model implementation
- **Impact:** Speculative decoding cannot use CUDA as target
- **Remediation:** Implement draft forward pass that returns logits for speculative verification

#### Issue 2: ForwardShardedLayer not fully implemented
- **Location:** `internal/engine/remote.go:78-92`
- **Problem:** Returns error "waiting for Flight DoGet response implementation"
- **Impact:** Tensor parallelism cannot actually distribute work to workers
- **Remediation:** Implement bidirectional Flight RPC: DoPut sends input, DoGet receives partial output

#### Issue 3: MasterDistributedEngine still delegates to primary
- **Location:** `internal/engine/distributed_master.go:51-57`
- **Problem:** `ForwardBatch` just calls `shards[0].ForwardBatch(batch)` - no actual parallel execution
- **Impact:** Distributed engine provides no speedup
- **Remediation:** Implement fan-out goroutines per shard using errgroup, then AllReduce the results

#### Issue 4: CPU Engine attention is O(n²) naive
- **Location:** `internal/engine/engine_cpu.go:487-537`
- **Problem:** Uses triple-nested loops for attention - will be extremely slow
- **Remediation:** Add optimized attention kernel using simd or cgo to Accelerate/OpenBLAS

#### Issue 5: CoW block copy is no-op
- **Location:** `internal/engine/kv_cache_paged.go:242-248`
- **Problem:** `copyBlockData` returns immediately without copying data
- **Impact:** Forked sequences get corrupted data
- **Remediation:** Implement actual device-side copy using C.memmove or device.CopyFromAt

---

## Phase 6: Transformers v5 Compatibility Tests

Tests are implemented as stubs in:
- `internal/engine/transformers_v5_compat_test.go`
- `internal/engine/transformers_v5_fuzz_test.go`
- `internal/metrics/v5_metrics_test.go`

These require model files to run.

---

#### Last updated: April 2026