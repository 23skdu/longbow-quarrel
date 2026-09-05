# Longbow-Quarrel - Next Steps & Roadmap

## v0.3.0 Improvement Plan

### Quality & Correctness Fixes
| Priority | Item | Files | Status |
|----------|------|-------|--------|
| P0 | BF16 tensor dequantization | `internal/gguf/dequant.go` | ✅ Done — `BF16ToFloat32`, `DequantizeBF16`, `MatVecMulBF16` |
| P0 | F16 IEEE 754 bit-conversion | `internal/gguf/dequant.go` | ✅ Done — `Float16ToFloat32` with subnormal/Inf/NaN handling |
| P0 | Top-P sampling correctness | `internal/engine/engine_cpu.go` | ✅ Done — sort+CDF+zero nucleus in `applyTopPCPU` |
| P0 | Chat template rendering | `internal/engine/prompt_wrapper.go` | ✅ Done — Jinja2 subset renderer with `{% for %}`, `{% if %}`, `{{ var }}` |
| P0 | CPU LoRA adapter loading | `internal/engine/engine_cpu.go`, `internal/engine/lora.go` | ✅ Done — merge-on-load with `A×B×α/r` delta fusion |
| P0 | PromptCache population | `internal/engine/engine_cpu.go`, `internal/engine/engine_cuda.go` | ✅ Done — `Insert` called after `CompleteSequence` in all engines |

### Performance & Scalability
| Priority | Item | Files | Status |
|----------|------|-------|--------|
| P0 | Zero-copy MatVec for Q2_K/Q3_K/Q4_0/Q5_0/Q5_K | `internal/gguf/dequant_simd.go`, `internal/engine/cpu_weights.go` | ✅ Done — parallel kernels via `matVecMulGeneric` |
| P0 | Parallel MatMul with SIMD dot products | `internal/simd/simd.go` | ✅ Done — B transpose + `VecDotF32` inner loop |
| P0 | Speculative decoding ForwardDraft | `internal/engine/engine_cpu.go` | ✅ Done — real multi-token forward pass with KV cache |
| P0 | Sliding-window KV cache wiring | `internal/engine/engine_cpu.go`, `internal/engine/kv_cache_sliding_window.go` | ✅ Done — `NewCPUKVCacheWithWindow` when `WindowSize > 0` |

### Architecture & Extensibility
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P1 | Remote worker engine completion | `internal/engine/remote.go` | ✅ Done — `Infer`, `InferWithCallback`, `ForwardShard`, `ForwardBatch` via Arrow Flight RPC |
| P1 | NCCL integration | `internal/device/multi_gpu.go`, `internal/device/nccl.go` | ✅ Done — real NCCL `allreduce`/`broadcast`/`allgather` with `nccl` build tag, stub fallback |
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

**All 10 P0 blockers have been resolved.** See the summary tables above for implementation details.

---

#### Last updated: September 2026 (v0.3.0 P0 Correctness, Completeness & Throughput Plan — All P0 items complete)