# Longbow-Quarrel - Next Steps & Roadmap

## 10-Part Improvement Plan (v0.6.0) — Active

> Verified 2026-09-08 against `main@8e96f74`. The v0.5.0 plan below was audited;
> all 10 parts confirmed NOT IMPLEMENTED (scaffolding exists but no end-to-end wiring).
> This v0.6.0 plan consolidates remaining v0.5.0 gaps with concrete acceptance criteria
> and new cross-cutting work (Dockerfile dedup, CI publish, PR closure, grammar hardening).
> Prior plans kept below as archive trail.

### Part 1: Complete zero-dequant GEMM coverage (Q6_K, Q2_K, Q3_K)
- **Status**: NOT IMPLEMENTED — `dequant_q6_k_kernel` (`cuda_kernels.cu:423`) is bulk dequant only; no GEMM variant.
  `cuda.go:601` gates zero-dequant on `Q8_0||Q4_K` only; Q6_K/Q2_K/Q3_K fall back to CPU.
- **Work**: add `cudaMatVecDequantQ6_K` (+ Q2_K/Q3_K if in scope), Go wrappers in `cuda.go`,
  extend zero-dequant branch, add `cuda_dequant_*_test.go` parity vs CPU.
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`
- **Accept**: `grep cudaMatVecDequantQ6_K` non-empty; Q6_K model loads via `LoadQuantizedRaw`
  with `RecordCUDAVRAMSaved`; parity test passes.

### Part 2: True tiled FlashAttention-2 + Mistral sliding-window wiring
- **Status**: NOT IMPLEMENTED — `flash_attention_prefill_kernel` (`cu:1751-1811`) streams KV one token at
  a time (`:1786`) with zero `__shared__` memory; `engine_cuda.go:771-774` hardcodes `windowSize=0`
  unless `IsGemma4`. Mistral's `WindowSize=4096` is parsed (`engine.go:226`) but never passed to CUDA.
- **Work**: add QK/V blocking with shared memory, keep `slidingWindow` arg (`cu:1757`,
  `startKV` `:1776-1778`); wire `config.WindowSize` for Mistral in `engine_cuda.go`.
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`, `internal/engine/engine_cuda.go`
- **Accept**: prefill benchmark shows expected speedup on multi-token prompts; Mistral
  window test asserts masked vs unmasked divergence.

### Part 3: Real preemption + priority scheduling in continuous batching
- **Status**: NOT IMPLEMENTED — `InferenceRequest` has no `Priority` field (`continuous_batching.go:11-32`);
  `RequestQueue` is plain FIFO; `RecordContinuousBatchIteration(...,preempted=true)` never called
  in production (`metrics_test.go:226` only); `AbortAll` is nuclear shutdown, not selective preemption.
- **Work**: add `Priority` to `InferenceRequest`, priority-aware `Push/PopUpTo` + `Step`
  admit (`:138-191`), block-pressure preemption (evict low-priority running → requeue),
  emit preemption metric (`metrics.go:1185`).
- **Files**: `internal/engine/continuous_batching.go`, `internal/engine/kv_cache_paged.go`, `internal/metrics/metrics.go`
- **Accept**: stress test with forced `LowWaterMark` pressure triggers preemption counter;
  high-priority request jumps FIFO queue in test.

### Part 4: End-to-end Gemma 4 VLM fusion (loader + server + CLI)
- **Status**: NOT IMPLEMENTED — `NewVLMDecoder` (`vlm_loader.go:22`) rejects `gemma4` (default case);
  `server.go:319` hardcodes `"clip"` and discards `Encode` output (`_, _ =`);
  `cmd/quarrel/main.go` has zero `--image` or VLM wiring.
- **Work**: add `gemma4` case to loader, fuse vision tensor into prompt/prefill in
  `ChatCompletionsHandler` (`server.go:254-321`), wire `--image` flag + `api.Server`
  startup in `cmd/quarrel/main.go`.
- **Files**: `internal/vlm/vlm_loader.go`, `internal/api/server.go`, `cmd/quarrel/main.go`
- **Accept**: `POST /v1/chat/completions` with base64 `image_url` returns vision-conditioned
  output (not text-only); CLI `--image` e2e test passes.

### Part 5: True AVX-512 VNNI / AMX intrinsics (replace Go fallback)
- **Status**: NOT IMPLEMENTED — `VecDotQ8_0_VNNI`/`VecDotQ4_K_VNNI` (`simd.go:286,330`) run identical
  Go scalar loops; zero `vpdpbusd` hits in `internal/simd/`; VNNI path never calls C kernels.
- **Work**: route VNNI path to `vpdpbusd`-based C kernels (extend `kernels_avx512.c`) or
  Go asm, add `HasAVXVNNI` dispatch benchmark proving >1.3x over scalar.
- **Files**: `internal/simd/simd.go`, `internal/simd/kernels_avx512.c`, `internal/simd/avx512.go`
- **Accept**: `go test -bench VNNI -run XXX ./internal/simd/` shows labeled VNNI faster than
  `fallback_*` on VNNI hardware; `grep vpdpbusd internal/simd/` non-empty.

### Part 6: Wire FP8/Q8_0 paged KV end-to-end + harden paged kernel
- **Status**: NOT IMPLEMENTED — `PagedAttentionQuantized` (`cuda.go:449`) has zero callers in `internal/engine/`;
  `SetBlockScale/GetBlockScales` have zero non-test callers; `paged_attention_kernel` is self-labeled
  "naive…stub" with `headDim<=128` early-return (`cu:1578-1580`).
- **Work**: call quantized path from CUDA engine when cache dtype is FP8/Q8_0, update
  per-block scales on store, fix "TurboQuant" misnomer comment, promote paged kernel past
  naive/stub (arbitrary headDim, block-table bounds checks).
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/engine/engine_cuda.go`, `internal/device/cuda.go`, `internal/device/cuda_kernels.cu`
- **Accept**: 64k-context run uses quantized pages with non-1.0 scales observed in test;
  `grep -rn PagedAttentionQuantized internal/engine` non-empty.

### Part 7: Harden distributed engine (stubs → real, pipeline chaining)
- **Status**: NOT IMPLEMENTED — `RollbackKV`/`ForwardDraft` (`remote.go:326-332`) are `return nil` stubs;
  `SyncWeights` is connect-only; no pipeline-stage chaining beyond tensor-parallel.
- **Work**: implement rollback/draft for remote workers, real weight sync/barrier,
  document pipeline vs tensor-parallel scope (or add stage chaining).
- **Files**: `internal/engine/remote.go`, `internal/engine/distributed.go`
- **Accept**: speculative + distributed integration test passes with remote draft; no
  `return nil // stub` remains in `remote.go`.

### Part 8: Wire observability into hot path (TTFT + governor auto-trigger)
- **Status**: NOT IMPLEMENTED — `RecordTTFTLatency`/`TriggerGovernor` have zero non-test callers;
  no TTFT histogram observations emitted from inference; no auto-trigger on memory pressure.
- **Work**: emit TTFT/inter-token spans + Prometheus observations from infer path,
  auto-call `CheckMemoryPressure/TriggerGovernor` on alloc pressure, add Governor action
  metrics (`metrics.go:1367`).
- **Files**: `internal/engine/engine*.go`, `internal/telemetry/telemetry.go`, `internal/metrics/metrics.go`, `internal/device/memory.go`
- **Accept**: sample inference produces TTFT histogram observations + trace events;
  simulated >92% pressure triggers defrag/offload in integration test.

### Part 9: Docker/CI build matrix + GHCR publish + Dockerfile dedup
- **Status**: NOT IMPLEMENTED — CI (`ci.yml`) has zero Docker build/publish steps; `docs/usage.md:172-179`
  references `ghcr.io/...:latest/:cuda-latest` with no publish workflow; root `Dockerfile` is
  byte-identical to `Dockerfile.cpu` (redundant).
- **Work**: add CI job building all 5 Dockerfiles (+ compose config lint), add GHCR publish
  workflow, remove redundant root `Dockerfile` (keep `Dockerfile.cpu`), unify `:cuda` vs `:nvidia`
  naming (`Makefile:150` vs compose).
- **Files**: `.github/workflows/ci.yml`, `docs/usage.md`, `Makefile`, `Dockerfile.*`, `docker-compose.*.yml`
- **Accept**: `docker build -f Dockerfile.{cpu,nvidia,webui,tpu} .` + `docker compose config`
  pass in CI; published tags match docs; no duplicate root Dockerfile.

### Part 10: PR hygiene + grammar strictness + test hardening
- **Status**: NOT IMPLEMENTED — PRs 1–5 (`refs/pull/{1..5}/head`) share no merge-base with `main`
  (distinct initial commits; 341 vs 22–141 commits); direct merge would delete ~60–90k lines.
  `GrammarTypeCFG` defaults allow-all (`:302-303`); `isTokenAllowedRegex` (`:288`) allows
  anything `<128` chars; `VocabularyTrie` is dead code (zero usages).
- **Work**: close PRs 1–5 as superseded in GitHub UI (requires `gh auth login`), strict CFG
  enforcement or remove `CFG` type, strict regex prefix-validation (remove `<128` loophole),
  delete or wire `VocabularyTrie`; replace `paged_attention_kernel` naive stub (see Part 6).
- **Files**: `internal/sampler/grammar.go`, `internal/engine/sampler_config.go`, `internal/device/cuda_kernels.cu`
- **Accept**: PRs 1–5 closed; regex/CFG negative tests reject invalid tokens; dead code gone;
  paged kernel passes headDim>128 test.

---

## v0.5.0 Plan Audit (2026-09-08) — All Parts NOT IMPLEMENTED

> Deep code analysis on 2026-09-08 confirmed all 10 parts of v0.5.0 remain NOT IMPLEMENTED.
> Scaffolding (struct definitions, metric hooks, CPUID detection, kernel declarations) exists
> throughout, but no end-to-end wiring, real intrinsics, or production callers were found.
> v0.6.0 supersedes v0.5.0 with consolidated acceptance criteria.

---

## Archive: v0.4.0 10-Part Plan — Completed (detail removed 2026-09-07)

v0.4.0 delivered: zero-dequant CUDA GEMM (Q8_0/Q4_K), CUDA prefill FlashAttention,
continuous batching + paged blocks, asymmetric speculative decoding, VLM encoder +
multimodal endpoint parsing, grammar PDA + bitmask sampling, VNNI/AMX dispatch labels,
FP8/Q8_0 paged cache structures, Arrow Flight sharded forward, OTel/Prometheus/Governor
instrumentation, plus P1 backlog (`run_all_tests.sh`, 80% CI gates, `make release`).
Gaps above are tracked as v0.6.0 Parts 1–10.

#### Last updated: September 2026 (v0.6.0 10-Part Plan Active; v0.5.0 superseded; v0.4.0 archived)
