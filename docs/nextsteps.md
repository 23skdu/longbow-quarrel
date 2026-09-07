# Longbow-Quarrel - Next Steps & Roadmap

## 10-Part Improvement Plan (v0.5.0) — Active

> Verified 2026-09-07 against `main@40c1588`. The v0.4.0 detailed checklist below was
> archived after code verification (all claimed files exist and core paths are real).
> Verification found 3 fully-complete areas (speculative decoding, grammar sampling,
> dev-experience backlog) and 7 partial areas where symbols exist but wiring/claims
> are overstated. This v0.5.0 plan closes exactly those gaps, plus Docker/CI and PR hygiene.
> Prior v0.4.0 detail removed; one-line archive kept at bottom.

### Part 1: Complete zero-dequant GEMM coverage (Q6_K, Q2_K, Q3_K)
- **Gap**: `dequant_q8_0_gemm_kernel` (`cuda_kernels.cu:1673`) and `dequant_q4_k_gemm_kernel`
  (`:1701`) are real, but no `dequant_q6*gemm` exists; `NewCUDAModel` (`cuda.go:600-611`)
  zero-dequant gates only `Q8_0||Q4_K`, Q6_K falls back to CPU `DequantizeQ6K_SIMD` (`:657-658`).
- **Work**: add `cudaMatVecDequantQ6_K` (+ Q2_K/Q3_K if in scope), Go wrappers in `cuda.go`,
  extend zero-dequant branch, add `cuda_dequant_*_test.go` parity vs CPU.
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`
- **Accept**: `grep cudaMatVecDequantQ6_K` non-empty; Q6_K model loads via `LoadQuantizedRaw`
  with `RecordCUDAVRAMSaved`; parity test passes.

### Part 2: True tiled FlashAttention-2 + Mistral sliding-window wiring
- **Gap**: `flash_attention_prefill_kernel` (`cu:1751-1811`) does online softmax correctly
  (`runningMax/Sum` `:1782-1784`, rescale `:1793-1801`) but streams KV one token at a time
  (`:1786`) — no shared-memory QK/V tiling, so "tiled FA-2" is overstated. Gemma-512 window
  is wired (`engine_cuda.go:179,771-778`), Mistral is not (zero `Mistral` hits in
  `cuda.go`/`cuda_kernels.cu`/`engine_cuda.go`; prefill forces `windowSize=0` unless Gemma4).
- **Work**: add QK/V blocking with shared memory, keep `slidingWindow` arg (`cu:1757`,
  `startKV` `:1776-1778`); wire `config.WindowSize/SlidingWindowSize` (`config.go:40,94`)
  into CUDA prefill for Mistral (mirror CPU path `engine_cuda.go:530-531`).
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`, `internal/engine/engine_cuda.go`
- **Accept**: prefill benchmark shows expected speedup on multi-token prompts; Mistral
  window test asserts masked vs unmasked divergence.

### Part 3: Real preemption + priority scheduling in continuous batching
- **Gap**: `ContinuousBatchManager` (`continuous_batching.go:89-277`) + `PagedKVCache`
  block alloc (`kv_cache_paged.go:317,383,395`) are real, but `preempt` appears only in
  comments/`AbortAll`, and `RequestQueue` (`:60-80`) is plain FIFO with no priority field;
  `RecordContinuousBatchIteration(...,preempted=true)` never called outside tests.
- **Work**: add `Priority` to `InferenceRequest`, priority-aware `Push/PopUpTo` + `Step`
  admit (`:138-191`), block-pressure preemption (evict low-priority running → requeue),
  emit preemption metric (`metrics.go:1185`).
- **Files**: `internal/engine/continuous_batching.go`, `internal/engine/kv_cache_paged.go`, `internal/metrics/metrics.go`
- **Accept**: stress test with forced `LowWaterMark` pressure triggers preemption counter;
  high-priority request jumps FIFO queue in test.

### Part 4: End-to-end Gemma 4 VLM fusion (loader + server + CLI)
- **Gap**: `VisionEncoder.Encode` (`vlm/encoder.go:49-126`) is real, but `NewVLMDecoder`
  (`vlm_loader.go:21-28`) rejects `gemma4` (`unsupported VLM architecture`); `server.go:319`
  hardcodes `"clip"` and discards `Encode` output (`_, _ =`); `cmd/quarrel/main.go` has zero
  `vlm|Vision|chat/completions` wiring.
- **Work**: add `gemma4` case to loader, fuse vision tensor into prompt/prefill in
  `ChatCompletionsHandler` (`server.go:254-321`), wire `--image` flag + `api.Server`
  startup in `cmd/quarrel/main.go`.
- **Files**: `internal/vlm/vlm_loader.go`, `internal/api/server.go`, `cmd/quarrel/main.go`
- **Accept**: `POST /v1/chat/completions` with base64 `image_url` returns vision-conditioned
  output (not text-only); CLI `--image` e2e test passes.

### Part 5: True AVX-512 VNNI / AMX intrinsics (replace Go fallback)
- **Gap**: detection (`cpuinfo.go:47-50,180,188`) + `VecDotQ8_0_VNNI`/`VecDotQ4_K_VNNI`
  (`simd.go:286,330`) exist and tests pass, but both branches run identical Go scalar loops
  (`simd.go:303,347`), differing only in metric label; zero `vpdpbusd|_mm512|tileloadd` hits.
  Real AVX-512 CGO exists in `avx512.go` but VNNI path never calls it.
- **Work**: route VNNI path to `vpdpbusd`-based C kernels (extend `kernels_avx512.c`) or
  Go asm, add `HasAVXVNNI` dispatch benchmark proving >1.3x over scalar.
- **Files**: `internal/simd/simd.go`, `internal/simd/kernels_avx512.c`, `internal/simd/avx512.go`
- **Accept**: `go test -bench VNNI -run XXX ./internal/simd/` shows labeled VNNI faster than
  `fallback_*` on VNNI hardware; `grep vpdpbusd internal/simd/` non-empty.

### Part 6: Wire FP8/Q8_0 paged KV end-to-end + harden paged kernel
- **Gap**: `kScales/vScales` (`kv_cache_paged.go:56-57,163-172`), `StoreKVQuantized`
  (`:682-690`), and `paged_attention_quantized_kernel` (`cu:1813-1970`) exist, but
  `PagedAttentionQuantized` (`cuda.go:449-469`) has zero callers in `internal/engine/`,
  `SetBlockScale/GetBlockScales` have zero non-test callers (scales stay `1.0`); the plain
  `paged_attention_kernel` is self-labeled `naive…stub` with `headDim<=128` early-return
  (`cu:1578-1579,1845`).
- **Work**: call quantized path from CUDA engine when cache dtype is FP8/Q8_0, update
  per-block scales on store, fix "TurboQuant" misnomer comment, promote paged kernel past
  naive/stub (arbitrary headDim, block-table bounds checks).
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/engine/engine_cuda.go`, `internal/device/cuda.go`, `internal/device/cuda_kernels.cu`
- **Accept**: 64k-context run uses quantized pages with non-1.0 scales observed in test;
  `grep -rn PagedAttentionQuantized internal/engine` non-empty.

### Part 7: Harden distributed engine (stubs → real, pipeline chaining)
- **Gap**: `DistributedEngine` interface (`distributed.go:11-26`), `ForwardShard`/
  `ForwardShardedLayer` via `DoPutTensor` (`remote.go:60,337`), and transfer metrics
  (`:127,135,187,195,288-289`) are real, but `RollbackKV`/`ForwardDraft` (`:326-332`) are
  `return nil` stubs and `SyncWeights` is connect-only; tensor-parallel sharding only, no
  pipeline-stage chaining.
- **Work**: implement rollback/draft for remote workers, real weight sync/barrier,
  document pipeline vs tensor-parallel scope (or add stage chaining).
- **Files**: `internal/engine/remote.go`, `internal/engine/distributed.go`
- **Accept**: speculative + distributed integration test passes with remote draft; no
  `return nil // stub` remains in `remote.go`.

### Part 8: Wire observability into hot path (TTFT + governor auto-trigger)
- **Gap**: `RecordTTFT`/`RecordInterTokenLatency` (`telemetry.go:27-96`),
  `quarrel_time_to_first_token_seconds` (`metrics.go:1286`), `MemoryGovernor>0.85/0.92`
  (`memory.go:52,68,136,159`), and `/metrics` exporter (`monitoring/health.go:115`) exist
  and are unit-tested, but `RecordTTFTLatency` ignores its duration param and both it and
  `TriggerGovernor` have zero non-test callers.
- **Work**: emit TTFT/inter-token spans + Prometheus observations from infer path,
  auto-call `CheckMemoryPressure/TriggerGovernor` on alloc pressure, add Governor action
  metrics (`metrics.go:1367`).
- **Files**: `internal/engine/engine*.go`, `internal/telemetry/telemetry.go`, `internal/metrics/metrics.go`, `internal/device/memory.go`
- **Accept**: sample inference produces TTFT histogram observations + trace events;
  simulated >92% pressure triggers defrag/offload in integration test.

### Part 9: Docker/Compose build matrix + GHCR publish + CI coverage
- **Gap (fixed in this commit, needs CI lock-in)**: `Dockerfile.cuda`→`Dockerfile.nvidia`
  refs, `cmd/webui/Dockerfile`→`Dockerfile.webui` refs, `12.4-runtime`→`12.4.0-runtime`,
  `.so`→`.a` kernel lib + missing `nvcc` step, `alpine:latest` pins, `o64-clang` guard,
  `pip`/`golang-go`/`libtpu0`, webui module build + CUDA skew + `curl` + `CMD` expansion,
  `docker/`+`ssl/` volumes, `:9090` collision, bogus `limits: nvidia.com/gpu`, obsolete
  `version:` keys — all corrected here. `docs/usage.md:172-179` references
  `ghcr.io/...:latest/:cuda-latest` with no publish workflow; CI builds no Docker images.
- **Work**: add CI job building all 5 Dockerfiles (+ compose config lint), add GHCR publish
  or fix `docs/usage.md` tags, unify `:cuda` vs `:nvidia` naming (`Makefile:150` vs compose).
- **Files**: `.github/workflows/ci.yml`, `docs/usage.md`, `Makefile`, `Dockerfile.*`, `docker-compose.*.yml`
- **Accept**: `docker build -f Dockerfile.{cpu,nvidia,webui,tpu} .` + `docker compose config`
  pass in CI; published tags match docs or docs corrected.

### Part 10: PR hygiene + grammar strictness + test hardening
- **Remote PRs (audited 2026-09-07, NOT merged — intentionally)**: `refs/pull/{1..5}/head`
  exist but share no merge-base with `main` (distinct `Initial commit`: `ca9b0ee` vs
  `c92d766`; `main` 341 commits vs PRs 22–141). Direct merge would delete ~60–90k lines
  (`git diff --stat main pr1/pr5`: 367–427 files, net −60k+). Tips (`pr1 metal reduction
  fixes`, `pr2/3/4 Mistral RoPE/KV`, `pr5 mock_client`) are already superseded —
  `internal/arrow_client/mock_client.go` exists on main. No `gh` auth in this env to
  close via API. **Action**: close PRs 1–5 as superseded in GitHub UI (or `gh pr close`),
  keep this note as audit trail. Do NOT merge.
- **Grammar**: `Grammar.Apply` bitmask (`sampler/grammar.go:98-120`) + PDA (`:148-207`) pass,
  but `GrammarTypeCFG` defaults allow-all (`:302-303`), `isTokenAllowedRegex` (`:288`)
  allows nearly anything `<128` chars, `VocabularyTrie` is dead code (zero usages).
- **Work**: strict CFG enforcement or remove `CFG` type, strict regex prefix-validation,
  delete or wire `VocabularyTrie`; replace `paged_attention_kernel` naive stub (see Part 6).
- **Files**: `internal/sampler/grammar.go`, `internal/engine/sampler_config.go`
- **Accept**: PRs 1–5 closed; regex/CFG negative tests reject invalid tokens; dead code gone.

---

## Archive: v0.4.0 10-Part Plan — Completed (detail removed 2026-09-07)

v0.4.0 delivered: zero-dequant CUDA GEMM (Q8_0/Q4_K), CUDA prefill FlashAttention,
continuous batching + paged blocks, asymmetric speculative decoding, VLM encoder +
multimodal endpoint parsing, grammar PDA + bitmask sampling, VNNI/AMX dispatch labels,
FP8/Q8_0 paged cache structures, Arrow Flight sharded forward, OTel/Prometheus/Governor
instrumentation, plus P1 backlog (`run_all_tests.sh`, 80% CI gates, `make release`).
Gaps above are tracked as v0.5.0 Parts 1–10.

#### Last updated: September 2026 (v0.5.0 10-Part Plan Active; v0.4.0 archived)
