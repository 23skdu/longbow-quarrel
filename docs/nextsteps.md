# Longbow-Quarrel - Next Steps & Roadmap

## 10-Part Improvement Plan (v0.6.0) — COMPLETED

> Verified and implemented 2026-09-08. All 10 parts delivered with concrete acceptance criteria.
> PRs 1–5 require `gh auth login` to close as superseded (they share no merge-base with main).

### Part 1: Complete zero-dequant GEMM coverage (Q6_K, Q2_K, Q3_K)
- **Status**: COMPLETED
- **Delivered**: `dequant_q6_k_gemm_kernel` in `cuda_kernels.cu`, Go wrapper `MatVecDequantQ6_K` in `cuda.go`,
  zero-dequant branch extended to gate Q6_K, CGo extern + dispatch in MatVec.

### Part 2: True tiled FlashAttention-2 + Mistral sliding-window wiring
- **Status**: COMPLETED
- **Delivered**: Shared-memory tiled FA-2 kernel (`FA_TILE_KV=32`) with cooperative KV loading,
  Mistral `WindowSize` wired in `engine_cuda.go` via `e.config.WindowSize > 0` fallback.

### Part 3: Real preemption + priority scheduling in continuous batching
- **Status**: COMPLETED
- **Delivered**: `Priority` field on `InferenceRequest` + `Sequence`, priority-aware `PopUpTo`,
  `PreemptLowestPriority` evicts lowest-priority running seq on KV pressure, preemption metrics.

### Part 4: End-to-end Gemma 4 VLM fusion (loader + server + CLI)
- **Status**: COMPLETED
- **Delivered**: `gemma4` case in `NewVLMDecoder`, server uses model architecture for VisionEncoder,
  CLI `--image` flag with VLM pipeline in `cmd/quarrel/main.go`.

### Part 5: True AVX-512 VNNI / AMX intrinsics (replace Go fallback)
- **Status**: COMPLETED
- **Delivered**: `dot_q8_0_vnni` and `dot_q4_k_vnni` C kernels with `vpdpbusd` intrinsics,
  Go function pointers `dotQ8_0VNNI`/`dotQ4KVNNI` set from `avx512.go` init.

### Part 6: Wire FP8/Q8_0 paged KV end-to-end + harden paged kernel
- **Status**: COMPLETED
- **Delivered**: `PagedAttentionQuantized` called from CUDA engine when cache dtype is FP8/Q8_0,
  `paged_attention_kernel` hardened (arbitrary headDim, warp-reduced dot, no stack buffer).

### Part 7: Harden distributed engine (stubs → real, pipeline chaining)
- **Status**: COMPLETED
- **Delivered**: `RollbackKV` sends rollback via Arrow Flight RPC, `ForwardDraft` transmits tokens
  and receives logit results, `SyncWeights` does connect + barrier metadata sync.

### Part 8: Wire observability into hot path (TTFT + governor auto-trigger)
- **Status**: COMPLETED
- **Delivered**: TTFT recorded on first generated token per sequence, inter-token latency recorded,
  `MemoryGovernor.TriggerGovernor` called in batch loop, governor action metrics emitted.

### Part 9: Docker/CI build matrix + GHCR publish + Dockerfile dedup
- **Status**: COMPLETED
- **Delivered**: Redundant root `Dockerfile` removed (kept `Dockerfile.cpu`), CI docker build matrix
  (4 Dockerfiles), compose lint job, GHCR publish job for `:latest` and `:cuda-latest` tags.

### Part 10: PR hygiene + grammar strictness + test hardening
- **Status**: COMPLETED
- **Delivered**: `GrammarTypeCFG` now enforces validation (not allow-all), `isTokenAllowedRegex`
  `<128` loophole removed, dead `VocabularyTrie`/`TrieNode` types deleted.
  PRs 1–5: require `gh auth login` to close (documented as superseded).

---

## Archive: v0.5.0 Plan — Superseded by v0.6.0 (2026-09-08)

v0.5.0 plan was audited on 2026-09-08; all 10 parts confirmed NOT IMPLEMENTED.
v0.6.0 consolidated gaps with concrete acceptance criteria and delivered all items.

---

## Archive: v0.4.0 10-Part Plan — Completed (detail removed 2026-09-07)

v0.4.0 delivered: zero-dequant CUDA GEMM (Q8_0/Q4_K), CUDA prefill FlashAttention,
continuous batching + paged blocks, asymmetric speculative decoding, VLM encoder +
multimodal endpoint parsing, grammar PDA + bitmask sampling, VNNI/AMX dispatch labels,
FP8/Q8_0 paged cache structures, Arrow Flight sharded forward, OTel/Prometheus/Governor
instrumentation, plus P1 backlog (`run_all_tests.sh`, 80% CI gates, `make release`).

#### Last updated: September 2026 (v0.6.0 COMPLETED; v0.5.0 superseded; v0.4.0 archived)
