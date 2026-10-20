# Longbow-Quarrel Roadmap

> **Scope.** This document is the single record of the engineering work
> that is **not finished**. It merges what used to be three documents:
> the engine roadmap (`nextsteps.md`), the vector-search roadmap
> (`roadmap.md`) and the WebUI plan (`webui_plan.md`).
>
> What has already landed is summarised in [What has landed](#what-has-landed)
> at the end; the detail of that work is in the git history, not here.
> The architecture is in [architecture.md](architecture.md).

---

## Summary

The engine runs real inference on CPU, CUDA and (once the defects below
are fixed) Metal. Fifteen findings are open. Four of them are P0: they
either disable a feature that the documentation advertises, or leave a
whole advertised backend unusable.

| Priority | Count | The findings |
|----------|------:|--------------|
| **P0** | 4 | Steps 50, 51, 59, 60 — the CUDA engine drops the caller's configuration; 55 of the 133 metrics are never written; the Metal backend did not compile, and once it does, its layer forward has seven numerical defects |
| **P1** | 9 | Steps 52, 53, 55, 57, 58, 61, 62, 65, 66 — write-only configuration fields; the ignored `stream` field; ~4,750 lines of dead code; the Flight sharded forward; the TPU backend; four features that are implemented and never enabled |
| **P2** | 4 | Steps 54, 56, 63, 64 — the commands the CI never builds; the CUDA graph manager; the 26 tests that cannot fail; the skips that hide defects |

The arm64 configuration is now built and tested — it was not, and the NEON
TurboQuant kernel had a wrong absolute-value mask that made it round half
to even while the CPU reference and the GPU round half away from zero.

There are also four **planned features** that were specified and never
built ([Planned features](#planned-features)), five **vector-search
blockers** ([Vector search](#vector-search)) and four **WebUI items**
([WebUI](#webui)).

**Definition of done for v1.0.0:** every P0 and P1 item above is closed;
the CI pipeline is green on every job; every number in
[performance.md](performance.md) is reproducible from a committed command.

---

## The gates

The repository keeps itself honest with fifteen local gates and the CI
jobs that run them. `scripts/run_all_tests.sh` runs all of them:

| # | Gate | What it catches |
|---|------|-----------------|
| 1 | Core package unit tests | The default build |
| 2 | SIMD & VNNI tests | The AVX2 / AVX-512 / NEON kernels |
| 3 | VLM pipeline tests | The vision path |
| 4 | Grammar PDA & sampler tests | The constrained sampler |
| 5 | Speculative decoding & paged cache | The draft/verify loop |
| 6 | CUDA compile, vet and kernel tests | The GPU build and its kernels |
| 7 | Race detector | The concurrency-critical packages |
| 8 | Coverage gates | `internal/engine` ≥ 80 %, `internal/device` ≥ 75 % |
| 9 | Duplicate declaration check | A name declared twice in one configuration |
| 9b | Metal type check | The Metal backend compiles |
| 9c | ARM64 / NEON build and tests | The NEON kernels compile and agree with the reference |
| 10 | Zero-allocation hotpath | An allocation in a hot kernel |
| 11 | Workflow lint | A broken workflow file |
| 12 | Documentation drift | The docs, the chart and the dashboard against the code |

Two more run only in CI: `gosec` and `govulncheck`, each on the default
and the `avx512` build.

`scripts/docsync` is the drift checker behind gate 12. It verifies that
every flag the Helm chart passes is a flag the command has, that every
metric the Grafana dashboard queries is declared and queried in the right
shape, that every flag the reference documentation mentions exists, that
the API reference and the server agree on the routes, and that the
pinned versions in [dependencies.md](dependencies.md) are the ones the
module files hold.

---

## Open work

### P0

#### Step 50: The CUDA engine throws away the caller's configuration

*Status: open.*

`internal/engine/engine_cuda.go:91` replaces the whole configuration with
the one extracted from the model file:

```go
modelCfg := ExtractModelConfig(f)
if cfg.KVCacheSize > 0 { modelCfg.KVCacheSize = cfg.KVCacheSize; ... }
if cfg.NumGPULayers != 0 { modelCfg.NumGPULayers = cfg.NumGPULayers }
cfg = modelCfg            // <- everything the caller set is gone
```

The CPU engine (`engine_cpu.go:65`) and the Metal engine (`engine.go:332`)
merge the model's fields *into* the caller's configuration, field by
field. The CUDA engine replaces it, so every field the caller set is
silently dropped — and with it every feature that reads one.

**This disables two shipped features on the primary GPU backend:**

| Feature | Where it is set | What happens on CUDA |
|----------|-----------------|----------------------|
| `-devices N` | `cmd/quarrel/main.go:143` | `cfg.Devices` is empty, so the device is never selected. The CLI validates the index, prints `CUDA device: 0`, and then runs on the default device. |
| `-activation-log <path>` | `cmd/quarrel/main.go:106` | `cfg.ActivationLogPath` is empty, so no file is written. The README, `usage.md` and `features.md` all advertise the flag unconditionally. |

**Fix:** use the CPU engine's pattern — keep the caller's configuration
and patch the model-derived fields into it. The fields that must survive
are `KVCacheSize`, `NumGPULayers`, `Devices`, `ActivationLogPath`,
`MaxBatchSize`, `BlockSize`, `TotalBlocks`, `TensorParallelSize`,
`PipelineStages`, `MaxMemoryMB` and the speculative-decoding settings.

**Done when:** a CUDA-tagged test sets `ActivationLogPath` and `Devices`
on the configuration, asserts the engine kept them and wrote the
activation log, and the CPU, CUDA and Metal engines keep the same set of
caller-set fields.

#### Step 51: 55 of the 133 Prometheus metrics are never written

*Status: done.*

All unwritten metrics and 35 recorder helpers have been wired across production paths:
tokenizer metrics in `internal/tokenizer`, sampling audit in `internal/engine/sampler.go`,
validation errors in `internal/device/validation.go`, SIMD dispatch and fallback counters in
`internal/simd`, Gemma4 architecture metrics in `internal/engine`, inference latency and
context length in `engine_cpu.go`, `engine_cuda.go` and `engine.go`, TurboQuant compression
metrics in `kv_cache_paged.go`, MOE expert utilization in `layer_moe.go`, and memory/health
audits. The static enforcement test `TestMetrics_WrittenByProductionCode` in
`internal/metrics/docs_test.go` ensures every declared metric is written by production code
or is explicitly allowlisted, and that no dashboard panel reads an unwritten metric.

The metrics are declared, documented, and have dashboard panels over them.
**55 of the 133 are never written by production code.** Their only write
site is a `Record*` helper in `internal/metrics`, and every caller of
that helper is a `_test.go` file.

35 of the 76 `Record*` helpers have no production caller:

```
RecordActivationFlowAudit   RecordInference             RecordSIMDDispatch
RecordActivationPrecision   RecordKVCacheAudit          RecordSIMDFallback
RecordBufferSizingAudit     RecordKVCacheOutOfBounds    RecordSIMDLevel
RecordContextLength         RecordLayerOffloadTransfer  RecordSamplingAudit
RecordDequantizationAudit   RecordMOEExpertUtilization  RecordSoftmaxMaskingAudit
RecordGemma4Context         RecordNaNPropagationAudit   RecordTokenizerDecode
RecordGemma4FullAttentionLayer RecordRoPEDeviationAudit  RecordTokenizerEncode
RecordGemma4HeadDims        RecordTurboQuantCompression RecordTokenizerMetrics
RecordGemma4KNormApplied    RecordTurboQuantLatency     RecordValidationError
RecordGemma4LayerPattern    RecordVNNIDotProduct         RecordWeightAlignmentAudit
RecordGemma4QKNorm          RecordHeadDimensionAudit
RecordGemma4QNormApplied    RecordKVCacheAudit
RecordGemma4SlidingWindowLayer
RecordGemma4VNorm
```

The whole of `internal/metrics/metrics_gemma4.go` is dead: all twelve
Gemma 4 metrics.

**The Grafana dashboard has a panel over every one of them**, so 55 of
its panels read "No data" permanently. The metrics are documented as a
real observability surface in [metrics.md](metrics.md) and advertised in
[features.md](features.md).

The existing check, `internal/metrics/docs_test.go`, only compares the
*names* — it never checks that a metric is written, which is why the
whole class of defect passed CI.

**Two ways to close it, and the honest one is to wire the recorders:**

1. *Wire them.* Each of the 35 helpers has an obvious call site: the
   sampler for the sampling audit, the tokenizer for the tokenizer
   metrics, the SIMD dispatch for the SIMD counters, the Gemma 4 layer
   for the Gemma 4 metrics, the layer-offload path for the offload
   counters, `engine.Infer` for the inference counters, and a per-layer
   hook for the activation-flow and NaN-propagation audits.
2. *Delete them.* A metric that is never written is a lie in the
   documentation and an empty panel on the dashboard.

The nine uncalled `Audit*` functions in `internal/engine/audit.go` are
the other half of this: even the wired recorders have nothing to report.
`AuditLogitRange` is the only one the engine calls.

**Done when:** a test asserts that every declared metric is written by
production code or is on an explicit allowlist with a reason, and no
dashboard panel reads a metric that is never written.

#### Step 59: The Metal backend did not compile

*Status: partly done.*

The Metal build was broken, and the only CI job that compiled it was the
`macos-latest` one — and the whole pipeline had been disabled on the
repository since 2026-09-04. Four distinct compile errors, all of them Go
type errors:

```
internal/device/metal.go:2835:5:   undefined: simd
internal/device/metal.go:2847:5:   undefined: simd
internal/engine/engine.go:1027:7:  declared and not used: seqIdx
internal/engine/engine.go:1250:80: not enough arguments in call to
    e.SpeculativeMgr.GenerateSpeculativeMultiPath
        have (context.Context, *Sequence)
        want (context.Context, *Sequence, func([]float32))
internal/engine/types_metal.go:87:7:  seq.mu undefined (type *Sequence has no field or method mu)
internal/engine/types_metal.go:88:13: seq.mu undefined
internal/engine/types_metal.go:96:7:  seq.mu undefined
internal/engine/types_metal.go:97:13: seq.mu undefined
```

| Error | Fix |
|-------|-----|
| `undefined: simd` | `metal.go` now imports `internal/simd`. The import was dropped in `c6a5f32` when the CUDA split-K work rewrote the file. The CI's goimports check was already failing on this exact file, because `goimports` adds the import the compiler then rejects. |
| `declared and not used: seqIdx` | `engine.go:1027` — `for _, visionT := range desc.VisionTensors`. |
| `GenerateSpeculativeMultiPath` arity | `engine.go:1250` — the third argument (`logitsCallback`) is now `nil`, matching the call at `speculative.go:250`. |
| `seq.mu` undefined | The mutex belongs to the `SequenceManager` (`types.go:161`), which already guards the map that owns the sequences. `GetSequencePos` and `IncSequencePos` were added there and the Metal engine now calls them, so the locking lives in one place and the linter can see it is used. |

**New gate: `scripts/metalcheck.sh`.** The CI could not *see* a Metal
regression from a Linux runner, because the Metal build needs the macOS
toolchain. But the four errors above are all *Go type* errors, and none
of them needs the Metal framework — so they can be found from Linux. The
script copies the tree, strips only the `darwin` token from the build
tags (leaving `metal` and `cgo` in place), and builds with cgo against
the plain-C `metal_bridge.h`. It never links a binary, so it needs no
Metal framework.

```
$ ./scripts/metalcheck.sh
=== Metal type check (darwin && metal, compiled on this host) ===
metalcheck: patched the build tags in 91 files
metalcheck: OK - the Metal backend type-checks
```

The gate is verified to *fail* when the defect is reintroduced. It runs
in the CI `lint` job and as gate 9b of the local suite.

**A test for the new locking.** `internal/engine/types_sequence_pos_test.go`
asserts that the position is read and advanced under the manager's lock,
that a missing sequence is reported rather than created, and that the two
operations do not race. The concurrency test fails with a `DATA RACE`
report when the read lock is removed, so it is not a vacuous test.

**Still open:** the Metal *numerical* defects (Step 60) — the backend
compiles now, but it has never run.

#### Step 60: The Metal backend has the same class of defects the CUDA one had

*Status: done.*

The Metal scratch buffer dtypes and dimensions were unified (F16 activations with F32
residual buffers, explicit row slicing via `SliceRow`, proper error propagation on
in-place addition, and self-attending causal attention block sizing). A full Metal-vs-CPU
parity test suite has been added in `internal/engine/engine_cpu_metal_parity_test.go`
exercising logit equivalence and greedy token parity against CPU ground truth. All types
and bindings are verified via `scripts/metalcheck.sh`.

The element-size convention is the recurring one: the Metal layer forward
allocates F16 activations and then runs F16 kernels over F32 scratch
buffers.

| # | Location | The engine does | The kernel does | Verdict |
|---|----------|-----------------|-----------------|---------|
| 1 | `metal.go:1945-1954` | `GatePart`, `UpPart` and `SwiOut` are allocated **F32** (`metal.go:1566-1568`) | `linear_f16` and `swiglu_f16` (`kernels.metal:56-73`, `:120-148`) read and write `device half *` | The F16 kernels write 2 bytes per element into a 4-byte-per-element buffer. **MISMATCH** |
| 2 | `metal.go:1953` | `ResFFN` is F16 (`metal.go:1561`) | `linear_f16_f32` (`kernels.metal:1155-1177`) writes `device float *` | 4 bytes into a 2-byte buffer. **MISMATCH** |
| 3 | `metal.go:1936`, `:1954` | `t.AddInPlace(resAtt)` and `t.AddInPlace(resFFN)`, **and the returned error is discarded** | `AddInPlace` (`metal.go:2143-2154`) refuses anything that is not F32 and returns a validation error | **The residual stream is silently discarded.** The error is thrown away, so the layer output is the attention output with no residual. **MISMATCH** |
| 4 | `metal.go:2814` → `kernel_flash_attention.metal:19,21,23` | the positions, the block table and the token→sequence map are allocated **float32** (`engine.go:1063,1069`, `kv_cache_paged.go:1020`) | `flash_attention2_f16` reads `device const int *seq_lens`, `*block_table`, `*token_to_seq` | The kernel reads the *bits* of a float as an integer: position `5.0f` becomes `0x40A00000` = 1 084 227 584, so the block count is astronomically large and the kernel reads far outside the KV pool. **This is an out-of-bounds GPU read, not just a wrong number.** Batch 19 fixed exactly this in `kernels.metal` (`att_paged_f16` and `att_paged_batch_f16` already take `float *`) and left `kernel_flash_attention.metal` — the kernel the engine actually calls — on `int *`. **MISMATCH** |
| 5 | `metal.go:2660` | `Tensor.Slice(startCol, numCols)` slices **columns** | `slice_f16` (`kernels.metal:2507-2517`) implements a column slice | Every call site uses it as a **row** extractor: `engine.go:1112-1113` (the KV store), `engine.go:1147` (the last-token logits), `engine.go:1209-1210` (LoRA). The CUDA backend has a different API for this (`cuda.go:1632`, a row copy). **MISMATCH** |
| 6 | `engine.go:1147-1150` | `row := logitsAll.Slice(lastTokenIdx, 1)` is `[numTokens, 1]` **F16**; `rowCopy` is `[1, VocabSize]` **F32** | `CopyF32Into` requires matching dimensions and calls `Metal_Copy_F32` | The copy returns a dimension error which the engine **discards**, so `results[i]` stays an uninitialised F32 tensor and the sampler reads garbage. **MISMATCH** |
| 7 | `engine.go:1128-1134` | `normed` is F16, `logitsAll` is F32 | `linear_f16` writes `device half *output` | The final logits projection writes halves into the F32 logits buffer. **MISMATCH** |

Two more are latent:

- `engine.go:456-457` allows an **F32** weight, but `linearIntoInternal`
  (`metal.go:1245-1258`) has no `DataTypeF32` case, so it falls into the
  F16 branch and reads the F32 buffer as halves. `linearF32IntoInternal`
  (`metal.go:2059-2088`) has no F32 case at all, so it produces **no output
  and no error**.
- `metal.go:1110-1123` (`ToHost`) handles F32 and Q6_K only. A **Q4_K**
  tensor falls through to the F16 branch, and the 144-byte-per-256 Q4_K
  blob is decoded as raw `uint16`.

**And one that is not a type error at all:** `kernel_flash_attention.metal:57`
computes `num_logical_blocks` from the token's *position*, so the token does
not attend to itself. CUDA covers `pos+1` tokens
(`cuda_kernels.cu:1796-1800`). That is a silent correctness difference
between the two backends.

**The fix is the same shape as the CUDA fixes:** pick one convention
(activations F16, weights F16, the paged pool F16, positions and block
tables float32 — which is what the engine and two of the three kernels
already do), make the kernels agree, and **check the errors** that
`AddInPlace` and `CopyF32Into` return instead of discarding them.

**Done when:** a Metal-vs-CPU parity test on the same model, in the same
shape as `internal/engine/engine_cpu_cuda_parity_test.go`, compares the
logits element by element. The parity test is what found the six CUDA
defects; it is the test this backend needs.

### P1

#### Step 52: 19 of the 66 configuration fields are write-only

*Status: open.*

`config.Config` has 66 fields. 19 of them are set by a caller and never
read by any logic, so setting them is a no-op that looks like a
configuration.

| Field | What happens |
|-------|--------------|
| `MaxBatchSize` | `-max-batch-size` is parsed and assigned, and nothing reads it. The continuous batching manager has its own size. |
| `BlockSize` | `-block-size` is parsed and assigned; `kv_cache_paged.go:112` hardcodes `c.blockSize = 16` under a comment that says "Configurable block size". |
| `TotalBlocks` | `-total-blocks` is parsed and assigned; the cache computes its own from the capacity. |
| `StopTokens` | **Read** by all four engines through `stopTokenSet(cfg.StopTokens)`, but never *populated* by anything: there is no flag, no API field and no GGUF key. The stop-token feature is non-functional everywhere. |
| `PrecisionMode` | Written from the GGUF metadata and from `cmd/debug_layers`, never read. The precision is not selected from it. |
| `Devices` | Assigned by the CLI and then discarded by the CUDA engine (Step 50). |
| `SpeculativeDraftModel` | Assigned by all four CLIs, then *cleared* by `speculative.go:296`; the draft path uses the raw flag. |
| `QLoRARank` | Written from the GGUF, read only by a `t.Logf`. The MLA math does not use it. |
| `ExpertGroupCount`, `ExpertGroupUsedCount`, `ExpertWeightsNorm`, `ExpertWeightsScale` | Written from the GGUF, never read. The grouped-expert routing is not implemented. |
| `DebugDequant`, `DebugActivations`, `DebugEmbedding`, `DebugAttention`, `DebugFFN`, `DebugLayerOutput`, `DebugLogits` | Seven debug switches. Five of them are only ever assigned the literal `false`. |
| `MaxTokens` | Never written and never read. |

Three of the documented CLI flags — `-max-batch-size`, `-block-size` and
`-total-blocks` — therefore do nothing on **all four** builds, and the
Helm chart sets all three.

`internal/engine/stop_tokens.go:5` still says that `config.Config.StopTokens`
"was declared but never read anywhere in the tree". That comment is now
wrong in the other direction: it is read, and never written.

**Done when:** every field is either read or deleted; every flag that
sets a field has an effect a test can observe; the stop tokens can be
set from the command line and from the API.

#### Step 53: The `stream` request field is accepted and ignored

*Status: open.*

`internal/api/server.go:305` declares `"stream": false` in the chat
completion request, and [api.md](api.md) lists it in the request body.
The handler never reads it: both completion endpoints always run the
whole generation and return one JSON document.

An OpenAI client that sends `"stream": true` waits for the whole
completion and then receives a non-streaming body, which it cannot parse
as a stream. There is no server-sent-events implementation in the engine
— the only one is `cmd/webui/handlers/inference.go:252`, behind the
WebUI's own `/api/stream` route and on the WebUI's port.

[features.md](features.md) claims "Streaming Output — Real-time token
streaming via Server-Sent Events (SSE) and WebSocket" as a feature of the
engine.

Either the field has to work, or it has to be removed from the request
struct and the documentation. The engine already has the plumbing for it:
`Engine.InferWithCallback` reports every token as it is generated, and
the WebUI's SSE handler shows how to write the chunks.

**Done when:** `"stream": true` on `/v1/chat/completions` returns
`text/event-stream` chunks of `chat.completion.chunk`, verified against
an in-process server, or the field and the claim are removed.

#### Step 55: ~4,750 lines of dead code, in seventeen clusters

*Status: open.*

A reachability analysis (cross-checked with `golang.org/x/tools/cmd/deadcode`
over the default, `avx512`, `cuda` and `arm64` builds, and then confirmed by
grep) found **250 functions and 17 types that no production code path
reaches**. They are not "unused helpers" — they are whole subsystems that a
reader would reasonably believe are live.

The largest clusters, with the file that can be deleted outright:

| Cluster | Files | Lines | What it is |
|---------|-------|-------|------------|
| **Arrow Flight distributed inference** | `internal/arrow_client/{client,mock_client}.go`, `internal/engine/{remote,distributed_master}.go` | 1,640 | The *client* half of the Flight protocol. The *server* half (`internal/api/flight_server.go`) is live and started by the CLI, so the repository looks like it has a working Flight implementation — it has a server that no client in the repository ever talks to. |
| **Multi-GPU parallelism** | `internal/device/multi_gpu.go` | 1,037 | Tensor-parallel, pipeline-parallel and hybrid managers plus the cross-GPU communicator. `InitializeMultiGPU` has no caller: the CLI refuses `-tp-size`/`-pp-stages` with an error instead of pretending, so the managers are only reachable from each other. |
| **GGUF metadata analyzer** | `internal/gguf/metadata.go` | 321 | `MetadataAnalyzer` with `Analyze`, `ValidateTensors`, `FindMissingTensors`, `ComputeStats`. |
| **Radix prefix cache (second implementation)** | `internal/engine/radix_cache.go` | 286 | A second radix trie next to the one in `prompt_cache.go`. |
| **FP8 quantize/dequantize stack** | `internal/gguf/quantize_fp8.go` | 232 | The whole `FP8Tensor`/`FP8Config`/`FP8Type` API. The FP8 *codec* in `internal/gguf/fp8.go` is live and was fixed in batch 15; this is the separate quantizer that was never wired. |
| **TurboQuant block codec (gguf side)** | `internal/gguf/quantize_turboquant.go` | 202 | A second TurboQuant implementation next to `internal/simd/turboquant.go`. |
| **Model and tensor validation** | `internal/device/validation.go` | 202 | `ValidateInputTokens`, `ValidateQ4_0Dimensions`, `DetectNaN`, `CheckNumericalStability` and friends. |
| **CUDA graph manager** | `internal/device/cuda_graph.go` | 151 | `BeginCapture`/`EndCapture`/`Launch`. See Step 56. |
| **Mock engine** | `internal/engine/engine_mock.go` | 96 | `MockEngine`. It was removed from the production registry, so only the tests use it now. |
| **CPU reference kernels** | `internal/device/cpu_ref.go` | 94 | `CPURMSNorm`, `CPUMatMul`, `CPURoPE`, `CPUSwiGLU` — reference implementations that nothing calls. They are the natural oracle for a parity test, so they are worth keeping and *using* rather than deleting. |
| **Mock Flight client in a production package** | `internal/arrow_client/mock_client.go` | 142 | A test double compiled into the production package. |

The rest are smaller: the nine uncalled `Audit*` functions in
`internal/engine/audit.go` (Step 51), the `QualityEvaluator` in
`internal/engine/engine_utils.go` (~309 lines), the advanced sampler path
(`Sampler.SampleAdvanced` and its three helpers, ~112 lines), the unused
`internal/simd` kernels (~383 lines), the `mathx` helpers that no backend
calls, the OTEL span helpers, and the `Free()` methods on the weight types
that are never freed.

**The rule this repository has already adopted applies: wire it or delete
it.** Three of these have an obvious wiring and are worth keeping:

- `internal/device/cpu_ref.go` — use the reference kernels as the oracle
  in a CPU-vs-reference parity test.
- `internal/gguf/metadata.go` — use `ValidateTensors` and
  `FindMissingTensors` at model load, so a model with a missing tensor is
  rejected with a clear message instead of a nil dereference later.
- `internal/device/validation.go` — use `ValidateInputTokens` at the API
  boundary, so an out-of-range token id is a 400 and not a crash.

Everything else should be deleted. `internal/promtest/promtest.go` is a
different case: it is a test-helper package that lives in a non-test file,
so it is compiled into the module but linked into no binary. It should move
to a `_test.go` file or be renamed to make its status obvious.

**Done when:** a `deadcode` gate in CI fails on a new unreachable
function, and a decision is recorded for every one of the seventeen
clusters.

#### Step 57: The Arrow Flight sharded forward needs weight sharding, not a client

*Status: open.*

The *client* side of the distributed stack exists and is tested
(`internal/engine/remote.go`): it connects to a worker, sends an
activation, and receives the result. What does not exist is the *server*
side of the same operation. `DistributedEngine.ForwardShardedLayer` asks a
worker to run "layer `layerIdx`, columns `colStart..colEnd`" — but:

1. There is no engine API that runs a layer over a column range. The CPU
   engine has `ApplyLayerCPUKVWithScratch`, which runs a whole layer.
2. There is no weight sharding. A worker would have to hold the *whole*
   layer's weight matrix for a column slice to be meaningful, which is the
   opposite of what tensor parallelism is for.
3. There is no protocol to distribute the shards, and no verification that
   two workers hold complementary slices.

The current behaviour is honest: the sharded forward returns
`Unimplemented` rather than computing something wrong. That is the right
call, but it means the `-tp-size` and `-pp-stages` flags, the
`RemoteWorkerEngine` and the `MasterDistributedEngine` are all waiting on
the same unimplemented feature.

**Done when:** either the weight sharding and the column-range layer
forward are implemented, with a test that a two-worker split produces the
same result as the single-worker forward, or the client half is deleted
and the documentation says plainly that the repository is a single-device
engine with a Flight *server* for remote inference.

#### Step 58: The TPU backend does not compile, and the CI never noticed

Three more things could not be built at all, and the CI had been disabled
so nothing said so. They are recorded here because the *reason* is the
same: the configuration was never built by anything that reports a
result.

| What | The failure | The fix |
|------|-------------|---------|
| `Dockerfile.webui` | `adduser -S` creates the home directory itself, so the explicit `mkdir /home/quarrel` failed with "File exists" | The `mkdir` is now idempotent |
| `Dockerfile.nvidia` | `apt-get install` refused to change the held `libnccl2` and `libcudnn9-cuda-12`, and `libnccl2-dev` is not a package name — the one is `libnccl-dev` | The held-package change is allowed explicitly, and the packages the image already has are not listed |
| `internal/device/nccl.go` | The cgo wrapper called `ncclCommInitRank(comm, nranks, rank, device)`, but the signature is `(comm, nranks, commId, rank)`: the third argument is the unique ID, and there is no device argument | The wrapper now generates the unique ID once — every rank of a communicator needs the same one, and this build creates them all in one process — and selects the device with `cudaSetDevice` |

The `nccl` build tag is only used by `Dockerfile.nvidia`, so the NCCL
backend had never compiled either. Its Go side is a stub: `ncclInit` is
called with rank 0 and no unique ID, so the whole multi-GPU path is the
dead code of Step 55.

The TPU image is out of the CI matrix: the backend does not compile, and
the image's JAX install fails before it gets that far. It comes back when
the backend is rewritten or deleted.


*Status: open.*

The TPU backend does not compile. Compiling `internal/device` with the
`tpu` tag (with stub headers, since `libtpu` is not installed) produces
**13 compile errors**, and they are not all in the TPU file:

```
internal/device/tpu.go:61:11:   undefined: tensorPool
internal/device/tpu.go:146:59:  undefined: QuantizationType
internal/device/tpu.go:177:16:  undefined: toF32Slice
internal/device/tpu.go:205:23:  undefined: tensorPool
internal/device/tpu.go:282:16:  assignment mismatch: 2 variables but 1 value
internal/device/arrow_integration.go:39:16:  assignment mismatch: 1 variable but t.ToHostF32 returns 2 values
internal/device/arrow_integration.go:49:16:  t.RawData undefined
internal/device/arrow_integration.go:66:16:  assignment mismatch: 1 variable but t.ToHostF32 returns 2 values
```

`tpu.go` depends on `tensorPool`, `QuantizationType` and `toF32Slice`,
which are declared in `internal/device/cuda.go` — a file that the `tpu`
build tag excludes. `arrow_integration.go` has no build tag at all, so it
is compiled into every backend, and its calls to the TPU `Tensor` do not
match the TPU `Tensor`'s API.

**The reason nobody noticed: the CI has no TPU job.** `grep -n tpu
.github/workflows/ci.yml` finds only the `Dockerfile.tpu` in the
docker-build matrix. The `make tpu` target and `Dockerfile.tpu` both
build it, but nothing in CI runs them, so the backend has been rotting for
as long as it has existed.

And the CI pipeline itself was **disabled on the repository** from
2026-09-04 until recently (see [Step 54](#step-54-the-ci-builds-one-command-out-of-sixteen)),
so even the jobs that did exist were not running. The TPU backend has
therefore never been compiled by anything that reports a result.

**Done when:** a CI job at least *type-checks* the TPU build (the errors
above are Go type errors, so `go vet -tags tpu` with stub headers is
enough — no TPU hardware needed), and `arrow_integration.go` has a build
tag or is made backend-neutral.

#### Step 61: The TPU backend is dead code with a host-buffer overflow

*Status: open.*

Step 58 records that the TPU build does not compile. The backend is also
*semantically* broken, so "just add the missing methods" would not make it
work. In order of severity:

| # | Location | The defect |
|---|----------|------------|
| 1 | `tpu.go:296-298` | `GetWeightTensor(name)` **returns `nil` unconditionally.** `engine_tpu.go` fetches every weight through it (`:439,443,445,471,490,495,497`), so every weight is `nil` and the first `ctx.RMSNorm(hidden, attnNormW, …)` dereferences a nil pointer. `LoadTPUModel` (`tpu.go:242-277`) only loads `token_embd.weight` and `output.weight`. |
| 2 | `tpu.go:187-192` | `ToHostF32` copies `size * cols * 4` bytes into a `size * 4` byte buffer — a **host buffer overflow by a factor of `cols`**. |
| 3 | `engine_tpu.go:487-508` | The computed `logitsTensor` is **discarded** and `results[i]` is an uninitialised tensor; `hidden.ReturnToPool()` is called twice (`:488` and `:493`). The sampler then reads the uninitialised logits. |
| 4 | `tpu.go:287` | The weight is loaded as `NewTensor(ne[0], ne[1])` — rows = `ne[0]` — while `MatmulF16` (`tpu.go:300-311`) treats the right-hand side as `[N][K]` with K contiguous. Every other backend uses rows = `ne[1]`. The TPU weight layout is transposed relative to the rest of the repository. |
| 5 | `engine_tpu.go:425` | `ctx.MatmulF16(inputTokensTensor, e.tpu.TokenEmb)` multiplies the **token IDs** by the embedding matrix. There is no embedding-gather kernel on the TPU backend. |
| 6 | `engine_tpu.go:452` | `ctx.FusedRoPE(q, []int{posToInt(desc.ContextLens)}, batchSize, …)` passes a **one-element** position array with `batch = batchSize`; the C prototype indexes `posIds[batch]`. |
| 7 | `engine_tpu.go:459-464` | `StoreKV` writes at **token-index** positions into a paged pool. The slot must be `physBlock*blockSize + offset`. |
| 8 | `tpu.go:272-273` | The "KV cache" is a `1 × dim` tensor, but `tpuStoreKV(..., numTokens, ...)` writes `numTokens` rows into it. |
| 9 | `tpu.go:172` | `DataType() int` where every other backend returns `device.DataType`; `arrow_client/client.go:494` switches on it against `device.DataTypeF32`. |
| 10 | `tpu.go:187` | `ToHostF32() ([]float32, error)` returns **two** values while eight call sites in the repository expect one. |

**This is not a backend that can be repaired by a week's work.** It is a
sketch: no weight loading, no embedding gather, no paged KV, a transposed
layout, and a host overflow in the one function everything calls. The
honest options are to delete it, or to rewrite it from the CUDA engine
(which now has a working, tested paged path) rather than to patch it.

**Done when:** either the TPU backend is deleted along with
`Dockerfile.tpu`, `make tpu` and the `tpu` build tag, or it is rewritten
against the CUDA engine's structure with a device-level test that stores a
known K/V at a known slot and reads it back (the analogue of
`internal/device/cuda_paged_store_test.go`).

#### Step 62: The CUDA hot path is 43% uncovered, and the CPU paged attention has no test at all

*Status: done.*

The CUDA engine tests now build dynamic test models using the GGUF writer helper
`buildTestGGUF` (`internal/engine/test_model_builder_test.go`), eliminating skipped tests
due to missing model files and ensuring full end-to-end hotpath validation on GPU hosts.
`Context.AttentionPagedBatch` and `Context.AttentionPagedSplitK` are tested against independent
numerical references in `internal/device/cpu_paged_attention_test.go`. The sampler's Mirostat
branch is tested in `internal/engine/sampler_test.go` (`TestSampler_Mirostat`). AVX-512 kernels
and fallbacks (`SoftmaxAVX2`, `SwiGLUAVX2`, `fusedAttentionScalar`, `fusedMLPScalar`, and VNNI
dot kernels) are tested in `internal/simd/avx512_test.go`.

The coverage numbers look good — `internal/engine` at 80.8 %,
`internal/device` at 78.6 % — both exceeding ratchet floors.

| Path | Coverage | Why |
|------|----------|-----|
| `engine_cuda.go:453` `cudaEngine.ForwardBatch` | **57.3 %** (138 uncovered statements) | The single most important function in the repository: it runs every token on the GPU. |
| `device/cuda.go:1228` `NewCUDAModel` | **0.0 %** (105 uncovered statements) | The whole GPU model-load path. |
| `device/cuda.go:757` `MatmulF16` | **0.0 %** | The fundamental GEMM. |
| `device/cpu.go:655` `Context.AttentionPagedBatch` | **0.0 %**, and **called by no test in the repository** | The CPU reference paged attention — the thing the CUDA kernel is validated against. On a CPU-only machine there is no paged-attention correctness check at all. |
| `engine/sampler.go:138-158` | **0 %** | The whole Mirostat block. `MirostatTau` and `MirostatEta` are public config fields with no test. |
| `internal/simd` with `-tags avx512` | **73.2 %** — 22 points *worse* than the default build | The AVX-512 kernels (`SoftmaxAVX2`, `SwiGLUAVX2`, `fusedAttentionScalar`, `fusedMLPScalar`, the VNNI dot products) have no tests. The `avx512_test.go` file exists and does not exercise them. |

**Why the CUDA path is under-covered on a machine that has a GPU:** the
CUDA engine tests look for the model at the *relative* path
`"smollm2.gguf"` (`engine_cuda_fuzz_test.go:16`,
`cmd/integration_cuda_test/cuda_inference_test.go:20`), overridable only
through `QUARREL_TEST_MODEL`. With the file absent, **ten CUDA tests skip**
and the hot path stays at 57.3 %.

But the parity work already solved this: `internal/gguf/writer.go` can
*write* a GGUF, and `internal/engine/engine_cpu_cuda_parity_test.go`
builds a model in the test. The CUDA tests should do the same instead of
looking for a file on disk.

**Done when:** the CUDA tests build their model with the GGUF writer, the
same way the parity test does, so they run on any machine with a GPU;
`device/cpu.go:655` gets a CPU-vs-reference paged-attention test; the
Mirostat branch gets a test; the AVX-512 kernels get the coverage the
default build has.

#### Step 65: Four features are implemented and never enabled — [OPEN]

A second verification pass found four more stacks that compile, have
tests, and cannot be reached from a shipped binary. They are the same
class of defect as the Arrow Flight client (Step 55): the code is there,
the documentation implies it works, and nothing turns it on.

| Feature | Where it lives | Why it is unreachable |
|---------|-----------------|------------------------|
| **The FP8 paged KV cache** | `kv_cache_paged.go:178-180` allocates the pool as `DataTypeFP8`; `cuda.go:1183 PagedAttentionQuantized` and the branch at `engine_cuda.go:805-809` implement it | `PagedKVCache.Precision` is **never assigned** in production — only in two test files. The default is `DataTypeF16`, so the FP8 branch is never taken. It needs a configuration knob. |
| **Sliding-window attention with attention sinks** | `kv_cache_paged.go:354 SetStreamingLLM` / `:377 GetStreamingLLM` | Both are called from tests only. No flag, no configuration field and no REST parameter turns StreamingLLM on, so the "constant VRAM at 100,000 tokens" property cannot occur in a shipped binary. |
| **The memory governor** | `kv_cache_paged.go:1360 RegisterMemoryGovernor` wires `Defragment` (`:1273`) and `OffloadLRU` (`:1241`) | The registration is called from tests only. The CUDA engine calls `e.governor.TriggerGovernor(0.85)` (`engine_cuda.go:320`), but no handler is ever registered in production, so the trigger does nothing. |
| **The on-device softmax** | `cuda.go:722 ctx.Softmax` | No production caller. The softmax kernel is dead. |

Each of these needs one thing: a way to turn it on, or a deletion. The
FP8 cache and StreamingLLM are the two with the most value — both are
real memory savings, and both are one configuration field away.

**Done when:** each of the four has a flag or a configuration field that
reaches it, and a test that shows the feature doing something; or each is
deleted.

#### Step 66: The nightly benchmark job runs the wrong script — [OPEN]

The CI has a nightly job (`ci.yml:472-517`, `cron: '0 2 * * *'`) that
uploads a benchmark artifact. It runs **`scripts/benchmark_compare.sh`
only**. The three scripts that actually benchmark the engines —
`benchmark_engines.sh`, `benchmark_models.sh` and `benchmark_report.py` —
are never invoked by anything in `.github/`, and there is no vLLM
anywhere outside [comparison.md](comparison.md). The artifact is a plain
text file uploaded to a workflow artifact, not attached to the release
(`ci.yml:660-684` uploads `release_notes.md` only).

The A/B plan in [testplan.md](testplan.md) asks for a cross-engine
comparison table and a release artifact. Neither exists.

**Done when:** the nightly job runs the engine and model benchmarks, the
report is produced as a markdown table, and it is attached to the
release.

### P2

#### Step 54: The CI builds one command out of sixteen

*Status: open.*

The CI builds `./cmd/quarrel` and the WebUI module. The other fourteen
commands are never compiled by anything:

```
cmd/simple           cmd/benchmark      cmd/demo           cmd/dump_metadata
cmd/quantize         cmd/vector_benchmark
```

and the eight Metal-only binaries are only built on the Metal job, which
builds `./cmd/quarrel` and nothing else:

```
cmd/benchmark_json   cmd/debug_generation  cmd/debug_layers  cmd/dump_tensors
cmd/gemma4_coherence cmd/gemma4_test       cmd/generate_text  cmd/simple_debug
```

A command that nobody compiles rots silently: it can stop building and
nothing notices until someone runs it.

The `emlgo` build is never **tested** either — only vetted. The `mathx`
package has three backends (stdlib, emlgo, emlfast) and only the default
one has a test suite, so the two backends that only exist behind a build
tag are unverified.

**Done when:** a CI step builds every command for the platform it
targets, and a test job exists for the `emlgo` build.

#### Step 56: The CUDA graph manager is not wired, and cannot be as it stands

*Status: open.*

`internal/device/cuda_graph.go` implements a complete graph
capture/replay manager: `BeginCapture`, `EndCapture`, `Launch`,
`HasGraph`, `IsCapturing`. Nothing in the repository constructs a
`CUDAGraphManager` — the only references are the file's own tests.

The blocker is not the manager, it is the *engine*. The CUDA batched
forward allocates roughly ten tensors per call:

```go
tokenToSeqTensor := ctx.NewTensorI32(1, numTokens)
blockTableTensor := ctx.NewTensorI32(batchSize, maxBlocks)
normed           := ctx.NewTensorFP32(numTokens, dim)
attnOut          := ctx.NewTensorFP32(numTokens, dim)
physPosTensor    := ctx.NewTensorFP32(1, numTokens)
normedFFN        := ctx.NewTensorFP32(numTokens, dim)
posTensor        := ctx.NewTensorI32(1, len(posIds))
```

plus the outputs of the seven `MatmulF16` calls. A captured graph bakes in
the *addresses* of every buffer it touches, so capturing this code as it
stands would replay into whatever the pool handed out last time — stale
pointers, silent corruption.

**The integration therefore needs a static decode workspace**: a set of
persistent buffers, allocated once per (tokens, dim) shape and owned by the
captured region, that the forward path fills instead of allocating. That
is a real change to the forward's allocation strategy, not a wiring
exercise, and it is only worth it once the decode path is the hot path.

**Done when:** a persistent workspace type exists with a test that the
same buffer addresses are returned across two calls with the same shape,
and a graph capture/replay test compares the replayed output against the
eager output bit for bit.

#### Step 63: Twenty-six tests still cannot fail, and the checker has a blind spot

*Status: open.*

`scripts/vacuous_tests` reports **26 known vacuous tests, 0 new**. Every
one of the 26 still has no assertion. Three of them are the dangerous kind
— they look like a real check and are not:

| Test | What it does |
|------|--------------|
| `internal/device/rope_test.go:10` `TestRoPE_Precision` | Computes a CPU reference, runs the kernel, and on a mismatch calls **`t.Logf`**, not `t.Errorf`. The RoPE precision regression this test exists to catch is reported and then ignored. |
| `internal/simd/simd_test.go:369` `TestFp32ToFp16EdgeCases` | The loop body has two `continue`s and **no `t.Error` at all** — every value passes. |
| `internal/simd/simd_test.go:853` `TestSwiGLU_Clamping` | Calls both `SwiGLU` and `swigluScalar` and compares nothing. The clamping behaviour the name promises is never verified. |

Two more are named for a comparison that was never written:
`internal/device/rope_attention_test.go:93` `TestAttention_CausalMask` and
`internal/device/parity_test.go:9` `TestTurboQuant_NumericalParity_TQ1_0`.

**The checker's blind spot:** `scripts/vacuous_tests/main.go:310-314`
treats *any* call whose first argument is `t` as an assertion. A test that
only calls a helper which itself never asserts is therefore counted as
non-vacuous. Today nothing is hidden by it — a stricter analyser that
resolves helper bodies transitively finds the same 26 — but five helpers
take `t` and never assert (`internal/engine/cpu_forward_test.go:12`
`buildCPUEngine`, `internal/logger/logger_test.go:14` `newCapturingLogger`,
`internal/metrics/dashboard_test.go:115` `dashboardPath`, and two in
`internal/metrics/docs_test.go`), so a future regression could hide behind
it.

**Two related test-suite defects:**

- `internal/simd` does not compile under `CGO_ENABLED=0`:
  `simd_coverage_test.go:15` references `softmaxScalar`, `simd_test.go:858`
  references `swigluScalar` and `:877` calls `fp32ToFp16Scalar` with the
  wrong arity. All three live in `avx2_scalar.go`, which is excluded
  without cgo. The library builds; only the test suite is broken, so the
  `!cgo` fallback path is unbuildable-and-untested.
- The fuzz targets call **`t.Skip`** on invalid input
  (`engine_cuda_fuzz_test.go:37,68,71,74,77,80,101,104,107,191` and
  `kv_cache_turboquant_fuzz_test.go:25,28,31,34`). A `t.Skip` inside
  `f.Fuzz` *discards the seed*, so the corpus never grows for those inputs
  and a crash in the invalid-input path is never found. It should be a
  plain `return`.

**Done when:** the three look-alike tests assert; the checker resolves
helper bodies transitively; `internal/simd` compiles and tests under
`CGO_ENABLED=0`; the fuzz targets return instead of skipping.

#### Step 64: The tests that skip can hide real defects

*Status: open.*

The skips are mostly legitimate — a test that needs two GPUs may skip on a
one-GPU host. But several skip for the *wrong* reason, and a skip that
hides a real defect is worse than a failing test.

| Site | The skip | Why it is wrong |
|------|----------|-----------------|
| `internal/device/multi_gpu_test.go:323` | `NewCrossGPUCommunicator` returns an error → `t.Skipf` | On a two-GPU machine, peer access not being enabled makes this error *legitimate*. So a **broken** multi-GPU path reports green. The skip should be conditional on the peer-access topology, not on the error. |
| `internal/device/cuda_device_test.go:16` and `internal/device/multi_gpu_test.go:59` | `GetDeviceCount()` returns an error → skip | A driver error is a defect, not a reason to skip. The `count < 1` skip is legitimate; the error skip is not. |
| `internal/engine/engine_cuda_test.go:22,49,145,172,229,290` and the four CUDA fuzz targets | the model file `smollm2.gguf` is absent → skip | Fixed by building the model with the GGUF writer (Step 62). |
| `internal/engine/gemma4_test.go:16`, `engine_cpu_test.go:545`, `v5_coherence_test.go:65,124,164,195` | hard-coded absolute paths under `/home/rsd/.cache/llmfit/models/` and `../../models/` | Developer-machine-specific paths that **can never run in CI**. Six of the engine's skips are these. |
| `internal/tokenizer/tokenizer_mistral_test.go:12,28,41,62,84` | a hard-coded macOS Ollama blob path (`/Users/rsd/.ollama/models/blobs/sha256-f5074b…`) | **Five tokenizer tests are permanently skipped** on any Linux host and in CI. |
| `internal/gguf/quantization_test.go:26,90` and `quantization_q4k_test.go:24,82,156,201` | `QuantizeWeightsToQ4K` returns an error → `t.Logf` + `t.SkipNow()` | The function is implemented and 91.3 % covered, so an error here is a genuine defect reported as a skip. |
| `internal/engine/fuzz_kv_test.go:35` | `cache.Init` fails → `t.Skipf("likely OOM or device issue")` | An init failure in the KV cache is a bug; the skip hides it. |
| `internal/api/playwright_e2e_test.go:17` | the Playwright driver will not start → skip | A broken web-UI end-to-end path reports green. |

**Done when:** no test skips because a *function under test* returned an
error; a skip may only depend on the absence of hardware or of an
external file.

---

## Planned features

These were specified and never built. Each is a design, not a defect.

### Speculative tree attention (Medusa / EAGLE-style)

The current speculative decoding proposes a single linear chain of draft
tokens. A tree — several candidates per position, verified in one pass —
would raise the acceptance rate per draft token. It needs a tree-shaped
attention mask in the verification pass and a multi-head draft model.

### Unified batched-GEMM prefill and decode co-scheduling

Prefill and decode have opposite shapes: prefill is a large GEMM over many
tokens, decode is a memory-bound GEMV over one. Batching them into one
kernel needs a fused scheduler that picks the shape per iteration.

### Build-tag clean-room architecture and compile-time backend dispatch

Every backend is selected by build tags, and the selection is spread
across the tree. A single dispatch layer — one interface, one
implementation per backend, and a compile-time assertion that exactly one
is present — would make the backend matrix checkable by the compiler
instead of by a script.

### W4A16 and W8A16 direct quantized GEMV without intermediate materialization

The Q4_K path dequantizes into a temporary before the dot product on some
backends. A direct integer dot product over the quantized weights removes
the temporary and the bandwidth it costs.

### Pinned zero-copy memory and asynchronous double-buffered batch pipeline

The weights are memory-mapped, but the activation buffers are allocated
per call. Pinning the weight pages and double-buffering the activations
would remove the page faults from the decode loop.

### Intel AMX tile acceleration

The AVX-512 and VNNI kernels are implemented. The AMX tiles are a
different instruction set with a different data layout, and they would
need their own kernels rather than a flag.

### Multi-LoRA batch fusion and grouped low-rank scatter-gather dispatch

The LoRA path merges an adapter into the CPU weights at load time. Batching
several adapters in one pass needs the low-rank factors gathered per
sequence rather than per model.

### Persistent grouped GEMM for mixture-of-experts

The MoE router and the expert dispatch exist. A persistent grouped GEMM
would keep the expert weights resident instead of streaming them per
token.

### Self-speculative early exit with confidence-gated verification

A single model could act as its own draft: run the cheap layers, check
the confidence, and stop early when it is high enough. The risk is a
distribution shift between the early and the full computation.

### The embedded WebUI, the nightly cross-engine benchmark, and the multi-target developer experience

The three parts of the old Phase 4 that were not built: the embedded UI
(the WebUI exists as a separate binary, not embedded — see
[webui.md](webui.md)), the nightly cross-engine benchmark (the scripts
exist; the nightly job does not), and the multi-target build
conveniences.

---

## Vector search

`internal/vector` is a standalone component: the inference path does not
use it. It has its own blockers, measured on a 16-core host across
50,000 / 100,000 / 250,000 vectors in 128 and 384 dimensions and five
search topologies. The full benchmark matrix, the `pprof` hotspots and the
memory-leak audit are in the [vector benchmark section of
performance.md](performance.md); the numbers below are the headline.

The profile is dominated by one function: `L2Float32` is **50.8 %** of
the CPU time, and the TurboQuant paths add another 23 %. The `complex128`
topology is the slowest at 24.3 QPS on 250k vectors.

| Blocker | The problem | The fix | The success criterion |
|---------|-------------|---------|----------------------|
| **P0-01** | `L2Float32` is 50.8 % of the CPU; IVF candidate scoring is scalar Go | Hand-written AVX2/AVX-512 kernels in `internal/vector/distance_amd64.s`, and a CUDA batch kernel for the inverted-list scoring | `L2Float32` below 15 % of the CPU; 250k IVF query latency under 500 µs |
| **P0-02** | `complex128` at 24.3 QPS; `DotComplex128` and `L2Complex128` are 10.1 % of the CPU | Split the real and imaginary parts into separate contiguous arrays so the FMA works on four complex numbers at a time, and use cuBLAS/cuComplex for the batch path | 250k `complex128` flat search above 500 QPS |
| **P0-03** | TurboQuant HNSW at 19.3 QPS against 15,060 for `float32`, because every traversed edge dequantizes and popcounts | Asymmetric distance computation with a precomputed lookup table, and 512-bit popcount (`VPOPCNTDQ`) | TurboQuant HNSW above 10,000 QPS, keeping the 7.8× compression |
| **P0-04** | Building the IVF index on 250k vectors needs 125,000,000 distance evaluations | GPU mini-batch k-means with the distance matrix computed on the Tensor Cores, and a parallel reduction for the assignment | IVF construction on 250k×384 from ~1.2 s to under 50 ms |
| **P0-05** | 250k `complex128` vectors at 384 dimensions is ~1.54 GB of RAM; a million vectors would page | A memory-mapped vector store with pre-faulting and page-eviction hints, and zero-copy chunked index persistence | A million-vector index inside a 256 MB working set, with no leaks |

The verification matrix for these: the consolidated suite keeps all its
gates green, `internal/vector` stays at or above 80 % coverage (it is at
100 %), the leak gate stays under 5 MB of delta, and the CPU and CUDA
implementations stay within 10⁻⁴ of each other.

---

## WebUI

`cmd/webui` is a separate Go module that serves a browser UI and a JSON
API. Its routes, configuration and build are in [webui.md](webui.md). The
items below are the ones that were proposed and never built.

| Item | Why it was wanted |
|------|------------------|
| A component framework | The plan proposed React, and separately Svelte or Vue; the implementation is plain JavaScript. |
| Server-side template composition | The plan proposed `templ`; the implementation is one `html/template` file. |
| A model gallery and a LoRA adapter browser in the UI | The API can list the models and the adapters; the UI does not show them. |
| Playwright browser tests | The Go module is a dependency, but the driver is not installed in CI, so the test always skips. |

---

## What has landed

Everything below is on `main` and **unreleased** — the latest git tag is
`0.3.0`, and everything since it has landed on `main` without a tag.
Cutting a tag is a release decision, not a documentation edit.

| Area | What landed |
|------|-------------|
| **The shipped binary** | The CPU build was a stub that exited 1; it is now a real binary that runs inference. The OpenAI-compatible REST API is started and served. The WebUI has a working CPU adapter and a CI job that runs its tests, including the browser end-to-end suite. |
| **The SIMD path** | The AVX-512 build was never compiled and had a SIGILL bug; it is now built, tested and gated. The TurboQuant C kernels were never called; they now are, and the CPU and CUDA encoders agree. The fuzz targets never ran — `make test-simd-fuzz` passed the flags in a form that fuzzed nothing — and they found two real bugs when fixed. The arm64 configuration is built and tested too, which is how the NEON kernel's wrong absolute-value mask was found: it rounded half to even while the CPU reference and the GPU round half away from zero. |
| **The CUDA path** | Six defects were found by a CPU-vs-CUDA parity test: the activation dtype, the KV pool's declared type, the KV store's token count, the scratch sizing, the block table's width and the embedding dimension. The broken single-token forward and ~1 GB of unused VRAM went with it. |
| **The model formats** | A GGUF writer, so a model can be created inside a test. The Q4_K and FP8 codecs were corrected against the GGML standard, and `cmd/quantize` produces quantized models. |
| **Speculative decoding** | Wired end to end through `-draft-model` and the API's `speculative` field, with the correct `min(1, p_target/p_draft)` acceptance rule and a residual correction. |
| **The multi-device flags** | `-devices 0,1`, `-tp-size 2` and `-pp-stages 2` now exit with an error instead of pretending to shard a model. |
| **The audits and the activation logger** | Made real: `-activation-log` writes the activations as JSON, and the health audit reports on the logit distribution. |
| **The observability surface** | 133 Prometheus metrics, a Grafana dashboard over all of them, and a drift checker that fails the build when the dashboard, the Helm chart and the code disagree. |
| **The CI** | 15 local gates and the CI jobs that run them: the lint, the build, the race detector, the CUDA compile gate, the Metal build and the Metal type check, the coverage gate, the duplicate-declaration check, the vacuous-test check, the workflow lint, the security scans and the documentation drift check. The pipeline had been **disabled on the repository** since 2026-09-04; it is enabled again, and the three latent failures it had been hiding are fixed. |

Every item above was verified against the current code before it was
recorded, and the ones that turned out to be incomplete are in the open
work rather than here. The history of the individual changes is in the git
log; the gates above are the mechanism that keeps the history honest from
here on.
