# Metrics Reference

Comprehensive Prometheus metrics reference for monitoring inference performance, GPU layer offloading, TurboQuant compression, model behavior, and system health in Longbow-Quarrel (v0.2.0).

---

## 1. Core Engine Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `inference_tokens_total` | Counter | Total tokens generated across all inference requests. |
| `inference_duration_seconds` | Summary | Total latency spent in the token generation loop. |
| `gpu_memory_allocated_bytes` | Gauge | Current bytes allocated in GPU VRAM. |
| `gpu_kernel_duration_seconds` | Histogram | Execution latency per individual GPU kernel (label: `kernel`). |
| `context_length_tokens` | Histogram | Distribution of sequence context lengths processed. |

---

## 2. GPU Layer Offloading Metrics (Hybrid VRAM / CPU Execution)

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_gpu_layers_active` | Gauge | Number of transformer layers currently offloaded to GPU VRAM. |
| `quarrel_cpu_layers_active` | Gauge | Number of transformer layers currently retained in CPU system RAM. |
| `quarrel_layer_offload_transfers_total` | Counter | Total host-device activation tensor roundtrip transfers. |
| `quarrel_layer_offload_duration_seconds` | Histogram | Latency histogram for CPU layer forward passes (`ApplyLayerCPU`). |

---

## 3. TurboQuant KV Cache Compression Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_turboquant_encode_calls_total` | Counter | Total number of TurboQuant encode operations (PolarQuant + QJL). |
| `quarrel_turboquant_decode_calls_total` | Counter | Total number of TurboQuant decode / reconstruction operations. |
| `quarrel_turboquant_encode_latency_seconds` | Histogram | Time spent in TurboQuant KV encoding. |
| `quarrel_turboquant_decode_latency_seconds` | Histogram | Time spent in TurboQuant KV decoding. |
| `quarrel_turboquant_compression_ratio` | Histogram | Compression ratio achieved over uncompressed FP16 baseline. |

---

## 4. CUDA Engine Initialization & Lifecycle Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_cuda_engine_initialized_total` | Counter | Total successful CUDA engine initializations (labels: `model`, `architecture`). |
| `quarrel_cuda_engine_failed_total` | Counter | Total CUDA initialization failures (labels: `model`, `error_type`). |
| `quarrel_cuda_memory_bytes` | Gauge | VRAM memory allocated and tracked by the CUDA engine (label: `model`). |

---

## 5. Paged KV Cache Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `kv_cache_capacity_bytes` | Gauge | Total pre-allocated KV cache memory capacity. |
| `kv_cache_used_bytes` | Gauge | Current KV cache memory utilization in bytes. |
| `kv_cache_hits_total` | Counter | Number of prefix tokens retrieved directly from prompt cache. |
| `kv_cache_misses_total` | Counter | Number of token positions requiring new attention computation. |
| `kv_cache_evictions_total` | Counter | Total blocks evicted under cache pressure. |
| `kv_cache_oob_total` | Counter | Count of prevented out-of-bounds KV cache access attempts. |
| `kv_cache_overlap_total` | Counter | Count of prevented position overlaps in block tables. |

---

## 6. Model Health & Stability Audit Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `logit_nan_count_total` | Counter | Count of NaN values detected in output logits. |
| `logit_extreme_values_total` | Counter | Logit values exceeding +/- 1000 in probability distribution. |
| `numerical_instability_total` | Counter | Instances of NaN/Inf in intermediate hidden states (label: `tensor`). |
| `activation_unhealthy_total` | Counter | Count of layers exhibiting collapsed or saturated activations. |
| `nan_detected_total` | Counter | NaN propagation events detected during transformer forward loop. |

---

## 7. Model Hot-Swapping & Speculative Decoding

| Metric | Type | Description |
|--------|------|-------------|
| `model_hot_swap_total` | Counter | Total model hot-swap operations initiated via API. |
| `model_hot_swap_duration_seconds` | Histogram | Time taken to load replacement model weights and resume engine. |
| `quarrel_moe_expert_selection_total` | Counter | Frequency of expert selection (labels: `layer`, `expert_id`). |
| `quarrel_moe_routing_latency_seconds` | Histogram | Latency of top-K MoE router gating. |

---

## 8. Tokenizer Performance

| Metric | Type | Description |
|--------|------|-------------|
| `tokenizer_encode_time_seconds` | Histogram | Latency of BPE / SentencePiece encoding process. |
| `tokenizer_decode_time_seconds` | Histogram | Latency of BPE / SentencePiece decoding process. |
| `tokenizer_unknown_tokens_total` | Counter | Total unknown token substitutions encountered. |
