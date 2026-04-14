# Metrics Reference

Comprehensive Prometheus metrics for monitoring inference performance, model behavior, and system health.

## Core Engine Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `inference_tokens_total` | Counter | Total tokens generated across all requests. |
| `inference_duration_seconds` | Summary | Total time spent in the inference step loop. |
| `gpu_memory_allocated_bytes` | Gauge | Current bytes allocated on the GPU. |
| `gpu_kernel_duration_seconds` | Histogram | Execution time per individual kernel (label: `kernel`). |
| `context_length_tokens` | Histogram | Distribution of sequence lengths processed. |

## Paged KV Cache Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `kv_cache_capacity_bytes` | Gauge | Total pre-allocated KV cache capacity. |
| `kv_cache_used_bytes` | Gauge | Current KV cache memory usage. |
| `kv_cache_hits_total` | Counter | Number of positions retrieved from the prompt cache. |
| `kv_cache_misses_total` | Counter | Number of positions requiring new computation. |
| `kv_cache_evictions_total` | Counter | Number of blocks evicted from the cache. |
| `kv_cache_oob_total` | Counter | Count of out-of-bounds KV cache access attempts (Stability Audit). |
| `kv_cache_overlap_total` | Counter | Count of position overlaps in the block table (Stability Audit). |

## Model Health & Audit Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `logit_nan_count_total` | Counter | Total count of NaN values detected in model output. |
| `logit_extreme_values_total` | Counter | Values exceeding +/- 1000 in the logit distribution. |
| `numerical_instability_total` | Counter | Instances of NaN/Inf in hidden states (label: `tensor`). |
| `activation_unhealthy_total` | Counter | Count of layers with collapsed or saturated activations. |
| `nan_detected_total` | Counter | Count of NaN propagation events detected during forward pass. |

## Speculative Decoding & Advanced Features

| Metric | Type | Description |
|--------|------|-------------|
| `model_hot_swap_total` | Counter | Number of times a model was swapped via API. |
| `model_hot_swap_duration_seconds` | Histogram | Time taken to load a new model and restart the engine. |
| `quarrel_moe_expert_selection_total` | Counter | frequency of selection per expert (labels: `layer`, `expert_id`). |
| `quarrel_moe_routing_latency_seconds` | Histogram | Latency of the MOE router top-k selection. |

## Quantization Accuracy (Audit)

| Metric | Type | Description |
|--------|------|-------------|
| `dequant_max_abs_error` | Histogram | Maximum error compared to FP16 reference during dequantization. |
| `dequant_fail_total` | Counter | Count of kernels failing accuracy gates (>0.1 error). |

## Tokenizer Performance

| Metric | Type | Description |
|--------|------|-------------|
| `tokenizer_encode_time_seconds` | Histogram | Latency of the BPE encoding process. |
| `tokenizer_decode_time_seconds` | Histogram | Latency of the BPE decoding process. |
| `tokenizer_unknown_tokens_total` | Counter | Number of unknown tokens encountered. |
