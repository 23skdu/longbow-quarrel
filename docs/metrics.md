# Metrics Reference (v0.3.0)

Comprehensive Prometheus metrics reference for monitoring inference performance, GPU layer offloading, TurboQuant compression, model behavior, and system health in Longbow-Quarrel.

---

## 1. Core Engine Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `inference_tokens_total` | Counter | Total tokens generated across all inference requests. |
| `inference_duration_seconds` | Summary | Total latency spent in the token generation loop. |
| `gpu_memory_allocated_bytes` | Gauge | Current bytes allocated in GPU VRAM. |
| `gpu_kernel_duration_seconds` | HistogramVec (kernel) | Execution latency per individual GPU kernel. |
| `context_length_tokens` | Histogram | Distribution of sequence context lengths processed. |
| `numerical_instability_total` | CounterVec (tensor, type) | NaN/Inf in intermediate hidden states. |
| `validation_errors_total` | CounterVec (operation, error_type) | Validation errors encountered during inference. |

---

## 2. GPU Layer Offloading Metrics (Hybrid VRAM / CPU Execution)

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_gpu_layers_active` | GaugeVec (model) | Number of transformer layers currently offloaded to GPU VRAM. |
| `quarrel_cpu_layers_active` | GaugeVec (model) | Number of transformer layers currently retained in CPU system RAM. |
| `quarrel_layer_offload_transfers_total` | CounterVec (model) | Total host-device activation tensor roundtrip transfers. |
| `quarrel_layer_offload_duration_seconds` | HistogramVec (model, phase) | Latency histogram for CPU layer forward passes (`ApplyLayerCPU`). |

---

## 3. TurboQuant KV Cache Compression

| Metric | Type | Description |
|--------|------|-------------|
| `longbow_turboquant_compression_ratio` | GaugeVec (layer) | Compression ratio achieved per layer over uncompressed FP16 baseline. |
| `longbow_turboquant_quantization_latency_seconds` | Histogram | Latency of TurboQuant quantization operations (PolarQuant + QJL). |

---

## 4. CUDA Engine Initialization & Lifecycle Metrics (build tag: linux && cuda)

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_cuda_engine_initialized_total` | CounterVec (model, architecture) | Total successful CUDA engine initializations. |
| `quarrel_cuda_engine_failed_total` | CounterVec (model, error_type) | Total CUDA initialization failures. |
| `quarrel_cuda_memory_bytes` | GaugeVec (model) | VRAM memory allocated and tracked by the CUDA engine. |

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
| `kv_cache_sliding_window_total` | Counter | Count of sliding window KV cache operations. |
| `kv_cache_unique_positions` | Histogram | Number of unique KV cache positions used. |

---

## 6. PromptCache

| Metric | Type | Description |
|--------|------|-------------|
| `kv_cache_hits_total` | Counter | Prefix tokens retrieved from prompt cache (skips prefill computation). |
| `kv_cache_misses_total` | Counter | Tokens requiring new attention computation. |
| `kv_cache_evictions_total` | Counter | Blocks evicted from prompt cache under memory pressure. |

---

## 7. Model Hot-Swapping & MoE

| Metric | Type | Description |
|--------|------|-------------|
| `model_hot_swap_total` | Counter | Total model hot-swap operations initiated via API. |
| `model_hot_swap_duration_seconds` | Histogram | Time taken to load replacement model weights and resume engine. |
| `model_hot_swap_errors_total` | Counter | Total number of failed model hot-swap operations. |

---

## 8. Speculative Decoding

| Metric | Type | Description |
|--------|------|-------------|
| `speculative_tokens_accepted_total` | Counter | Total number of draft tokens accepted by the target model. |
| `speculative_tokens_rejected_total` | Counter | Total number of draft tokens rejected by the target model. |

---

## 9. LoRA Dispatch

| Metric | Type | Description |
|--------|------|-------------|
| `lora_dispatch_groups_per_batch` | Histogram | Number of unique LoRA adapter groups per batch. |

---

## 10. Sampling Quality

| Metric | Type | Description |
|--------|------|-------------|
| `sampling_temperature` | Histogram | Temperature values used in sampling. |
| `sampling_top_k` | Histogram | Top-K values used in sampling. |
| `sampling_top_p` | Histogram | Top-P values used in sampling. |
| `sampling_entropy` | Histogram | Logit entropy as a quality metric. |
| `sampling_repetition_penalty` | Histogram | Repetition penalty values used. |
| `sampling_nan_handling_total` | Counter | NaN/Inf values handled during sampling. |
| `sampling_seed_reproducible_total` | Counter | Reproducible seeded sampling operations. |

---

## 11. Tokenizer Performance

| Metric | Type | Description |
|--------|------|-------------|
| `tokenizer_encode_time_seconds` | Histogram | Latency of BPE / SentencePiece encoding process. |
| `tokenizer_decode_time_seconds` | Histogram | Latency of BPE / SentencePiece decoding process. |
| `tokenizer_unknown_tokens_total` | Counter | Total unknown token substitutions encountered. |

---

## 12. Numerical Stability Audit

| Metric | Type | Description |
|--------|------|-------------|
| `logit_nan_count_total` | Counter | NaN values detected in output logits. |
| `logit_extreme_values_total` | Counter | Logit values exceeding +/- 1000 in probability distribution. |
| `logit_flat_distribution_total` | Counter | Flat logit distributions detected (low entropy). |
| `logit_max_value` | Histogram | Maximum logit value observed. |
| `logit_min_value` | Histogram | Minimum logit value observed. |
| `logit_mean_value` | Histogram | Mean logit value observed. |

---

## 13. Activation Health

| Metric | Type | Description |
|--------|------|-------------|
| `activation_healthy_total` | Counter | Healthy activation flows (no collapse, saturation, or jumps). |
| `activation_unhealthy_total` | Counter | Collapsed or saturated activation flows detected. |
| `activation_jumps_total` | Counter | Large activation jumps detected between layers. |
| `activation_rmsnorm_max` | Histogram | Maximum activation value after RMSNorm. |
| `activation_swiglu_max` | Histogram | Maximum activation value after SwiGLU. |
| `activation_residual_max` | Histogram | Maximum activation value after residual addition. |

---

## 14. NaN Propagation Audit

| Metric | Type | Description |
|--------|------|-------------|
| `nan_detected_total` | Counter | NaN propagation events detected during forward pass. |
| `nan_pattern_gradual_total` | Counter | Gradual NaN propagation patterns (slowly increasing). |
| `nan_pattern_sudden_total` | Counter | Sudden NaN propagation patterns (abrupt appearance). |
| `nan_pattern_scattered_total` | Counter | Scattered NaN propagation patterns (intermittent across layers). |

---

## 15. SIMD Kernel Performance

| Metric | Type | Description |
|--------|------|-------------|
| `simd_level_detected` | Gauge | SIMD level detected: 0=scalar, 1=AVX2, 2=AVX-512. |
| `simd_kernel_duration_seconds` | HistogramVec (kernel, size_class) | Duration of SIMD kernel operations. |
| `simd_softmax_duration_seconds` | Histogram | Duration of softmax SIMD operation. |
| `simd_rmsnorm_duration_seconds` | Histogram | Duration of RMSNorm SIMD operation. |
| `simd_matmul_duration_seconds` | Histogram | Duration of matmul SIMD operation. |
| `simd_fallback_count_total` | CounterVec (from, to) | Number of times SIMD fell back to lower implementation. |

---

## 16. MOE (Mixture of Experts)

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_moe_layer_latency_seconds` | Histogram | MOE layer forward pass latency. |
| `quarrel_moe_expert_selection_total` | CounterVec (layer, expert_id) | Frequency of expert selection per layer. |
| `quarrel_moe_routing_latency_seconds` | Histogram | Latency of top-K MoE router gating. |
| `quarrel_moe_expert_utilization` | GaugeVec (layer, expert_id) | Expert utilization rate (selections / total tokens). |

---

## 17. Gemma4 Architecture

| Metric | Type | Description |
|--------|------|-------------|
| `gemma4_sliding_window_layers_total` | Counter | Total number of sliding window attention layers processed. |
| `gemma4_full_attention_layers_total` | Counter | Total number of full attention layers processed. |
| `gemma4_q_norm_applied_total` | Counter | Count of Q normalization operations applied. |
| `gemma4_k_norm_applied_total` | Counter | Count of K normalization operations applied. |
| `gemma4_v_norm_applied_total` | Counter | Count of V normalization operations applied (with_scale=false). |

---

## 18. Dequantization Audit

| Metric | Type | Description |
|--------|------|-------------|
| `dequant_max_abs_error` | Histogram | Maximum absolute dequantization error. |
| `dequant_max_rel_error` | Histogram | Maximum relative dequantization error. |
| `dequant_pass_total` | Counter | Passing dequantization accuracy checks. |
| `dequant_fail_total` | Counter | Failing dequantization accuracy checks. |

---

## 19. Weight Alignment Audit

| Metric | Type | Description |
|--------|------|-------------|
| `weight_aligned_total` | Counter | Properly aligned weight tensors (no padding required). |
| `weight_not_aligned_total` | Counter | Misaligned weight tensors requiring padding. |
| `weight_padding_bytes` | Histogram | Number of padding bytes in weight tensors. |

---

## 20. Buffer Sizing Audit

| Metric | Type | Description |
|--------|------|-------------|
| `buffer_gqa_ratio` | Histogram | GQA ratio (heads / kv_heads). |
| `buffer_alignment_total` | Counter | Properly aligned buffer allocations. |
| `buffer_invalid_total` | Counter | Invalid buffer configurations detected. |

---

## 21. Arrow Flight & Batch Queue

| Metric | Type | Description |
|--------|------|-------------|
| `arrow_flight_bytes_transferred_total` | Counter | Total bytes transferred via Arrow Flight. |
| `batch_queue_depth` | Gauge | Current number of requests waiting for inference. |
| `batch_running_sequences` | Gauge | Current number of active sequences being processed. |

---

## 22. TPU Engine (build tag: linux && tpu)

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_tpu_engine_initialized_total` | CounterVec (model, architecture) | Total TPU engine initializations. |
| `quarrel_tpu_inference_total` | CounterVec (model) | Total TPU inference calls. |
| `quarrel_tpu_tokens_generated_total` | CounterVec (model) | Total tokens generated on TPU. |
| `quarrel_tpu_memory_bytes` | GaugeVec (model) | Current TPU memory usage. |
