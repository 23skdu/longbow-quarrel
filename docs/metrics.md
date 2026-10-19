# Metrics Reference (v0.3.0)

> **Version scope.** This document describes release **v0.3.0**, the latest
> git tag. `main` carries additional unreleased work; see
> [nextsteps.md](nextsteps.md) for what has landed and what is planned.

Comprehensive Prometheus metrics reference for monitoring inference performance, GPU layer offloading, TurboQuant compression, model behavior, and system health in Longbow-Quarrel.

---

## 1. Core Engine Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `inference_tokens_total` | Counter | Total tokens generated across all inference requests. |
| `inference_duration_seconds` | Histogram | Total latency spent in the token generation loop. |
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

---

## 10. Sampling Quality

| Metric | Type | Description |
|--------|------|-------------|
| `sampling_temperature` | Histogram | Temperature values used in sampling. |
| `sampling_top_k` | Histogram | Top-K values used in sampling. |
| `sampling_top_p` | Histogram | Top-P values used in sampling. |
| `sampling_entropy` | Histogram | Logit entropy as a quality metric. |
| `sampling_repetition_penalty` | Histogram | Repetition penalty values used. |

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

---

## 19. Weight Alignment Audit

| Metric | Type | Description |
|--------|------|-------------|
| `weight_padding_bytes` | Histogram | Number of padding bytes in weight tensors. |

---

## 20. Buffer Sizing Audit

| Metric | Type | Description |
|--------|------|-------------|
| `buffer_gqa_ratio` | Histogram | GQA ratio (heads / kv_heads). |

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

---

## 23. Metrics Not Previously Documented

These metrics are registered by the code and were missing from the
reference above. The `internal/metrics` test suite now checks that this
table and the registry agree in both directions, so a metric cannot be
documented without existing, nor exist without being documented.

| Metric | Type | Description |
|--------|------|-------------|
| `activation_collapsed_layers` | Gauge | Layers whose activations collapsed during the last forward pass. |
| `activation_saturated_layers` | Gauge | Layers whose activations saturated during the last forward pass. |
| `arrow_flight_embeddings_pushed_total` | Counter | Total embedding batches pushed over Arrow Flight. |
| `gemma4_context_length_tokens` | Gauge | Configured Gemma 4 context length. |
| `gemma4_full_head_dim` | Gauge | Configured head dimension of the full-attention layers. |
| `gemma4_layer_pattern_ratio` | Gauge | Ratio of sliding-window to full-attention layers. |
| `gemma4_partial_rope_factor` | Gauge | Partial RoPE factor used by the full-attention layers. |
| `gemma4_rope_theta` | Gauge | RoPE theta configured for the Gemma 4 model. |
| `gemma4_sliding_head_dim` | Gauge | Configured head dimension of the sliding-window layers. |
| `gemma4_sliding_window_size` | Gauge | Configured sliding-window size. |
| `head_dim_threadgroup_size` | Gauge | Thread-group size used for the head-dimension kernels. |
| `host_memory_allocated_bytes` | Gauge | Current bytes allocated in host memory by the inference engine. |
| `logit_rms` | Gauge | Root-mean-square of the sampled logits. |
| `nan_layer_end` | Gauge | Index of the last layer in which a NaN was detected. |
| `nan_layer_start` | Gauge | Index of the first layer in which a NaN was detected. |
| `nan_total_count` | Counter | Total number of NaN values detected. |
| `quarrel_active_prefill_tokens` | Gauge | Tokens currently being prefilled. |
| `quarrel_continuous_batch_active_requests` | Gauge | Requests currently active in the continuous batch. |
| `quarrel_continuous_batch_iteration_duration_seconds` | Histogram | Duration of one continuous batching iteration. |
| `quarrel_continuous_batch_preemptions_total` | Counter | Total sequences preempted by the continuous batch scheduler. |
| `quarrel_flash_attention_prefill_duration_seconds` | Histogram | Duration of a Flash Attention prefill. |
| `quarrel_flash_attention_sliding_window_tokens_total` | Counter | Total tokens attended with a sliding window. |
| `quarrel_grammar_constrained_tokens_total` | Counter | Total tokens produced under a grammar constraint. |
| `quarrel_grammar_filter_duration_seconds` | Histogram | Duration of a grammar filter pass. |
| `quarrel_inter_token_latency_seconds` | Histogram | Latency between two consecutive generated tokens. |
| `quarrel_prefill_chunk_latency_seconds` | Histogram | Latency of a single prefill chunk. |
| `quarrel_simd_vnni_dot_product_duration_seconds` | Histogram | Duration of a SIMD VNNI dot product. |
| `quarrel_speculative_acceptance_rate` | Gauge | Fraction of drafted tokens that were accepted. |
| `quarrel_speculative_accepted_tokens_total` | Counter | Total drafted tokens that were accepted. |
| `quarrel_speculative_draft_tokens_total` | Counter | Total tokens produced by the draft model. |
| `quarrel_speculative_dynamic_draft_length` | Gauge | Current dynamic draft length. |
| `quarrel_time_to_first_token_seconds` | Histogram | Time to the first generated token. |
| `quarrel_vlm_image_encoding_duration_seconds` | Histogram | Duration of a vision encoder invocation. |
| `quarrel_vlm_patches_processed_total` | Counter | Total image patches processed by the vision encoder. |
| `rope_deviation` | Gauge | Maximum deviation of the RoPE output from the reference. |
| `rope_deviation_ratio` | Gauge | Ratio of the RoPE deviation to the signal magnitude. |
| `rope_fail_total` | Counter | RoPE parity checks that failed. |
| `rope_pass_total` | Counter | RoPE parity checks that passed. |
| `sampling_top_token_probability` | Gauge | Probability of the most likely sampled token. |
| `sampling_unique_samples` | Gauge | Number of distinct tokens observed while sampling. |
| `simd_attention_duration_seconds` | Histogram | Duration of a SIMD attention kernel. |
| `simd_fused_mlp_duration_seconds` | Histogram | Duration of a SIMD fused MLP kernel. |
| `simd_rope_duration_seconds` | Histogram | Duration of a SIMD RoPE kernel. |
| `simd_swiglu_duration_seconds` | Histogram | Duration of a SIMD SwiGLU kernel. |
| `softmax_mask_value` | Gauge | The value used to mask out the softmax denominator. |
| `tokenizer_encode_length` | Histogram | Number of tokens produced by the tokenizer. |
| `tokenizer_vocab_size` | Gauge | Size of the tokenizer vocabulary. |

---

## 24. Metrics Declared in Other Packages

The metrics below are declared outside `internal/metrics` (in
`internal/device`, `internal/simd` and `cmd/webui/handlers`) and were
missing from the reference. The `internal/metrics` test suite now checks
the table and the source in both directions.

| Metric | Type | Description |
|--------|------|-------------|
| `quarrel_cuda_dequant_gemm_duration_seconds` | Histogram | Duration of a CUDA dequantization GEMM. |
| `quarrel_cuda_vram_saved_bytes` | Gauge | VRAM saved by keeping weights quantized. |
| `quarrel_distributed_transfer_bytes_total` | Counter | Total bytes transferred between distributed workers. |
| `quarrel_distributed_transfer_duration_seconds` | Histogram | Duration of a distributed tensor transfer. |
| `quarrel_kv_cache_compression_ratio` | Gauge | Ratio of the compressed KV cache size to the uncompressed one. |
| `quarrel_kv_cache_quantized_pages_total` | Counter | Total KV cache pages stored in quantized form. |
| `quarrel_memory_governor_actions_total` | Counter | Total actions taken by the memory governor. |
| `quarrel_memory_pressure_events_total` | Counter | Total memory pressure events observed. |
| `quarrel_memory_pressure_ratio` | Gauge | Ratio of used to available memory. |
| `quarrel_tpu_attention_seconds` | Histogram | Attention time on the TPU backend. |
| `quarrel_tpu_batch_size` | Gauge | Current batch size on the TPU backend. |
| `quarrel_tpu_dequantization_seconds` | Histogram | Dequantization time on the TPU backend. |
| `quarrel_tpu_engine_failed_total` | Counter | Total TPU engine initialisation failures. |
| `quarrel_tpu_inference_duration_seconds` | Histogram | Duration of a TPU inference call. |
| `quarrel_tpu_layer_latency_seconds` | Histogram | Latency of a single TPU layer. |
| `quarrel_tpu_sampling_seconds` | Histogram | Sampling time on the TPU backend. |
| `quarrel_tpu_tokens_per_second` | Gauge | Token generation rate on the TPU backend. |
| `quarrel_webui_connections_active` | Gauge | Active WebUI connections. |
| `quarrel_webui_errors_total` | Counter | Total WebUI errors. |
| `quarrel_webui_inference_duration_seconds` | Histogram | Duration of a WebUI inference request. |
| `quarrel_webui_requests_total` | Counter | Total WebUI requests. |
| `quarrel_webui_tokens_total` | Counter | Total tokens generated through the WebUI. |
| `simd_dispatch_total` | Counter | Total SIMD kernel dispatches. |
| `simd_inf_count_total` | Counter | Total Inf values detected in SIMD kernel outputs. |
| `simd_kernel_errors_total` | Counter | Total SIMD kernel errors. |
| `simd_nan_count_total` | Counter | Total NaN values detected in SIMD kernel outputs. |
