# Metrics Reference

## Overview

Comprehensive Prometheus metrics for monitoring inference performance, model behavior, and system health.

## Engine Metrics

### Kernel Performance

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `gpu_kernel_duration_seconds` | Histogram | `kernel` | Metal kernel execution time |
| `gpu_memory_allocated_bytes` | Gauge | N/A | Current GPU memory in use |

### Inference

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `inference_tokens_total` | Counter | N/A | Total tokens generated |
| `inference_duration_seconds` | Summary | N/A | Inference step duration |

## Model-Specific Metrics

### Gemma4 Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gemma4_sliding_window_layers_total` | Counter | Sliding window attention layers processed |
| `gemma4_full_attention_layers_total` | Counter | Full attention layers processed |
| `gemma4_sliding_window_size` | Gauge | Configured sliding window (default: 512) |
| `gemma4_partial_rope_factor` | Histogram | p-RoPE rotation factor (0.25 = 25%) |
| `gemma4_rope_theta` | Histogram | RoPE theta (10K sliding, 1M full) |
| `gemma4_q_norm_applied_total` | Counter | Q normalization operations |
| `gemma4_k_norm_applied_total` | Counter | K normalization operations |
| `gemma4_layer_pattern_ratio` | Histogram | Sliding:full attention ratio (5:1) |
| `gemma4_sliding_head_dim` | Gauge | Sliding attention head dim (256) |
| `gemma4_full_head_dim` | Gauge | Full attention head dim (512) |
| `gemma4_context_length_tokens` | Histogram | Context length distribution |

### KV Cache

| Metric | Type | Description |
|--------|------|-------------|
| `kv_cache_sliding_window_total` | Counter | Sliding window cache operations |
| `kv_cache_overlap_total` | Counter | Cache position overlaps |
| `kv_cache_hits_total` | Counter | Cache hits |
| `kv_cache_misses_total` | Counter | Cache misses |

### Quantization

| Metric | Type | Description |
|--------|------|-------------|
| `dequant_max_abs_error` | Histogram | Max absolute dequantization error |
| `dequant_pass_total` | Counter | Passing accuracy checks |
| `dequant_fail_total` | Counter | Failing accuracy checks |

## Query Examples

```promql
# Tokens per second
rate(inference_tokens_total[5m])

# Kernel performance
histogram_quantile(0.95, rate(gpu_kernel_duration_seconds_bucket[5m]))

# KV cache effectiveness
kv_cache_hits_total / (kv_cache_hits_total + kv_cache_misses_total)
```
