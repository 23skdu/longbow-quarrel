# Prometheus Metrics Reference

## Overview

Longbow-Quarrel WebUI exposes comprehensive Prometheus metrics at `http://localhost:9090/metrics`. These metrics enable monitoring of inference performance, WebSocket connections, and system health.

## WebUI Metrics

### HTTP Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_webui_http_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests |
| `quarrel_webui_http_request_duration_seconds` | Histogram | `method`, `endpoint` | HTTP request duration |
| `quarrel_webui_http_request_size_bytes` | Histogram | `method`, `endpoint` | HTTP request body size |
| `quarrel_webui_http_response_size_bytes` | Histogram | `method`, `endpoint` | HTTP response body size |

### WebSocket Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_webui_connections_active` | Gauge | N/A | Active WebSocket connections |
| `quarrel_webui_connections_total` | Counter | `status` | Total WebSocket connections |
| `quarrel_webui_ws_messages_total` | Counter | `direction` | WebSocket messages sent/received |
| `quarrel_webui_ws_message_duration_seconds` | Histogram | `type` | WebSocket message processing duration |

### Inference Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_webui_inference_requests_total` | Counter | `model`, `status` | Total inference requests |
| `quarrel_webui_inference_duration_seconds` | Histogram | `model` | Inference request duration |
| `quarrel_webui_tokens_total` | Counter | `model`, `type` | Total tokens generated |
| `quarrel_webui_tokens_per_second` | Gauge | `model` | Current tokens per second |
| `quarrel_webui_inference_queue_size` | Gauge | `model` | Current queue size |
| `quarrel_webui_inference_queue_full_total` | Counter | `model` | Queue full rejections |

### Model Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_webui_models_loaded` | Gauge | `model`, `backend` | Number of loaded models |
| `quarrel_webui_model_memory_bytes` | Gauge | `model` | Memory used by model |
| `quarrel_webui_model_load_duration_seconds` | Histogram | `model` | Model load time |

### Connection Manager Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_webui_active_connections` | Gauge | `model` | Active connections per model |
| `quarrel_webui_connections_rate_limited` | Counter | N/A | Rate-limited connections |
| `quarrel_webui_broadcast_messages_total` | Counter | N/A | Broadcast messages sent |

---

## Engine Metrics

### CUDA Engine Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_cuda_memory_allocated_bytes` | Gauge | `device` | GPU memory allocated |
| `quarrel_cuda_memory_total_bytes` | Gauge | `device` | GPU memory total |
| `quarrel_cuda_kernel_duration_seconds` | Histogram | `kernel` | CUDA kernel execution time |
| `quarrel_cuda_stream_in_flight` | Gauge | `device` | Operations in flight |
| `quarrel_cuda_gemm_ops_total` | Counter | `device`, `precision` | GEMM operations performed |
| `quarrel_cuda_attention_ops_total` | Counter | `device`, `heads` | Attention operations performed |

### Metal Engine Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_metal_memory_used_bytes` | Gauge | `device` | Metal memory in use |
| `quarrel_metal_command_queue_in_flight` | Gauge | `device` | Command queue depth |
| `quarrel_metal_kernel_duration_seconds` | Histogram | `kernel` | Metal kernel execution time |
| `quarrel_metal_buffer_bytes` | Gauge | `type` | Buffer memory by type |

### CPU Engine Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_cpu_gemm_ops_total` | Counter | `precision` | GEMM operations |
| `quarrel_cpu_attention_ops_total` | Counter | N/A | Attention operations |
| `quarrel_cpu_memory_bytes` | Gauge | `type` | Memory by type |

---

## Inference Metrics

### Token Generation

```
# Tokens generated per model
quarrel_tokens_total{model="smollm2", type="generated"}

# Tokens per second (rolling average)
quarrel_tokens_per_second{model="smollm2"}

# Total prompt tokens
quarrel_tokens_total{model="smollm2", type="prompt"}
```

### Inference Latency

```
# Time to first token (TTFT)
quarrel_inference_ttft_seconds{model="smollm2"}

# Time per output token (TPOT)
quarrel_inference_tpot_seconds{model="smollm2"}

# Total inference time
quarrel_inference_duration_seconds{model="smollm2"}
```

### Sampling Parameters

```
# Effective temperature used
quarrel_sampling_temperature{effective="0.72"}

# Top-K value used
quarrel_sampling_top_k{value="40"}

# Top-P value used  
quarrel_sampling_top_p{effective="0.95"}
```

---

## KV Cache Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quarrel_kv_cache_size_bytes` | Gauge | `model`, `layer` | KV cache size per layer |
| `quarrel_kv_cache_hits_total` | Counter | `model` | KV cache hits |
| `quarrel_kv_cache_misses_total` | Counter | `model` | KV cache misses |
| `quarrel_kv_cache_evictions_total` | Counter | `model` | KV cache evictions |
| `quarrel_kv_cache_hit_ratio` | Gauge | `model` | Cache hit ratio |

---

## System Metrics

### Memory

```
# Process memory
quarrel_memory_rss_bytes
quarrel_memory_alloc_bytes
quarrel_memory_heap_bytes
quarrel_memory_stack_bytes

# GPU memory (if available)
quarrel_gpu_memory_used_bytes{device="0"}
quarrel_gpu_memory_total_bytes{device="0"}
```

### CPU

```
# Goroutines
quarrel_goroutines_total

# CPU usage
quarrel_cpu_usage_percent

# GC metrics
quarrel_gc_pause_seconds_total
quarrel_gc_runs_total
```

### Network

```
# WebSocket connections
quarrel_ws_connections_active
quarrel_ws_connections_total

# Message rates
quarrel_ws_messages_received_total
quarrel_ws_messages_sent_total
```

---

## Example Queries

### Basic Queries

```promql
# Requests per second over last 5 minutes
rate(quarrel_webui_http_requests_total[5m])

# Error rate
rate(quarrel_webui_http_requests_total{status=~"5.."}[5m]) 
  / 
rate(quarrel_webui_http_requests_total[5m])

# Active WebSocket connections
quarrel_webui_connections_active
```

### Inference Performance

```promql
# Average tokens per second
avg(rate(quarrel_tokens_total[5m])) by (model)

# Inference duration percentiles
histogram_quantile(0.95, 
  rate(quarrel_inference_duration_seconds_bucket[5m]))

# Queue latency
histogram_quantile(0.99,
  rate(quarrel_inference_queue_latency_seconds_bucket[5m]))
```

### Memory Usage

```promql
# GPU memory usage
quarrel_cuda_memory_allocated_bytes / quarrel_cuda_memory_total_bytes

# Process memory growth
delta(quarrel_memory_rss_bytes[1h])

# Memory by model
quarrel_model_memory_bytes by (model)
```

### Cache Performance

```promql
# KV cache hit ratio
quarrel_kv_cache_hits_total / 
  (quarrel_kv_cache_hits_total + quarrel_kv_cache_misses_total)
```

---

## Grafana Dashboards

Pre-configured dashboards available in `docker/grafana/provisioning/dashboards/`:

### WebUI Overview
- HTTP request rate and latency
- WebSocket connections
- Active models

### Inference Performance
- Tokens per second by model
- Inference latency percentiles
- Queue size and wait times

### Resource Usage
- GPU/CPU memory usage
- Goroutine count
- GC pause times

### Model Details
- Per-model metrics
- Cache hit ratios
- Kernel performance

---

## Alert Rules

Example Prometheus alerting rules:

```yaml
groups:
- name: longbow-quarrel
  rules:
  - alert: HighErrorRate
    expr: |
      rate(quarrel_webui_http_requests_total{status=~"5.."}[5m]) 
      / rate(quarrel_webui_http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate on {{ $labels.endpoint }}"

  - alert: ModelQueueFull
    expr: quarrel_inference_queue_full_total > 100
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Model queue frequently full"

  - alert: GPU memory critical
    expr: |
      quarrel_cuda_memory_allocated_bytes / quarrel_cuda_memory_total_bytes > 0.9
    labels:
      severity: critical
    annotations:
      summary: "GPU memory above 90%"

  - alert: InferenceSlow
    expr: |
      histogram_quantile(0.95, 
        rate(quarrel_inference_duration_seconds_bucket[5m])) > 10
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "P95 inference time above 10s"
```

---

## Metric Collection Configuration

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'longbow-quarrel-webui'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 15s
    scrape_timeout: 10s
```

### Remote Write (Optional)

```yaml
remote_write:
  - url: "https://prometheus-remote-write.example.com/api/v1/write"
    basic_auth:
      username: prometheus
      password_file: /etc/prometheus/password
```

---

## Performance Benchmarks

### Expected Performance Ranges

| Metric | Typical | Warning | Critical |
|--------|---------|---------|----------|
| HTTP latency (p95) | <100ms | <500ms | >1s |
| TTFT | <50ms | <200ms | >500ms |
| Tokens/sec (GPU) | 50-100 | 20-50 | <20 |
| Tokens/sec (CPU) | 5-20 | 1-5 | <1 |
| Queue wait time | <1s | <5s | >10s |

---

## Troubleshooting Metrics

### High Latency Diagnosis

```promql
# Isolate bottleneck
rate(quarrel_inference_duration_seconds[5m])
  by (stage)

# Possible stages:
# - queue_wait
# - model_load
# - prompt_processing
# - token_generation
# - post_processing
```

### Memory Issues

```promql
# Memory leak detection
delta(quarrel_memory_rss_bytes[1h]) > 100*1024*1024

# GPU memory growth
rate(quarrel_cuda_memory_allocated_bytes[1h]) > 50*1024*1024
```

### Cache Effectiveness

```promql
# Low cache hit ratio
quarrel_kv_cache_hit_ratio < 0.5

# High eviction rate
rate(quarrel_kv_cache_evictions_total[5m]) > 100
```
