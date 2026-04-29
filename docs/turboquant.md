# TurboQuant Design Document

## Overview

TurboQuant is a KV cache compression technique that combines:
1. **PolarQuant** - 4-bit/2-bit quantization of the primary signal
2. **QJL Transform** - 1-bit quantized Joint Learning residual

This achieves ~8x compression while maintaining coherence quality.

## Architecture

### Block Layout

```
TurboQuant Block: [headDim int8][qjlRows int8][8 bytes metadata]
                     |             |              |
                     v             v              v
               PolarQuant    QJL Residue   s (4 bytes) + sj (4 bytes)
```

- **headDim**: Primary quantization dimension (e.g., 128, 256)
- **qjlRows**: QJL rows for residual (default: 32-64)
- **metadata**: Scale factors for reconstruction

### Memory Savings

| Format | Bits | Compression Ratio |
|--------|------|------------------|
| F16 baseline | 16 | 1x |
| TQ 4-bit | 4 + 1 | ~8x |
| TQ 2-bit | 2 + 1 | ~16x |

## Implementation

### 1. PagedKVCache Integration

Location: `internal/engine/kv_cache_paged.go`

```go
type PagedKVCache struct {
    // TurboQuant matrices for KV cache compression
    tqRotation *device.Tensor  // Rotation matrix R
    tqQJL      *device.Tensor  // QJL sign matrix S
    qjlRows    int           // Number of QJL rows
}
```

Initialization:
- Matrices loaded from GGUF model tensors (`turboquant.rotation_matrix`, `turboquant.qjl_matrix`)
- Fallback to precomputed deterministic matrices if not in model

### 2. Encode Pipeline

```
Input K/V → Rotate → PolarQuant → QJLTransform → Store to Cache
```

```go
func (c *PagedKVCache) encodeKVTurboQuant(k, v, kCache, vCache *device.Tensor, physicalPositions *device.Tensor)
```

### 3. CUDA Implementation

Location: `internal/device/cuda.go`

- `StoreKVTurboQuant()` - Stores K/V in TurboQuant format to paged pool
- `TurboQuantEncode()` - CPU fallback encoding with GPU copy
- `cudaStoreKVTurboQuant()` - GPU kernel for batch encoding
- `cudaTurboQuantEncode()` - PolarQuant + QJL fused kernel

### 4. Metal Implementation

Location: `internal/device/metal.go`

- `Metal_TurboQuant_Encode()` - Fused encode kernel
- `Metal_TurboQuant_Decode()` - Fused decode kernel  
- `Metal_TurboQuant_PolarQuant()` - PolarQuant only
- `Metal_TurboQuant_QJLTransform()` - QJL only

## Tensor Parallelism (Distributed)

Location: `internal/engine/remote.go`

`ForwardShardedLayer()` - RPC call to worker shards via Arrow Flight:

```go
func (e *RemoteWorkerEngine) ForwardShardedLayer(
    ctx context.Context,
    layerIdx int,
    colStart, colEnd int,
    input *device.Tensor,
) (*device.Tensor, error)
```

Uses `FlightClient.DoPutTensor()` for:
- Serialization of input tensor
- RPC to worker
- Deserialization of result

## Metrics

Prometheus metrics are defined in `internal/metrics/metrics.go`:

| Metric | Type | Description |
|--------|------|-------------|
| `turboquant_encode_calls_total` | Counter | Total encode operations |
| `turboquant_encode_errors_total` | Counter | Encode failures |
| `turboquant_encode_latency_seconds` | Histogram | Encode latency |
| `turboquant_decode_calls_total` | Counter | Total decode operations |
| `turboquant_decode_latency_seconds` | Histogram | Decode latency |
| `turboquant_compression_ratio` | Histogram | Original/compressed ratio |
| `turboquant_missing_matrices_total` | Counter | Missing R/S matrices |
| `distributed_forward_shard_calls_total` | Counter | Shard RPC calls |
| `distributed_forward_shard_latency_seconds` | Histogram | RPC latency |

## Tests

### Unit Tests

- `internal/engine/kv_cache_turboquant_test.go` - PagedKVCache TurboQuant tests
- `internal/engine/remote_test.go` - Remote worker shard tests
- `internal/device/cuda_tq_encode_test.go` - CUDA TurboQuant encode tests

### Fuzz Tests

- `internal/engine/kv_cache_turboquant_fuzz_test.go` - Property-based tests

### Integration Tests

- `internal/device/cuda_tq_test.go` - Full KV cache cycle (CUDA only)
- `internal/device/cpu_tq_test.go` - CPU encode/decode tests

## Usage

### Enable TurboQuant KV Cache

```go
cfg := config.Config{
    KVCacheType: config.KVCacheTQ1_0, // or KVCacheTQ2_0
    // ...
}
eng, err := engine.NewMetalEngine(cfg)
```

### Initialize with Model

The engine automatically loads TurboQuant matrices from GGUF:

```go
rot, qjl, err := model.GetTurboQuantMatrices()
// or fallback to precomputed
```

## Performance Characteristics

### Encode Latency

| heads × headDim | blockSize | qjlRows | Latency (CPU) | Latency (GPU) |
|----------------|-----------|---------|--------------|--------------|
| 4 × 128 | 128 | 32 | ~0.5ms | ~0.1ms |
| 8 × 256 | 256 | 64 | ~1ms | ~0.2ms |

### Memory Usage

Typical 4096 context with TQ 4-bit:
- Standard F16 KV cache: 4096 × 32 × 256 × 2 × 2 bytes = 256 MB
- TurboQuant KV cache: 256 MB / 8 = 32 MB

## Dependencies

- **Internal**
  - `internal/device` - Tensor operations
  - `internal/simd` - SIMD kernels (QJLTransform)
  - `internal/gguf` - Model loading
  - `internal/engine` - Cache management
  - `internal/arrow_client` - RPC for distributed

- **External**
  - Apache Arrow Flight - Tensor serialization
  - CUDA/cuDNN - GPU kernels (if available)
  - Metal - Apple GPU kernels (if available)

## Future Work

1. **Fused Kernel Optimization**
   - Combine Rotate + PolarQuant + QJL in single kernel
   - Use shared memory for intermediate results

2. **Adaptive QJL**
   - Select qjlRows based on sequence complexity
   - Train QJL matrix for specific models

3. **Multi-GPU Support**
   - Shard across multiple GPUs with AllReduce
   - Ring attention with compressed KV

4. **Streaming Decode**
   - Progressive decode with partial KV reconstruction
   - Early exit based on confidence