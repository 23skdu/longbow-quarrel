# Longbow-Quarrel: Cloud-Native Deployment Guide (Phase 4)

This guide provides the necessary infrastructure patterns for deploying Longbow-Quarrel within a Kubernetes-orchestrated environment.

## 1. Kubernetes Resource Limits

To ensure system stability, strictly define memory limits. The engine's fault tolerance logic will engage at 95% of these limits to prevent OOM kills.

```yaml
resources:
  limits:
    memory: "16Gi" # Zero-copy inference drastically reduces memory pressure
    nvidia.com/gpu: 1 # For CUDA nodes (optional, or use CPU SIMD mode)
  requests:
    cpu: "4"
    memory: "8Gi"
```

> **Note on Zero-Copy Inference (v0.3.0):** With direct memory-mapped matrix-vector multiplication (`MatVecMulQ8_0`, `MatVecMulQ4_K`, `MatVecMulQ6_K`, `MatVecMulQ2_K`, `MatVecMulQ3_K`, `MatVecMulQ4_0`, `MatVecMulQ5_0`, `MatVecMulQ5_K`, `MatVecMulBF16`), RAM allocations on the Go heap remain `< 50 MB` even for 4B–8B models. The requested memory primarily accommodates OS page cache and the paged KV cache. When GPU VRAM is limited, use `-ngl <layers>` to offload a portion of layers to GPU while running remaining layers on host CPU.


## 2. Health & Readiness Probes

### Liveness Probe (`/healthz`)
Ensures the process is responsive and not stuck in a deadlock.

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
```

### Readiness Probe (`/readyz`)
Ensures the engine is fully initialized and weights are loaded before receiving traffic.

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8080
  initialDelaySeconds: 60
  periodSeconds: 5
```

## 3. Graceful Degradation

Longbow-Quarrel implements **Proactive Fault Tolerance**. When memory pressure exceeds 95% of the configured `MaxMemory`, the engine will:
1.  Return `503 Service Unavailable` for new inference requests.
2.  Maintain responsiveness of the `/healthz` endpoint to prevent the pod from being restarted.
3.  Automatically resume normal operations when memory is freed (e.g., after long-running sequences finish).

## 4. Distributed Sharding (Arrow Flight)

For large models requiring multi-GPU sharding, use the following environment configuration:

- `QUARREL_SHARD_ROLE`: `master` or `worker`
- `QUARREL_WORKER_ADDRS`: Comma-separated list of worker Flight endpoints.

The master will coordinate tensor sharding across workers using zero-copy Arrow Flight streams.
