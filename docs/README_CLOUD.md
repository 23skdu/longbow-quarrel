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

> **Note on Zero-Copy Inference (v0.3.0):** With the direct matrix-vector
> multiplication over the memory-mapped weights (the formats are in
> [features.md](features.md#2-quantization-and-formats)), the allocations on
> the Go heap stay under 50 MB even for 4B–8B models, and under 20 MB
> for a 4B one. The requested memory accommodates the page cache and the
> paged KV cache. When the VRAM is limited, use `-ngl <layers>` to
> offload part of the layers to the GPU and run the rest on the host.


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

## 4. The Helm Chart

The chart in `helm/quarrel` deploys the server, the API and the metrics
service, the probes and the ServiceMonitor.

```bash
helm install quarrel helm/quarrel \
  --set model.name=smollm2:135m-instruct \
  --set engine.ngl=32
```

The values that matter:

| Value | Default | What it does |
|-------|---------|--------------|
| `model.name` | `smollm2:135m-instruct` | The model to load |
| `service.port` | `8080` | The API listener (`-api-addr`) |
| `service.metricsPort` | `9090` | The metrics listener (`-metrics`) |
| `engine.ngl` | `-1` | The layers to offload to the GPU |
| `engine.maxBatchSize` | `16` | The continuous batching size |
| `engine.kvCache.blockSize` | `16` | The paged attention block size |
| `engine.speculative.draftModel` | `""` | A draft model for speculative decoding |
| `engine.activationLog` | `""` | Where to write the activation log |

The API and the metrics are two separate listeners, and the ServiceMonitor
scrapes the metrics one. A chart that pointed the monitor at the API port
would have scraped nothing.

## 5. Distributed and multi-device inference

> **Not available, and the failure is loud.** Earlier revisions of this
> document named `QUARREL_SHARD_ROLE` and `QUARREL_WORKER_ADDRS`. Neither
> variable exists: no part of the code reads them, so setting them did
> nothing at all.
>
> The Arrow Flight server is started with `-flight <addr>` and serves
> single-request inference over `DoGet`. The *sharded* forward — the
> part that would spread a model over several machines — is not
> implemented: the worker rejects the tensor forward with `Unimplemented`
> rather than silently returning zeroed activations. The Flight client
> that used to live in the repository is gone; a client is the Arrow
> library's own.
>
> The tensor-parallel and pipeline-parallel managers were unit-tested and
> never used: the engine loads the whole model onto one device and never
> calls a collective. They have been deleted, so there is no code that
> looks like it shards a model and does not. `-devices` with a *single*
> device id does work — it selects the CUDA device the engine runs on —
> but `-devices 0,1`, `-tp-size 2` and `-pp-stages 2` exit with an error
> rather than printing a topology the engine never applies.
>
> To spread a model over several machines, run one process per machine
> behind a load balancer.
