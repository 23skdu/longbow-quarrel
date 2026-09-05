# Longbow-Quarrel Usage Guide (v0.3.0)

Comprehensive guide for running inference, configuring GPU layer offloading, loading LoRA adapters, leveraging universal model discovery, and deploying Longbow-Quarrel.

---

## 1. CLI Commands & Binaries

Quarrel provides two primary CLI interfaces:

1. `cmd/simple` — Lightweight, multi-platform runner supporting CPU, Metal, and CUDA with zero-copy inference and universal model resolution.
2. `cmd/quarrel` — Full-featured production server and CLI with fused CUDA kernels, continuous batching, and Prometheus telemetry.

---

## 2. Universal Model Resolution

Quarrel eliminates the requirement to type out long, cumbersome file paths. When you supply a model name to `-model`, Quarrel searches:

1. Exact filesystem path (relative or absolute, e.g. `./models/qwen.gguf`)
2. `~/.cache/llmfit/models/`
3. `~/.cache/llama.cpp/`
4. `~/.cache/huggingface/hub/`
5. `~/.ollama/models/` (using Ollama registry resolution)

### Examples
```bash
# Finds ~/.cache/llmfit/models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf
./quarrel -model Huihui-Qwen3.5

# Finds ~/.ollama/models/blobs/sha256-... for mistral:latest
./quarrel -model mistral

# Finds Hugging Face cached snapshot
./quarrel -model Llama-3.2-3B-Instruct
```

---

## 3. Partial GPU Layer Offloading (`-ngl` / `-gpu-layers`)

If a model is too large to fit entirely into GPU VRAM, you can offload a subset of layers to the GPU while retaining the remaining layers in CPU system memory. Activations roundtrip dynamically between GPU and CPU host memory with zero memory leaks.

```bash
# Offload 16 layers to GPU (remaining layers execute via CPU zero-copy SIMD):
./quarrel -model Qwen3.5 -ngl 16 -prompt "Why is the sky blue?"

# Offload all layers to GPU (if VRAM permits):
./quarrel -model mistral -gpu-layers 32
```

---

## 4. `cmd/simple` Reference

### Build
```bash
# Standard CPU build (AVX-512 / AVX2 / ARM NEON auto-detected)
go build -o quarrel-simple ./cmd/simple/

# Apple Silicon Metal build
go build -tags darwin,metal -o quarrel-simple ./cmd/simple/
```

### Options
```text
--model string       Model path or fuzzy name (e.g. "Qwen3.5", "mistral:latest")
--prompt string      Text prompt (default: "The capital of France is")
--n int              Number of tokens to generate (default: 50)
--temp float         Sampling temperature 0.0-1.0 (default: 0.7)
--topk int           Top-K filtering (default: 40)
--topp float         Top-P nucleus sampling (default: 0.95)
--min-p float        Min-P sampling threshold (default: 0.0)
--rep-penalty float  Repetition penalty (default: 1.1)
--presence-penalty float  Presence penalty (default: 0.0)
--frequency-penalty float  Frequency penalty (default: 0.0)
--ngl int            Number of GPU layers to offload (-1 for all, default: -1)
--lora string        Path to LoRA adapter .gguf file (optional)
--seed int           Random seed (default: 0 = auto)
--stream             Stream tokens as they are generated
--max-memory int     Max memory in MB (0 = no limit)
```

---

## 5. `cmd/quarrel` Reference (CUDA Production CLI & Server)

### Build
```bash
# Compiles internal/device/libcuda_kernels.a and links bin/quarrel-linux-amd64-cuda
make nvidia
```

### Options
```text
-model string       Model name or GGUF path
-prompt string      Text prompt for single inference
-n int              Tokens to generate (default: 20)
-ngl int            Number of GPU layers to offload (default: -1 for all)
-gpu-layers int     Alias for -ngl
-kv-cache int       Context window / KV cache capacity in tokens (default: 2048)
-temp float         Sampling temperature (default: 0.7)
-topk int           Top-K sampling (default: 40)
-topp float         Top-P nucleus sampling (default: 0.95)
-min-p float        Min-P sampling threshold (default: 0.0)
-rep-penalty float  Repetition penalty (default: 1.1)
-presence-penalty float  Presence penalty (default: 0.0)
-frequency-penalty float  Frequency penalty (default: 0.0)
-seed int           Random seed (0 for auto)
-stream             Stream tokens as they are generated
-lora string        Path to LoRA adapter .gguf file (optional)
-metrics string     Prometheus metrics address (default: ":9090")
-flight string      Arrow Flight server address (default: ":50051")
-max-batch-size int Maximum batch size (default: 16)
-block-size int     Paged attention block size (default: 16)
-total-blocks int   Total paged cache blocks (default: 256)
```

### Example
```bash
./bin/quarrel-linux-amd64-cuda -model Qwen3.5 -ngl 20 -kv-cache 4096
```

---

## 6. LoRA Adapter Loading

LoRA (Low-Rank Adaptation) adapters allow fine-tuning without replacing base model weights. Quarrel supports loading `.gguf` sidecar LoRA files at runtime.

### Via CLI
```bash
# Load a LoRA adapter with the simple CLI
./quarrel-simple -model Llama-3.2-3B -lora /path/to/adapter.gguf -prompt "Hello!"

# Load a LoRA adapter with the CUDA CLI
./quarrel -model Llama-3.2-3B -ngl 32 -lora /path/to/adapter.gguf
```

### Via API
```bash
# Load adapter at runtime
curl -X POST http://localhost:8080/v1/adapters/load \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/adapter.gguf", "id": "my-lora"}'

# List loaded adapters
curl http://localhost:8080/v1/adapters/list
```

### CPU Merge-on-Load
On CPU, LoRA deltas (`A × B × α/r`) are fused directly into weight matrices at load time, resulting in zero inference-time overhead.

---

## 7. Supported Architectures & Quantization

| Model Architecture | Layers / Details | Supported Quantizations |
|--------------------|------------------|-------------------------|
| **Qwen 3.5** | Hybrid GatedDeltaNet SSM + Full MHA (every 4th layer) | Q8_0, Q4_K, Q6_K, Q4_0, Q5_0, Q5_K, Q2_K, Q3_K, BF16, FP16 |
| **Llama 3 / 3.1 / 3.2** | Full Self-Attention + RoPE + GQA | Q8_0, Q4_K, Q6_K, Q4_0, Q5_0, Q5_K, Q2_K, Q3_K, BF16, FP16, FP32 |
| **Mistral** | Sliding Window (4096) + Full Attention | Q8_0, Q4_K, Q6_K, Q4_0, Q5_0, Q5_K, Q2_K, Q3_K, BF16, FP16 |
| **Gemma 4** | Hybrid Attention (5 sliding + 1 full per 6 layers) | Q8_0, Q4_K, Q6_K, Q4_0, Q5_0, Q5_K, Q2_K, Q3_K, BF16, FP16 |
| **SmolLM2 / TinyLlama** | Standard Transformer | Q8_0, Q4_K, Q6_K, Q4_0, Q5_0, Q5_K, Q2_K, Q3_K, BF16, FP16 |
| **Phi3 / Granite** | Standard Transformer | Q8_0, Q4_K, Q6_K, Q4_0, Q5_0, Q5_K, Q2_K, Q3_K, BF16, FP16 |

---

## 8. Docker Deployment

```bash
# CPU Image with Zero-Copy SIMD
docker run -p 8080:8080 -v ~/.cache:/root/.cache \
  ghcr.io/23skdu/longbow-quarrel:latest \
  -model Qwen3.5 -prompt "Hello!"

# NVIDIA CUDA GPU Image
docker run --gpus all -p 8080:8080 -v ~/.cache:/root/.cache \
  ghcr.io/23skdu/longbow-quarrel:cuda-latest \
  -model mistral -gpu-layers 32
```

---

## 9. Cloud Deployment (Kubernetes)

### Resource Limits
Define strict memory limits to prevent OOM kills. The engine engages fault tolerance at 95% of configured limits.

```yaml
resources:
  limits:
    memory: "16Gi"
    nvidia.com/gpu: 1
  requests:
    cpu: "4"
    memory: "8Gi"
```

> **Note:** With zero-copy inference, RAM allocations remain `< 50 MB` even for 4B–8B models. Requested memory accommodates OS page cache and the paged KV cache.

### Health & Readiness Probes
```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /readyz
    port: 8080
  initialDelaySeconds: 60
  periodSeconds: 5
```

### Graceful Degradation
When memory pressure exceeds 95% of `MaxMemory`, the engine returns `503` for new requests while keeping `/healthz` alive to prevent pod restarts. Normal operation resumes automatically when memory is freed.

### Distributed Sharding (Arrow Flight)
For multi-GPU sharding, configure:
- `QUARREL_SHARD_ROLE`: `master` or `worker`
- `QUARREL_WORKER_ADDRS`: Comma-separated list of worker Flight endpoints