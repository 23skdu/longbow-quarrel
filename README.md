
<img width="2784" height="1536" alt="quarrel_logo" src="https://github.com/user-attachments/assets/e1ab45ae-f4de-4f68-91a5-fe931a720c21" />

# longbow-quarrel

High-performance LLM inference engine written in Go with Metal GPU acceleration for Apple Silicon. Integrates with the Longbow ecosystem via Apache Arrow Flight gRPC for zero-copy tensor transfer between services.

## Architecture Integration

```
longbow-fletcher (embeddings)  -->  Apache Arrow Flight  -->  longbow-quarrel (inference)
                                           gRPC                          |
                                           (tensor transfer)              v
                                    zero-copy serialization         Text Output
```

- **Input**: Accepts pre-computed embeddings via Arrow Flight gRPC or direct GGUF loading
- **Output**: Returns logits or generated text via gRPC streaming or HTTP/WebSocket
- **Protocol**: Apache Arrow Flight for efficient tensor transfer, HTTP/REST for client integration

## Supported Models

| Architecture | Status | Notes |
|--------------|--------|-------|
| Llama 3 / 3.1 | Supported | Full attention + sliding window |
| Llama 3.2 | Supported | 1B-8B variants |
| Mistral | Supported | Sliding window attention |
| Gemma 4 | Beta | Hybrid attention (unit tests passing) |
| Qwen2 | Supported | |
| SmolLM2 | Supported | 135M/360M variants |

## Supported Quantizations

| Type | Status | Use Case |
|------|--------|----------|
| Q4_K_M | Full acceleration | Default, best quality/size |
| Q4_K_S | Full acceleration | Smaller models |
| Q6_K | Full acceleration | Higher precision |
| Q3_K | Full acceleration | Memory constrained |
| IQ4_NL | Full acceleration | Improved 4-bit |
| FP16 | Full acceleration | High quality |
| FP32 | Supported | Reference |

## Performance

- Custom Metal compute kernels (MatMul, RMSNorm, RoPE, SwiGLU, Attention, KV cache)
- Fused kernel optimizations (RMSNorm+Linear, attention scaling)
- Sliding window attention for Mistral/Gemma4 architectures
- Grouped Query Attention (GQA) support
- Thread-safe async GPU dispatch with tensor pooling and memory budget

## Technical Stack

- **Language**: Go with CGO for Metal interop
- **GPU**: Metal framework (macOS/Apple Silicon)
- **Protocol**: Apache Arrow Flight gRPC, HTTP/1.1, WebSocket
- **Metrics**: Prometheus export at `/metrics`

## Quick Start

```bash
# Run with Ollama model
go run -tags darwin,metal ./cmd/generate_text/main.go -model gemma4:e4b -prompt "Hello"

# Or with GGUF file
./generate_text -model /path/to/model.gguf -prompt "Your prompt"
```

## Testing

```bash
# Unit tests
go test -tags darwin,metal ./internal/device/...

# Model loading tests
go test -tags darwin,metal ./internal/engine/...
```

## Ecosystem

Part of the Longbow LLM infrastructure:
- `longbow-fletcher`: Embedding generation
- `longbow-quarrel`: Text generation inference

See `docs/` for detailed API reference, metrics, and deployment guides.

## Gemma4 Inference (In Progress)

Gemma4 uses a hybrid attention mechanism requiring specific handling in the forward pass:

1. **Q/K Normalization**: Apply `attn_q_norm` and `attn_k_norm` RMSNorm before Q/K projections
2. **Hybrid Attention**: Per-layer switching between sliding window (local) and full attention (global)
3. **Partial RoPE (p-RoPE)**: Apply rotation to only 25% of dimensions for full attention layers

### Gemma4 Architecture Details

| Layer Type | Sliding Window | RoPE | Head Dim | KV Heads |
|------------|----------------|------|----------|----------|
| Sliding (5x) | 512 | Full (10K theta) | 256 | More |
| Full (1x) | N/A | Partial (0.25, 1M theta) | 512 | Fewer |

Unit tests for Gemma4 operations are passing. Full inference integration requires implementing the hybrid attention pattern in the `Layer()` method.
