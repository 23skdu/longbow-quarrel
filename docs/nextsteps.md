# Longbow-Quarrel - Project Status

## Completed Features (v0.1.0)

All major features from the original 10-part plan have been implemented:

| Feature | Status | Location |
|--------|--------|----------|
| Metal GPU Backend | ✅ Complete | `internal/device/metal.go` |
| CUDA GPU Backend | ✅ Complete | `internal/device/cuda.go` |
| GGUF Model Loading | ✅ Complete | `internal/gguf/` |
| Sliding Window Attention | ✅ Complete | Mistral 4096 tokens |
| Gemma4 Hybrid Attention | ✅ Complete | 5 sliding + 1 full per 6 layers |
| OpenAI API Endpoints | ✅ Complete | `/v1/chat/completions`, `/v1/completions` |
| Benchmark Tool | ✅ Complete | `cmd/benchmark` |
| Output Validation | ✅ Complete | `compareTokenSequences()` |
| cuDNN Flash Attention | ✅ Complete | `internal/device/cudnn.go` |
| Fused Kernels | ✅ Complete | `cudaFusedAttention`, `cudaFusedRoPE` |

## Unit Tests

```bash
# GPU kernel tests
go test -tags darwin,metal ./internal/device/... -v

# Engine tests
go test ./internal/engine/... -v
```

## Performance

- Custom Metal compute kernels
- Fused kernel optimizations
- Grouped Query Attention (GQA)
- Tensor pooling and memory budget management

## Future Work

Possible enhancements for future releases:

1. **More Quantizations**: Q5_K, Q2_K support
2. **Multi-GPU**: NCCL-based distributed inference
3. **FlashAttention v2**: Further optimization
4. **Speculative Decoding**: Faster token generation

## Docker

```bash
# Linux CPU
docker run ghcr.io/23skdu/longbow-quarrel:latest --model model.gguf -prompt "Hello"

# Linux CUDA (requires build from source with cuda_kernels)
docker build -f Dockerfile.cuda -t longbow-quarrel:cuda .
```

---

*Last updated: April 2026*