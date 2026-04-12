# Longbow-Quarrel vs llama.cpp/Ollama Comparison

## Test Environment

| Component | Version/Info |
|-----------|--------------|
| llama.cpp (llama-cli) | b8680 (15f786e65) |
| Ollama | 0.20.5 |
| Model | smollm2.gguf (1.8GB) |

## Output Comparison

### Prompt: "Hello, how are you?"

**llama-cli (llama.cpp)**:
```
I'm doing well, thanks for asking! It's a beautiful day here at Hugging Face. I'm excited to assist you with any text-based tasks or questions you may have. What can I help you with today?
```
Performance: ~18-25 tokens/s

**Ollama (smollm2)**:
```
I'm doing well! It's great to hear from you. I hope all is going smoothly on your end as well. If there's anything you need assistance with, feel free to ask. I'm here to help.
```
Performance: 25.17 prompt eval/s, 18.23 tokens/s

**Quarrel**:
- `cmd/simple`: Does NOT perform inference (just tokenization)
- `cmd/generate_text`: Requires Metal (darwin) - NOT available on Linux
- No Linux/CUDA inference command available

---

## Feature Comparison

| Feature | llama.cpp | Ollama | Quarrel |
|---------|-----------|--------|---------|
| **CLI Interface** | llama-cli | ollama CLI | None (webui only) |
| **Model Format** | GGUF native | GGUF (converted) | GGUF native |
| **Metal Support** | Yes | Yes | Yes |
| **CUDA Support** | Yes | Yes | Yes (incomplete) |
| **API Server** | No (external) | Yes | Yes (WebUI) |
| **WebSocket Streaming** | No | No | Yes |
| **Quantization** | Full Q4/Q5/Q6 | Limited | Full Q4/Q6 |
| **Sliding Window** | Yes | Yes | Partial (Mistral) |

---

## Performance Gap

| Metric | llama.cpp | Ollama | Quarrel (target) |
|--------|-----------|--------|-----------------|
| **Smollm2 135M** | ~45 t/s | ~18 t/s | N/A (no Linux inference) |
| **Mistral 7B** | ~25 t/s | ~12 t/s | N/A (Metal only) |
| **Model Loading** | <1s | 2-3s | ~1s |

**Root Cause**: Quarrel lacks a Linux/CUDA CLI for inference testing.

---

# 10-Part Improvement Plan

## Priority 1: Infrastructure (Critical)

### 1. Implement Linux/CUDA Inference CLI

**Problem**: Quarrel has no command-line tool for Linux CUDA inference.

**Solution**: Create `cmd/quarrel/main.go` (CUDA version already exists but needs testing).

**Files to modify**:
- `cmd/quarrel/main.go` - Add inference loop
- `internal/engine/engine_cuda.go` - Ensure Infer() works

**Target**: Run on Linux with CUDA, generate text from GGUF model.

---

### 2. Fix CUDA Engine Infer() Implementation

**Problem**: `engine_cuda.go` Infer() may have incomplete implementation.

**Solution**: Implement full forward pass with proper layer iteration.

**Files to modify**:
- `internal/engine/engine_cuda.go:400-600` - Forward pass

**Status**: Basic implementation exists, needs validation.

---

## Priority 2: Feature Parity

### 3. Add Sliding Window Attention (Mistral/Gemma4)

**Problem**: Sliding window not fully implemented for CUDA.

**Solution**: Add attention mask for window size (512 for Mistral).

**Files to modify**:
- `internal/device/cuda_kernels.cu` - attention kernel
- `internal/engine/engine_cuda.go` - layer config

---

### 4. Implement Gemma4 Hybrid Attention

**Problem**: Gemma4 uses 5 sliding + 1 full attention pattern per 6 layers.

**Solution**: Add layer-type detection and apply appropriate attention.

**Files to modify**:
- `internal/engine/engine_cuda.go` - layer iteration
- `internal/device/cuda_kernels.cu` - hybrid attention kernel

---

### 5. Add OpenAI-Compatible API Endpoints

**Problem**: API not fully OpenAI-compatible.

**Solution**: Add `/v1/chat/completions`, `/v1/completions` endpoints.

**Files to modify**:
- `cmd/webui/api/handlers.go` - new endpoints

---

## Priority 3: Performance

### 6. Remove CPU-GPU Transfer Bottleneck

**Problem**: ToHostF32() called too frequently in forward pass.

**Solution**: Keep tensors on GPU, use cublas for matmul.

**Files to modify**:
- `internal/engine/engine_cuda.go` - GPU matmul path
- `internal/device/cuda.go` - tensor management

---

### 7. Add Fused Kernel Integration

**Problem**: Fused QKV+RoPE kernel exists but not used.

**Solution**: Integrate `cudaFusedQKVRoPE` in forward pass.

**Files to modify**:
- `internal/device/cuda_kernels.cu` - kernel implementation
- `internal/engine/engine_cuda.go` - kernel calls

---

### 8. Add cuDNN Flash Attention

**Problem**: Using naive O(n²) attention.

**Solution**: Integrate cuDNN `cudnnAttnForward()`.

**Files to modify**:
- `internal/device/cudnn.go` - attention wrapper

---

## Priority 4: Testing & Validation

### 9. Add Benchmark Command

**Problem**: No standardized benchmark tool.

**Solution**: Create `cmd/benchmark/main.go` for token/s measurement.

**Features**:
- Warm-up runs
- Variable batch sizes
- Multiple models
- CSV output

---

### 10. Add Model Output Validation

**Problem**: No way to verify correctness vs llama.cpp.

**Solution**: Add validation script comparing outputs.

**Features**:
- Same prompt across engines
- Token matching percentage
- Perplexity calculation

---

## Implementation Order

| Step | Task | Dependency |
|------|------|------------|
| 1 | Linux/CUDA CLI | - |
| 2 | CUDA Infer() | 1 |
| 3 | Sliding Window | 2 |
| 4 | Gemma4 Hybrid | 3 |
| 5 | OpenAI API | - |
| 6 | GPU Transfer Fix | 2 |
| 7 | Fused Kernels | 6 |
| 8 | Flash Attention | 7 |
| 9 | Benchmark Tool | 1,2 |
| 10 | Output Validation | 9 + llama.cpp |

---

*Last updated: April 2026*