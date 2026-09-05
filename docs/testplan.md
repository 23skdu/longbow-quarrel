# Longbow-Quarrel Test Plan: Verification vs llama.cpp

Comprehensive test plan for verifying speed, correctness, and accuracy of Longbow-Quarrel against llama.cpp as the reference implementation.

---

## 1. Test Environment

### Hardware
- **CPU:** AMD/Intel x86_64 with AVX-512 or AVX2 support (or Apple Silicon for Metal)
- **GPU:** NVIDIA GPU with 8+ GB VRAM (for CUDA tests)
- **RAM:** 32 GB minimum
- **Storage:** SSD with 50 GB free

### Software
- **Quarrel:** v0.3.0 (latest build from `main`)
- **llama.cpp:** Latest release (b4500+)
- **OS:** Ubuntu 22.04+ or macOS 14+

### Models (Same GGUF Files for Both)

| Model | Quantization | Size | Purpose |
|-------|-------------|------|---------|
| Qwen3.5-4B | Q8_0 | ~4.4 GB | Primary benchmark model |
| Qwen3.5-4B | Q4_K | ~2.5 GB | Quantized comparison |
| Llama-3.2-3B | Q8_0 | ~3.2 GB | Cross-architecture |
| Mistral-7B | Q4_K | ~4.1 GB | Sliding window test |
| Gemma-4-9B | BF16 | ~18 GB | BF16 correctness |

---

## 2. Speed Benchmarks

### 2.1 Throughput (Tokens/Second)

**Method:** Generate 512 tokens from a fixed prompt, measure total time.

```bash
# Quarrel
./quarrel -model Qwen3.5-4B-Q8_0.gguf -prompt "$(cat prompts/benchmark.txt)" -n 512 -temp 0

# llama.cpp
./llama-cli -m Qwen3.5-4B-Q8_0.gguf -p "$(cat prompts/benchmark.txt)" -n 512 -t 0
```

**Measurements:**

| Metric | Quarrel | llama.cpp | Delta |
|--------|---------|-----------|-------|
| Prompt eval (tok/s) | | | |
| Generation (tok/s) | | | |
| Total time (s) | | | |

### 2.2 Time-to-First-Token (TTFT)

**Method:** Measure latency from prompt submission to first generated token.

```bash
# Quarrel (with PromptCache - second call should be faster)
./quarrel -model Qwen3.5-4B-Q8_0.gguf -prompt "Short prompt" -n 1 -stream

# llama.cpp
./llama-cli -m Qwen3.5-4B-Q8_0.gguf -p "Short prompt" -n 1
```

**Measurements:**

| Metric | Quarrel (Cold) | Quarrel (Cached) | llama.cpp |
|--------|---------------|------------------|-----------|
| TTFT (ms) | | | |

### 2.3 Batch Prefill Performance

**Method:** Measure prefill speed for varying prompt lengths (128, 512, 2048, 8192 tokens).

| Prompt Length | Quarrel (tok/s) | llama.cpp (tok/s) | Speedup |
|--------------|-----------------|-------------------|---------|
| 128 | | | |
| 512 | | | |
| 2048 | | | |
| 8192 | | | |

### 2.4 Memory Usage

**Method:** Peak RSS during 512-token generation.

| Metric | Quarrel | llama.cpp |
|--------|---------|-----------|
| Peak RSS (MB) | | |
| VRAM Used (MB) | | |
| Heap Allocated (MB) | | |

### 2.5 Partial GPU Offloading

**Method:** Vary `-ngl` and measure throughput on CUDA builds.

| GPU Layers | Quarrel (tok/s) | llama.cpp -ngl (tok/s) |
|-----------|-----------------|----------------------|
| 0 (CPU only) | | |
| 8 | | |
| 16 | | |
| 24 | | |
| 32 (full GPU) | | |

---

## 3. Correctness Tests

### 3.1 Token-Level Parity

**Method:** Generate 256 tokens with temperature=0 (deterministic) and compare output token IDs.

```bash
# Both should produce identical token sequences with same seed
./quarrel -model Qwen3.5-4B-Q8_0.gguf -prompt "The capital of France is" -n 256 -temp 0 -seed 42
./llama-cli -m Qwen3.5-4B-Q8_0.gguf -p "The capital of France is" -n 256 -temp 0 -s 42
```

**Expected:** 100% token-level match for deterministic generation.

### 3.2 Quantization Format Correctness

**Method:** For each quantization format, generate 100 tokens at temperature=0 and compare against llama.cpp reference.

| Format | Token Match Rate | Notes |
|--------|-----------------|-------|
| Q8_0 | | |
| Q4_K | | |
| Q6_K | | |
| Q4_0 | | |
| Q5_0 | | |
| Q5_K | | |
| Q2_K | | |
| Q3_K | | |
| BF16 | | |
| FP16 | | |

### 3.3 F16/BF16 Weight Correctness

**Method:** Load a BF16 model, generate text, verify output is coherent (not garbage/zeros).

```bash
# Quarrel - BF16 model should produce coherent text
./quarrel -model Gemma-4-9B-BF16.gguf -prompt "Explain quantum computing" -n 100

# llama.cpp reference
./llama-cli -m Gemma-4-9B-BF16.gguf -p "Explain quantum computing" -n 100
```

**Expected:** Both produce identical or near-identical coherent text.

### 3.4 Chat Template Correctness

**Method:** Send multi-turn chat prompts and verify correct formatting.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0
  }'
```

**Expected:** Model receives properly formatted prompt with correct delimiters for the model family (e.g., `<|im_start|>system` for Qwen, `<|begin_of_text|>` for Llama 3).

### 3.5 Top-P Sampling Correctness

**Method:** Verify applyTopPCPU correctly filters tokens outside the nucleus.

```bash
# Generate with very low top_p (0.1) - should produce very repetitive text
./quarrel -model Qwen3.5-4B-Q8_0.gguf -prompt "Write a poem" -n 100 -temp 1.0 -topp 0.1
```

**Expected:** Output vocabulary is highly restricted, consistent with top-10% probability mass.

---

## 4. Accuracy Tests

### 4.1 Coherence Evaluation

**Method:** Generate 512 tokens on 10 standard prompts and rate coherence on 1-5 scale.

Prompts:
1. "Explain the theory of relativity in simple terms"
2. "Write a Python function to sort a list"
3. "What are the benefits of exercise?"
4. "Describe the water cycle"
5. "Write a haiku about technology"
6. "What is the capital of France?"
7. "How does photosynthesis work?"
8. "Explain machine learning to a child"
9. "Write a short story about a robot"
10. "What are the three laws of thermodynamics?"

| Prompt | Quarrel Score | llama.cpp Score | Notes |
|--------|--------------|-----------------|-------|
| 1 | | | |
| 2 | | | |
| ... | | | |

### 4.2 Instruction Following

**Method:** Test instruction-following with structured prompts.

```bash
# List generation
./quarrel -model Qwen3.5-4B-Q8_0.gguf -prompt "List 5 fruits, one per line:" -n 50 -temp 0

# JSON generation
./quarrel -model Qwen3.5-4B-Q8_0.gguf -prompt "Return a JSON object with keys name and age:" -n 50 -temp 0
```

**Expected:** Quarrel produces structured output matching the instruction format.

### 4.3 Multi-Turn Conversation

**Method:** Verify context retention across 5 conversation turns via API.

**Expected:** Model correctly references information from earlier turns.

---

## 5. Feature-Specific Tests

### 5.1 Speculative Decoding

**Method:** Enable speculative decoding and measure acceptance rate.

```bash
# Via API
curl -X POST http://localhost:8080/v1/completions \
  -d '{"model": "default", "prompt": "Write a story", "max_tokens": 256, "speculative": true, "draft_k": 4}'
```

**Expected:** Acceptance rate >= 50% with matched draft model; throughput improvement over non-speculative.

### 5.2 LoRA Adapter Loading

**Method:** Load a LoRA adapter and verify output changes.

```bash
# Without LoRA
./quarrel -model Llama-3.2-3B.gguf -prompt "Hello, my name is" -n 20 -temp 0

# With LoRA
./quarrel -model Llama-3.2-3B.gguf -lora adapter.gguf -prompt "Hello, my name is" -n 20 -temp 0
```

**Expected:** Output differs between base and LoRA-adapted model.

### 5.3 PromptCache TTFT Reduction

**Method:** Send the same prompt twice, measure TTFT difference.

**Expected:** Second call TTFT is significantly lower than first call (target: >50% reduction).

### 5.4 Sliding Window Attention

**Method:** Generate 4096+ tokens with Mistral-7B and verify memory stability.

```bash
./quarrel -model Mistral-7B-Q4_K.gguf -prompt "$(python3 -c 'print("hello " * 2000)')" -n 4096
```

**Expected:** No OOM, memory usage stays bounded by window size.

---

## 6. Regression Tests

### 6.1 Existing Fuzz Tests

Run all existing fuzz test corpora to verify no regressions:

```bash
go test ./internal/simd/... -fuzz FuzzMatMul -fuzztime 60s
go test ./internal/engine/... -fuzz FuzzApplyLayerCPU -fuzztime 60s
go test ./internal/gguf/... -fuzz FuzzDequantizeQ4K_SIMD -fuzztime 60s
go test ./internal/engine/... -fuzz FuzzSampler -fuzztime 60s
```

### 6.2 Data Race Detection

```bash
go test -race ./internal/...
```

**Expected:** 0 data races.

### 6.3 Security Audit

```bash
go vet ./...
gosec ./...
```

**Expected:** 0 issues.

---

## 7. Automated Benchmark Script

```bash
#!/bin/bash
# scripts/benchmark_vs_llamacpp.sh

QUARREL=./quarrel
LLAMACLI=./llama-cli
MODEL=$1
PROMPT="The quick brown fox jumps over the lazy dog. "

echo "=== Quarrel vs llama.cpp Benchmark ==="
echo "Model: $MODEL"
echo ""

# Quarrel throughput
echo "--- Quarrel ---"
$QUARREL -model "$MODEL" -prompt "$PROMPT" -n 512 -temp 0 2>&1 | grep -E "tokens|tok/s"

# llama.cpp throughput
echo "--- llama.cpp ---"
$LLAMACLI -m "$MODEL" -p "$PROMPT" -n 512 -t 0 2>&1 | grep -E "eval|speed"
```

---

## 8. Acceptance Criteria

| Category | Criterion | Target |
|----------|-----------|--------|
| Speed | Generation throughput | Within 20% of llama.cpp (CPU), within 15% (GPU) |
| Speed | TTFT (cached) | >50% reduction vs cold start |
| Correctness | Token-level parity (temp=0) | >=99% match for Q8_0 |
| Correctness | BF16/F16 weight decoding | Coherent output, no zeros/garbage |
| Accuracy | Coherence score | >=4.0/5.0 average |
| Memory | Peak RSS (4B model) | <25% of llama.cpp RSS |
| Stability | Data races | 0 |
| Security | gosec vulnerabilities | 0 |
