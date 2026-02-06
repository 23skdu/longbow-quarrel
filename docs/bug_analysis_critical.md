# Multi-Token Corruption Debugging Report

**Status:** CRITICAL ISSUE IDENTIFIED - Model produces all-zero outputs

## Summary

Initial fix applied to Phase 2 logits path (`scratch.Normed.LinearToFP32_Into()`), but testing revealed a **deeper issue**:

### Critical Finding

**All layer outputs are zeros:**
```
[Layer 1 RMSNorm Output] Min: -0.0000 Max: -0.0000 Mean: 0.0000 RMS: 0.0000 Zeros: 4096/4096
```

This causes the model to consistently output `<unk>` (likely token ID 0) instead of meaningful text.

### Evidence

| Test | Prompt | Expected | Actual | Status |
|---|---|---|---|---|
| Single token | "The" | "The" | "<unk>" | 🚨 ALL ZEROS |
| Multi-token | "The capital of France is" | "Paris" | "<unk><unk><unk>" | 🚨 ALL ZEROS |

### Root Cause Analysis

The issue is **NOT just multi-token generation** - the entire model forward pass is producing zero outputs. Possible causes:

1. **Embedding Lookup** - Fetching wrong embedding or failing silently
2. **Attention Mechanism** - Producing zero attention outputs
3. **FFN Layers** - All forward/middle/down passes producing zeros
4. **Position Embeddings (RoPE)** - Incorrect position encoding
5. **Model Weights** - Corruption during loading or dequantization
6. **Layer Normalization** - RMSNorm producing zeros

### Fixes Attempted

| Fix | File | Status | Impact |
|---|---|---|---|
| GPU buffer zeroing | `metal.go:354, 438` | ✅ Applied | Prevents stale data |
| KV cache zeroing | `kv_cache_sliding_window.go:83,92` | ✅ Applied | Prevents cache corruption |
| Phase 2 logits path | `engine.go:1734` | ✅ Applied | Matches Phase 1 |
| Sampler investigation | `sampler.go:243-260` | 🟡 Identified | In-place modification noted |

### Next Required Actions

This is now a **deep model debugging issue** requiring:

1. **Verify embedding lookup** - Check if `TokenEmb.EmbeddingLookup()` returns correct values
2. **Debug attention** - Verify Q/K/V computation produces non-zero outputs
3. **Debug FFN** - Verify gate/up/down operations produce non-zero outputs
4. **Verify RoPE** - Check position encoding
5. **Compare with llama.cpp** - Validate model weights are correct

### Timeline

- [x] Applied Phase 2 logits fix (5 min)
- [ ] Identified all-zero outputs issue (current)
- [ ] Deep model debugging required (next)

### Recommendations

1. **Use Xcode GPU Frame Capture** - Inspect Metal kernel execution and memory state
2. **Add intermediate value logging** - Log embedding values, attention outputs, FFN outputs at each layer
3. **Disable quantization testing** - Test with unquantized weights to isolate quantization issues
4. **Verify model file integrity** - Compare GGUF hashes with known good versions

---

**Generated:** 2026-01-29
**Status:** CRITICAL - Model not functioning correctly, produces all-zero outputs
