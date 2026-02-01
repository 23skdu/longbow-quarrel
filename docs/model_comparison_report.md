# Multi-Model Comparison Results

## Finding: System-Wide Issue

**Date:** 2026-01-29

### Models Tested

| Model | Path | Architecture | Quantization | Status |
|---|---|---|---|---|
| **TinyLlama 1.1B** | `/Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816` | Llama | Q4_0 | 🚨 ALL UNK |
| **Granite 4B** | `/Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf4c8da9534ed72cc632d005aeed6547f1e8662ccdfae688364e` | Llama | Q4_0 | 🚨 ALL UNK |
| **Mistral 7B** | `/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f` | Llama | Q4_K | 🚨 ALL UNK |

### Test Results

#### Common Test: Simple Prompt "The"

**TinyLlama:**
```
=== Prompt 1: "The" ===
Token ID: 0
Token Text: "<unk>"
⚠️  WARNING: Token ID 0 (likely UNK)
```

**Granite:**
```
=== Prompt 1: "The" ===
Token ID: 0
Token Text: "!"
⚠️  WARNING: Token ID 0 (likely UNK)
```

**Mistral:**
```
=== Prompt 1: "The" ===
Token ID: 0
Token Text: "<unk>"
⚠️  WARNING: Token ID 0 (likely UNK)
```

### Analysis

**Key Finding:** **All three models produce IDENTICAL broken behavior:**

1. ✅ All return token ID 0 (UNK token)
2. ✅ All output meaningless text ("<unk>" or "!")
3. ✅ All suffer from same all-zero layer outputs
4. ✅ All different architectures (1.1B, 4B, 7B)
5. ✅ All different quantizations (Q4_0, Q4_0, Q6_K)

**Conclusion:** This is a **SYSTEM-WIDE FUNDAMENTAL ISSUE**, not model-specific.

### Root Cause Hypotheses

Since issue affects all models regardless of architecture/quantization, likely causes:

1. **Embedding lookup failure** - `TokenEmb.EmbeddingLookup()` returns all zeros
2. **Metal kernel bug** - All GPU operations produce zeros
3. **RMSNorm failure** - Normalization produces all-zero outputs
4. **RoPE (Position Embeddings)** - Position encoding broken
5. **Shared scratch buffers** - Scratch tensor reuse causing corruption

### Fixes Applied (No Impact)

| Fix | Files | Status |
|---|---|---|
| GPU buffer zeroing | `metal.go:354, 438` | ✅ Applied, no change |
| KV cache zeroing | `kv_cache_sliding_window.go:83, 92` | ✅ Applied, no change |
| Phase 2 logits path | `engine.go:1734` | ✅ Applied, no change |

### Comparison with llama.cpp

- **llama.cpp** also hangs/timeouts on this model
- Suggests **model file corruption** or **fundamental GPU issue**
- Not a quantization difference (both use Q4_0/Q6_K)

### Next Steps

1. **Verify GGUF integrity** - Check if model file is corrupted
2. **Test with different model files** - Try a fresh download
3. **Metal GPU Frame Capture** - Use Xcode Instruments
4. **Add intermediate logging** - Log embedding values, Q/K/V matrices
5. **Isolate layer-by-layer** - Test each layer individually
6. **Check Metal shader compilation** - Verify no warnings/errors

### Status

**SEVERITY:** CRITICAL - All models completely broken, produce UNK tokens only

**RECOMMENDATION:** This requires **deep Metal GPU debugging** or **model file replacement**. Simple fixes have no impact.

---

**Generated:** 2026-01-29
**Models tested:** 3 (TinyLlama, Granite, Mistral)
**All models:** Identical broken behavior
