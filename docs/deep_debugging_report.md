# Deep Debugging Results - Multi-Token Corruption

## Critical Issue Identified

**Status:** MODEL COMPLETELY BROKEN - All outputs are UNK (token ID 0)

### Evidence

| Test | Prompt | Expected Output | Actual Output | Token ID |
|---|---|---|---|---|
| Our engine | "The" | "The" | "<unk>" | 0 |
| Our engine | "The capital of" | "The capital of" | "<unk>" | 0 |
| Our engine | "The capital of France is" | "Paris" | "<unk>" | 0 |
| llama.cpp | "The" | (timed out) | N/A | N/A |

### Summary

- **ALL generated tokens are ID 0** (UNK token)
- This affects ALL prompts, even single-token generation
- llama.cpp also hangs/timeouts on this model
- This suggests a **fundamental model loading or weight corruption issue**

### Root Cause Analysis

The issue is **NOT**:
1. ❌ Phase 2 logits path (we fixed this)
2. ❌ GPU buffer initialization (we fixed this)
3. ❌ KV cache initialization (we fixed this)

**The issue IS:**
1. ⚠️ **Model weights are corrupted or incorrectly loaded**
2. ⚠️ **Embedding lookup fails silently**
3. ⚠️ **All layer operations produce zeros**

### Tests Performed

| Test | Result | Notes |
|---|---|---|
| Single-token generation | Produces UNK | Token ID 0 |
| Multi-token generation | Produces UNK | Token ID 0 |
| Layer debug output | All zeros at every layer | Min=0.0, Max=0.0 |
| llama.cpp comparison | Timed out | Also broken? |
| GPU buffer zeroing | Applied | No change in output |
| KV cache zeroing | Applied | No change in output |
| Phase 2 logits fix | Applied | No change in output |

### Fixes Attempted

#### Fix 1: GPU Buffer Zeroing ✅
**File:** `internal/device/metal.go` (lines 354-380, 431-449)
**Change:** Added `Metal_ZeroBufferGPU()` after `Metal_Alloc()`
**Status:** Applied successfully
**Impact:** Prevented stale data in reused buffers
**Result:** No improvement in output

#### Fix 2: KV Cache Zeroing ✅
**File:** `internal/engine/kv_cache_sliding_window.go` (lines 83, 92)
**Change:** Added `k.ZeroInit()` and `v.ZeroInit()` after allocation
**Status:** Applied successfully
**Impact:** Prevented stale data in KV cache
**Result:** No improvement in output

#### Fix 3: Phase 2 Logits Path ✅
**File:** `internal/engine/engine.go` (line 1734)
**Change:** Changed from `normedF32.ToF32() → normedF32.LinearF32_Into()` to `scratch.Normed.LinearToFP32_Into()`
**Status:** Applied successfully
**Impact:** Matches Phase 1 direct computation path
**Result:** No improvement in output

### Remaining Issues

The model is fundamentally broken. All layer outputs are zeros, causing the output to be UNK tokens.

**Possible causes:**
1. **GGUF model file corruption** - The model file may be corrupted
2. **Weight loading bug** - Dequantization may be failing silently
3. **Embedding table corruption** - Token embedding lookup returns zeros
4. **Metal kernel bug** - All kernel operations produce zeros
5. **Model file mismatch** - GGUF may not match the expected architecture

### Recommendations

#### Immediate Actions

1. **Verify GGUF file integrity**
   ```bash
   # Check if GGUF file is corrupted
   python3 -c "import gguf; f = gguf.GGUF('model.gguf'); print(f'Tensors: {len(f.tensors)}, version: {f.version}')"
   ```

2. **Test with different model**
   - Try a different GGUF model file (not TinyLlama)
   - Determine if issue is model-specific or system-wide

3. **Disable quantization completely**
   - Force FP16 weights to rule out dequantization bugs
   - Requires unquantized GGUF file or temporary FP16 conversion

4. **Metal GPU Frame Capture**
   - Use Xcode Instruments > GPU Capture
   - Run model through GPU capture
   - Inspect kernel execution and memory state

5. **Compare llama.cpp build**
   - Build llama.cpp from source with Metal support
   - Test same GGUF file
   - Compare behavior

6. **Check embedding table**
   - Verify token embeddings are loaded correctly
   - Test embedding lookup for known token IDs
   - Confirm embeddings are non-zero

#### Deep Debugging

If the issue persists after above checks:

1. **Add verbose Metal logging**
   - Modify Metal kernels to log intermediate values
   - Check Q/K/V matrices at each layer
   - Verify attention scores are non-zero

2. **Isolate layer by layer**
   - Test with only 1 layer
   - Incrementally add layers to find which layer fails
   - Determine if issue is in specific layer type

3. **Verify Metal shader compilation**
   - Check Metal shaders compile correctly
   - Verify no compilation warnings or errors
   - Test on different GPU architectures

4. **Memory leak detection**
   - Run with `-race` detector
   - Check for undefined behavior
   - Verify GPU memory is properly released

---

**Generated:** 2026-01-29
**Status:** CRITICAL - Model outputs all UNK tokens, fundamental issue not resolved
