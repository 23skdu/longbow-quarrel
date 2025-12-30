# Quarrel Next Steps - Weight Pipeline Debugging

## Current Status

**✅ Confirmed Root Cause Area:**

- llama.cpp produces "Paris" ✅
- Quarrel produces "̣uszNV" ❌
- Logit gap: 8.5 (correct -1.27, wrong 7.10)
- Issue is NOT RoPE, NOT tensor offsets, NOT basic Q4K kernel

**🔍 Leading Hypothesis:**
Q4K weight dequantization has subtle precision/accumulation issues at model scale despite passing small unit tests.

---

## Next 10 Steps: Weight Pipeline Tracing

### Step 1: Extract Real Q4K Block from Mistral Model

**Goal:** Get actual Q4K weight data from blk.0.attn_q.weight to test with

**Tasks:**

- [ ] Load GGUF file and locate blk.0.attn_q.weight tensor
- [ ] Extract first Q4K block (144 bytes)
- [ ] Save to test file for repeatability
- [ ] Log d, dmin, scales values from actual data

**Implementation:**

```go
// cmd/extract_q4k_block/main.go
func main() {
    f := gguf.LoadFile(modelPath)
    tensor := findTensor(f, "blk.0.attn_q.weight")
    block := tensor.Data[0:144]
    os.WriteFile("mistral_q4k_block_0.bin", block, 0644)
}
```

### Step 2: Validate CPU Q4K Dequantization Reference

**Goal:** Ensure Go reference implementation matches llama.cpp exactly

**Tasks:**

- [ ] Port llama.cpp's dequantize_row_q4_K verbatim to Go
- [ ] Test with actual Mistral block from Step 1
- [ ] Compare element-wise with known-good llama.cpp output
- [ ] Document any differences

**Expected:** Should match llama.cpp within float epsilon

### Step 3: Test GPU Kernel with Real Block

**Goal:** Compare GPU Q4K dequant vs CPU reference on real data

**Test:**

```go
func TestQ4K_RealMistralBlock(t *testing.T) {
    block := loadFile("mistral_q4k_block_0.bin")
    
    cpuResult := DequantizeQ4K_Reference(block)
    
    // GPU via dot product
    gpuWeight := NewQ4KTensor(1, 256)
    gpuWeight.LoadRaw(block)
    input := onesVector(256)
    gpuDot := LinearInto(input, gpuWeight)
    
    expectedDot := sum(cpuResult)
    
    if abs(gpuDot - expectedDot) > 0.001 {
        t.Errorf("GPU != CPU")
    }
}
```

### Step 4: Add Dequant Logging to Metal Kernel

# Immediate Next Steps: Solving the Weight Scaling Mystery

We have confirmed that the Mistral model weights (specifically Q4K/Q6K scales `d`) appear to be **100x smaller** than expected (`~0.01` vs `~1.0`), causing the model to freeze. A manual 100x boost restored reactivity. Our goal is to find the **origin** of this scaling factor.

## 10-Step Investigation Plan

1. **GGUF Header Inspection**:
    - Create `cmd/inspect_gguf` to dump all Key-Value pairs in the GGUF header.
    - Look for `general.quantization_version`, `tokenizer.ggml.model`, or any custom `scale` attributes.

2. **Analyze `llama.cpp` Loading Logic**:
    - Audit `llama.cpp/ggml-quants.c` and `llama.cpp/llama.cpp`.
    - Does `llama.cpp` apply a hidden global scaling factor during load?
    - Does it apply `1.0 / 100.0` or similar?

3. **Inspect `output_norm.weight` Scale**:
    - Use `extract_q4k_block` to read `output_norm.weight` (Q6K).
    - If `d` is also ~0.01, the issue is **Global** (All weights).
    - If `d` is ~1.0, the issue is **Layer-Specific** (Attention/FFN only).

4. **Verify `Float16ToFloat32` Subnormal Handling**:
    - We previously suspected subnormal bugs.
    - Verify if `0.01` corresponds to a specific subnormal/normal boundary in FP16 that we are misinterpreting.
    - Compare our `Float16ToFloat32` against a C reference for the specific `d` bits found in the model.

5. **Check for `LayerScale` Tensors**:
    - Does the model contain tensors named `blk.N.layer_scale`?
    - If so, we might be ignoring them, and their default value (or inverse) is the missing factor.

6. **RMSNorm Kernel Epsilon Check**:
    - Verify the `eps` (epsilon) value passed to kernels.
    - If `eps` is huge (e.g. 1e-4 vs 1e-5) or handled wrongly, it could squash the norm.
    - (Note: Weight scaling `~0.01` suggests the weight itself is small, independent of the kernel).

7. **Raw Hex Validation**:
    - Use `hexdump -C -s <offset> -n 32 <model_file>` to verify the raw bytes on disk match what `extract_q4k_block` reports.
    - Rule out `mmap` or file reading corruption.

8. **Activation Snapshot Comparison**:
    - Run `llama-cli` with `--trace` or instrument it to print the norm of the first embedding vector.
    - Compare with `Longbow-Quarrel` embedding norm.
    - If Embedding Norm is equal but Attention Norm is 100x off, the issue is in the Quantized Weights.

9. **Q4K Block Layout verification**:
    - Re-read `k-quants` spec.
    - Are we reading `d` from the correct offset? (We used offset 208 for Q6K, consistent with spec, but triple check).

10. **Implement the Fix**:
    - **Scenario A (Global Scale)**: Add a `GlobalScale` config parameter (default 100.0?) or auto-detect based on `attn_norm` mean.
    - **Scenario B (Bug)**: Fix the specific bug (e.g. wrong offset, wrong cast).

**Hypothesis:** If layer 0 is wrong, issue is in weights/embedding

### Step 9: Test with F16 Model (if available)

**Goal:** Rule out Q4K quantization entirely

**Tasks:**

- [ ] Find/download Mistral F16 GGUF
- [ ] Run same prompt
- [ ] If F16 works → Q4K bug confirmed
- [ ] If F16 fails → deeper architecture issue

### Step 10: Cross-Reference with llama.cpp Code

**Goal:** Find what llama.cpp does differently

**Tasks:**

- [ ] Review llama.cpp's Q4K dequant implementation
- [ ] Check for any special handling of subnormals
- [ ] Look for scale factor adjustments
- [ ] Identify any Metal-specific optimizations

---

## Next 20 Steps: Tokenizer & Sampling Refinement

### Phase 1: Output Quality & Tokenizer (Steps 1-5)

1. **Decode Top Token**: Investigate token 31980 ("Invisible Separator") and 9445 ("ogy"). Why are they top?
2. **Tokenizer Flags**: Verify `AddBOS` and `SpecialTokens` handling in `internal/tokenizer`.
3. **Prompt Formatting**: Validate strict Mistral `[INST]` template compliance.
4. **Stop Token Handling**: Ensure EOS (2) and other stop tokens are correctly recognized.
5. **Detokenization Test**: Round-trip test (Encode -> Decode) to ensure no artifacts are added.

### Phase 2: Sampling Strategy (Steps 6-10)

6. **Validate Temperature**: Ensure distinct outputs at Temp=0.7 vs Temp=0.0.
2. **Implement Top-K**: Verify Top-K sampling implementation clamps the tail.
3. **Implement Top-P (Nucleus)**: Verify Top-P logic works combined with Temperature.
4. **Repetition Penalty**: Tune penalty (currently 1.1) to prevent loops without killing coherence.
5. **Greedy Sampling (ArgMax)**: Verify strict deterministic output when Temp=0.

### Phase 3: Code Cleanup & Performance (Steps 11-15)

11. **Remove Verification Logs**: Strip all `DEBUG_MAX`, `DEBUG_SCAN`, `DEBUG_Q4K_HEX` logs for speed.
2. **Benchmark Throughput**: Measure tokens/sec on Metal with scaling fixes.
3. **Memory Audit**: Check for leaks in `AutoreleasePool` usage during long generation.
4. **Preallocation**: Optimize scratch buffer reuse in `Layer` loop.
5. **Concurrency**: Verify thread safety of `Engine` for multiple requests (if applicable).

### Phase 4: Full Model Validation (Steps 16-20)

16. **MMLU Mini-Test**: Run a small set of QA questions to grade accuracy.
2. **Long Context Test**: Verify coherence at 512+ tokens context.
3. **Llama-3 Support**: Test Llama-3 8B with the same scaling logic (does it need it?).
4. **Q6K Verification**: Ensure Q6K weights (Output Head) are also scaled correctly.
5. **Release Candidate**: Tag `v0.2.0-beta` with Metal Q4K support.

**Total:** 7-10 hours focused debugging to root cause
