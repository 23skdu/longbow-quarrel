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

**Goal:** See actual d, dmin, d_val values inside GPU kernel

**Tasks:**

- [ ] Add Metal printf to log first block's d, dmin
- [ ] Log d_val, m_val for first group
- [ ] Compare with CPU reference values
- [ ] Identify any divergence in conversion

**Check:** Are subnormal values being preserved correctly?

### Step 5: Test Full Matrix Dequantization

**Goal:** Verify 4096x4096 Q4K matrix dequantizes correctly

**Tasks:**

- [ ] Load entire blk.0.attn_q.weight (4096x4096, Q4K)
- [ ] Dequantize on GPU
- [ ] Spot-check values against CPU dequant
- [ ] Measure min/max/mean of dequantized weights

**Expected:** Min ~-0.5, Max ~0.5 (typical for Q4K)

### Step 6: Trace First Linear Transformation

**Goal:** Verify blk.0.attn_q produces correct Q projection

**Tasks:**

- [ ] Get actual embedding from token 1782 ("The")
- [ ] Run through blk.0.attn_q on CPU (reference)
- [ ] Run through blk.0.attn_q on GPU
- [ ] Compare output element-wise

**Critical:** This is where divergence likely occurs

### Step 7: Add Checkpoint Logging in Layer Method

**Goal:** Capture intermediate values at each step

**Add logging after:**

- [ ] Embedding lookup
- [ ] Each RMSNorm
- [ ] Q/K/V projections
- [ ] Attention output
- [ ] FFN output

**Output:** JSON with max/mean/sample for each checkpoint

### Step 8: Binary Search for Divergence Layer

**Goal:** Find exact layer where output diverges

**Tasks:**

- [ ] Run Mistral, capture outputs after each layer
- [ ] Compare layer N output with expected pattern
- [ ] Binary search: if layer 16 wrong, check layer 8, etc.
- [ ] Identify first bad layer

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

## Quick Wins to Try First

### A. Test Simpler Model

Try Qwen-2.5-0.5B or TinyLlama to see if issue is Mistral-specific

### B. Disable Q4K Altogether

Force dequant Q4K → F16 on load, test if that fixes it

### C. Compare Metal vs CPU

Run first layer on CPU, rest on GPU - does it work?

---

## Success Criteria

**Phase 1 (Steps 1-3):** Confirm GPU Q4K matches CPU reference  
**Phase 2 (Steps 4-6):** Identify where first divergence occurs  
**Phase 3 (Steps 7-10):** Root cause identified and fixed  

**Final:** Mistral outputs "Paris" with logit > 7.0

---

## Tools & Scripts Needed

1. `cmd/extract_q4k_block/` - Extract real weight data
2. `cmd/trace_layer_outputs/` - Checkpoint logging
3. `scripts/compare_with_llama.py` - Diff analysis
4. `internal/device/q4k_reference.go` - Exact llama.cpp port

---

## Time Estimates

- Steps 1-3: 2-3 hours (weight extraction & validation)
- Steps 4-6: 3-4 hours (tracing pipeline)
- Steps 7-10: 2-3 hours (binary search & fix)

**Total:** 7-10 hours focused debugging to root cause
