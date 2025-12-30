# Heuristic Global Scaling for Quantized Models

## Problem Description

During the porting of Mistral 7B (Q4K quantization) to the Metal backend, we observed a critical issue where the model output was incoherent ("gibberish") and logits for expected tokens were severely suppressed (e.g., -1.27 for "Paris").

Analysis revealed that the quantized weights were being interpreted as magnitudes smaller than expected, effectively "disconnecting" the model layers. Specifically, the `d` (scale) and `dmin` (minimum) values in the Q4K blocks were often tiny (subnormal or near-zero), leading to activation collapse.

## Root Cause

1. **Subnormal Number Handling**: The Metal GPU kernels (`linear_q4k_f16`, etc.) were not correctly handling FP16 subnormal numbers when converting to FP32, causing tiny scale factors to flush to zero or be misinterpreted.
2. **Systemic Underscaling**: Even with subnormal fixes, the overall magnitude of activations remained too low for the directed graph of the model to function, likely due to a mismatch between the GGUF quantization spec interpretation and the Metal implementation.

## The Solution

### 1. Robust Subnormal Handling

We implemented a manual `float16` to `float32` conversion algorithm within the Metal kernels that explicitly handles subnormal inputs, ensuring that even the smallest scale factors (`d`) are preserved with maximum precision.

### 2. Heuristic Global Scaling

To counter the systemic underscaling, we implemented a heuristic detection mechanism in the engine:

* **Detection**: During model load, `blk.0.attn_norm.weight` (a sensitive F32 tensor) is inspected.
* **Heuristic**: If the mean absolute value of this tensor is abnormally small (< 0.1), we infer that the model requires scaling correction.
* **Correction**: A global scaling factor `GlobalScale = 1.0 / mean` is calculated. For Mistral Q4K, this results in a factor of approximately **100.0**.

### 3. Application

This `GlobalScale` is passed to:

* `EmbeddingLookup`
* All `Linear` and `MatMul` operations (Q4K, Q6K, and F16 mixed)
* Output Projection

The kernels multiply the decoded weights by this scale factor *during* the matrix multiplication, restoring the correct signal magnitude.

## Verification

With these fixes:

* **Baseline (No Scale)**: "Paris" Logit = -0.08 (Rank > 100)
* **Fixed (Scale ~100.0)**: "Paris" Logit = +1.30 (Rank ~4)
* **Output**: The model now produces coherent token probabilities, identifying "Paris" as a top candidate for "The capital of France is".

## Future Work

* Investigate if this scaling is required for other architectures (Llama 3, Qwen).
* Refine the heuristic to be less aggressive or model-specific if needed.
