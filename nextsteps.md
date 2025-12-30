# Next Steps: Metal Backend Refinement

## High Priority

- [ ] **Tune Sampling Parameters:** Verify text quality with non-zero temperature and penalties.
- [ ] **Verify RoPE Rotation:** Confirm if "Half-Half" is optimal or if "Interlaced" (NeoX) is required for Mistral v0.3.
- [ ] **Validate Output Norm:** Check if `FinalRMSNorm` handles F32 residuals correctly with F32 weights.

## Performance

- [ ] **Optimize Q4K Kernels:** Revisit `linear_q4k` for performance tuning (simd groups).
- [ ] **Async Dispatch:** Re-enable async dispatch with proper synchronization points.

## Testing

- [ ] **Run Full Benchmark:** Execute `benchmark.sh` to measure Tokens/Sec.
- [ ] **Unit Tests:** Add regression tests for RMSNorm F32->F16 conversion.
