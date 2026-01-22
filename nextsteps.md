# Mistral Coherence Recovery Plan

## Debugging Strategy (10-Point Plan)

1. [ ] **RoPE Logic Re-Verification**: Confirm calling `rope_f16` twice (for Q and K) with same `pos` is correct for Mistral.
2. [ ] **GQA Ratio Handling**: Verify `att_scores_f16` and `att_values_f16` correctly utilize the `heads / kv_heads` ratio.
3. [x] **Activation Trace Analysis**: Run `ScanMax` tracking across all 32 layers for the first token to identify collapse/saturation.
4. [ ] **Logit Range Audit**: Inspect raw logit distribution; check for flatness or extreme values.
5. [ ] **KV Cache Audit**: Ensure `CachePos` logic doesn't cause overwrites or misindexing.
6. [ ] **Scratch Buffer Sizing**: Validate `Scores` buffer sizing (`heads * seqLen * 4`) and heap non-overlap.
7. [ ] **Dequantization Accuracy**: Verify CPU-side Q6_K dequantization matches reference outputs.
8. [ ] **Weight Padding/Alignment**: Investigate `token_embd.weight` zero-padding and alignment offsets.
9. [ ] **Softmax Attention Masking**: Ensure `softmax_f16` strictly masks tokens beyond `pos`.
10. [ ] **Head Dimension Logic**: Confirm `headDim=128` handling in kernels is correct for threadgroups.

## High Priority

- [ ] Revert experimental RoPE kernel changes (Step 790 regression).
- [x] Implement enhanced `ScanMax` tracking in `engine.go`.
- [ ] Analyze Layer 0 -> Layer 31 activation flow.

## Verification

- [ ] Success Condition: `./quarrel -model mistral:latest` responds with coherent "Paris" for France prompt.
