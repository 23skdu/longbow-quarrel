package metrics

import (
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/promtest"
)

func TestMetricsExistence(t *testing.T) {
	tokens := promtest.Value(t, "inference_tokens_total", InferenceTokensTotal)
	kernels := promtest.Value(t, "gpu_kernel_duration_seconds", KernelDuration.WithLabelValues("existence_probe"))

	RecordInference(10, 100*time.Millisecond)
	RecordGPUMemory(1024 * 1024)
	RecordKernelDuration("existence_probe", 5*time.Millisecond)

	promtest.RequireDelta(t, "inference_tokens_total", InferenceTokensTotal, tokens, 10)
	promtest.RequireValue(t, "gpu_memory_allocated_bytes", GPUMemoryAllocated, 1024*1024)
	promtest.RequireDelta(t, "gpu_kernel_duration_seconds", KernelDuration.WithLabelValues("existence_probe"), kernels, 1)
}

func TestRecordInferenceMultiple(t *testing.T) {
	before := promtest.Value(t, "inference_tokens_total", InferenceTokensTotal)

	RecordInference(5, 50*time.Millisecond)
	RecordInference(10, 100*time.Millisecond)
	RecordInference(3, 30*time.Millisecond)

	promtest.RequireDelta(t, "inference_tokens_total", InferenceTokensTotal, before, 18)
}

func TestRecordGPUMemoryChanges(t *testing.T) {
	RecordGPUMemory(1024 * 1024 * 1024) // 1GB
	RecordGPUMemory(512 * 1024 * 1024)  // 512MB - gauge must update, not accumulate

	promtest.RequireValue(t, "gpu_memory_allocated_bytes", GPUMemoryAllocated, 512*1024*1024)
}

func TestRecordKernelDurationHistogram(t *testing.T) {
	kernel := KernelDuration.WithLabelValues("histogram_probe")
	before := promtest.Value(t, "gpu_kernel_duration_seconds", kernel)

	RecordKernelDuration("histogram_probe", 10*time.Millisecond)
	RecordKernelDuration("histogram_probe", 20*time.Millisecond)
	RecordKernelDuration("histogram_probe", 30*time.Millisecond)

	promtest.RequireDelta(t, "gpu_kernel_duration_seconds", kernel, before, 3)
}

func TestRecordNumericalInstability(t *testing.T) {
	nan := NumericalInstability.WithLabelValues("tensor1", "nan")
	inf := NumericalInstability.WithLabelValues("tensor2", "inf")
	nanBefore := promtest.Value(t, "numerical_instability_total", nan)
	infBefore := promtest.Value(t, "numerical_instability_total", inf)

	RecordNumericalInstability("tensor1", 5, 0) // 5 NaNs
	RecordNumericalInstability("tensor2", 0, 3) // 3 Infs

	promtest.RequireDelta(t, "numerical_instability_total/nan", nan, nanBefore, 5)
	promtest.RequireDelta(t, "numerical_instability_total/inf", inf, infBefore, 3)
}

func TestRecordValidationError(t *testing.T) {
	bounds := ValidationErrors.WithLabelValues("decode", "bounds_check")
	dtype := ValidationErrors.WithLabelValues("decode", "dtype_mismatch")
	boundsBefore := promtest.Value(t, "validation_errors_total", bounds)
	dtypeBefore := promtest.Value(t, "validation_errors_total", dtype)

	RecordValidationError("decode", "bounds_check")
	RecordValidationError("decode", "dtype_mismatch")

	promtest.RequireDelta(t, "validation_errors_total/bounds_check", bounds, boundsBefore, 1)
	promtest.RequireDelta(t, "validation_errors_total/dtype_mismatch", dtype, dtypeBefore, 1)
}

func TestRecordContextLength(t *testing.T) {
	before := promtest.Value(t, "context_length_tokens", ContextLengthHistogram)

	RecordContextLength(512)
	RecordContextLength(1024)
	RecordContextLength(2048)
	RecordContextLength(4096)

	promtest.RequireDelta(t, "context_length_tokens", ContextLengthHistogram, before, 4)
}

func TestRecordLogitAuditBasic(t *testing.T) {
	max := promtest.Value(t, "logit_max_value", LogitMaxValue)
	nan := promtest.Value(t, "logit_nan_count_total", LogitNaNCount)

	RecordLogitAudit(10.0, -5.0, 2.5, 3.0, false, false, false)

	promtest.RequireDelta(t, "logit_max_value", LogitMaxValue, max, 1)
	promtest.RequireDelta(t, "logit_nan_count_total", LogitNaNCount, nan, 0)
}

func TestRecordLogitAuditWithIssues(t *testing.T) {
	nan := promtest.Value(t, "logit_nan_count_total", LogitNaNCount)
	extreme := promtest.Value(t, "logit_extreme_values_total", LogitExtremeValues)
	flat := promtest.Value(t, "logit_flat_distribution_total", LogitFlatDistribution)

	RecordLogitAudit(1000.0, -1000.0, 0.0, 500.0, true, true, true)

	promtest.RequireDelta(t, "logit_nan_count_total", LogitNaNCount, nan, 1)
	promtest.RequireDelta(t, "logit_extreme_values_total", LogitExtremeValues, extreme, 1)
	promtest.RequireDelta(t, "logit_flat_distribution_total", LogitFlatDistribution, flat, 1)
}

func TestRecordNaNPropagationAuditPatterns(t *testing.T) {
	gradual := promtest.Value(t, "nan_pattern_gradual_total", NaNPatternGradual)
	sudden := promtest.Value(t, "nan_pattern_sudden_total", NaNPatternSudden)
	scattered := promtest.Value(t, "nan_pattern_scattered_total", NaNPatternScattered)
	detected := promtest.Value(t, "nan_detected_total", NaNDetected)

	RecordNaNPropagationAudit(2, 8, 256, "gradual")
	RecordNaNPropagationAudit(0, 4, 128, "sudden")
	RecordNaNPropagationAudit(1, 6, 64, "scattered")

	promtest.RequireDelta(t, "nan_pattern_gradual_total", NaNPatternGradual, gradual, 1)
	promtest.RequireDelta(t, "nan_pattern_sudden_total", NaNPatternSudden, sudden, 1)
	promtest.RequireDelta(t, "nan_pattern_scattered_total", NaNPatternScattered, scattered, 1)
	promtest.RequireDelta(t, "nan_detected_total", NaNDetected, detected, 3)
}

func TestRecordRoPEDeviationAudit(t *testing.T) {
	pass := promtest.Value(t, "rope_pass_total", RoPEPass)
	fail := promtest.Value(t, "rope_fail_total", RoPEFail)

	RecordRoPEDeviationAudit(0.001, 1.0, true)
	RecordRoPEDeviationAudit(0.1, 1.5, false)

	promtest.RequireDelta(t, "rope_pass_total", RoPEPass, pass, 1)
	promtest.RequireDelta(t, "rope_fail_total", RoPEFail, fail, 1)
}

func TestRecordKVCacheSlidingWindow(t *testing.T) {
	window := promtest.Value(t, "kv_cache_sliding_window_total", KVCacheSlidingWindow)
	overlap := promtest.Value(t, "kv_cache_overlap_total", KVCacheOverlap)

	RecordKVCacheSlidingWindow(4096, 100, false)
	RecordKVCacheSlidingWindow(4096, 4100, true) // wrapped

	promtest.RequireDelta(t, "kv_cache_sliding_window_total", KVCacheSlidingWindow, window, 2)
	promtest.RequireDelta(t, "kv_cache_overlap_total", KVCacheOverlap, overlap, 1)
}

func TestRecordKVCacheOutOfBounds(t *testing.T) {
	before := promtest.Value(t, "kv_cache_oob_total", KVCacheOutOfBounds)

	RecordKVCacheOutOfBounds(5000, 4096)

	promtest.RequireDelta(t, "kv_cache_oob_total", KVCacheOutOfBounds, before, 1)
}

func TestRecordKVCacheStats(t *testing.T) {
	RecordKVCacheStats(1024*1024*1024, 256*1024*1024)

	promtest.RequireValue(t, "kv_cache_capacity_bytes", KVCacheCapacityBytes, 1024*1024*1024)
	promtest.RequireValue(t, "kv_cache_used_bytes", KVCacheUsedBytes, 256*1024*1024)
}

func TestRecordActivationFlowAudit(t *testing.T) {
	healthy := promtest.Value(t, "activation_healthy_total", ActivationHealthy)
	unhealthy := promtest.Value(t, "activation_unhealthy_total", ActivationUnhealthy)
	jumps := promtest.Value(t, "activation_jumps_total", ActivationJumps)

	RecordActivationFlowAudit(0, 0, 0) // healthy
	RecordActivationFlowAudit(3, 2, 5) // unhealthy

	promtest.RequireDelta(t, "activation_healthy_total", ActivationHealthy, healthy, 1)
	promtest.RequireDelta(t, "activation_unhealthy_total", ActivationUnhealthy, unhealthy, 1)
	promtest.RequireDelta(t, "activation_jumps_total", ActivationJumps, jumps, 1)
}

func TestRecordSamplingAudit(t *testing.T) {
	temperature := promtest.Value(t, "sampling_temperature", SamplingTemperature)
	topK := promtest.Value(t, "sampling_top_k", SamplingTopK)
	topP := promtest.Value(t, "sampling_top_p", SamplingTopP)
	entropy := promtest.Value(t, "sampling_entropy", SamplingEntropy)

	RecordSamplingAudit(map[string]interface{}{
		"temperature":    0.7,
		"top_k":          40,
		"top_p":          0.95,
		"top_token_prob": 0.25,
		"unique_samples": 12,
		"entropy":        2.0,
		"rep_penalty":    1.1,
	})

	promtest.RequireDelta(t, "sampling_temperature", SamplingTemperature, temperature, 1)
	promtest.RequireDelta(t, "sampling_top_k", SamplingTopK, topK, 1)
	promtest.RequireDelta(t, "sampling_top_p", SamplingTopP, topP, 1)
	promtest.RequireDelta(t, "sampling_entropy", SamplingEntropy, entropy, 1)
}

func TestRecordTokenizerMetrics(t *testing.T) {
	length := promtest.Value(t, "tokenizer_encode_length", TokenizerEncodeLength)
	vocab := promtest.Value(t, "tokenizer_vocab_size", TokenizerVocabSize)

	RecordTokenizerMetrics(100, 32000, 5*time.Millisecond)

	promtest.RequireDelta(t, "tokenizer_encode_length", TokenizerEncodeLength, length, 1)
	promtest.RequireDelta(t, "tokenizer_vocab_size", TokenizerVocabSize, vocab, 1)
}

func TestRecordActivationPrecision(t *testing.T) {
	rmsnorm := promtest.Value(t, "activation_rmsnorm_max", ActivationRMSNormMax)
	swiglu := promtest.Value(t, "activation_swiglu_max", ActivationSwiGLUMax)
	residual := promtest.Value(t, "activation_residual_max", ActivationResidualMax)

	RecordActivationPrecision("rmsnorm", 1.5)
	RecordActivationPrecision("swiglu", 50.0)
	RecordActivationPrecision("residual", 2.0)

	promtest.RequireDelta(t, "activation_rmsnorm_max", ActivationRMSNormMax, rmsnorm, 1)
	promtest.RequireDelta(t, "activation_swiglu_max", ActivationSwiGLUMax, swiglu, 1)
	promtest.RequireDelta(t, "activation_residual_max", ActivationResidualMax, residual, 1)
}

func TestRecordMOELayerLatency(t *testing.T) {
	before := promtest.Value(t, "quarrel_moe_layer_latency_seconds", MOELayerLatency)

	RecordMOELayerLatency(50 * time.Millisecond)
	RecordMOELayerLatency(100 * time.Millisecond)

	promtest.RequireDelta(t, "quarrel_moe_layer_latency_seconds", MOELayerLatency, before, 2)
}

func TestRecordMOERoutingLatency(t *testing.T) {
	before := promtest.Value(t, "quarrel_moe_routing_latency_seconds", MOERoutingLatency)

	RecordMOERoutingLatency(5 * time.Millisecond)
	RecordMOERoutingLatency(10 * time.Millisecond)

	promtest.RequireDelta(t, "quarrel_moe_routing_latency_seconds", MOERoutingLatency, before, 2)
}

func TestRecordMOEExpertSelection(t *testing.T) {
	// One child per (layer, expert) pair, read before and after so the test
	// proves every selection landed on its own series.
	type pair struct{ layer, expert string }
	pairs := []pair{{"2", "0"}, {"2", "3"}, {"2", "5"}, {"5", "1"}, {"5", "7"}}
	before := make([]float64, len(pairs))
	for i, p := range pairs {
		before[i] = promtest.Value(t, "quarrel_moe_expert_selection_total",
			MOEExpertSelection.WithLabelValues(p.layer, p.expert))
	}

	RecordMOEExpertSelection(2, []int32{0, 3, 5})
	RecordMOEExpertSelection(5, []int32{1, 7})

	for i, p := range pairs {
		promtest.RequireDelta(t, "quarrel_moe_expert_selection_total["+p.layer+","+p.expert+"]",
			MOEExpertSelection.WithLabelValues(p.layer, p.expert), before[i], 1)
	}
}

func TestRecordMOEExpertUtilization(t *testing.T) {
	RecordMOEExpertUtilization(0, 0, 0.5)
	RecordMOEExpertUtilization(1, 5, 0.75)

	promtest.RequireValue(t, "quarrel_moe_expert_utilization[0,0]",
		MOEExpertUtilization.WithLabelValues("0", "0"), 0.5)
	promtest.RequireValue(t, "quarrel_moe_expert_utilization[1,5]",
		MOEExpertUtilization.WithLabelValues("1", "5"), 0.75)
}

func TestTotalTokensAtomic(t *testing.T) {
	// Test atomic operations
	initial := totalTokens.Load()
	RecordInference(1, time.Millisecond)
	after := totalTokens.Load()
	if after != initial+1 {
		t.Errorf("Expected totalTokens to increment by 1, got %d -> %d", initial, after)
	}
}

func TestRecordGemma4SlidingWindowLayer(t *testing.T) {
	layers := promtest.Value(t, "gemma4_sliding_window_layers_total", Gemma4SlidingWindowLayers)

	RecordGemma4SlidingWindowLayer(512, 10*time.Millisecond)
	RecordGemma4SlidingWindowLayer(4096, 15*time.Millisecond)

	promtest.RequireDelta(t, "gemma4_sliding_window_layers_total", Gemma4SlidingWindowLayers, layers, 2)
	promtest.RequireValue(t, "gemma4_sliding_window_size", Gemma4SlidingWindowSize, 4096)
}

func TestRecordGemma4FullAttentionLayer(t *testing.T) {
	layers := promtest.Value(t, "gemma4_full_attention_layers_total", Gemma4FullAttentionLayers)

	RecordGemma4FullAttentionLayer(0.25, 1000000.0, 20*time.Millisecond)
	RecordGemma4FullAttentionLayer(0.25, 1000000.0, 25*time.Millisecond)

	promtest.RequireDelta(t, "gemma4_full_attention_layers_total", Gemma4FullAttentionLayers, layers, 2)
}

func TestRecordGemma4QKNorm(t *testing.T) {
	qNorm := promtest.Value(t, "gemma4_q_norm_applied_total", Gemma4QNormApplied)
	kNorm := promtest.Value(t, "gemma4_k_norm_applied_total", Gemma4KNormApplied)

	RecordGemma4QKNorm(true, false)
	RecordGemma4QKNorm(false, true)
	RecordGemma4QKNorm(true, true)

	promtest.RequireDelta(t, "gemma4_q_norm_applied_total", Gemma4QNormApplied, qNorm, 2)
	promtest.RequireDelta(t, "gemma4_k_norm_applied_total", Gemma4KNormApplied, kNorm, 2)
}

func TestRecordGemma4VNorm(t *testing.T) {
	before := promtest.Value(t, "gemma4_v_norm_applied_total", Gemma4VNormApplied)

	RecordGemma4VNorm()
	RecordGemma4VNorm()

	promtest.RequireDelta(t, "gemma4_v_norm_applied_total", Gemma4VNormApplied, before, 2)
}

func TestRecordGemma4LayerPattern(t *testing.T) {
	before := promtest.Value(t, "gemma4_layer_pattern_ratio", Gemma4LayerPatternRatio)

	RecordGemma4LayerPattern(5, 1)
	RecordGemma4LayerPattern(10, 2)
	RecordGemma4LayerPattern(20, 4)

	promtest.RequireDelta(t, "gemma4_layer_pattern_ratio", Gemma4LayerPatternRatio, before, 3)
}

func TestRecordGemma4HeadDims(t *testing.T) {
	RecordGemma4HeadDims(256, 512)
	RecordGemma4HeadDims(128, 256)

	promtest.RequireValue(t, "gemma4_sliding_head_dim", Gemma4SlidingHeadDim, 128)
	promtest.RequireValue(t, "gemma4_full_head_dim", Gemma4FullHeadDim, 256)
}

func TestRecordGemma4Context(t *testing.T) {
	before := promtest.Value(t, "gemma4_context_length_tokens", Gemma4ContextLength)

	RecordGemma4Context(512)
	RecordGemma4Context(4096)
	RecordGemma4Context(32768)
	RecordGemma4Context(131072)

	promtest.RequireDelta(t, "gemma4_context_length_tokens", Gemma4ContextLength, before, 4)
}

func TestRecordAdditionalMetrics(t *testing.T) {
	hotSwap := promtest.Value(t, "model_hot_swap_total", ModelHotSwapTotal)
	hotSwapErrors := promtest.Value(t, "model_hot_swap_errors_total", ModelHotSwapErrors)
	unknownTokens := promtest.Value(t, "tokenizer_unknown_tokens_total", TokenizerUnknownTokens)
	simdFallback := promtest.Value(t, "simd_fallback_count_total", SIMDFallbackCount.WithLabelValues("avx512", "avx2"))
	offloadTransfers := promtest.Value(t, "quarrel_layer_offload_transfers_total",
		LayerOffloadTransfersTotal.WithLabelValues("modelX"))
	queueBefore := batchQueueHotpath.Load()
	runningBefore := batchRunningHotpath.Load()

	RecordBatchStats(4, 128, 64)
	RecordModelHotSwap(2*time.Second, true)
	RecordModelHotSwap(2*time.Second, false)
	RecordKVCacheAudit(map[string]interface{}{"UniquePositions": 1024})
	RecordBufferSizingAudit(map[string]interface{}{"GQARatio": 4})
	RecordDequantizationAudit(map[string]interface{}{"MaxAbsError": 0.05, "MaxRelError": 0.01})
	RecordWeightAlignmentAudit(map[string]interface{}{"PaddingBytes": 64})
	RecordSoftmaxMaskingAudit(map[string]interface{}{"MaxMaskValue": 0.0})
	RecordHeadDimensionAudit(map[string]interface{}{"ThreadgroupSize": 32})
	RecordTokenizerEncode(100, 2)
	RecordTokenizerDecode(100, 5*time.Millisecond)
	RecordGemma4QNormApplied()
	RecordGemma4KNormApplied()

	RecordSIMDLevel(2)
	RecordSIMDKernelDuration("test_kernel", 10*time.Millisecond, "medium")
	RecordSIMDSoftmaxDuration(2 * time.Millisecond)
	RecordSIMDRMSNormDuration(3 * time.Millisecond)
	RecordSIMDMatmulDuration(15 * time.Millisecond)
	RecordSIMDAttentionDuration(20 * time.Millisecond)
	RecordSIMDFusedMLPDuration(25 * time.Millisecond)
	RecordSIMDRoPEDuration(5 * time.Millisecond)
	RecordSIMDSwiGLUDuration(8 * time.Millisecond)
	RecordSIMDFallback("avx512", "avx2")
	RecordSIMDNaN("gemm")
	RecordSIMDInf("softmax")
	RecordSIMDKernelError("rope", "overflow")

	RecordLayerOffload("modelX", 16, 16)
	RecordLayerOffloadTransfer("modelX", 5*time.Millisecond)
	RecordLayerOffloadCPUDuration("modelX", 12*time.Millisecond)

	RecordTurboQuantCompression("layer0", 0.25)
	RecordTurboQuantLatency(0.015)

	promtest.RequireDelta(t, "model_hot_swap_total", ModelHotSwapTotal, hotSwap, 2)
	promtest.RequireDelta(t, "model_hot_swap_errors_total", ModelHotSwapErrors, hotSwapErrors, 1)
	promtest.RequireDelta(t, "tokenizer_unknown_tokens_total", TokenizerUnknownTokens, unknownTokens, 2)
	promtest.RequireDelta(t, "simd_fallback_count_total", SIMDFallbackCount.WithLabelValues("avx512", "avx2"), simdFallback, 1)
	promtest.RequireDelta(t, "quarrel_layer_offload_transfers_total",
		LayerOffloadTransfersTotal.WithLabelValues("modelX"), offloadTransfers, 1)

	if got := batchQueueHotpath.Load(); got-queueBefore != 4 {
		t.Errorf("batch queue depth changed by %d, want 4", got-queueBefore)
	}
	if got := batchRunningHotpath.Load(); got-runningBefore != 192 {
		t.Errorf("running+prefill changed by %d, want 192", got-runningBefore)
	}
}

func TestRecordV040Metrics(t *testing.T) {
	accepted := promtest.Value(t, "quarrel_speculative_accepted_tokens_total", SpeculativeAcceptedTokensTotal)
	draft := promtest.Value(t, "quarrel_speculative_draft_tokens_total", SpeculativeDraftTokensTotal)
	patches := promtest.Value(t, "quarrel_vlm_patches_processed_total", VLMPatchesProcessedTotal)
	transferBytes := promtest.Value(t, "quarrel_distributed_transfer_bytes_total",
		DistributedTransferBytesTotal.WithLabelValues("egress", "arrow_flight"))
	pressure := promtest.Value(t, "quarrel_memory_pressure_events_total",
		MemoryPressureEventsTotal.WithLabelValues("vram", "defrag"))
	preemptions := promtest.Value(t, "quarrel_continuous_batch_preemptions_total", ContinuousBatchPreemptionsTotal)
	ttft := promtest.Value(t, "quarrel_time_to_first_token_seconds", TimeDurationTTFT)
	interToken := promtest.Value(t, "quarrel_inter_token_latency_seconds", InterTokenLatencyHistogram)

	RecordCUDADequantGEMM("Q4_K", 2*time.Millisecond)
	RecordCUDAVRAMSaved("gemma-4", 1024*1024*1024)
	RecordFlashAttentionPrefill(15*time.Millisecond, 512)
	RecordContinuousBatchIteration(8, 12*time.Millisecond, true)
	RecordContinuousBatchIteration(4, 8*time.Millisecond, false)
	RecordSpeculativeStep(5, 4, 6)
	RecordVLMEncoding(256, 30*time.Millisecond)
	RecordGrammarFilter(500 * time.Microsecond)
	RecordVNNIDotProduct(100 * time.Microsecond)
	RecordKVCacheQuantization("FP8", 32, 2.0)
	RecordDistributedTransfer("egress", "arrow_flight", 4096, 5*time.Millisecond)
	RecordMemoryPressureEvent("vram", "defrag", 0.89)
	RecordTTFTLatency(45 * time.Millisecond)
	RecordInterTokenLatency(12 * time.Millisecond)

	promtest.RequireDelta(t, "quarrel_speculative_accepted_tokens_total", SpeculativeAcceptedTokensTotal, accepted, 4)
	promtest.RequireDelta(t, "quarrel_speculative_draft_tokens_total", SpeculativeDraftTokensTotal, draft, 5)
	promtest.RequireDelta(t, "quarrel_vlm_patches_processed_total", VLMPatchesProcessedTotal, patches, 256)
	promtest.RequireDelta(t, "quarrel_distributed_transfer_bytes_total",
		DistributedTransferBytesTotal.WithLabelValues("egress", "arrow_flight"), transferBytes, 4096)
	promtest.RequireDelta(t, "quarrel_memory_pressure_events_total",
		MemoryPressureEventsTotal.WithLabelValues("vram", "defrag"), pressure, 1)
	promtest.RequireDelta(t, "quarrel_continuous_batch_preemptions_total", ContinuousBatchPreemptionsTotal, preemptions, 1)
	promtest.RequireDelta(t, "quarrel_time_to_first_token_seconds", TimeDurationTTFT, ttft, 1)
	promtest.RequireDelta(t, "quarrel_inter_token_latency_seconds", InterTokenLatencyHistogram, interToken, 1)
	promtest.RequireValue(t, "quarrel_cuda_vram_saved_bytes", CUDAVRAMSavedBytesTotal.WithLabelValues("gemma-4"),
		1024*1024*1024)
}
