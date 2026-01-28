package metrics

import (
	"fmt"
	"os"
	"sync"
	"time"

	"sync/atomic"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var totalTokens atomic.Int64

var (
	InferenceTokensTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "inference_tokens_total",
		Help: "The total number of tokens generated",
	})

	InferenceDuration = promauto.NewSummary(prometheus.SummaryOpts{
		Name: "inference_duration_seconds",
		Help: "Duration of inference steps",
	})

	GPUMemoryAllocated = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gpu_memory_allocated_bytes",
		Help: "Current bytes allocated on GPU",
	})

	KernelDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "gpu_kernel_duration_seconds",
		Help:    "Histogram of kernel execution times",
		Buckets: prometheus.DefBuckets,
	}, []string{"kernel"})

	NumericalInstability = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "numerical_instability_total",
		Help: "Total number of NaN/Inf values detected",
	}, []string{"tensor", "type"})

	ValidationErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "validation_errors_total",
		Help: "Total number of validation errors",
	}, []string{"operation", "error_type"})

	ContextLengthHistogram = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "context_length_tokens",
		Help:    "Distribution of context lengths processed",
		Buckets: []float64{100, 500, 1000, 2000, 4000, 8000, 16000, 32000},
	})

	// ===== New Audit Metrics =====

	// Logit Range Audit Metrics
	LogitMaxValue = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "logit_max_value",
		Help:    "Maximum logit value observed",
		Buckets: []float64{-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500, 1000},
	})

	LogitMinValue = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "logit_min_value",
		Help:    "Minimum logit value observed",
		Buckets: []float64{-1000, -500, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100},
	})

	LogitMeanValue = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "logit_mean_value",
		Help:    "Mean logit value observed",
		Buckets: []float64{-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100},
	})

	LogitRMS = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "logit_rms",
		Help:    "Root mean square of logit values",
		Buckets: []float64{0, 1, 2, 5, 10, 20, 50, 100, 200, 500},
	})

	LogitFlatDistribution = promauto.NewCounter(prometheus.CounterOpts{
		Name: "logit_flat_distribution_total",
		Help: "Count of flat logit distributions detected",
	})

	LogitNaNCount = promauto.NewCounter(prometheus.CounterOpts{
		Name: "logit_nan_count_total",
		Help: "Total count of NaN values in logits",
	})

	LogitExtremeValues = promauto.NewCounter(prometheus.CounterOpts{
		Name: "logit_extreme_values_total",
		Help: "Count of extreme logit values detected",
	})

	// KV Cache Audit Metrics
	KVCacheOverlap = promauto.NewCounter(prometheus.CounterOpts{
		Name: "kv_cache_overlap_total",
		Help: "Count of KV cache position overlaps detected",
	})

	KVCacheOutOfBounds = promauto.NewCounter(prometheus.CounterOpts{
		Name: "kv_cache_oob_total",
		Help: "Count of KV cache out-of-bounds accesses detected",
	})

	KVCacheUniquePositions = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "kv_cache_unique_positions",
		Help:    "Number of unique KV cache positions used",
		Buckets: []float64{1, 10, 100, 500, 1000, 2000, 4000, 8000},
	})

	KVCacheSlidingWindow = promauto.NewCounter(prometheus.CounterOpts{
		Name: "kv_cache_sliding_window_total",
		Help: "Count of sliding window KV cache operations",
	})

	KVCacheCapacityBytes = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "kv_cache_capacity_bytes",
		Help: "Total capacity of KV cache in bytes",
	})

	KVCacheUsedBytes = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "kv_cache_used_bytes",
		Help: "Current bytes used in KV cache",
	})

	KVCacheHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "kv_cache_hits_total",
		Help: "Total number of KV cache hits",
	})

	KVCacheMisses = promauto.NewCounter(prometheus.CounterOpts{
		Name: "kv_cache_misses_total",
		Help: "Total number of KV cache misses",
	})

	KVCacheEvictions = promauto.NewCounter(prometheus.CounterOpts{
		Name: "kv_cache_evictions_total",
		Help: "Total number of KV cache evictions",
	})

	// Buffer Sizing Audit Metrics
	BufferScoresSize = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "buffer_scores_size_bytes",
		Help:    "Scores buffer size in bytes",
		Buckets: []float64{32768, 65536, 131072, 262144, 524288, 1048576, 2097152},
	})

	BufferGQARatio = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "buffer_gqa_ratio",
		Help:    "GQA ratio (heads / kv_heads)",
		Buckets: []float64{1, 2, 4, 8, 16, 32},
	})

	BufferAlignment = promauto.NewCounter(prometheus.CounterOpts{
		Name: "buffer_alignment_total",
		Help: "Count of properly aligned buffers",
	})

	BufferInvalid = promauto.NewCounter(prometheus.CounterOpts{
		Name: "buffer_invalid_total",
		Help: "Count of invalid buffer configurations",
	})

	BufferNonOverlap = promauto.NewCounter(prometheus.CounterOpts{
		Name: "buffer_non_overlap_total",
		Help: "Count of non-overlapping buffer allocations",
	})

	// Dequantization Audit Metrics
	DequantMaxAbsError = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dequant_max_abs_error",
		Help:    "Maximum absolute dequantization error",
		Buckets: []float64{0, 0.001, 0.01, 0.1, 1.0, 10.0},
	})

	DequantMaxRelError = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dequant_max_rel_error",
		Help:    "Maximum relative dequantization error",
		Buckets: []float64{0, 0.0001, 0.001, 0.01, 0.1, 1.0},
	})

	DequantPass = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dequant_pass_total",
		Help: "Count of passing dequantization accuracy checks",
	})

	DequantFail = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dequant_fail_total",
		Help: "Count of failing dequantization accuracy checks",
	})

	DequantMismatches = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dequant_mismatches_total",
		Help: "Total number of dequantization mismatches",
	})

	// Weight Alignment Audit Metrics
	WeightPadding = promauto.NewCounter(prometheus.CounterOpts{
		Name: "weight_padding_total",
		Help: "Count of tensors with padding detected",
	})

	WeightAligned = promauto.NewCounter(prometheus.CounterOpts{
		Name: "weight_aligned_total",
		Help: "Count of properly aligned weight tensors",
	})

	WeightNotAligned = promauto.NewCounter(prometheus.CounterOpts{
		Name: "weight_not_aligned_total",
		Help: "Count of misaligned weight tensors",
	})

	WeightPaddingBytes = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "weight_padding_bytes",
		Help:    "Number of padding bytes in weight tensors",
		Buckets: []float64{0, 128, 256, 512, 1024, 4096, 16384},
	})

	WeightValid = promauto.NewCounter(prometheus.CounterOpts{
		Name: "weight_valid_total",
		Help: "Count of valid weight tensors",
	})

	WeightInvalid = promauto.NewCounter(prometheus.CounterOpts{
		Name: "weight_invalid_total",
		Help: "Count of invalid weight tensors",
	})

	// Softmax Masking Audit Metrics
	SoftmaxStrictMask = promauto.NewCounter(prometheus.CounterOpts{
		Name: "softmax_strict_mask_total",
		Help: "Count of strictly masked softmax operations",
	})

	SoftmaxNotStrict = promauto.NewCounter(prometheus.CounterOpts{
		Name: "softmax_not_strict_total",
		Help: "Count of non-strict softmax masking operations",
	})

	SoftmaxMaskedCount = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "softmax_masked_count",
		Help:    "Number of masked positions in softmax",
		Buckets: []float64{0, 10, 100, 500, 1000, 2000, 4000, 8000},
	})

	SoftmaxUnmaskedCount = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "softmax_unmasked_count",
		Help:    "Number of unmasked positions in softmax",
		Buckets: []float64{0, 10, 100, 500, 1000, 2000, 4000, 8000},
	})

	SoftmaxMaskValue = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "softmax_mask_value",
		Help:    "Value used for masking in softmax",
		Buckets: []float64{-1000000, -100000, -10000, -1000, -100, 0},
	})

	SoftmaxOutOfBounds = promauto.NewCounter(prometheus.CounterOpts{
		Name: "softmax_oob_total",
		Help: "Count of out-of-bounds positions in softmax",
	})

	// Head Dimension Audit Metrics
	HeadDimPowerOf2 = promauto.NewCounter(prometheus.CounterOpts{
		Name: "head_dim_power_of_2_total",
		Help: "Count of power-of-2 head dimensions",
	})

	HeadDimNotPowerOf2 = promauto.NewCounter(prometheus.CounterOpts{
		Name: "head_dim_not_power_of_2_total",
		Help: "Count of non-power-of-2 head dimensions",
	})

	HeadDimThreadgroupSize = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "head_dim_threadgroup_size",
		Help:    "Threadgroup size for head dimension",
		Buckets: []float64{32, 64, 128, 256, 512},
	})

	HeadDimOptimal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "head_dim_optimal_total",
		Help: "Count of optimal head dimension configurations",
	})

	HeadDimNotOptimal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "head_dim_not_optimal_total",
		Help: "Count of non-optimal head dimension configurations",
	})

	// Activation Flow Audit Metrics
	ActivationHealthy = promauto.NewCounter(prometheus.CounterOpts{
		Name: "activation_healthy_total",
		Help: "Count of healthy activation flows",
	})

	ActivationUnhealthy = promauto.NewCounter(prometheus.CounterOpts{
		Name: "activation_unhealthy_total",
		Help: "Count of unhealthy activation flows",
	})

	ActivationCollapsedLayers = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "activation_collapsed_layers",
		Help:    "Number of collapsed layers detected",
		Buckets: []float64{0, 1, 2, 4, 8, 16, 32},
	})

	ActivationSaturatedLayers = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "activation_saturated_layers",
		Help:    "Number of saturated layers detected",
		Buckets: []float64{0, 1, 2, 4, 8, 16, 32},
	})

	ActivationJumps = promauto.NewCounter(prometheus.CounterOpts{
		Name: "activation_jumps_total",
		Help: "Count of large activation jumps detected",
	})

	// NaN Propagation Audit Metrics
	NaNDetected = promauto.NewCounter(prometheus.CounterOpts{
		Name: "nan_detected_total",
		Help: "Count of NaN detection events",
	})

	NaNLayerStart = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "nan_layer_start",
		Help:    "Layer where NaN propagation starts",
		Buckets: []float64{0, 4, 8, 12, 16, 20, 24, 28, 32},
	})

	NaNLayerEnd = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "nan_layer_end",
		Help:    "Layer where NaN propagation ends",
		Buckets: []float64{0, 4, 8, 12, 16, 20, 24, 28, 32},
	})

	NaNTotalCount = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "nan_total_count",
		Help:    "Total number of NaN values detected",
		Buckets: []float64{1, 10, 100, 1000, 4096, 16384},
	})

	NaNPatternGradual = promauto.NewCounter(prometheus.CounterOpts{
		Name: "nan_pattern_gradual_total",
		Help: "Count of gradual NaN propagation patterns",
	})

	NaNPatternSudden = promauto.NewCounter(prometheus.CounterOpts{
		Name: "nan_pattern_sudden_total",
		Help: "Count of sudden NaN propagation patterns",
	})

	NaNPatternScattered = promauto.NewCounter(prometheus.CounterOpts{
		Name: "nan_pattern_scattered_total",
		Help: "Count of scattered NaN propagation patterns",
	})

	// RoPE Deviation Audit Metrics
	RoPEDeviation = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "rope_deviation",
		Help:    "RoPE deviation from reference",
		Buckets: []float64{0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0},
	})

	RoPEPass = promauto.NewCounter(prometheus.CounterOpts{
		Name: "rope_pass_total",
		Help: "Count of passing RoPE deviation checks",
	})

	RoPEFail = promauto.NewCounter(prometheus.CounterOpts{
		Name: "rope_fail_total",
		Help: "Count of failing RoPE deviation checks",
	})

	RoPEDeviationRatio = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "rope_deviation_ratio",
		Help:    "Ratio of actual to expected RoPE deviation",
		Buckets: []float64{0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0},
	})

	// Sampling Audit Metrics
	SamplingTemperature = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "sampling_temperature",
		Help:    "Temperature values used in sampling",
		Buckets: []float64{0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0},
	})

	SamplingTopK = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "sampling_top_k",
		Help:    "Top-K values used in sampling",
		Buckets: []float64{1, 5, 10, 20, 40, 50, 100},
	})

	SamplingTopP = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "sampling_top_p",
		Help:    "Top-P values used in sampling",
		Buckets: []float64{0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0},
	})

	SamplingTopTokenProbability = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "sampling_top_token_probability",
		Help:    "Probability mass on top token after temperature scaling",
		Buckets: []float64{0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0},
	})

	SamplingUniqueSamples = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "sampling_unique_samples",
		Help:    "Number of unique tokens sampled in 100 trials",
		Buckets: []float64{1, 2, 5, 10, 20, 50, 100},
	})

	SamplingNaNHandling = promauto.NewCounter(prometheus.CounterOpts{
		Name: "sampling_nan_handling_total",
		Help: "Count of NaN/Inf values handled during sampling",
	})

	SamplingSeedReproducible = promauto.NewCounter(prometheus.CounterOpts{
		Name: "sampling_seed_reproducible_total",
		Help: "Count of reproducible seeded sampling operations",
	})

	SamplingEntropy = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "sampling_entropy",
		Help:    "Logit entropy as quality metric",
		Buckets: []float64{0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0},
	})

	SamplingRepetitionPenalty = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "sampling_repetition_penalty",
		Help:    "Repetition penalty values used",
		Buckets: []float64{1.0, 1.1, 1.2, 1.5, 2.0},
	})

	// Tokenizer Audit Metrics
	TokenizerEncodeLength = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "tokenizer_encode_length",
		Help:    "Length of encoded token sequences",
		Buckets: []float64{1, 5, 10, 20, 50, 100, 200, 500},
	})

	TokenizerVocabSize = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "tokenizer_vocab_size",
		Help:    "Vocabulary size of tokenizers",
		Buckets: []float64{1000, 5000, 10000, 30000, 50000, 100000, 200000},
	})

	TokenizerMergeCount = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "tokenizer_merge_count",
		Help:    "Number of BPE merges in tokenizer",
		Buckets: []float64{0, 100, 500, 1000, 5000, 10000},
	})

	TokenizerUnknownTokens = promauto.NewCounter(prometheus.CounterOpts{
		Name: "tokenizer_unknown_tokens_total",
		Help: "Count of unknown tokens encountered during encoding",
	})

	TokenizerEncodeTime = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "tokenizer_encode_time_seconds",
		Help:    "Time to encode text",
		Buckets: prometheus.DefBuckets,
	})

	TokenizerDecodeTime = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "tokenizer_decode_time_seconds",
		Help:    "Time to decode token IDs",
		Buckets: prometheus.DefBuckets,
	})

	// Activation precision metrics
	ActivationRMSNormMax = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "activation_rmsnorm_max",
		Help:    "Maximum activation value after RMSNorm",
		Buckets: []float64{0, 1, 2, 5, 10, 20, 50, 100},
	})

	ActivationSwiGLUMax = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "activation_swiglu_max",
		Help:    "Maximum activation value after SwiGLU",
		Buckets: []float64{0, 10, 50, 100, 500, 1000, 5000},
	})

	ActivationResidualMax = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "activation_residual_max",
		Help:    "Maximum activation value after residual addition",
		Buckets: []float64{0, 1, 2, 5, 10, 20, 50, 100},
	})

	// MOE Metrics
	MOELayerLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_moe_layer_latency_seconds",
		Help:    "MOE layer forward pass latency",
		Buckets: prometheus.DefBuckets,
	})

	MOEExpertSelection = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_moe_expert_selection_total",
		Help: "Total number of times an expert was selected",
	}, []string{"layer", "expert_id"})

	MOERoutingLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_moe_routing_latency_seconds",
		Help:    "Expert routing (topk selection) latency",
		Buckets: prometheus.DefBuckets,
	})

	MOEExpertUtilization = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_moe_expert_utilization",
		Help: "Expert utilization rate (selections / total tokens across time)",
	}, []string{"layer", "expert_id"})
)

func RecordInference(tokens int, duration time.Duration) {
	InferenceTokensTotal.Add(float64(tokens))
	totalTokens.Add(int64(tokens))
	InferenceDuration.Observe(duration.Seconds())
}

func RecordGPUMemory(bytes int64) {
	GPUMemoryAllocated.Set(float64(bytes))
}

func RecordKernelDuration(name string, duration time.Duration) {
	KernelDuration.WithLabelValues(name).Observe(duration.Seconds())
}

func RecordNumericalInstability(name string, nanCount, infCount int) {
	if nanCount > 0 {
		NumericalInstability.WithLabelValues(name, "nan").Add(float64(nanCount))
	}
	if infCount > 0 {
		NumericalInstability.WithLabelValues(name, "inf").Add(float64(infCount))
	}
}

func RecordValidationError(operation, errorType string) {
	ValidationErrors.WithLabelValues(operation, errorType).Inc()
}

func RecordContextLength(tokens int) {
	ContextLengthHistogram.Observe(float64(tokens))
}

// ===== New Audit Recording Functions =====

// RecordLogitAudit records logit range audit results
func RecordLogitAudit(max, min, mean, rms float32, hasNaN, hasExtreme, isFlat bool) {
	LogitMaxValue.Observe(float64(max))
	LogitMinValue.Observe(float64(min))
	LogitMeanValue.Observe(float64(mean))
	LogitRMS.Observe(float64(rms))
	if isFlat {
		LogitFlatDistribution.Inc()
	}
	if hasNaN {
		LogitNaNCount.Inc()
	}
	if hasExtreme {
		LogitExtremeValues.Inc()
	}
}

// RecordKVCacheAudit records KV cache position audit results
func RecordKVCacheAudit(audit interface{}) {
	// Type assertion to get audit results
	// In practice, this would use the actual audit result type
	KVCacheUniquePositions.Observe(0) // Placeholder - would use actual unique count
}

// RecordBufferSizingAudit records scratch buffer sizing audit results
func RecordBufferSizingAudit(audit interface{}) {
	BufferGQARatio.Observe(0) // Placeholder - would use actual GQA ratio
}

// RecordDequantizationAudit records dequantization accuracy audit results
func RecordDequantizationAudit(audit interface{}) {
	DequantMaxAbsError.Observe(0)   // Placeholder
	DequantMaxRelError.Observe(0.0) // Placeholder
}

// RecordWeightAlignmentAudit records weight padding/alignment audit results
func RecordWeightAlignmentAudit(audit interface{}) {
	WeightPaddingBytes.Observe(0) // Placeholder
}

// RecordSoftmaxMaskingAudit records softmax masking audit results
func RecordSoftmaxMaskingAudit(audit interface{}) {
	SoftmaxMaskValue.Observe(0) // Placeholder
}

// RecordHeadDimensionAudit records head dimension logic audit results
func RecordHeadDimensionAudit(audit interface{}) {
	HeadDimThreadgroupSize.Observe(0) // Placeholder
}

// RecordNaNPropagationAudit records NaN propagation detection results
func RecordNaNPropagationAudit(layerStart, layerEnd, totalCount int, pattern string) {
	if layerStart > 0 {
		NaNLayerStart.Observe(float64(layerStart))
	}
	if layerEnd > 0 {
		NaNLayerEnd.Observe(float64(layerEnd))
	}
	if totalCount > 0 {
		NaNTotalCount.Observe(float64(totalCount))
		NaNDetected.Inc()
	}
	switch pattern {
	case "gradual":
		NaNPatternGradual.Inc()
	case "sudden":
		NaNPatternSudden.Inc()
	case "scattered":
		NaNPatternScattered.Inc()
	}
}

// RecordRoPEDeviationAudit records RoPE deviation audit results
func RecordRoPEDeviationAudit(maxDeviation, deviationRatio float32, passed bool) {
	RoPEDeviation.Observe(float64(maxDeviation))
	RoPEDeviationRatio.Observe(float64(deviationRatio))
	if passed {
		RoPEPass.Inc()
	} else {
		RoPEFail.Inc()
	}
}

// RecordKVCacheSlidingWindow records sliding window KV cache operations
func RecordKVCacheSlidingWindow(windowSize, position int, wrapped bool) {
	KVCacheSlidingWindow.Inc()
	KVCacheUniquePositions.Observe(float64(windowSize))
	if wrapped {
		KVCacheOverlap.Inc()
	}
}

// RecordKVCacheOutOfBounds records out-of-bounds KV cache access attempts
func RecordKVCacheOutOfBounds(position, windowSize int) {
	KVCacheOutOfBounds.Inc()
}

// RecordKVCacheStats records KV cache capacity and usage
func RecordKVCacheStats(capacity, used int64) {
	KVCacheCapacityBytes.Set(float64(capacity))
	KVCacheUsedBytes.Set(float64(used))
}

// RecordActivationFlowAudit records activation flow analysis results
func RecordActivationFlowAudit(collapsedLayers, saturatedLayers int, jumps int) {
	ActivationCollapsedLayers.Observe(float64(collapsedLayers))
	ActivationSaturatedLayers.Observe(float64(saturatedLayers))
	if jumps > 0 {
		ActivationJumps.Inc()
	}
	if collapsedLayers == 0 && saturatedLayers == 0 && jumps == 0 {
		ActivationHealthy.Inc()
	} else {
		ActivationUnhealthy.Inc()
	}
}

// RecordSamplingAudit records sampling operation metrics
func RecordSamplingAudit(data map[string]interface{}) {
	if temp, ok := data["temperature"].(float64); ok {
		SamplingTemperature.Observe(temp)
	}
	if topK, ok := data["top_k"].(int); ok {
		SamplingTopK.Observe(float64(topK))
	}
	if topP, ok := data["top_p"].(float64); ok {
		SamplingTopP.Observe(topP)
	}
	if prob, ok := data["top_token_prob"].(float64); ok {
		SamplingTopTokenProbability.Observe(prob)
	}
	if samples, ok := data["unique_samples"].(int); ok {
		SamplingUniqueSamples.Observe(float64(samples))
	}
	if entropy, ok := data["entropy"].(float64); ok {
		SamplingEntropy.Observe(entropy)
	}
	if repPenalty, ok := data["rep_penalty"].(float64); ok {
		SamplingRepetitionPenalty.Observe(repPenalty)
	}
}

// RecordTokenizerMetrics records tokenizer operation metrics
func RecordTokenizerMetrics(encodeLen, vocabSize int, encodeTime time.Duration) {
	TokenizerEncodeLength.Observe(float64(encodeLen))
	TokenizerVocabSize.Observe(float64(vocabSize))
	TokenizerEncodeTime.Observe(encodeTime.Seconds())
}

// RecordTokenizerEncode records tokenizer encoding metrics
func RecordTokenizerEncode(length int, unknownCount int) {
	TokenizerEncodeLength.Observe(float64(length))
	if unknownCount > 0 {
		TokenizerUnknownTokens.Add(float64(unknownCount))
	}
}

// RecordTokenizerDecode records tokenizer decoding metrics
func RecordTokenizerDecode(length int, decodeTime time.Duration) {
	TokenizerEncodeLength.Observe(float64(length))
	TokenizerDecodeTime.Observe(decodeTime.Seconds())
}

// RecordActivationPrecision records activation precision metrics
func RecordActivationPrecision(activationType string, maxValue float32) {
	switch activationType {
	case "rmsnorm":
		ActivationRMSNormMax.Observe(float64(maxValue))
	case "swiglu":
		ActivationSwiGLUMax.Observe(float64(maxValue))
	case "residual":
		ActivationResidualMax.Observe(float64(maxValue))
	}
}

// RecordMOELayerLatency records the latency of an MOE layer forward pass
func RecordMOELayerLatency(duration time.Duration) {
	MOELayerLatency.Observe(duration.Seconds())
}

// RecordMOERoutingLatency records the latency of MOE expert routing
func RecordMOERoutingLatency(duration time.Duration) {
	MOERoutingLatency.Observe(duration.Seconds())
}

var moeExpertCounts sync.Map // map[string]*atomic.Int64

// RecordMOEExpertSelection records which experts were selected for a layer
func RecordMOEExpertSelection(layerIdx int, expertIndices []int32) {
	layerStr := fmt.Sprintf("%d", layerIdx)
	total := totalTokens.Load()

	for _, idx := range expertIndices {
		expertStr := fmt.Sprintf("%d", idx)
		MOEExpertSelection.WithLabelValues(layerStr, expertStr).Inc()

		// Update internal counters for utilization calculation
		key := layerStr + ":" + expertStr
		actual, _ := moeExpertCounts.LoadOrStore(key, &atomic.Int64{})
		counter := actual.(*atomic.Int64)
		count := counter.Add(1)

		if total > 0 {
			utilization := float64(count) / float64(total)
			MOEExpertUtilization.WithLabelValues(layerStr, expertStr).Set(utilization)
		}
	}
	fmt.Fprintf(os.Stderr, "METRIC: Layer %d selected experts: %v\n", layerIdx, expertIndices)
}

// RecordMOEExpertUtilization updates the utilization rate for an expert
func RecordMOEExpertUtilization(layerIdx int, expertID int, utilization float64) {
	layerStr := fmt.Sprintf("%d", layerIdx)
	expertStr := fmt.Sprintf("%d", expertID)
	MOEExpertUtilization.WithLabelValues(layerStr, expertStr).Set(utilization)
}
