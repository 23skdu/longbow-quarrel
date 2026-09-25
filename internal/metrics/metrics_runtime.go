package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ===== v0.4.0 10-Part Plan Metrics =====

var (
	// Part 1: Native CUDA Quantized Matrix Multiplication
	CUDADequantGEMMDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_dequant_gemm_duration_seconds",
		Help:    "Duration of on-the-fly CUDA dequantizing GEMM kernels",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05},
	}, []string{"quant_type"})

	CUDAVRAMSavedBytesTotal = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_cuda_vram_saved_bytes",
		Help: "Bytes of VRAM saved via zero-dequant weights",
	}, []string{"model"})

	// Part 2: CUDA Prefill Flash Attention with Sliding Window
	FlashAttentionPrefillDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_flash_attention_prefill_duration_seconds",
		Help:    "Duration of CUDA FlashAttention prefill kernel",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
	})

	FlashAttentionSlidingWindowTokens = promauto.NewCounter(prometheus.CounterOpts{
		Name: "quarrel_flash_attention_sliding_window_tokens_total",
		Help: "Number of tokens processed within sliding window mask",
	})

	// Chunked Prefill & Iteration Scheduling
	PrefillChunkDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_prefill_chunk_latency_seconds",
		Help:    "Execution latency of chunked prompt prefill operations",
		Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0},
	})

	ActivePrefillTokens = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "quarrel_active_prefill_tokens",
		Help: "Current count of active tokens in flight for prefill chunks",
	})

	// Part 3: Continuous Batching & Iteration Scheduling
	ContinuousBatchActiveRequests = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "quarrel_continuous_batch_active_requests",
		Help: "Current number of requests active in the continuous batching scheduler",
	})

	ContinuousBatchIterationDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_continuous_batch_iteration_duration_seconds",
		Help:    "Execution time per continuous batching iteration step",
		Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1},
	})

	ContinuousBatchPreemptionsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "quarrel_continuous_batch_preemptions_total",
		Help: "Total count of request preemptions due to KV page memory pressure",
	})

	// Part 4: Asymmetric Speculative Decoding
	SpeculativeDraftTokensTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "quarrel_speculative_draft_tokens_total",
		Help: "Total number of draft tokens proposed by draft engine",
	})

	SpeculativeAcceptedTokensTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "quarrel_speculative_accepted_tokens_total",
		Help: "Total number of draft tokens accepted by target verification engine",
	})

	SpeculativeAcceptanceRate = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "quarrel_speculative_acceptance_rate",
		Help: "Moving average acceptance rate of speculative tokens",
	})

	SpeculativeDynamicDraftLength = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "quarrel_speculative_dynamic_draft_length",
		Help: "Dynamically adjusted speculative draft length",
	})

	// Part 5: Gemma 4 VLM Vision-Language Pipeline
	VLMImageEncodingDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_vlm_image_encoding_duration_seconds",
		Help:    "Duration of vision transformer patch embedding and projection",
		Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5},
	})

	VLMPatchesProcessedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "quarrel_vlm_patches_processed_total",
		Help: "Total image patches processed by VLM projection layer",
	})

	// Part 6: Grammar-Constrained Sampling (CFG & Regex)
	GrammarConstraintFilterDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_grammar_filter_duration_seconds",
		Help:    "Time spent computing valid token bitmask from pushdown automaton",
		Buckets: []float64{0.00001, 0.00005, 0.0001, 0.0005, 0.001},
	})

	GrammarConstrainedTokensTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "quarrel_grammar_constrained_tokens_total",
		Help: "Tokens generated under active grammar or regex constraints",
	})

	// Part 7: AVX-512 VNNI & AMX Quantized Dot Products
	SIMDVNNIDotProductDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_simd_vnni_dot_product_duration_seconds",
		Help:    "Execution duration of AVX-512 VNNI vpdpbusd dot products",
		Buckets: []float64{0.00001, 0.00005, 0.0001, 0.0005, 0.001},
	})

	KVCacheQuantizedPagesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_kv_cache_quantized_pages_total",
		Help: "Total number of KV cache pages quantized",
	}, []string{"precision"})

	KVCacheCompressionRatioGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_kv_cache_compression_ratio",
		Help: "Compression ratio achieved by quantized KV cache pages",
	}, []string{"precision"})

	// Part 9: Distributed Pipeline & Tensor Parallelism (Arrow Flight)
	DistributedTransferBytesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_distributed_transfer_bytes_total",
		Help: "Total bytes transferred across distributed pipeline stages",
	}, []string{"direction", "protocol"})

	DistributedTransferDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_distributed_transfer_duration_seconds",
		Help:    "Latency of cross-node Arrow Flight tensor transfers",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
	}, []string{"protocol"})

	// Part 10: Telemetry & Memory Governor
	MemoryPressureEventsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_memory_pressure_events_total",
		Help: "Proactive memory governor trigger events",
	}, []string{"resource", "action"})

	MemoryPressureRatioGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_memory_pressure_ratio",
		Help: "Current memory utilization ratio (0.0 - 1.0)",
	}, []string{"resource"})

	MemoryGovernorActionsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_memory_governor_actions_total",
		Help: "Actions taken by the memory governor (defrag, offload, evict)",
	}, []string{"action"})

	TimeDurationTTFT = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_time_to_first_token_seconds",
		Help:    "Time to First Token (TTFT) latency distribution",
		Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5},
	})

	InterTokenLatencyHistogram = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "quarrel_inter_token_latency_seconds",
		Help:    "Inter-token generation latency distribution",
		Buckets: []float64{0.001, 0.005, 0.01, 0.02, 0.05, 0.1},
	})
)

// RecordCUDADequantGEMM records duration of a CUDA dequant GEMM call
func RecordCUDADequantGEMM(quantType string, duration time.Duration) {
	CUDADequantGEMMDuration.WithLabelValues(quantType).Observe(duration.Seconds())
}

// RecordCUDAVRAMSaved records bytes saved by zero-dequant storage
func RecordCUDAVRAMSaved(model string, bytesSaved int64) {
	CUDAVRAMSavedBytesTotal.WithLabelValues(model).Set(float64(bytesSaved))
}

// RecordFlashAttentionPrefill records prefill kernel latency
func RecordFlashAttentionPrefill(duration time.Duration, slidingWindowTokens int) {
	FlashAttentionPrefillDuration.Observe(duration.Seconds())
	if slidingWindowTokens > 0 {
		FlashAttentionSlidingWindowTokens.Add(float64(slidingWindowTokens))
	}
}

// RecordPrefillChunk records chunked prefill latency and token volume
func RecordPrefillChunk(tokens int, duration time.Duration) {
	PrefillChunkDuration.Observe(duration.Seconds())
	ActivePrefillTokens.Set(float64(tokens))
}

// RecordContinuousBatchIteration records one continuous batching scheduling step
func RecordContinuousBatchIteration(activeRequests int, duration time.Duration, preempted bool) {
	ContinuousBatchActiveRequests.Set(float64(activeRequests))
	ContinuousBatchIterationDuration.Observe(duration.Seconds())
	if preempted {
		ContinuousBatchPreemptionsTotal.Inc()
	}
}

// RecordSpeculativeStep records token counts and acceptance rate for a speculative step
func RecordSpeculativeStep(draftTokens, acceptedTokens int, dynamicLength int) {
	SpeculativeDraftTokensTotal.Add(float64(draftTokens))
	SpeculativeAcceptedTokensTotal.Add(float64(acceptedTokens))
	if draftTokens > 0 {
		rate := float64(acceptedTokens) / float64(draftTokens)
		SpeculativeAcceptanceRate.Set(rate)
	}
	SpeculativeDynamicDraftLength.Set(float64(dynamicLength))
}

// RecordVLMEncoding records image patch encoding duration and count
func RecordVLMEncoding(patches int, duration time.Duration) {
	VLMPatchesProcessedTotal.Add(float64(patches))
	VLMImageEncodingDuration.Observe(duration.Seconds())
}

// RecordGrammarFilter records grammar constraint filter duration and generation
func RecordGrammarFilter(duration time.Duration) {
	GrammarConstraintFilterDuration.Observe(duration.Seconds())
	GrammarConstrainedTokensTotal.Inc()
}

// RecordVNNIDotProduct records SIMD VNNI execution duration
func RecordVNNIDotProduct(duration time.Duration) {
	SIMDVNNIDotProductDuration.Observe(duration.Seconds())
}

// RecordKVCacheQuantization records quantized page storage
func RecordKVCacheQuantization(precision string, pages int, compressionRatio float64) {
	KVCacheQuantizedPagesTotal.WithLabelValues(precision).Add(float64(pages))
	KVCacheCompressionRatioGauge.WithLabelValues(precision).Set(compressionRatio)
}

// RecordDistributedTransfer records distributed Arrow Flight transfer
func RecordDistributedTransfer(direction, protocol string, bytes int64, duration time.Duration) {
	DistributedTransferBytesTotal.WithLabelValues(direction, protocol).Add(float64(bytes))
	DistributedTransferDuration.WithLabelValues(protocol).Observe(duration.Seconds())
}

// RecordMemoryPressureEvent records proactive memory governor events
func RecordMemoryPressureEvent(resource, action string, ratio float64) {
	MemoryPressureEventsTotal.WithLabelValues(resource, action).Inc()
	MemoryPressureRatioGauge.WithLabelValues(resource).Set(ratio)
	MemoryGovernorActionsTotal.WithLabelValues(action).Inc()
}

// RecordTTFTLatency records TTFT latency in seconds
func RecordTTFTLatency(duration time.Duration) {
	TimeDurationTTFT.Observe(duration.Seconds())
}

// RecordInterTokenLatency records inter-token latency in seconds
func RecordInterTokenLatency(duration time.Duration) {
	InterTokenLatencyHistogram.Observe(duration.Seconds())
}
