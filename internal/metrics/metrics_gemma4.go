package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ===== Gemma4-Specific Metrics =====

var (
	// Hybrid Attention Metrics
	Gemma4SlidingWindowLayers = promauto.NewCounter(prometheus.CounterOpts{
		Name: "gemma4_sliding_window_layers_total",
		Help: "Total number of sliding window attention layers processed",
	})

	Gemma4FullAttentionLayers = promauto.NewCounter(prometheus.CounterOpts{
		Name: "gemma4_full_attention_layers_total",
		Help: "Total number of full attention layers processed",
	})

	Gemma4SlidingWindowSize = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gemma4_sliding_window_size",
		Help: "Configured sliding window size for Gemma4",
	})

	Gemma4PartialRoPE = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "gemma4_partial_rope_factor",
		Help:    "Partial RoPE rotation factor (0.25 = 25% rotated)",
		Buckets: []float64{0.0, 0.125, 0.25, 0.5, 0.75, 1.0},
	})

	Gemma4RoPETheta = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "gemma4_rope_theta",
		Help:    "RoPE theta values (10K for sliding, 1M for full)",
		Buckets: []float64{1000, 10000, 100000, 1000000, 10000000},
	})

	// Gemma4 Q/K Norm Metrics
	Gemma4QNormApplied = promauto.NewCounter(prometheus.CounterOpts{
		Name: "gemma4_q_norm_applied_total",
		Help: "Count of Q normalization operations applied",
	})

	Gemma4KNormApplied = promauto.NewCounter(prometheus.CounterOpts{
		Name: "gemma4_k_norm_applied_total",
		Help: "Count of K normalization operations applied",
	})

	Gemma4VNormApplied = promauto.NewCounter(prometheus.CounterOpts{
		Name: "gemma4_v_norm_applied_total",
		Help: "Count of V normalization operations applied (with_scale=false)",
	})

	// Gemma4 Attention Layer Pattern
	Gemma4LayerPatternRatio = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "gemma4_layer_pattern_ratio",
		Help:    "Ratio of sliding to full attention layers (e.g., 5:1)",
		Buckets: []float64{1, 2, 4, 5, 6, 8, 10},
	})

	// Gemma4 Head Dimension Metrics
	Gemma4SlidingHeadDim = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gemma4_sliding_head_dim",
		Help: "Head dimension for sliding attention layers",
	})

	Gemma4FullHeadDim = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gemma4_full_head_dim",
		Help: "Head dimension for full attention layers",
	})

	// Gemma4 Context Length
	Gemma4ContextLength = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "gemma4_context_length_tokens",
		Help:    "Context length used for Gemma4 inference",
		Buckets: []float64{128, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131072},
	})
)

// RecordGemma4SlidingWindowLayer records a sliding window attention layer forward pass
func RecordGemma4SlidingWindowLayer(windowSize int, layerLatency time.Duration) {
	Gemma4SlidingWindowLayers.Inc()
	Gemma4SlidingWindowSize.Set(float64(windowSize))
}

// RecordGemma4FullAttentionLayer records a full attention layer forward pass
func RecordGemma4FullAttentionLayer(partialRotaryFactor float64, ropeTheta float64, layerLatency time.Duration) {
	Gemma4FullAttentionLayers.Inc()
	Gemma4PartialRoPE.Observe(partialRotaryFactor)
	Gemma4RoPETheta.Observe(ropeTheta)
}

// RecordGemma4QKNorm records Q/K normalization operations
func RecordGemma4QKNorm(qNormApplied, kNormApplied bool) {
	if qNormApplied {
		Gemma4QNormApplied.Inc()
	}
	if kNormApplied {
		Gemma4KNormApplied.Inc()
	}
}

// RecordGemma4VNorm records V normalization (with_scale=false)
func RecordGemma4VNorm() {
	Gemma4VNormApplied.Inc()
}

// RecordGemma4LayerPattern records the sliding/full attention layer ratio
func RecordGemma4LayerPattern(slidingCount, fullCount int) {
	if fullCount > 0 {
		ratio := float64(slidingCount) / float64(fullCount)
		Gemma4LayerPatternRatio.Observe(ratio)
	}
}

// RecordGemma4HeadDims records the head dimensions for sliding and full attention
func RecordGemma4HeadDims(slidingHeadDim, fullHeadDim int) {
	Gemma4SlidingHeadDim.Set(float64(slidingHeadDim))
	Gemma4FullHeadDim.Set(float64(fullHeadDim))
}

// RecordGemma4Context records context length for Gemma4 inference
func RecordGemma4Context(length int) {
	Gemma4ContextLength.Observe(float64(length))
}

func RecordGemma4QNormApplied() {
	Gemma4QNormApplied.Inc()
}

func RecordGemma4KNormApplied() {
	Gemma4KNormApplied.Inc()
}
