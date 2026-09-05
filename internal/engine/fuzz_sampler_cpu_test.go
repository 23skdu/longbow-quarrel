package engine

import (
	"math"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

func FuzzSamplerQualityControl(f *testing.F) {
	f.Add(float32(0.7), 40, float32(0.95), float32(1.1), float32(0.5), float32(0.2), float32(0.05), int64(42))
	f.Add(float32(0.0), 0, float32(0.0), float32(0.0), float32(0.0), float32(0.0), float32(0.0), int64(0))
	f.Add(float32(2.5), 100, float32(1.0), float32(2.0), float32(1.0), float32(1.0), float32(0.9), int64(12345))
	f.Add(float32(-1.0), -5, float32(-0.5), float32(-1.0), float32(-2.0), float32(-2.0), float32(-0.1), int64(-1))

	f.Fuzz(func(t *testing.T, temp float32, topK int, topP, repPenalty, presencePenalty, frequencyPenalty, minP float32, seed int64) {
		samplerCfg := SamplerConfig{
			Temperature:      float64(temp),
			TopK:             topK,
			TopP:             float64(topP),
			RepPenalty:       float64(repPenalty),
			PresencePenalty:  float64(presencePenalty),
			FrequencyPenalty: float64(frequencyPenalty),
			MinP:             float64(minP),
			Seed:             seed,
		}

		vocabSize := 64
		logits := make([]float32, vocabSize)
		for i := range logits {
			logits[i] = float32(i%10) - 5.0
		}
		// Introduce some extreme values
		logits[0] = -1e9
		logits[vocabSize-1] = 50.0

		ctxTokens := []int{0, 1, 2, 2, 2, 5, 10}

		sampler := NewSampler(samplerCfg)
		tok := sampler.Sample(logits, ctxTokens)

		if tok < 0 || tok >= vocabSize {
			t.Errorf("sampled token %d out of bounds [0, %d)", tok, vocabSize)
		}
	})
}

func FuzzCPUKVCache_Attention(f *testing.F) {
	f.Add(1, 0, 16, 2, 2, 8)
	f.Add(2, 5, 32, 4, 2, 8)
	f.Add(1, 10, 16, 2, 1, 16)

	f.Fuzz(func(t *testing.T, layers, pos, dim, heads, kvHeads, headDim int) {
		if layers <= 0 || layers > 4 || heads <= 0 || heads > 8 || kvHeads <= 0 || kvHeads > heads || headDim <= 0 || headDim > 32 {
			return
		}
		if pos < 0 || pos > 50 {
			return
		}
		if dim != heads*headDim {
			dim = heads * headDim
		}

		_ = config.Config{
			Dim:       dim,
			Layers:    layers,
			VocabSize: 32,
			Heads:     heads,
			KVHeads:   kvHeads,
			HeadDim:   headDim,
			HiddenDim: dim * 2,
			RopeTheta: 10000.0,
			Eps:       1e-5,
		}

		kv := NewCPUKVCache(layers)
		q := make([]float32, heads*headDim)
		k := make([]float32, kvHeads*headDim)
		v := make([]float32, kvHeads*headDim)
		for i := range q {
			q[i] = float32(i%7) * 0.1
		}
		for i := range k {
			k[i] = float32(i%5) * 0.1
		}
		for i := range v {
			v[i] = float32(i%3) * 0.1
		}

		attn := attentionCPUKV(q, k, v, 0, pos, kv, heads, kvHeads, headDim)
		if len(attn) != heads*headDim {
			t.Fatalf("expected attention length %d, got %d", heads*headDim, len(attn))
		}

		for i, val := range attn {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Fatalf("attention produced non-finite value at %d: %v", i, val)
			}
		}

		// Verify metrics hook works
		metrics.RecordInference(1, 100*time.Microsecond)
	})
}
