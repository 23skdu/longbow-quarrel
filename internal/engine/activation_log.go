package engine

import (
	"math"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

// logLayerActivations records one layer's activations in the activation log.
//
// The logger wants the peak magnitude of each tensor, a short sample of
// it, and the number of NaNs and Infs it contains. The NaN and Inf counts
// are the useful part: they say which layer first produced a non-finite
// value, which is the question you usually have when a model produces
// garbage.
func logLayerActivations(al *ActivationLogger, c *LayerActivationCollector) {
	if al == nil || c == nil || !al.IsEnabled() {
		return
	}

	al.LogLayer(
		c.LayerIdx,
		maxAbs(c.Q),
		maxAbs(c.K),
		maxAbs(c.V),
		maxAbs(c.AttnOut),
		maxAbs(c.FFNOut),
		sampleHead(c.Q),
		sampleHead(c.K),
		sampleHead(c.V),
		c.Q,
		c.K,
		c.V,
		c.AttnOut,
		c.FFNOut,
	)
}

// maxAbs returns the largest absolute value in a tensor, or zero for an
// empty one.
func maxAbs(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	max := float32(0)
	for _, v := range data {
		if mathx.IsNaN(float64(v)) {
			// A NaN is not a magnitude, but it is a problem, and the
			// NaN count reports it separately.
			continue
		}
		a := math.Abs(float64(v))
		if float32(a) > max {
			max = float32(a)
		}
	}
	return max
}

// topTokenIDs returns the ids of the n tokens with the highest logits.
//
// The activation log records the final logits of a few tokens, and the
// interesting ones are the ones the sampler would have picked.
func topTokenIDs(logits []float32, n int) []int {
	if len(logits) == 0 || n <= 0 {
		return nil
	}
	ids := make([]int, 0, n)
	used := make([]bool, len(logits))
	for i := 0; i < n && i < len(logits); i++ {
		best := -1
		for j, l := range logits {
			if used[j] {
				continue
			}
			if best == -1 || l > logits[best] {
				best = j
			}
		}
		if best < 0 {
			break
		}
		used[best] = true
		ids = append(ids, best)
	}
	return ids
}

// sampleHead returns the first few values of a tensor, which is enough to
// eyeball what a layer produced.
func sampleHead(data []float32) []float32 {
	const limit = 8
	if len(data) == 0 {
		return nil
	}
	n := len(data)
	if n > limit {
		n = limit
	}
	out := make([]float32, n)
	copy(out, data[:n])
	return out
}
