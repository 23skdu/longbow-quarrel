//go:build darwin && metal

package device

import "math"

// ActivationStats contains comprehensive activation statistics
type ActivationStats struct {
	Max    float32
	Min    float32
	Mean   float32
	RMS    float32
	Zeros  int
	NaNs   int
	Infs   int
	Sample []float32 // First 16 values
}

// GetStats reads a tensor and returns comprehensive activation statistics
func (t *Tensor) GetStats(sampleSize int) ActivationStats {
	data := t.ToHost()

	maxVal := float32(0)
	minVal := float32(0)
	sum := float32(0)
	sumSq := float64(0)
	zeros := 0
	nans := 0
	infs := 0

	if len(data) > 0 {
		maxVal = data[0]
		minVal = data[0]
	}

	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nans++
			continue
		}
		if math.IsInf(float64(v), 0) {
			infs++
			continue
		}
		if v == 0 {
			zeros++
		}
		if v > maxVal {
			maxVal = v
		}
		if v < minVal {
			minVal = v
		}
		sum += v
		sumSq += float64(v) * float64(v)
	}

	n := len(data) - nans - infs
	mean := float32(0)
	rms := float32(0)
	if n > 0 {
		mean = sum / float32(n)
		rms = float32(math.Sqrt(sumSq / float64(n)))
	}

	limit := sampleSize
	if limit > 32 {
		limit = 32
	}
	if len(data) < limit {
		limit = len(data)
	}
	sample := make([]float32, limit)
	copy(sample, data[:limit])

	return ActivationStats{
		Max:    maxVal,
		Min:    minVal,
		Mean:   mean,
		RMS:    rms,
		Zeros:  zeros,
		NaNs:   nans,
		Infs:   infs,
		Sample: sample,
	}
}
