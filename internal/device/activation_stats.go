//go:build darwin && metal

package device

// GetActivationStats extracts max and sample values from a tensor
type ActivationStats struct {
	Max    float32
	Sample []float32 // First 10 values
}

// GetStats reads a tensor and returns activation statistics
func (t *Tensor) GetStats(sampleSize int) ActivationStats {
	data := t.ToHost()

	// Find max
	maxVal := float32(0)
	for _, v := range data {
		absV := v
		if absV < 0 {
			absV = -absV
		}
		if absV > maxVal {
			maxVal = absV
		}
	}

	// Get sample
	limit := sampleSize
	if len(data) < limit {
		limit = len(data)
	}
	sample := make([]float32, limit)
	copy(sample, data[:limit])

	return ActivationStats{
		Max:    maxVal,
		Sample: sample,
	}
}
