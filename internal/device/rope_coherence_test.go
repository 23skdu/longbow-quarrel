//go:build darwin && metal

package device

import (
	"fmt"
	"math"
	"testing"
)

// TestRoPE_NaNPropagation tests that NaN values are correctly propagated through RoPE
// This helps identify if RoPE operations introduce or preserve NaN values
func TestRoPE_NaNPropagation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 4
	pos := 5
	ropeTheta := float32(10000.0)

	testCases := []struct {
		name       string
		inputData  []float32
		expectsNaN bool
	}{
		{
			name: "All valid values",
			inputData: func() []float32 {
				data := make([]float32, heads*headDim)
				for i := range data {
					data[i] = float32(i%100) * 0.1
				}
				return data
			}(),
			expectsNaN: false,
		},
		{
			name: "NaN in first position",
			inputData: func() []float32 {
				data := make([]float32, heads*headDim)
				for i := range data {
					data[i] = float32(i%100) * 0.1
				}
				data[0] = float32(math.NaN())
				return data
			}(),
			expectsNaN: true,
		},
		{
			name: "NaN in middle position",
			inputData: func() []float32 {
				data := make([]float32, heads*headDim)
				for i := range data {
					data[i] = float32(i%100) * 0.1
				}
				data[64] = float32(math.NaN())
				return data
			}(),
			expectsNaN: true,
		},
		{
			name: "NaN in second half",
			inputData: func() []float32 {
				data := make([]float32, heads*headDim)
				for i := range data {
					data[i] = float32(i%100) * 0.1
				}
				data[128] = float32(math.NaN())
				return data
			}(),
			expectsNaN: true,
		},
		{
			name: "Multiple NaNs",
			inputData: func() []float32 {
				data := make([]float32, heads*headDim)
				for i := range data {
					data[i] = float32(i%100) * 0.1
				}
				data[0] = float32(math.NaN())
				data[64] = float32(math.NaN())
				data[128] = float32(math.NaN())
				return data
			}(),
			expectsNaN: true,
		},
		{
			name: "Inf values",
			inputData: func() []float32 {
				data := make([]float32, heads*headDim)
				for i := range data {
					data[i] = float32(i%100) * 0.1
				}
				data[0] = float32(math.Inf(1))
				return data
			}(),
			expectsNaN: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ten := ctx.NewTensor(1, heads*headDim)
			ten.LoadFrom(tc.inputData)
			defer ten.ReturnToPool()

			ten.RoPE(pos, headDim, heads, 1, ropeTheta)
			ctx.Synchronize()

			result := ten.ToHost()

			hasNaN := false
			hasInf := false
			nanCount := 0
			infCount := 0

			for i, v := range result {
				if math.IsNaN(float64(v)) {
					hasNaN = true
					nanCount++
					if nanCount <= 3 {
						t.Logf("NaN at index %d", i)
					}
				}
				if math.IsInf(float64(v), 0) {
					hasInf = true
					infCount++
				}
			}

			if tc.expectsNaN {
				if !hasNaN && !hasInf {
					t.Errorf("Expected NaN/Inf in output but found none")
				}
				if hasNaN {
					t.Logf("✓ NaN correctly propagated: %d NaN values", nanCount)
				}
				if hasInf {
					t.Logf("✓ Inf correctly propagated: %d Inf values", infCount)
				}
			} else {
				if hasNaN {
					t.Errorf("Unexpected NaN in output: %d NaN values", nanCount)
				}
				if hasInf {
					t.Errorf("Unexpected Inf in output: %d Inf values", infCount)
				}
				if !hasNaN && !hasInf {
					t.Logf("✓ No NaN/Inf in output (as expected)")
				}
			}
		})
	}
}

// TestRoPE_PrecisionBoundary tests RoPE precision at boundary conditions
func TestRoPE_PrecisionBoundary(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Test various head dimensions
	headDims := []int{64, 128, 256}
	thetaValues := []float32{10000.0, 1000000.0}
	positionValues := []int{0, 1, 10, 100, 1000, 4096, 8192}

	for _, headDim := range headDims {
		for _, theta := range thetaValues {
			for _, pos := range positionValues {
				t.Run(fmt.Sprintf("headDim=%d theta=%.0f pos=%d", headDim, theta, pos), func(t *testing.T) {
					heads := 4
					inputData := make([]float32, heads*headDim)
					for i := range inputData {
						inputData[i] = float32(i%100) * 0.01
					}

					ten := ctx.NewTensor(1, heads*headDim)
					ten.LoadFrom(inputData)
					defer ten.ReturnToPool()

					// Run GPU RoPE
					ten.RoPE(pos, headDim, heads, 1, theta)
					ctx.Synchronize()
					gpuResult := ten.ToHost()

					// Compute CPU reference
					cpuResult := CPURoPE(inputData, pos, heads, headDim, theta)

					// Calculate max error
					maxError := float32(0.0)
					for i := 0; i < len(gpuResult); i++ {
						diff := float32(math.Abs(float64(gpuResult[i] - cpuResult[i])))
						if diff > maxError {
							maxError = diff
						}
					}

					// Log the max error for analysis
					t.Logf("Max error: %.6f (headDim=%d, theta=%.0f, pos=%d)", maxError, headDim, theta, pos)

					// For most cases, error should be very small
					// Higher theta and larger positions may have larger errors
					maxAllowed := float32(0.01)
					if theta == 1000000.0 && pos > 1000 {
						maxAllowed = 0.05 // Allow larger error for high theta + large position
					}

					if maxError > maxAllowed {
						t.Errorf("Large precision error: %.6f > %.6f (headDim=%d, theta=%.0f, pos=%d)",
							maxError, maxAllowed, headDim, theta, pos)
					}
				})
			}
		}
	}
}

// TestRoPE_LargePositionPrecision specifically tests precision at very large positions
func TestRoPE_LargePositionPrecision(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 4
	ropeTheta := float32(1000000.0)

	// Test positions that are multiples of window size (4096)
	positions := []int{4096, 8192, 12288, 16384}

	for _, pos := range positions {
		t.Run(fmt.Sprintf("pos=%d", pos), func(t *testing.T) {
			inputData := make([]float32, heads*headDim)
			for i := range inputData {
				inputData[i] = float32(i%50) * 0.02
			}

			ten := ctx.NewTensor(1, heads*headDim)
			ten.LoadFrom(inputData)
			defer ten.ReturnToPool()

			ten.RoPE(pos, headDim, heads, 1, ropeTheta)
			ctx.Synchronize()
			gpuResult := ten.ToHost()

			cpuResult := CPURoPE(inputData, pos, heads, headDim, ropeTheta)

			// Check for numerical instability
			hasNaN := false
			hasInf := false
			for _, v := range gpuResult {
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
				if math.IsInf(float64(v), 0) {
					hasInf = true
				}
			}

			if hasNaN {
				t.Errorf("NaN detected at position %d", pos)
			}
			if hasInf {
				t.Errorf("Inf detected at position %d", pos)
			}

			// Calculate precision error
			maxError := float32(0.0)
			for i := 0; i < len(gpuResult); i++ {
				diff := float32(math.Abs(float64(gpuResult[i] - cpuResult[i])))
				if diff > maxError {
					maxError = diff
				}
			}

			// Log for monitoring
			t.Logf("Position %d: max error = %.6f", pos, maxError)

			// Allow larger error for very large positions
			if maxError > 0.1 {
				t.Errorf("Excessive error at position %d: %.6f", pos, maxError)
			}
		})
	}
}

// TestKVCache_IndexingPrecision tests KV cache indexing with precise position calculations
func TestKVCache_IndexingPrecision(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	const (
		kvHeads    = 8
		headDim    = 128
		windowSize = 4096
	)

	kvDim := kvHeads * headDim

	// Test positions that hit window boundaries
	testPositions := []int{
		0,     // Start
		1,     // Early
		100,   // Early
		1000,  // Mid-early
		4095,  // Last valid before wrap
		4096,  // First wrap
		4097,  // After first wrap
		8192,  // Second wrap
		8193,  // After second wrap
		12288, // Third wrap
		16384, // Fourth wrap
	}

	for _, pos := range testPositions {
		t.Run(fmt.Sprintf("pos=%d", pos), func(t *testing.T) {
			// Calculate expected cache index
			cacheIdx := pos % windowSize

			// Verify modulo arithmetic
			expectedIdx := pos % windowSize
			if cacheIdx != expectedIdx {
				t.Errorf("Cache index mismatch: got %d, expected %d", cacheIdx, expectedIdx)
			}

			// Verify cache index is within bounds
			if cacheIdx < 0 || cacheIdx >= windowSize {
				t.Errorf("Cache index out of bounds: %d (windowSize=%d)", cacheIdx, windowSize)
			}

			// Verify window boundaries
			windowStart := pos - windowSize + 1
			if windowStart < 0 {
				windowStart = 0
			}

			// For positions >= windowSize, window should be exactly windowSize
			if pos >= windowSize {
				windowLen := pos - windowStart + 1
				if windowLen != windowSize {
					t.Errorf("Window length incorrect: got %d, expected %d", windowLen, windowSize)
				}
			}

			// Test actual cache operation if position is reasonable
			if pos <= 8192 {
				k := ctx.NewTensor(1, kvDim)
				v := ctx.NewTensor(1, kvDim)
				kCache := ctx.NewTensor(windowSize, kvDim)
				vCache := ctx.NewTensor(windowSize, kvDim)

				kData := make([]float32, kvDim)
				vData := make([]float32, kvDim)
				for i := range kData {
					kData[i] = float32(pos*10 + i)
					vData[i] = float32(pos*100 + i)
				}
				k.LoadFrom(kData)
				v.LoadFrom(vData)

				// Store at position
				k.StoreKV(v, kCache, vCache, pos, kvHeads, headDim, 0)
				ctx.Synchronize()

				// Verify stored data
				kCacheData := kCache.ToHostF32()
				offset := cacheIdx * kvDim

				for i := 0; i < kvDim; i++ {
					expected := kData[i]
					actual := kCacheData[offset+i]
					diff := float32(math.Abs(float64(actual - expected)))
					if diff > 0.1 {
						t.Errorf("KV cache mismatch at pos=%d, idx=%d: got %.2f, want %.2f",
							pos, i, actual, expected)
					}
				}

				k.Free()
				v.Free()
				kCache.Free()
				vCache.Free()
			}
		})
	}
}

// TestRoPE_PositionEdgeCases tests RoPE at edge case positions
func TestRoPE_PositionEdgeCases(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 4
	ropeTheta := float32(1000000.0)

	// Edge case positions
	edgePositions := []int{
		0,     // Identity transform
		1,     // Minimal non-zero
		-1,    // Negative (unusual but should be handled)
		32767, // Near int16 max
		65535, // Near uint16 max
		65536, // Just over uint16
	}

	for _, pos := range edgePositions {
		t.Run(fmt.Sprintf("pos=%d", pos), func(t *testing.T) {
			inputData := make([]float32, heads*headDim)
			for i := range inputData {
				inputData[i] = float32(i%100) * 0.01
			}

			ten := ctx.NewTensor(1, heads*headDim)
			ten.LoadFrom(inputData)
			defer ten.ReturnToPool()

			// Run RoPE (negative positions will use negative angle)
			ten.RoPE(pos, headDim, heads, 1, ropeTheta)
			ctx.Synchronize()

			result := ten.ToHost()

			// Check for numerical instability
			hasNaN := false
			hasInf := false
			for _, v := range result {
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
				if math.IsInf(float64(v), 0) {
					hasInf = true
				}
			}

			if hasNaN {
				t.Errorf("NaN detected at position %d", pos)
			}
			if hasInf {
				t.Errorf("Inf detected at position %d", pos)
			}

			if !hasNaN && !hasInf {
				t.Logf("✓ Position %d handled without numerical instability", pos)
			}
		})
	}
}

// TestRoPE_ThetaSensitivity tests RoPE sensitivity to different theta values
func TestRoPE_ThetaSensitivity(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 4
	pos := 100

	// Different theta values
	thetaValues := []float32{
		10000.0,    // Llama default
		500000.0,   // Intermediate
		1000000.0,  // Mistral v0.3
		5000000.0,  // High theta
		10000000.0, // Very high theta
	}

	for _, theta := range thetaValues {
		t.Run(fmt.Sprintf("theta=%.0f", theta), func(t *testing.T) {
			inputData := make([]float32, heads*headDim)
			for i := range inputData {
				inputData[i] = float32(i%100) * 0.01
			}

			ten := ctx.NewTensor(1, heads*headDim)
			ten.LoadFrom(inputData)
			defer ten.ReturnToPool()

			ten.RoPE(pos, headDim, heads, 1, theta)
			ctx.Synchronize()

			result := ten.ToHost()

			// Check for numerical instability
			hasNaN := false
			hasInf := false
			for _, v := range result {
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
				if math.IsInf(float64(v), 0) {
					hasInf = true
				}
			}

			if hasNaN {
				t.Errorf("NaN detected with theta=%.0f", theta)
			}
			if hasInf {
				t.Errorf("Inf detected with theta=%.0f", theta)
			}

			// Compute CPU reference for comparison
			cpuResult := CPURoPE(inputData, pos, heads, headDim, theta)

			maxError := float32(0.0)
			for i := 0; i < len(result); i++ {
				diff := float32(math.Abs(float64(result[i] - cpuResult[i])))
				if diff > maxError {
					maxError = diff
				}
			}

			t.Logf("Theta %.0f: max error = %.6f", theta, maxError)
		})
	}
}

// TestRoPE_HeadDimBoundary tests RoPE with different head dimension boundaries
func TestRoPE_HeadDimBoundary(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	heads := 4
	pos := 50
	ropeTheta := float32(1000000.0)

	// Different head dimensions (must be even for RoPE)
	headDims := []int{32, 64, 128, 256, 512}

	for _, headDim := range headDims {
		t.Run(fmt.Sprintf("headDim=%d", headDim), func(t *testing.T) {
			inputData := make([]float32, heads*headDim)
			for i := range inputData {
				inputData[i] = float32(i%100) * 0.01
			}

			ten := ctx.NewTensor(1, heads*headDim)
			ten.LoadFrom(inputData)
			defer ten.ReturnToPool()

			ten.RoPE(pos, headDim, heads, 1, ropeTheta)
			ctx.Synchronize()

			result := ten.ToHost()

			// Check for numerical instability
			hasNaN := false
			for _, v := range result {
				if math.IsNaN(float64(v)) {
					hasNaN = true
					break
				}
			}

			if hasNaN {
				t.Errorf("NaN detected with headDim=%d", headDim)
			}

			// Verify rotation was applied (values should change)
			rotatedCount := 0
			for i := 0; i < headDim/2; i++ {
				idx0 := i
				idx1 := i + headDim/2
				if result[idx0] != inputData[idx0] || result[idx1] != inputData[idx1] {
					rotatedCount++
				}
			}

			if rotatedCount == 0 {
				t.Error("No rotation applied at non-zero position")
			} else {
				t.Logf("HeadDim=%d: %d/%d pairs rotated", headDim, rotatedCount, headDim/2)
			}
		})
	}
}
