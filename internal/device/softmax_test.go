//go:build darwin && metal

package device

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmax_Precision(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Test configurations
	testCases := []struct {
		name     string
		pos      int
		stride   int
		maxRange float32
	}{
		{"Short_Normal", 31, 32, 5.0},
		{"Long_Extreme", 1023, 1024, 50.0}, // Mistral/Llama usually max out at 1024 or higher
		{"Large_Stride", 127, 256, 10.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Generate Input
			inputData := make([]float32, tc.stride)
			for i := 0; i <= tc.pos; i++ {
				// Mix of large and small values
				inputData[i] = (rand.Float32() - 0.5) * tc.maxRange
			}
			// Mask values beyond pos with large negative if needed,
			// though the kernel should respect 'pos'.

			// 2. CPU Reference (FP64 for maximum precision)
			expected := make([]float32, tc.stride)
			maxVal := -math.MaxFloat64
			for i := 0; i <= tc.pos; i++ {
				if float64(inputData[i]) > maxVal {
					maxVal = float64(inputData[i])
				}
			}
			sum := 0.0
			for i := 0; i <= tc.pos; i++ {
				e := math.Exp(float64(inputData[i]) - maxVal)
				expected[i] = float32(e)
				sum += e
			}
			for i := 0; i <= tc.pos; i++ {
				expected[i] /= float32(sum)
			}

			// 3. GPU Execution
			scoresTen := ctx.NewTensorFP32(1, tc.stride)
			scoresTen.LoadFrom(inputData)

			scoresTen.AttSoftmax(tc.pos, 1, tc.stride)
			ctx.Synchronize()

			gpuOut := scoresTen.ToHost()

			// 4. Verification
			for i := 0; i <= tc.pos; i++ {
				diff := math.Abs(float64(expected[i] - gpuOut[i]))
				if diff > 1e-5 {
					t.Errorf("Mismatch at index %d: Expected %f, Got %f, Diff %f", i, expected[i], gpuOut[i], diff)
				}
			}

			// Verify sum is 1.0
			gpuSum := 0.0
			for i := 0; i <= tc.pos; i++ {
				gpuSum += float64(gpuOut[i])
			}
			if math.Abs(gpuSum-1.0) > 1e-4 {
				t.Errorf("Sum mismatch: %f (expected 1.0)", gpuSum)
			}
		})
	}
}
