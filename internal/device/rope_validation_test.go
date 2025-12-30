package device

import (
	"math"
	"testing"
)

// Reference CPU RoPE implementation
func ropeCPU(input []float32, headDim, pos int, theta float64) []float32 {
	output := make([]float32, len(input))
	copy(output, input)

	// Process each head (assuming 1 head for unit test usually, or flattened)
	// Input should be [Heads * HeadDim] or just [HeadDim]
	// If input > headDim, we loop

	for i := 0; i < len(input); i += headDim {
		for j := 0; j < headDim/2; j++ {
			idx0 := i + j
			idx1 := i + j + headDim/2
			
			// theta_i = theta ^ (-2i / d)
			expVar := -2.0 * float32(j) / float32(headDim)
			freq := float32(pos) * float32(math.Pow(theta, float64(expVar)))
			
			cos := float32(math.Cos(float64(freq)))
			sin := float32(math.Sin(float64(freq)))
			
			x0 := input[idx0]
			x1 := input[idx1]
			
			output[idx0] = x0*cos - x1*sin
			output[idx1] = x0*sin + x1*cos
		}
	}
	return output
}

func TestRoPE_Precision_HighTheta(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Mistral v0.3 parameters
	headDim := 128
	theta := 1000000.0 // 1 Million
	pos := 10          // Arbitrary position > 0

	// Create Input (Random-ish)
	inputData := make([]float32, headDim)
	for i := 0; i < headDim; i++ {
		inputData[i] = float32(i) * 0.1
	}

	// 1. Calculate Expected CPU Result
	expected := ropeCPU(inputData, headDim, pos, theta)

	// 2. Run GPU Kernel
	tIn := ctx.NewTensor(1, headDim)
	tIn.LoadFrom(inputData)
	
	// RoPE(posOffset, headDim, numHeads, seqLen int, ropeTheta float32)
	tIn.RoPE(pos, headDim, 1, 1, float32(theta))
	
	gpuRes := tIn.ToHost()

	// 3. Compare with strict tolerance
	// With 1e6 theta, precision loss in exp/pow might be significant in float16/float32 mixed
	maxErr := float32(0.0)
	for i := 0; i < headDim; i++ {
		diff := float32(math.Abs(float64(gpuRes[i] - expected[i])))
		if diff > maxErr {
			maxErr = diff
		}
		if diff > 1e-3 {
			t.Errorf("Mismatch at index %d: Got %f, Want %f (Diff: %f)", i, gpuRes[i], expected[i], diff)
		}
	}
	
	t.Logf("Max Error with Theta=1e6: %e", maxErr)
	t.Logf("Sample GPU[0:4]: %v", gpuRes[:4])
	t.Logf("Sample CPU[0:4]: %v", expected[:4])
}
