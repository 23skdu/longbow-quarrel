//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestRoPE_Precision(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	numHeads := 32
	pos := 5
	ropeTheta := float32(10000.0)

	// 1. Prepare Input Data
	inputData := make([]float32, numHeads*headDim)
	for i := range inputData {
		inputData[i] = float32(i) / 100.0
	}

	ten := ctx.NewTensor(1, numHeads*headDim)
	ten.LoadFrom(inputData)

	// 2. CPU Reference
	expected := make([]float32, numHeads*headDim)
	copy(expected, inputData)

	for h := 0; h < numHeads; h++ {
		for i := 0; i < headDim/2; i++ {
			theta := float64(pos) * math.Pow(float64(ropeTheta), -2.0*float64(i)/float64(headDim))
			ct := math.Cos(theta)
			st := math.Sin(theta)

			off := h * headDim
			idx0 := off + i
			idx1 := off + i + headDim/2

			v0 := float64(expected[idx0])
			v1 := float64(expected[idx1])

			expected[idx0] = float32(v0*ct - v1*st)
			expected[idx1] = float32(v0*st + v1*ct)
		}
	}

	// 3. Run Kernel
	ten.RoPE(pos, headDim, numHeads, 1, ropeTheta)
	ctx.Synchronize()

	// 4. Verify
	got := ten.ToHost()

	for i := 0; i < len(got); i++ {
		if math.Abs(float64(got[i]-expected[i])) > 1e-3 {
			t.Errorf("Mismatch at index %d: got %f, want %f", i, got[i], expected[i])
			if i > 10 {
				break
			} // Don't spam
		}
	}
}
