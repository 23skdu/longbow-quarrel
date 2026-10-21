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
	if len(got) != len(expected) {
		t.Fatalf("the kernel returned %d values, want %d", len(got), len(expected))
	}

	// The mismatch used to be reported with t.Logf, which does not fail the
	// test: the RoPE precision regression this test exists to catch was
	// reported and then ignored.
	mismatches := 0
	for i := 0; i < len(got); i++ {
		if math.Abs(float64(got[i]-expected[i])) > 1.5e-2 {
			mismatches++
			if mismatches <= 10 {
				t.Errorf("mismatch at index %d: got %f, want %f", i, got[i], expected[i])
			}
		}
	}
	if mismatches > 0 {
		t.Errorf("%d of %d values differ from the reference", mismatches, len(got))
	}
}
