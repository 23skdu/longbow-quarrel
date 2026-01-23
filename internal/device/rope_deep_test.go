//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestRoPE_Mistral_Deep(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Mistral Params
	dim := 128         // Head Dim
	theta := 1000000.0 // 1M Theta

	rows := 1 // 1 token

	// Create Tensors
	q := ctx.NewTensor(rows, dim) // F16
	// out := ctx.NewTensor(rows, dim) // Removed, RoPE is in-place

	// Data: All 1.0s to simplify check
	data := make([]float32, dim)
	for i := range data {
		data[i] = 1.0
	}
	q.LoadFrom(data)

	// Run RoPE
	// Pos 0
	// Signature: RoPE(posOffset, headDim, numHeads, seqLen, theta)
	// For our test, we have 1 row (seqLen=1), 1 head (numHeads=1), headDim=128
	q.RoPE(0, dim, 1, 1, float32(theta))

	outData := q.ToHost()

	// Verify Pos 0: Reference should be just 1.0 (Rotation by 0)
	// cos(0) = 1, sin(0) = 0
	// out[i] = x[i]*1 - x[i+half]*0 = 1
	// out[i+half] = x[i]*0 + x[i+half]*1 = 1

	for i := 0; i < dim; i++ {
		if math.Abs(float64(outData[i])-1.0) > 1e-3 {
			t.Errorf("Pos 0 RoPE failed at %d: got %f, expected 1.0", i, outData[i])
		}
	}

	// Pos 1
	// Reset data
	q.LoadFrom(data)
	q.RoPE(1, dim, 1, 1, float32(theta))
	outData = q.ToHost()

	// Calculate Reference
	// RoPE rotates pairs (i, i+dim/2) by angle theta_i * pos
	// theta_i = base^(-2*i/dim)

	nRot := dim / 2
	for i := 0; i < nRot; i++ {
		freq := 1.0 / math.Pow(theta, float64(2*i)/float64(dim))
		// x[i] = 1, y[i] = 1

		angle := freq * 1.0 // pos=1
		cos := math.Cos(angle)
		sin := math.Sin(angle)

		// x_new = x * cos - y * sin
		// y_new = x * sin + y * cos
		// Here x=1, y=1 (since we set all to 1.0)

		expectedX := 1.0*cos - 1.0*sin
		expectedY := 1.0*sin + 1.0*cos

		gotX := float64(outData[i])
		gotY := float64(outData[i+nRot]) // Pair is at i + dim/2

		if math.Abs(gotX-expectedX) > 1e-3 {
			t.Errorf("Pos 1 RoPE X failed at %d: got %f, expected %f (freq=%f)", i, gotX, expectedX, freq)
		}
		if math.Abs(gotY-expectedY) > 1e-3 {
			t.Errorf("Pos 1 RoPE Y failed at %d: got %f, expected %f", i+nRot, gotY, expectedY)
		}
	}
}
