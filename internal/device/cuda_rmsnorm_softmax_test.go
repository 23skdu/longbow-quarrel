//go:build linux && cuda

package device

import (
	"math"
	"math/rand"
	"testing"
)

func TestCUDA_WarpShuffle_RMSNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testDims := []struct {
		rows int
		cols int
		eps  float32
	}{
		{rows: 1, cols: 128, eps: 1e-5},
		{rows: 2, cols: 512, eps: 1e-5},
		{rows: 4, cols: 4096, eps: 1e-5}, // Standard Mistral/Llama hidden dimension
		{rows: 1, cols: 4096, eps: 1e-5}, // Single-token decode hotpath
	}

	for _, td := range testDims {
		inputData := make([]float32, td.rows*td.cols)
		weightData := make([]float32, td.cols)
		for i := range inputData {
			inputData[i] = (rand.Float32() - 0.5) * 4.0
		}
		for i := range weightData {
			weightData[i] = rand.Float32()*1.5 + 0.5
		}

		inTensor := ctx.NewTensorFP32(td.rows, td.cols)
		defer inTensor.Free()
		inTensor.LoadFrom(inputData)

		wTensor := ctx.NewTensorFP32(1, td.cols)
		defer wTensor.Free()
		wTensor.LoadFrom(weightData)

		outTensor := ctx.NewTensorFP32(td.rows, td.cols)
		defer outTensor.Free()

		ctx.RMSNorm(inTensor, wTensor, outTensor, td.rows, td.cols, td.eps)
		ctx.Synchronize()

		outData := outTensor.ToHost()

		// Verify against CPU reference
		for r := 0; r < td.rows; r++ {
			var sumSq float64
			for c := 0; c < td.cols; c++ {
				v := float64(inputData[r*td.cols+c])
				sumSq += v * v
			}
			rms := math.Sqrt(sumSq/float64(td.cols) + float64(td.eps))

			for c := 0; c < td.cols; c++ {
				expected := (float64(inputData[r*td.cols+c]) / rms) * float64(weightData[c])
				actual := float64(outData[r*td.cols+c])
				diff := math.Abs(expected - actual)
				if diff > 1e-3 {
					t.Fatalf("RMSNorm mismatch at row %d, col %d: expected %f, got %f, diff %e", r, c, expected, actual, diff)
				}
			}
		}
	}
}

func BenchmarkCUDA_WarpShuffle_RMSNorm_4096(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	cols := 4096
	rows := 1
	inTensor := ctx.NewTensorFP32(rows, cols)
	defer inTensor.Free()
	wTensor := ctx.NewTensorFP32(1, cols)
	defer wTensor.Free()
	outTensor := ctx.NewTensorFP32(rows, cols)
	defer outTensor.Free()

	b.ResetTimer()
	for b.Loop() {
		ctx.RMSNorm(inTensor, wTensor, outTensor, rows, cols, 1e-5)
	}
	ctx.Synchronize()
}
