package device

import (
	"math"
	"math/rand"
	"testing"
)

func TestRMSNorm_Mistral_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows := 2
	cols := 4096 // Mistral Dim
	eps := float32(1e-5) // Mistral Epsilon

	// Create Tensors
	x := ctx.NewTensorFP32(rows, cols) // F32 Input
	w := ctx.NewTensorFP32(1, cols)    // F32 Weights (Corrected)
	
	// Create Random Data
	xData := make([]float32, rows*cols)
	wData := make([]float32, cols)
	
	for i := range xData {
		xData[i] = (rand.Float32() - 0.5) * 10.0 // +/- 5.0
	}
	for i := range wData {
		wData[i] = rand.Float32() * 2.0 // 0..2.0
	}
	
	x.LoadFrom(xData)
	w.LoadFrom(wData)
	
	// Use scratch for output (F16)
	out := ctx.NewTensor(rows, cols) // F16 default if using NewTensor? 
	// NewTensor allocates F16.
	
	// Run Kernel
	// We need to call the exact method used in Engine: RMSNormFP32_ToF16_Into
	x.RMSNormFP32_ToF16_Into(w, eps, out)
	
	// Get Result
	outData := out.ToHost() // F16 -> F32 auto conversion on host
	
	// Compute Reference
	for r := 0; r < rows; r++ {
		// 1. Calc sum of squares
		sum := 0.0
		for c := 0; c < cols; c++ {
			val := float64(xData[r*cols + c])
			sum += val * val
		}
		
		// 2. Mean
		mean := sum / float64(cols)
		
		// 3. Scale
		scale := 1.0 / math.Sqrt(mean + float64(eps))
		
		// 4. Transform
		for c := 0; c < cols; c++ {
			expected := float64(xData[r*cols + c]) * scale * float64(wData[c])
			got := float64(outData[r*cols + c])
			
			// F16 precision tolerance
			if math.Abs(got - expected) > (math.Abs(expected)*0.01 + 0.01) {
				t.Errorf("Row %d Col %d: got %f, expected %f (scale=%f, input=%f, weight=%f)",
					r, c, got, expected, scale, xData[r*cols+c], wData[c])
				// Fail fast
				if r == 0 && c < 10 { return }
			}
		}
	}
}
