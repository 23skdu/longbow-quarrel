package device

import (
	"math"
	"math/rand"
	"testing"
)

// CPU reference for RoPE
func cpuRoPE(data []float32, rows, headDim, heads int, pos int, theta float32) []float32 {
	out := make([]float32, len(data))
	copy(out, data)

	// data is [rows, heads * headDim]
	// but tensor is flat.
	// We only verify single row for now as per kernel limitation
	r := 0
	rowOff := r * (heads * headDim)
	p := pos + r 

	for h := 0; h < heads; h++ {
		headOff := rowOff + h*headDim

		for i := 0; i < headDim/2; i++ {
			// theta^(-2i/dim)
			freq := float32(float64(p) * math.Pow(float64(theta), -2.0*float64(i)/float64(headDim)))
			cos := float32(math.Cos(float64(freq)))
			sin := float32(math.Sin(float64(freq)))

			idx1 := headOff + i
			idx2 := headOff + i + headDim/2

			x1 := data[idx1]
			x2 := data[idx2]

			// Rotation: [x1*c - x2*s, x1*s + x2*c]
			out[idx1] = x1*cos - x2*sin
			out[idx2] = x1*sin + x2*cos
		}
	}
	return out
}

func TestRoPEKernel(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows := 1 // Test single row
	heads := 2
	headDim := 64
	cols := heads * headDim
	pos := 10
	theta := float32(10000.0)

	// Create random input
	inputData := make([]float32, rows*cols)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}

	// CPU Result
	cpuOut := cpuRoPE(inputData, rows, headDim, heads, pos, theta)

	// Metal Result
	tRow := ctx.NewTensor(rows, cols)
	tRow.LoadFrom(inputData)
	
	// Pass 1 as batch/seqLen
	tRow.RoPE(pos, headDim, heads, 1, theta) 
	tRow.ctx.Synchronize()
	
	metalOut := tRow.ToHost()
	
	// Compare
	for i, v := range metalOut {
		// FP16 precision tolerance
		if math.Abs(float64(v - cpuOut[i])) > 1e-2 {
			t.Errorf("Mismatch at %d: CPU %f, Metal %f (Diff: %f)", i, cpuOut[i], v, v-cpuOut[i])
			return 
		}
	}
}
