//go:build linux && cuda

package device

import (
	"encoding/binary"
	"math/rand"
	"testing"
)

func float32ToBytes(f32 []float32) []byte {
	buf := make([]byte, len(f32)*4)
	for i, v := range f32 {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	return buf
}

func TestCUDARMSNorm(t *testing.T) {
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatalf("Failed to create CUDA context: %v", err)
	}
	defer ctx.Free()

	rows, cols := 4, 2048
	input := make([]float32, rows*cols)
	for i := range input {
		input[i] = float32(rand.Float64() * 2)
	}
	weight := make([]float32, cols)
	for i := range weight {
		weight[i] = 1.0 + float32(rand.Float64()*0.1)
	}

	inputTensor, _ := ctx.NewTensorFromData(rows, cols, DataTypeF32, float32ToBytes(input))
	weightTensor, _ := ctx.NewTensorFromData(1, cols, DataTypeF32, float32ToBytes(weight))
	outputTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)
	hiddenTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)

	eps := float32(1e-5)
	ctx.FusedRMSNormAdd(inputTensor, hiddenTensor, weightTensor, outputTensor, rows, cols, eps)
	ctx.Synchronize()

	output := outputTensor.ToHostF32()

	if len(output) != len(input) {
		t.Errorf("Output length mismatch: got %d, want %d", len(output), len(input))
	}

	t.Logf("RMSNorm completed - output length: %d", len(output))
}

func TestCUDASwiGLU(t *testing.T) {
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatalf("Failed to create CUDA context: %v", err)
	}
	defer ctx.Free()

	rows, size := 4, 8192
	gate := make([]float32, rows*size)
	up := make([]float32, rows*size)
	for i := range gate {
		gate[i] = float32(rand.Float64()*2 - 1)
		up[i] = float32(rand.Float64()*2 - 1)
	}

	gateTensor, _ := ctx.NewTensorFromData(rows, size, DataTypeF32, float32ToBytes(gate))
	upTensor, _ := ctx.NewTensorFromData(rows, size, DataTypeF32, float32ToBytes(up))
	downTensor, _ := ctx.NewTensor(rows, size, DataTypeF32)
	downWeight, _ := ctx.NewTensor(size*4, size, DataTypeF32)
	inputTensor, _ := ctx.NewTensor(rows, size, DataTypeF32)

	ctx.FusedSwiGLU(inputTensor, gateTensor, upTensor, downWeight, downTensor, rows, size, size*4)
	ctx.Synchronize()

	output := downTensor.ToHostF32()

	if len(output) != len(gate) {
		t.Errorf("Output length mismatch: got %d, want %d", len(output), len(gate))
	}

	t.Logf("SwiGLU completed - output length: %d", len(output))
}

func TestCUDALayerScratch(t *testing.T) {
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatalf("Failed to create CUDA context: %v", err)
	}
	defer ctx.Free()

	scratch := ctx.NewLayerScratch(4096, 4096, 14336, 32, 32, 128, 2048, 49152)

	if scratch == nil {
		t.Errorf("Layer scratch is nil")
	}

	t.Logf("Layer scratch structure created: Logits len=%d", len(scratch.Logits))
}

func BenchmarkCUDAKernel(b *testing.B) {
	ctx, err := NewCUDAContext()
	if err != nil {
		b.Fatalf("Failed to create CUDA context: %v", err)
	}
	defer ctx.Free()

	rows, cols := 32, 4096
	input := make([]float32, rows*cols)
	weight := make([]float32, cols)
	for i := range input {
		input[i] = float32(rand.Float64() * 2)
	}
	for i := range weight {
		weight[i] = 1.0 + float32(rand.Float64()*0.1)
	}

	inputTensor, _ := ctx.NewTensorFromData(rows, cols, DataTypeF32, float32ToBytes(input))
	weightTensor, _ := ctx.NewTensorFromData(1, cols, DataTypeF32, float32ToBytes(weight))
	outputTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)
	hiddenTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)

	eps := float32(1e-5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx.FusedRMSNormAdd(inputTensor, hiddenTensor, weightTensor, outputTensor, rows, cols, eps)
		ctx.Synchronize()
	}
}
