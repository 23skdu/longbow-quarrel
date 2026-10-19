//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/simd"
)

// TestQ4K_CPUGPUParity checks that the CPU and the GPU decode the same
// Q4_K weights the same way.
//
// The two used to disagree in two ways. The CPU unpacked the six-bit
// scales and the minimums with a different convention from the CUDA
// kernel, and the CUDA matrix-vector kernel read the four-bit values in
// an interleaved order while the CUDA dequantization kernel next to it
// read them in the GGML order. A Q4_K model therefore produced different
// results on the CPU and on the GPU.
func TestQ4K_CPUGPUParity(t *testing.T) {
	const rows = 64
	const cols = 512 // two Q4_K blocks per row

	ctx := NewContext()
	defer ctx.Free()

	// A weight matrix with a spread of values.
	weights := make([]float32, rows*cols)
	for i := range weights {
		weights[i] = float32((i*37)%101-50) / 50.0
	}

	quantized, err := gguf.QuantizeWeightsToQ4K(weights, len(weights))
	if err != nil {
		t.Fatalf("failed to quantize: %v", err)
	}

	// The input vector.
	input := make([]float32, cols)
	for i := range input {
		input[i] = float32((i*13)%17-8) / 8.0
	}

	// The CPU: the SIMD Q4_K matrix-vector product.
	cpu := simd.MatVecMulQ4_K_VNNI(quantized, input, rows, cols)

	// The GPU: the CUDA Q4_K matrix-vector product.
	weight, err := ctx.NewRawTensor(quantized)
	if err != nil {
		t.Fatalf("failed to create the Q4_K tensor: %v", err)
	}
	defer weight.Free()

	x := ctx.NewTensorFP32(1, cols)
	if x == nil {
		t.Fatal("failed to create the input tensor")
	}
	defer x.Free()
	if err := x.LoadFrom(input); err != nil {
		t.Fatalf("failed to load the input: %v", err)
	}

	y := ctx.NewTensorFP32(rows, 1)
	if y == nil {
		t.Fatal("failed to create the output tensor")
	}
	defer y.Free()

	ctx.MatVecDequantQ4_K(weight, x, y, rows, cols)
	ctx.Synchronize()

	gpu := y.ToHostF32()
	if len(gpu) != rows {
		t.Fatalf("the GPU produced %d results, want %d", len(gpu), rows)
	}

	// The two have to agree to within the precision of the format. The
	// reference is the float32 product, and the quantization error is
	// bounded by the step of the coarsest group.
	for i := range cpu {
		ref := 0.0
		for j := 0; j < cols; j++ {
			ref += float64(weights[i*cols+j] * input[j])
		}
		diff := math.Abs(float64(cpu[i]) - ref)
		if diff > 5.0 {
			t.Fatalf("the CPU result %d is %v, want about %v (difference %v)", i, cpu[i], ref, diff)
		}
		gpuDiff := math.Abs(float64(gpu[i]) - ref)
		if gpuDiff > 5.0 {
			t.Fatalf("the GPU result %d is %v, want about %v (difference %v)", i, gpu[i], ref, gpuDiff)
		}
		cpuGPU := math.Abs(float64(cpu[i]) - float64(gpu[i]))
		if cpuGPU > 0.05 {
			t.Fatalf("the CPU and the GPU disagree on row %d: %v against %v", i, cpu[i], gpu[i])
		}
	}
}

// TestQ4K_DecodeMatchesTheReference checks the pure Go decoder against
// the reference implementation of the GGML layout.
func TestQ4K_DecodeMatchesTheReference(t *testing.T) {
	const n = 256
	values := make([]float32, n)
	for i := range values {
		values[i] = float32((i*29)%61-30) / 30.0
	}

	quantized, err := gguf.QuantizeWeightsToQ4K(values, n)
	if err != nil {
		t.Fatalf("failed to quantize: %v", err)
	}

	got := gguf.DequantizeQ4K(quantized, n)
	if len(got) != n {
		t.Fatalf("the decoder produced %d values, want %d", len(got), n)
	}

	// The reference decoder, which is the GGML layout.
	want := DequantizeQ4K_Reference(quantized[:144])
	for i := range want {
		if math.Abs(float64(got[i])-float64(want[i])) > 1e-4 {
			t.Fatalf("the value %d is %v, want %v", i, got[i], want[i])
		}
	}

	// The values have to survive the round trip to within the step of
	// the format.
	step := 0.0
	for i := 1; i < n; i++ {
		if d := math.Abs(float64(values[i] - values[i-1])); d > step {
			step = d
		}
	}
	for i := range values {
		if math.Abs(float64(values[i])-float64(got[i])) > 0.2 {
			t.Fatalf("the value %d is %v, want about %v", i, got[i], values[i])
		}
	}
}
