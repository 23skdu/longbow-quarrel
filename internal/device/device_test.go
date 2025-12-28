//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestMetalAdd(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	count := 512
	a := make([]float32, count)
	b := make([]float32, count)
	expected := make([]float32, count)

	for i := 0; i < count; i++ {
		a[i] = float32(i)
		b[i] = float32(i * 2)
		expected[i] = a[i] + b[i]
	}

	rows, cols := count, 1
	tA := ctx.NewTensor(rows, cols)
	tA.LoadFrom(a)
	tB := ctx.NewTensor(rows, cols)
	tB.LoadFrom(b)

	tC := tA.Add(tB)
	result := tC.ToHost()

	for i := 0; i < count; i++ {
		if math.Abs(float64(result[i]-expected[i])) > 1e-3 {
			t.Fatalf("Add mismatch at %d: got %f, want %f", i, result[i], expected[i])
		}
	}
	
	tA.Free()
	tB.Free()
	tC.Free()
}

func TestMetalScale(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	count := 1024
	a := make([]float32, count)
	expected := make([]float32, count)
	scale := float32(3.5)

	for i := 0; i < count; i++ {
		a[i] = float32(i)
		expected[i] = a[i] * scale
	}

	tA := ctx.NewTensor(count, 1)
	tA.LoadFrom(a)

	tC := tA.Scale(scale)
	result := tC.ToHost()

	for i := 0; i < count; i++ {
		// Looser tolerance for FP16 multiplication
		if math.Abs(float64(result[i]-expected[i])) > 1e-1 {
			if expected[i] != 0 && math.Abs(float64(result[i]-expected[i])/float64(expected[i])) > 0.05 {
				t.Fatalf("Scale mismatch at %d: got %f, want %f", i, result[i], expected[i])
			}
		}
	}

	tA.Free()
	tC.Free()
}

func TestMetalMatMul(t *testing.T) {
	// ... (Existing F16 test)
}



func TestMetalRMSNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows, cols := 2, 4
	input := []float32{1, 1, 1, 1, 2, 2, 2, 2}
	weight := []float32{1, 1, 1, 1}

	tIn := ctx.NewTensor(rows, cols)
	tIn.LoadFrom(input)
	tW := ctx.NewTensor(1, cols)
	tW.LoadFrom(weight)

	res := tIn.RMSNorm(tW, 1e-5)
	out := res.ToHost()

	for _, v := range out {
		if math.Abs(float64(v-1.0)) > 1e-2 {
			t.Fatalf("RMSNorm mismatch: got %f, want 1.0", v)
		}
	}
}

func TestMetalRoPE(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// SeqLen=1, Heads=1, HeadDim=2
	// Theta = 10000.0
	// Pos = 0 (offset)
	// Input: [1.0, 0.0] -> Rotate by 0 -> [1.0, 0.0]
	// Pos = 1
	// Theta = 1 * 10000^(-0) = 1.
	// Cos(1) = 0.5403, Sin(1) = 0.8415
	// [1, 0] -> [1*cos - 0*sin, 1*sin + 0*cos] = [0.5403, 0.8415]
	
	headDim := 2
	rows := 1
	input := []float32{1.0, 0.0}
	
	tIn := ctx.NewTensor(rows, headDim)
	tIn.LoadFrom(input)
	
	// Test Pos 0
	tIn.RoPE(0, headDim, 1, 1, 10000.0)
	out := tIn.ToHost()
	if math.Abs(float64(out[0]-1.0)) > 1e-3 || math.Abs(float64(out[1]-0.0)) > 1e-3 {
		t.Errorf("RoPE pos 0 mismatch: %v", out)
	}
	
	// Reset
	tIn.LoadFrom(input)
	// Test Pos 1
	tIn.RoPE(1, headDim, 1, 1, 10000.0) // Offset=1
	out = tIn.ToHost()
	
	expected0 := float32(math.Cos(1.0))
	expected1 := float32(math.Sin(1.0))
	
	if math.Abs(float64(out[0]-expected0)) > 1e-3 {
		t.Errorf("RoPE pos 1 mismatch [0]: got %f, want %f", out[0], expected0)
	}
	if math.Abs(float64(out[1]-expected1)) > 1e-3 {
		t.Errorf("RoPE pos 1 mismatch [1]: got %f, want %f", out[1], expected1)
	}
}

func TestMetalSwiGLU(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	// Val = [2.0], Gate = [0.0]
	// Silu(0.0) = 0.0 / (1 + exp(0)) = 0.0 / 2 = 0.0
	// Out = 2.0 * 0.0 = 0.0
	
	// Val = [2.0], Gate = [10.0] (Sigmoid(10) ~ 1.0) -> Silu(10) ~ 10.0
	// Out = 2.0 * 10.0 = 20.0
	
	rows := 2
	interSize := 1
	valData := []float32{2.0, 2.0}
	gateData := []float32{0.0, 10.0}
	
	tVal := ctx.NewTensor(rows, interSize) // [2, 1]
	tVal.LoadFrom(valData)
	tGate := ctx.NewTensor(rows, interSize) // [2, 1]
	tGate.LoadFrom(gateData)
	
	res := tVal.SwiGLU(tGate)
	out := res.ToHost()
	
	// Check 0
	if math.Abs(float64(out[0]-0.0)) > 1e-3 {
		t.Errorf("SwiGLU [0] mismatch: got %f, want 0.0", out[0])
	}
	// Check 1
	// Silu(10) = 10 / (1 + exp(-10)) = 10 / (1 + 4.5e-5) ~ 10
	if math.Abs(float64(out[1]-20.0)) > 1e-1 {
		t.Errorf("SwiGLU [1] mismatch: got %f, want ~20.0", out[1])
	}
}
