//go:build darwin && metal

package device

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"encoding/binary"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestQ4K_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Parameters
	cols := 256 // Must be multiple of 256 for Q4_K
	rows := 1   // Single row test (Vector-Vector dot product essentially)

	// 1. Generate Random Q4_K Data (144 bytes per 256 elements)
	// We use random bytes, which is a valid bit pattern (raw noise).
	numBlocks := (rows * cols) / 256
	dataSize := numBlocks * 144
	q4kData := make([]byte, dataSize)
	rand.Seed(time.Now().UnixNano())
	rand.Read(q4kData) // Randomize scales, mins, quants

	// FIX: Ensure d (bytes 0-1) and dmin (bytes 2-3) are valid Float16s
	// Random bytes can form NaNs (exponent=31, mantissa!=0) or Infs.
	for i := 0; i < numBlocks; i++ {
		offset := i * 144
		// Generate valid random float16s for d and dmin
		// Small positive numbers to prevent explosion
		d := Float32ToFloat16(rand.Float32() * 0.1)
		dmin := Float32ToFloat16(rand.Float32() * 0.1)

		// Write back
		binary.LittleEndian.PutUint16(q4kData[offset:], d)
		binary.LittleEndian.PutUint16(q4kData[offset+2:], dmin)
	}

	// 2. CPU Reference: Dequantize to F32 then Dot Product
	weightsF32 := gguf.DequantizeQ4K(q4kData, rows*cols)

	// Generate Random Input Vector (F32)
	inputF32 := make([]float32, cols)
	for i := range inputF32 {
		inputF32[i] = (rand.Float32() - 0.5) * 0.05 // Small range to avoid F16 saturation
	}

	// Calculate Expected Dot Product (CPU)
	expected := float32(0.0)
	for i := 0; i < cols; i++ {
		expected += weightsF32[i] * inputF32[i]
	}

	// 3. GPU Execution
	// Create Q4K Weight Tensor
	wTen, err := ctx.NewQ4KTensor(rows, cols)
	if err != nil {
		t.Fatalf("Failed to create Q4K tensor: %v", err)
	}
	if err := wTen.LoadFromRaw(q4kData); err != nil {
		t.Fatalf("Failed to load Q4K data: %v", err)
	}

	// Create Input Vector (F16)
	// MatMul expects B as [M x K] where M is batch dims.
	// We want M=1.
	inTen := ctx.NewTensor(1, cols)
	inTen.LoadFrom(inputF32)

	// Run MatMul: C = A(1x256) * B(256x1) = 1x1
	outTen := wTen.MatMul(inTen)

	if err := ctx.WaitWithTimeout(2 * time.Second); err != nil {
		t.Fatal(err)
	}

	res := outTen.ToHost()

	wTen.Free()
	inTen.Free()
	outTen.Free()

	if len(res) != 1 {
		t.Fatalf("Expected 1 result element, got %d", len(res))
	}

	got := res[0]

	t.Logf("CPU Reference: %f", expected)
	t.Logf("GPU Result:    %f", got)

	// 4. Comparison
	// Tolerance: F16 precision loss + Q4_K decoding differences?
	// The CPU ref performs float32 arithmetic.
	// The GPU one performs half arithmetic (mostly).
	// But `dequant.go` converts `float16ToFloat32` explicitly.
	// Metal `kernel` likely promotes to float for mul then accum?
	// linear_q4k_f16 uses `float sum` accumulator. And `float` intermediate for dequant.
	// So precision should be decent.

	diff := float32(math.Abs(float64(got - expected)))
	// Relative error might be better if value is large.
	// But `expected` could be near 0.

	// With 256 elements, errors can accumulate.
	// Tolerance 1.0? or 1%?
	// Let's check magnitude.

	if diff > 1.0 && diff > float32(math.Abs(float64(expected)))*0.05 {
		t.Errorf("Mismatch. Expected %f, Got %f, Diff %f", expected, got, diff)
	}
}

func TestQ3K_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	cols := 256
	rows := 1

	// Q3_K: 110 bytes per 256
	numBlocks := (rows * cols) / 256
	dataSize := numBlocks * 110
	q3kData := make([]byte, dataSize)
	rand.Seed(time.Now().UnixNano())
	rand.Read(q3kData)

	// FIX: Ensure d (bytes 108-109) is a valid and safe Float16
	for i := 0; i < numBlocks; i++ {
		offset := i * 110
		d := Float32ToFloat16(rand.Float32() * 0.1)
		binary.LittleEndian.PutUint16(q3kData[offset+108:], d)
	}

	// CPU Reference
	weightsF32 := gguf.DequantizeQ3K(q3kData, rows*cols)

	inputF32 := make([]float32, cols)
	for i := range inputF32 {
		inputF32[i] = (rand.Float32() - 0.5) * 2.0
	}

	expected := float32(0.0)
	for i := 0; i < cols; i++ {
		expected += weightsF32[i] * inputF32[i]
	}

	// GPU Execution
	wTen, err := ctx.NewQ3KTensor(rows, cols)
	if err != nil {
		t.Fatalf("Failed to create Q3K tensor: %v", err)
	}
	if err := wTen.LoadFromRaw(q3kData); err != nil {
		t.Fatalf("Failed to load Q3K data: %v", err)
	}

	inTen := ctx.NewTensor(cols, 1)
	inTen.LoadFrom(inputF32)

	// Explicit Dispatch to allow ZeroInit
	c := ctx.NewTensor(rows, 1)
	c.ZeroInit() // Start with 0.0

	ctx.RunQ3K_Explicit(wTen, inTen, c)

	if err := ctx.WaitWithTimeout(2 * time.Second); err != nil {
		t.Fatal(err)
	}

	res := c.ToHost()

	wTen.Free()
	inTen.Free()
	c.Free()

	if len(res) != 1 {
		t.Fatalf("Expected 1 result element, got %d", len(res))
	}

	got := res[0]

	t.Logf("Q3K CPU Ref: %f", expected)
	t.Logf("Q3K GPU Res: %f", got)

	diff := float32(math.Abs(float64(got - expected)))

	// Q3 is lower precision, heavier quantization.
	// Error tolerance might need to be slightly higher?
	// But dequantization itself is exact logic on bits.
	// Only float adds are inexact.

	if diff > 1.0 && diff > float32(math.Abs(float64(expected)))*0.05 {
		t.Errorf("Mismatch. Expected %f, Got %f, Diff %f", expected, got, diff)
	}
}

func TestQ6K_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	cols := 256
	rows := 1

	// Q6_K: 210 bytes per 256
	numBlocks := (rows * cols) / 256
	dataSize := numBlocks * 210
	q6kData := make([]byte, dataSize)
	rand.Read(q6kData)

	// Ensure some non-zero scales for stability
	for i := 0; i < numBlocks; i++ {
		offset := i * 210
		// scales are at 192 (16 bytes)
		for j := 0; j < 16; j++ {
			q6kData[offset+192+j] = byte(rand.Intn(10) + 1)
		}
		// d is at 208 (2 bytes f16 f16)
		d := Float32ToFloat16(0.01)
		binary.LittleEndian.PutUint16(q6kData[offset+208:], d)
	}

	// CPU Reference
	weightsF32 := gguf.DequantizeQ6K(q6kData, rows*cols)
	inputF32 := make([]float32, cols)
	for i := range inputF32 {
		inputF32[i] = rand.Float32() - 0.5
	}

	var expected float32
	for i := 0; i < cols; i++ {
		expected += weightsF32[i] * inputF32[i]
	}

	// GPU Result
	wTen, err := ctx.NewQ6KTensor(rows, cols)
	if err != nil {
		t.Fatalf("Failed to create Q6K tensor: %v", err)
	}
	if err := wTen.LoadFromRaw(q6kData); err != nil {
		t.Fatalf("Failed to load Q6K data: %v", err)
	}
	inTen := ctx.NewTensor(1, cols)
	inTen.LoadFrom(inputF32)

	outTen := wTen.MatMul(inTen)
	if err := ctx.WaitWithTimeout(2 * time.Second); err != nil {
		t.Fatal(err)
	}
	res := outTen.ToHost()

	wTen.Free()
	inTen.Free()
	outTen.Free()

	if len(res) != 1 {
		t.Fatalf("Expected 1 result element, got %d", len(res))
	}
	got := res[0]

	t.Logf("Q6K CPU Ref: %f", expected)
	t.Logf("Q6K GPU Res: %f", got)

	diff := math.Abs(float64(expected - got))
	if diff > 1e-1 {
		t.Errorf("Mismatch. Expected %f, Got %f, Diff %f", expected, got, diff)
	}
}
