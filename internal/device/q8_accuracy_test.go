//go:build darwin && metal

package device

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestQ8_0_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	cols := 256
	rows := 1

	// 1. Generate Q8_0 data
	numBlocks := (rows * cols) / 32
	dataSize := numBlocks * 34
	q8Data := make([]byte, dataSize)
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numBlocks; i++ {
		offset := i * 34
		// Scale (f16)
		d := Float32ToFloat16(rand.Float32() * 0.1)
		binary.LittleEndian.PutUint16(q8Data[offset:], d)
		// Quants (32 * int8)
		for j := 0; j < 32; j++ {
			q8Data[offset+2+j] = byte(rand.Intn(256) - 128)
		}
	}

	// 2. CPU Reference
	weightsF32 := gguf.DequantizeQ8_0(q8Data, rows*cols)
	inputF32 := make([]float32, cols)
	for i := range inputF32 {
		inputF32[i] = rand.Float32() - 0.5
	}

	var expected float32
	for i := 0; i < cols; i++ {
		expected += weightsF32[i] * inputF32[i]
	}

	// 3. GPU Execution
	// First use NewQ8_0Tensor
	wTen, err := ctx.NewQ8_0Tensor(rows, cols)
	if err != nil {
		t.Fatalf("Failed to create Q8_0 tensor: %v", err)
	}
	if err := wTen.LoadFromRaw(q8Data); err != nil {
		t.Fatalf("Failed to load Q8_0 data: %v", err)
	}

	inTen := ctx.NewTensor(1, cols)
	inTen.LoadFrom(inputF32)

	// LinearInto uses weight.dataType to dispatch
	outTen := ctx.NewTensor(1, 1)
	outTen.ZeroInit()
	// Weight should be second arg in LinearInto?
	// Wait, LinearInto signature: func (t *Tensor) LinearInto(weight *Tensor, out *Tensor, scale float32) error
	// So inTen.LinearInto(wTen, outTen, 1.0)
	if err := inTen.LinearInto(wTen, outTen, 1.0); err != nil {
		t.Fatalf("LinearInto failed: %v", err)
	}

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

	t.Logf("Q8_0 CPU Ref: %f", expected)
	t.Logf("Q8_0 GPU Res: %f", got)

	diff := math.Abs(float64(expected - got))
	// Q8_0 dequantization in metal uses float sum and float scale, should be quite accurate.
	if diff > 1e-1 {
		t.Errorf("Mismatch. Expected %f, Got %f, Diff %f", expected, got, diff)
	}
}
