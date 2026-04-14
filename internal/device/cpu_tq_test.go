//go:build !cuda
package device

import (
	"fmt"
	"math"
	"testing"
)

func TestCPU_StoreKV_Float32(t *testing.T) {
	ctx := NewContext()
	heads := 8
	headDim := 128
	windowSize := 10

	kCache := ctx.NewTensorWithType(windowSize, heads*headDim, DataTypeF32)
	vCache := ctx.NewTensorWithType(windowSize, heads*headDim, DataTypeF32)

	k := ctx.NewTensor(1, heads*headDim)
	v := ctx.NewTensor(1, heads*headDim)

	// Fill with test data
	for i := 0; i < heads*headDim; i++ {
		k.Data()[i] = float32(i + 1)
		v.Data()[i] = float32(i + 100)
	}

	pos := 2
	k.StoreKV(v, kCache, vCache, pos, heads, headDim, windowSize)

	// Verify
	off := pos * heads * headDim
	for i := 0; i < heads*headDim; i++ {
		if kCache.Data()[off+i] != float32(i+1) {
			t.Errorf("kCache mismatch at %d: got %f, want %f", off+i, kCache.Data()[off+i], float32(i+1))
		}
		if vCache.Data()[off+i] != float32(i+100) {
			t.Errorf("vCache mismatch at %d: got %f, want %f", off+i, vCache.Data()[off+i], float32(i+100))
		}
	}
}

func TestCPU_TurboQuant_Roundtrip(t *testing.T) {
	ctx := NewContext()
	blockSize := 128
	qjlRows := 64
	numBlocks := 1
	bits := 2

	input := ctx.NewTensor(1, blockSize*numBlocks)
	rotation := ctx.NewTensor(blockSize, blockSize)
	qjl := ctx.NewTensor(qjlRows, blockSize)
	
	// Create identity rotation matrix and random QJL
	for i := 0; i < blockSize; i++ {
		rotation.Data()[i*blockSize+i] = 1.0
	}
	// Fill QJL with random-ish signs
	for i := 0; i < qjlRows*blockSize; i++ {
		if math.Sin(float64(i)) >= 0 {
			qjl.Data()[i] = 1.0
		} else {
			qjl.Data()[i] = -1.0
		}
	}
	ctx.TQRotation = rotation
	ctx.TQQJL = qjl

	// Fill input with some pattern
	for i := 0; i < blockSize; i++ {
		input.Data()[i] = float32(math.Sin(float64(i) * 0.1))
	}

	output := ctx.NewTensorWithType(1, blockSize*numBlocks, DataTypeTQ1_0)
	scaleOut := ctx.NewTensor(1, numBlocks)
	qjlScaleOut := ctx.NewTensor(1, numBlocks)

	ctx.TurboQuantEncode(input, rotation, qjl, output, scaleOut, qjlScaleOut, blockSize, qjlRows, bits)

	// Verify rawData is filled
	if len(output.RawData()) == 0 {
		t.Fatal("output.rawData is empty")
	}

	decoded := ctx.NewTensor(1, blockSize*numBlocks)
	ctx.TurboQuantDecode(output, rotation, qjl, decoded, scaleOut, blockSize, qjlRows)

	// In TQ1_0, the error should be small if QJL is reasonably effective, but since QJL is 1-bit,
	// it can have sign flips for very small residuals or specific projection angles.
	// We check for a slightly higher threshold and overall sign preservation.
	mismatches := 0
	for i := 0; i < blockSize; i++ {
		if (input.Data()[i] > 0.3 && decoded.Data()[i] < -0.1) || (input.Data()[i] < -0.3 && decoded.Data()[i] > 0.1) {
			mismatches++
		}
	}
	if mismatches > blockSize/8 {
		t.Errorf("Too many sign mismatches in TQ roundtrip: %d/%d", mismatches, blockSize)
	}
}

func TestCPU_StoreKV_TurboQuant(t *testing.T) {
	ctx := NewContext()
	heads := 4
	headDim := 128
	windowSize := 5
	qjlRows := 64

	// Setup rotation matrices in context
	rot := ctx.NewTensor(headDim, headDim)
	for i := 0; i < headDim; i++ {
		rot.Data()[i*headDim+i] = 1.0
	}
	qjl := ctx.NewTensor(qjlRows, headDim)
	ctx.TQRotation = rot
	ctx.TQQJL = qjl

	kCache := ctx.NewTurboTensor(windowSize, heads*headDim, DataTypeTQ2_0, headDim, qjlRows)
	vCache := ctx.NewTurboTensor(windowSize, heads*headDim, DataTypeTQ2_0, headDim, qjlRows)

	k := ctx.NewTensor(1, heads*headDim)
	v := ctx.NewTensor(1, heads*headDim)

	for i := 0; i < heads*headDim; i++ {
		k.Data()[i] = float32(math.Cos(float64(i)))
		v.Data()[i] = float32(math.Sin(float64(i)))
	}
	fmt.Printf("DEBUG Test: k.data[0:4] = %v\n", k.Data()[0:4])

	pos := 1
	k.StoreKV(v, kCache, vCache, pos, heads, headDim, windowSize)

	// Verify raw data was stored
	bytesPerBlock := headDim + qjlRows + 8
	off := (pos % windowSize) * heads * bytesPerBlock
	
	// Read scale using getFloat32 (need to make it accessible or copy logic)
	// For the test, we'll just check if something was written to the scale area
	scaleArea := kCache.RawData()[off+headDim+qjlRows : off+headDim+qjlRows+4]
	if scaleArea[0] == 0 && scaleArea[1] == 0 && scaleArea[2] == 0 && scaleArea[3] == 0 {
		t.Errorf("Scale is zero at offset %d", off+headDim+qjlRows)
	}

	fmt.Printf("TQ StoreKV Test: rawData sample [0:8] = %v\n", kCache.RawData()[off:off+8])

	// Test FetchKV
	kOut := ctx.NewTensor(1, heads*headDim)
	vOut := ctx.NewTensor(1, heads*headDim)
	kOut.FetchKV(vOut, kCache, vCache, pos, heads, headDim, windowSize)

	// Since TQ is lossy, we only check that we got some non-zero values back
	// that roughly match the signs of the input.
	for i := 0; i < heads*headDim; i++ {
		if (k.Data()[i] > 0.1 && kOut.Data()[i] <= 0) || (k.Data()[i] < -0.1 && kOut.Data()[i] >= 0) {
			t.Errorf("FetchKV: numerical mismatch at k[%d]: got %f, want sign of %f", i, kOut.Data()[i], k.Data()[i])
		}
	}
}
