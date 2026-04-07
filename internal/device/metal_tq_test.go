//go:build metal

package device

import (
	"math"
	"testing"
)

func TestMetal_TurboQuant_Fused(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	qjlRows := 64
	numHeads := 8
	kvHeads := 2
	pos := 10
	ctxLen := 32

	// 1. Setup TurboQuant Global Matrices
	rotationData := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		rotationData[i*headDim+i] = 1.0 // Identity
	}
	rotation := ctx.NewTensorFP32(headDim, headDim)
	rotation.LoadFromF32(rotationData)
	ctx.TQRotation = rotation

	qjlData := make([]float32, qjlRows*headDim)
	for i := 0; i < qjlRows*headDim; i++ {
		qjlData[i] = 1.0 
	}
	qjl := ctx.NewTensorFP32(qjlRows, headDim)
	qjl.LoadFromF32(qjlData)
	ctx.TQQJL = qjl

	// 2. Create and fill Query (use F32 then convert to F16 to avoid overflow in LoadFromF32)
	qData := make([]float32, numHeads*headDim)
	for i := 0; i < numHeads*headDim; i++ {
		qData[i] = float32(math.Sin(float64(i) * 0.1))
	}
	qF32 := ctx.NewTensorFP32(1, numHeads*headDim)
	qF32.LoadFromF32(qData)
	q := qF32.CopyToF16()

	// 3. Create and fill TQ KV Cache
	kCache := ctx.NewTurboTensor(ctxLen, numHeads*headDim, DataTypeTQ1_0, headDim, qjlRows)
	vCache := ctx.NewTurboTensor(ctxLen, numHeads*headDim, DataTypeTQ1_0, headDim, qjlRows)
	
	kData := make([]byte, kCache.SizeBytes())
	// Fill KV cache with dynamic data
	for b := 0; b < ctxLen*numHeads; b++ {
		// [headDim int8][qjlRows int8][float scale][float sj]
		offset := b * (headDim + qjlRows + 8)
		if offset + headDim + qjlRows + 4 > len(kData) { break }
		for i := 0; i < headDim; i++ {
			kData[offset+i] = byte(int8((i % 127))) // Dynamic K
		}
		for i := 0; i < qjlRows; i++ {
			kData[offset+headDim+i] = byte(int8(((i + (b % numHeads)) % 127))) // Dynamic QJL
		}
		// Scale = 1.0
		kData[offset+headDim+qjlRows] = 0x00
		kData[offset+headDim+qjlRows+1] = 0x00
		kData[offset+headDim+qjlRows+2] = 0x80
		kData[offset+headDim+qjlRows+3] = 0x3F
	}
	
	ctx.LoadBuffer(kCache, kData)
	ctx.LoadBuffer(vCache, kData)

	// 4. Run Attention
	res := q.Attention(kCache, vCache, pos, numHeads, kvHeads, headDim, ctxLen, 0)
	if res == nil {
		t.Fatal("Attention returned nil")
	}
	defer res.Free()

	// 5. Basic verification: output should not be all zeros
	output := res.ToHostF32()
	nonZero := false
	for _, v := range output {
		if v != 0 {
			nonZero = true
			break
		}
	}
	
	if !nonZero {
		t.Error("Attention output is all zeros")
	}
	
	// Since we haven't filled the cache with meaningful data yet, it might be zero if initial memory is zero.
	// But the kernel should have executed.
	t.Logf("Success: Fused TurboQuant Attention executed. Sample output[0:4]: %v", output[0:4])
}
func BenchmarkMetal_TurboQuant_Attention(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	numHeads := 32
	kvHeads := 8
	pos := 1024
	ctxLen := 32768
	qjlRows := 64

	// Setup TurboQuant Global Matrices
	rotation := ctx.NewTensorFP32(headDim, headDim)
	ctx.TQRotation = rotation
	qjl := ctx.NewTensorFP32(qjlRows, headDim)
	ctx.TQQJL = qjl

	qData := make([]float32, numHeads*headDim)
	qF32 := ctx.NewTensorFP32(1, numHeads*headDim)
	qF32.LoadFromF32(qData)
	q := qF32.CopyToF16()

	kCache := ctx.NewTurboTensor(ctxLen, numHeads*headDim, DataTypeTQ1_0, headDim, qjlRows)
	vCache := ctx.NewTurboTensor(ctxLen, numHeads*headDim, DataTypeTQ1_0, headDim, qjlRows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := q.Attention(kCache, vCache, pos, numHeads, kvHeads, headDim, ctxLen, 0)
		res.Free()
	}
}
