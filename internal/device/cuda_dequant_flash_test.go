//go:build linux && cuda

package device

import (
	"math"
	"testing"
)

func TestCUDA_ZeroDequantGEMM_Q8_0(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// M=4 rows, K=32 cols
	M := 4
	K := 32
	// For Q8_0: 1 block of 32 weights takes 34 bytes per row.
	// Row bytes: 34 bytes.
	// Let's create raw Q8_0 data for 4 rows.
	rawData := make([]byte, M*34)
	for r := 0; r < M; r++ {
		off := r * 34
		// Scale = 1.0 in FP16 (0x3C00)
		rawData[off] = 0x00
		rawData[off+1] = 0x3C
		// 32 int8 weights: alternating 1 and -1
		for j := 0; j < 32; j++ {
			if (r+j)%2 == 0 {
				rawData[off+2+j] = 1
			} else {
				rawData[off+2+j] = 0xFF // -1 in two's complement
			}
		}
	}

	weightTensor := ctx.NewTensorWithType(M, K, DataTypeQ8_0)
	defer weightTensor.Free()
	_ = weightTensor.LoadFrom(rawData)

	x := ctx.NewTensorFP32(1, K)
	defer x.Free()
	xHost := make([]float32, K)
	for i := range xHost {
		xHost[i] = 1.0
	}
	_ = x.LoadFrom(xHost)

	y := ctx.NewTensorFP32(1, M)
	defer y.Free()

	ctx.MatVecDequantQ8_0(weightTensor, x, y, M, K)
	ctx.Synchronize()

	yHost := y.ToHostF32()
	if len(yHost) != M {
		t.Fatalf("expected %d elements, got %d", M, len(yHost))
	}
	for r := 0; r < M; r++ {
		// 16 ones and 16 minus ones -> sum = 0
		if math.Abs(float64(yHost[r])) > 1e-4 {
			t.Errorf("row %d: expected 0, got %f", r, yHost[r])
		}
	}
}

func TestCUDA_FlashAttentionPrefill_SlidingWindow(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	batch := 1
	heads := 2
	kvHeads := 2
	seqLen := 4
	headDim := 32
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	slidingWindow := 2

	q := ctx.NewTensorFP32(batch*heads*seqLen, headDim)
	defer q.Free()
	k := ctx.NewTensorFP32(batch*kvHeads*seqLen, headDim)
	defer k.Free()
	v := ctx.NewTensorFP32(batch*kvHeads*seqLen, headDim)
	defer v.Free()
	output := ctx.NewTensorFP32(batch*heads*seqLen, headDim)
	defer output.Free()

	qData := make([]float32, batch*heads*seqLen*headDim)
	kData := make([]float32, batch*kvHeads*seqLen*headDim)
	vData := make([]float32, batch*kvHeads*seqLen*headDim)
	for i := range qData {
		qData[i] = 0.1
		kData[i] = 0.1
		vData[i] = float32(i % 10)
	}
	_ = q.LoadFrom(qData)
	_ = k.LoadFrom(kData)
	_ = v.LoadFrom(vData)

	ctx.FlashAttentionPrefill(q, k, v, output, batch, heads, kvHeads, seqLen, seqLen, headDim, scale, slidingWindow)
	ctx.Synchronize()

	outHost := output.ToHostF32()
	if len(outHost) != len(qData) {
		t.Fatalf("expected %d output elements, got %d", len(qData), len(outHost))
	}
	for i, val := range outHost {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Fatalf("NaN/Inf in output at index %d", i)
		}
	}
}
