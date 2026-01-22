//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestRoPE_Large_Context_Support(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 4
	kvHeads := 4
	headDim := 128

	testCases := []struct {
		name        string
		pos         int
		expectError bool
	}{
		{"Small context (512)", 511, false},
		{"Medium context (2048)", 2047, false},
		{"Near original limit (3999)", 3999, false},
		{"At original limit (4000)", 4000, false},
		{"Above original limit (8000)", 7999, false},
		{"Large context (16000)", 15999, false},
		{"Very large context (32000)", 31999, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.pos > 32000 {
				t.Skip("Skipping very large context test")
			}

			q := ctx.NewTensor(1, numHeads*headDim)
			kCache := ctx.NewTensor(tc.pos+1, kvHeads*headDim)
			vCache := ctx.NewTensor(tc.pos+1, kvHeads*headDim)
			out := ctx.NewTensor(1, numHeads*headDim)

			qData := make([]float32, numHeads*headDim)
			for i := range qData {
				qData[i] = float32(i%10) * 0.01
			}
			q.LoadFrom(qData)

			kvData := make([]float32, (tc.pos+1)*kvHeads*headDim)
			for i := range kvData {
				kvData[i] = float32(i%10) * 0.01
			}
			kCache.LoadFrom(kvData)
			vCache.LoadFrom(kvData)

			err := q.AttFused(kCache, vCache, out, tc.pos, numHeads, kvHeads, headDim, 0)
			ctx.Synchronize()

			if tc.expectError && err == nil {
				t.Errorf("Expected error at pos=%d but got success", tc.pos)
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error at pos=%d: %v", tc.pos, err)
			}

			if err == nil {
				outData := out.ToHost()
				for i, v := range outData {
					if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
						t.Errorf("Numerical instability at output[%d]: %f", i, v)
					}
				}
			}
		})
	}
}

func TestRoPE_Extended_Context_Precision(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 8
	kvHeads := 8
	headDim := 64
	pos := 8192

	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	vCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	out := ctx.NewTensor(1, numHeads*headDim)

	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = float32(i%100) * 0.001
	}
	q.LoadFrom(qData)

	kvData := make([]float32, (pos+1)*kvHeads*headDim)
	for i := range kvData {
		kvData[i] = float32(i%100) * 0.001
	}
	kCache.LoadFrom(kvData)
	vCache.LoadFrom(kvData)

	err := q.AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim, 0)
	ctx.Synchronize()

	if err != nil {
		t.Fatalf("AttFused failed at pos=%d: %v", pos, err)
	}

	outData := out.ToHost()
	hasNaN := false
	hasInf := false
	for _, v := range outData {
		if math.IsNaN(float64(v)) {
			hasNaN = true
		}
		if math.IsInf(float64(v), 0) {
			hasInf = true
		}
	}

	if hasNaN {
		t.Error("Output contains NaN values")
	}
	if hasInf {
		t.Error("Output contains Inf values")
	}

	_ = outData
}

func TestRoPE_Windowed_Attention_Large_Context(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 4
	kvHeads := 4
	headDim := 64
	windowSize := 4096
	pos := 8192

	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(windowSize, kvHeads*headDim)
	vCache := ctx.NewTensor(windowSize, kvHeads*headDim)
	out := ctx.NewTensor(1, numHeads*headDim)

	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = 0.1
	}
	q.LoadFrom(qData)

	kvData := make([]float32, windowSize*kvHeads*headDim)
	for i := range kvData {
		kvData[i] = 0.1
	}
	kCache.LoadFrom(kvData)
	vCache.LoadFrom(kvData)

	err := q.AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim, windowSize)
	ctx.Synchronize()

	if err != nil {
		t.Fatalf("AttFused with window failed at pos=%d: %v", pos, err)
	}

	outData := out.ToHost()
	for _, v := range outData {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Error("Output contains numerical instability")
			break
		}
	}
}

func TestRoPE_GQA_Large_Context(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 32
	kvHeads := 8
	headDim := 64
	pos := 4096

	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	vCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	out := ctx.NewTensor(1, numHeads*headDim)

	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = float32(i%8) * 0.01
	}
	q.LoadFrom(qData)

	kvData := make([]float32, (pos+1)*kvHeads*headDim)
	for i := range kvData {
		kvData[i] = float32(i%8) * 0.01
	}
	kCache.LoadFrom(kvData)
	vCache.LoadFrom(kvData)

	err := q.AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim, 0)
	ctx.Synchronize()

	if err != nil {
		t.Fatalf("AttFused with GQA failed at pos=%d: %v", pos, err)
	}

	outData := out.ToHost()
	for i, v := range outData {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Errorf("Numerical instability at output[%d]: %f", i, v)
		}
	}
}

func TestRoPE_Large_HeadDim_Context(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 4
	kvHeads := 4
	headDim := 256
	pos := 2048

	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	vCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	out := ctx.NewTensor(1, numHeads*headDim)

	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = float32(i%16) * 0.005
	}
	q.LoadFrom(qData)

	kvData := make([]float32, (pos+1)*kvHeads*headDim)
	for i := range kvData {
		kvData[i] = float32(i%16) * 0.005
	}
	kCache.LoadFrom(kvData)
	vCache.LoadFrom(kvData)

	err := q.AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim, 0)
	ctx.Synchronize()

	if err != nil {
		t.Fatalf("AttFused with large headDim failed at pos=%d: %v", pos, err)
	}

	outData := out.ToHost()
	for i, v := range outData {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Errorf("Numerical instability at output[%d]: %f", i, v)
		}
	}
}
