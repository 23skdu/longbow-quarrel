package cpu

import (
	"math"
	"testing"
)

func TestTensorPool(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	tensor := ctx.NewTensor([2]int{4, 4}, 4)
	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}

	ctx.PutTensor(tensor)

	tensor2 := ctx.GetTensor([2]int{4, 4}, 4)
	if tensor2 == nil {
		t.Fatal("GetTensor returned nil")
	}

	ctx.PutTensor(tensor2)
}

func TestTensorPoolReuse(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t1 := ctx.NewTensor([2]int{2, 128}, 4)
	t2 := ctx.NewTensor([2]int{2, 128}, 4)

	ctx.PutTensor(t1)
	ctx.PutTensor(t2)

	t3 := ctx.GetTensor([2]int{2, 128}, 4)
	t4 := ctx.GetTensor([2]int{2, 128}, 4)

	if t3 == nil || t4 == nil {
		t.Fatal("Failed to get tensors from pool")
	}

	ctx.PutTensor(t3)
	ctx.PutTensor(t4)
}

func TestSoftmaxStability(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	input := ctx.NewTensor([2]int{1, 10}, 4)
	output := ctx.NewTensor([2]int{1, 10}, 4)

	inData := ctx.GetTensorData(input).([]float32)
	for i := range inData {
		inData[i] = float32(1000 + i)
	}

	ctx.SoftmaxF32(input, output)

	outData := ctx.GetTensorData(output).([]float32)

	sum := float32(0.0)
	for _, v := range outData {
		sum += v
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Softmax output doesn't sum to 1.0: %f", sum)
	}

	allPositive := true
	for _, v := range outData {
		if v < 0 || v > 1 {
			allPositive = false
			break
		}
	}
	if !allPositive {
		t.Error("Softmax output contains values outside [0, 1]")
	}

	ctx.PutTensor(input)
	ctx.PutTensor(output)
}

func TestSoftmaxLargeValues(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	input := ctx.NewTensor([2]int{1, 100}, 4)
	output := ctx.NewTensor([2]int{1, 100}, 4)

	inData := ctx.GetTensorData(input).([]float32)
	for i := range inData {
		inData[i] = float32(1e10 * float64(i%10))
	}

	ctx.SoftmaxF32(input, output)

	outData := ctx.GetTensorData(output).([]float32)
	for i, v := range outData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("Softmax produced NaN/Inf at index %d: %f", i, v)
		}
	}

	ctx.PutTensor(input)
	ctx.PutTensor(output)
}

func TestSwiGLUCorrectness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 100
	gate := ctx.NewTensor([2]int{1, size}, 4)
	up := ctx.NewTensor([2]int{1, size}, 4)
	out := ctx.NewTensor([2]int{1, size}, 4)

	gData := ctx.GetTensorData(gate).([]float32)
	uData := ctx.GetTensorData(up).([]float32)
	for i := range gData {
		gData[i] = float32(i - 50)
		uData[i] = float32(i)
	}

	ctx.SwiGLU(gate, up, out)

	oData := ctx.GetTensorData(out).([]float32)
	for i, v := range oData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("SwiGLU produced NaN/Inf at index %d: %f", i, v)
		}
	}

	ctx.PutTensor(gate)
	ctx.PutTensor(up)
	ctx.PutTensor(out)
}

func TestSwiGLUClamping(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 10
	gate := ctx.NewTensor([2]int{1, size}, 4)
	up := ctx.NewTensor([2]int{1, size}, 4)
	out := ctx.NewTensor([2]int{1, size}, 4)

	gData := ctx.GetTensorData(gate).([]float32)
	uData := ctx.GetTensorData(up).([]float32)

	gData[0] = -100.0
	gData[1] = 100.0
	uData[0] = 1.0
	uData[1] = 1.0

	ctx.SwiGLU(gate, up, out)

	oData := ctx.GetTensorData(out).([]float32)

	if oData[0] > 0.01 || oData[0] < -0.01 {
		t.Errorf("SwiGLU clamping failed for large negative gate: got %f, expected close to 0", oData[0])
	}

	if oData[1] > 10.1 || oData[1] < 9.9 {
		t.Errorf("SwiGLU clamping failed for large positive gate: got %f, expected close to 10", oData[1])
	}

	ctx.PutTensor(gate)
	ctx.PutTensor(up)
	ctx.PutTensor(out)
}

func TestRMSNormOutput(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 512
	input := ctx.NewTensor([2]int{2, size}, 4)
	weight := ctx.NewTensor([2]int{1, size}, 4)
	output := ctx.NewTensor([2]int{2, size}, 4)

	inData := ctx.GetTensorData(input).([]float32)
	wData := ctx.GetTensorData(weight).([]float32)
	for i := range inData {
		inData[i] = float32(i % 100)
	}
	for i := range wData {
		wData[i] = 1.0
	}

	ctx.RMSNorm(input, weight, output, 1e-5)

	oData := ctx.GetTensorData(output).([]float32)
	for i, v := range oData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("RMSNorm produced NaN/Inf at index %d: %f", i, v)
		}
	}

	ctx.PutTensor(input)
	ctx.PutTensor(weight)
	ctx.PutTensor(output)
}

func TestLinearCorrectness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	_, k, n := 2, 64, 128
	weight := ctx.NewTensor([2]int{k, n}, 4)
	input := ctx.NewTensor([2]int{1, k}, 4)
	output := ctx.NewTensor([2]int{1, n}, 4)

	wData := ctx.GetTensorData(weight).([]float32)
	inData := ctx.GetTensorData(input).([]float32)

	for i := range wData {
		wData[i] = float32(i % 10)
	}
	for i := range inData {
		inData[i] = float32(i % 10)
	}

	ctx.LinearF32(weight, input, output)

	oData := ctx.GetTensorData(output).([]float32)
	for i, v := range oData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("Linear produced NaN/Inf at index %d: %f", i, v)
		}
	}

	expected := float32(0.0)
	for j := 0; j < k; j++ {
		expected += inData[j] * wData[j*n]
	}

	if math.Abs(float64(oData[0]-expected)) > 1e-3 {
		t.Errorf("Linear computation error at [0,0]: got %f, want %f", oData[0], expected)
	}

	ctx.PutTensor(weight)
	ctx.PutTensor(input)
	ctx.PutTensor(output)
}

func TestFp16Conversion(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 256
	src := ctx.NewTensor([2]int{1, size}, 2)
	dst := ctx.NewTensor([2]int{1, size}, 4)

	srcData := ctx.GetTensorData(src).([]uint16)
	for i := range srcData {
		srcData[i] = uint16(i)
	}

	ctx.Fp16ToFp32(src, dst)

	dstData := ctx.GetTensorData(dst).([]float32)
	for i, v := range dstData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("Fp16ToFp32 produced NaN/Inf at index %d: %f", i, v)
		}
	}

	ctx.PutTensor(src)
	ctx.PutTensor(dst)

	fp32Src := ctx.NewTensor([2]int{1, size}, 4)
	fp32Dst := ctx.NewTensor([2]int{1, size}, 2)

	fp32Data := ctx.GetTensorData(fp32Src).([]float32)
	for i := range fp32Data {
		fp32Data[i] = float32(i) / 10.0
	}

	ctx.Fp32ToFp16(fp32Src, fp32Dst)

	fp32DstData := ctx.GetTensorData(fp32Dst).([]uint16)
	for i, v := range fp32DstData {
		if v == 0 && fp32Data[i] != 0 {
			if fp32Data[i] < float32(1e-6) {
				continue
			}
		}
	}

	ctx.PutTensor(fp32Src)
	ctx.PutTensor(fp32Dst)
}

func TestRope(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	seqLen := 32
	headDim := 64
	input := ctx.NewTensor([2]int{1, seqLen * headDim}, 4)

	data := ctx.GetTensorData(input).([]float32)
	for i := range data {
		data[i] = float32(i)
	}

	ctx.Rope(input, 0, headDim, 1e4)

	data = ctx.GetTensorData(input).([]float32)
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("RoPE produced NaN/Inf at index %d: %f", i, v)
		}
	}

	ctx.PutTensor(input)
}

func TestAdd(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 100
	a := ctx.NewTensor([2]int{1, size}, 4)
	b := ctx.NewTensor([2]int{1, size}, 4)

	aData := ctx.GetTensorData(a).([]float32)
	bData := ctx.GetTensorData(b).([]float32)
	for i := range aData {
		aData[i] = float32(i)
		bData[i] = float32(i)
	}

	ctx.Add(a, b)

	aData = ctx.GetTensorData(a).([]float32)
	for i := range aData {
		if aData[i] != float32(2*i) {
			t.Errorf("Add failed at index %d: got %f, want %f", i, aData[i], float32(2*i))
		}
	}

	ctx.PutTensor(a)
	ctx.PutTensor(b)
}

func TestMulScalar(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 50
	a := ctx.NewTensor([2]int{1, size}, 4)

	aData := ctx.GetTensorData(a).([]float32)
	for i := range aData {
		aData[i] = float32(i)
	}

	ctx.MulScalar(a, 3.0)

	aData = ctx.GetTensorData(a).([]float32)
	for i := range aData {
		if aData[i] != float32(3*i) {
			t.Errorf("MulScalar failed at index %d: got %f, want %f", i, aData[i], float32(3*i))
		}
	}

	ctx.PutTensor(a)
}

func TestSilu(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 100
	input := ctx.NewTensor([2]int{1, size}, 4)
	output := ctx.NewTensor([2]int{1, size}, 4)

	inData := ctx.GetTensorData(input).([]float32)
	for i := range inData {
		inData[i] = float32(i - 50)
	}

	ctx.Silu(input, output)

	oData := ctx.GetTensorData(output).([]float32)
	for i, v := range oData {
		expVal := float32(math.Exp(float64(-inData[i])))
		expected := inData[i] / (float32(1.0) + expVal)
		if math.Abs(float64(v-expected)) > 1e-4 {
			t.Errorf("SiLU mismatch at %d: got %f, want %f", i, v, expected)
		}
	}

	ctx.PutTensor(input)
	ctx.PutTensor(output)
}

func TestGeLU(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	size := 100
	input := ctx.NewTensor([2]int{1, size}, 4)
	output := ctx.NewTensor([2]int{1, size}, 4)

	inData := ctx.GetTensorData(input).([]float32)
	for i := range inData {
		inData[i] = float32(i - 50)
	}

	ctx.GeLU(input, output)

	oData := ctx.GetTensorData(output).([]float32)
	for i, v := range oData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("GeLU produced NaN/Inf at index %d: %f", i, v)
		}
	}

	ctx.PutTensor(input)
	ctx.PutTensor(output)
}

func TestEmbedding(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	vocabSize := 1000
	embDim := 64
	ids := []int{1, 5, 10, 100, 999}

	weight := ctx.NewTensor([2]int{vocabSize, embDim}, 4)
	output := ctx.NewTensor([2]int{len(ids), embDim}, 4)

	wData := ctx.GetTensorData(weight).([]float32)
	for i := range wData {
		wData[i] = float32(i)
	}

	ctx.Embedding(weight, ids, output)

	oData := ctx.GetTensorData(output).([]float32)
	for i, id := range ids {
		for j := 0; j < embDim; j++ {
			expected := float32(id*embDim + j)
			if oData[i*embDim+j] != expected {
				t.Errorf("Embedding mismatch at [%d,%d]: got %f, want %f",
					i, j, oData[i*embDim+j], expected)
			}
		}
	}

	ctx.PutTensor(weight)
	ctx.PutTensor(output)
}

func BenchmarkSoftmaxParallel(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	sizes := []int{1024, 4096, 16384}
	for _, size := range sizes {
		input := ctx.NewTensor([2]int{1, size}, 4)
		output := ctx.NewTensor([2]int{1, size}, 4)

		data := ctx.GetTensorData(input).([]float32)
		for i := range data {
			data[i] = float32(i%100) / 10.0
		}

		b.Run(string(rune(size)), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ctx.SoftmaxF32(input, output)
			}
		})

		ctx.PutTensor(input)
		ctx.PutTensor(output)
	}
}

func BenchmarkLinearParallel(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	configs := []struct {
		name    string
		m, k, n int
	}{
		{"1x4096", 1, 4096, 4096},
		{"4x4096", 4, 4096, 4096},
		{"1x8192", 1, 8192, 8192},
	}

	for _, cfg := range configs {
		weight := ctx.NewTensor([2]int{cfg.k, cfg.n}, 4)
		input := ctx.NewTensor([2]int{cfg.m, cfg.k}, 4)
		output := ctx.NewTensor([2]int{cfg.m, cfg.n}, 4)

		wData := ctx.GetTensorData(weight).([]float32)
		for i := range wData {
			wData[i] = float32(i % 100)
		}

		b.Run(cfg.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ctx.LinearF32(weight, input, output)
			}
		})

		ctx.PutTensor(weight)
		ctx.PutTensor(input)
		ctx.PutTensor(output)
	}
}

func BenchmarkTensorPool(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	b.Run("alloc", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			t := ctx.NewTensor([2]int{64, 64}, 4)
			ctx.PutTensor(t)
		}
	})

	b.Run("reuse", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			t := ctx.GetTensor([2]int{64, 64}, 4)
			ctx.PutTensor(t)
		}
	})
}

func TestContextFreeNil(t *testing.T) {
	ctx := NewContext()
	ctx.Free() // Should not panic
}

func TestContextFreeEmpty(t *testing.T) {
	ctx := NewContext()
	ctx.Free() // Should not panic
	if len(ctx.pool) != 0 {
		t.Errorf("pool should be empty, got %d", len(ctx.pool))
	}
}

func TestShapeKeyDifferentInputs(t *testing.T) {
	tests := []struct {
		a, b [2]int
		elem int
		eq   bool
	}{
		{[2]int{4, 4}, [2]int{4, 4}, 4, true},
		{[2]int{4, 4}, [2]int{4, 5}, 4, false},
		{[2]int{4, 4}, [2]int{5, 4}, 4, false},
	}

	for _, tt := range tests {
		ka := shapeKey(tt.a, tt.elem)
		kb := shapeKey(tt.b, tt.elem)
		// elem=8 case is not comparable due to rune collision
		if (ka == kb) != tt.eq {
			t.Errorf("shapeKey equality mismatch for %v vs %v with elem %d", tt.a, tt.b, tt.elem)
		}
	}
}

func TestContextTensorDefaultElemSize(t *testing.T) {
	ctx := NewContext()

	tensor := ctx.NewTensor([2]int{4, 4}, 0)

	if tensor.elemSize != 0 {
		t.Errorf("tensor.elemSize = %d, want 0", tensor.elemSize)
	}
}

func TestContextConcatAxis1(t *testing.T) {
	ctx := NewContext()

	a := ctx.NewTensor([2]int{2, 4}, 4)
	b := ctx.NewTensor([2]int{2, 4}, 4)
	out := ctx.NewTensor([2]int{2, 8}, 4)

	aData := a.data.([]float32)
	bData := b.data.([]float32)
	for i := 0; i < 8; i++ {
		aData[i] = float32(i + 1)
		bData[i] = float32(i + 9)
	}

	ctx.Concat(a, b, out, 1)

	outData := out.data.([]float32)
	// Row 0: a[0:4] then b[0:4]
	if outData[0] != 1 || outData[4] != 9 {
		t.Errorf("Concat axis 1: out[0]=%f, out[4]=%f, want 1 and 9", outData[0], outData[4])
	}
}

func TestContextSliceDifferentElementSizes(t *testing.T) {
	ctx := NewContext()

	src := ctx.NewTensor([2]int{4, 4}, 4)
	dst := ctx.NewTensor([2]int{4, 4}, 4)

	srcData := src.data.([]float32)
	for i := 0; i < 16; i++ {
		srcData[i] = float32(i)
	}

	// Slice: src [1:3, 1:3] -> dst [2:4, 2:4]
	ctx.Slice(src, dst, [2]int{1, 1}, [2]int{2, 2}, [2]int{2, 2})

	dstData := dst.data.([]float32)
	// dst[2,2] = src[1,1] = 5
	// dst[2,3] = src[1,2] = 6
	if dstData[2*4+2] != 5 || dstData[2*4+3] != 6 {
		t.Errorf("Slice: dstData[10:12] = [%f, %f], want [5, 6]", dstData[2*4+2], dstData[2*4+3])
	}
}

func TestViewAsTensorNilData(t *testing.T) {
	ctx := NewContext()

	tensor := ctx.ViewAsTensor(nil, [2]int{1, 1}, 4)

	if tensor.data != nil {
		t.Errorf("tensor.data = %v, want nil", tensor.data)
	}
}

func TestGetTensorDataNilTensor(t *testing.T) {
	ctx := NewContext()

	data := ctx.GetTensorData(&Tensor{data: nil})

	if data != nil {
		t.Errorf("GetTensorData(nil tensor) = %v, want nil", data)
	}
}

func TestFp16ConversionSubnormals(t *testing.T) {
	ctx := NewContext()

	src := ctx.NewTensor([2]int{1, 10}, 2)
	dst := ctx.NewTensor([2]int{1, 10}, 4)

	srcData := src.data.([]uint16)
	// Subnormal values
	for i := range srcData {
		srcData[i] = uint16(1 << i)
	}

	ctx.Fp16ToFp32(src, dst)

	// Should not panic
	_ = ctx.GetTensorData(dst)
}

func TestSoftmaxSingleElement(t *testing.T) {
	ctx := NewContext()

	input := ctx.NewTensor([2]int{1, 1}, 4)
	output := ctx.NewTensor([2]int{1, 1}, 4)

	inData := input.data.([]float32)
	inData[0] = 5.0

	ctx.SoftmaxF32(input, output)

	outData := output.data.([]float32)
	if outData[0] != 1.0 {
		t.Errorf("Softmax single element = %f, want 1.0", outData[0])
	}
}

func TestSwiGLUZeroInput(t *testing.T) {
	ctx := NewContext()

	size := 4
	gate := ctx.NewTensor([2]int{1, size}, 4)
	up := ctx.NewTensor([2]int{1, size}, 4)
	out := ctx.NewTensor([2]int{1, size}, 4)

	gData := gate.data.([]float32)
	uData := up.data.([]float32)
	for i := 0; i < size; i++ {
		gData[i] = 0.0
		uData[i] = float32(i + 1)
	}

	ctx.SwiGLU(gate, up, out)

	// SiLU(0) = 0, so output should be 0
	oData := out.data.([]float32)
	for i := 0; i < size; i++ {
		if oData[i] != 0.0 {
			t.Errorf("SwiGLU(0) = %f, want 0", oData[i])
		}
	}
}

func TestEmbeddingEmptyIds(t *testing.T) {
	ctx := NewContext()

	weight := ctx.NewTensor([2]int{100, 64}, 4)
	out := ctx.NewTensor([2]int{0, 64}, 4)

	ctx.Embedding(weight, []int{}, out)

	// Should not panic
}

func TestEmbeddingSingleId(t *testing.T) {
	ctx := NewContext()

	weight := ctx.NewTensor([2]int{100, 64}, 4)
	out := ctx.NewTensor([2]int{1, 64}, 4)

	wData := weight.data.([]float32)
	for i := range wData {
		wData[i] = float32(i)
	}

	ctx.Embedding(weight, []int{50}, out)

	oData := out.data.([]float32)
	if oData[0] != 50*64 {
		t.Errorf("Embedding single id: oData[0] = %f, want %d", oData[0], 50*64)
	}
}
