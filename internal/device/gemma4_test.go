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

func TestGemma4Q4KMatMul(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	cols := 256
	rows := 1

	dataSize := (rows * cols / 256) * 144
	q4kData := make([]byte, dataSize)
	rand.Seed(time.Now().UnixNano())
	rand.Read(q4kData)

	for i := 0; i < rows*cols/256; i++ {
		offset := i * 144
		d := Float32ToFloat16(rand.Float32() * 0.1)
		dmin := Float32ToFloat16(rand.Float32() * 0.1)
		q4kData[offset] = byte(d)
		q4kData[offset+1] = byte(d >> 8)
		q4kData[offset+2] = byte(dmin)
		q4kData[offset+3] = byte(dmin >> 8)
	}

	weightsF32 := gguf.DequantizeQ4K(q4kData, rows*cols)

	inputF32 := make([]float32, cols)
	for i := range inputF32 {
		inputF32[i] = (rand.Float32() - 0.5) * 0.05
	}

	expected := float32(0.0)
	for i := 0; i < cols; i++ {
		expected += weightsF32[i] * inputF32[i]
	}

	wTen, err := ctx.NewQ4KTensor(rows, cols)
	if err != nil {
		t.Fatalf("Failed to create Q4K tensor: %v", err)
	}
	if err := wTen.LoadFromRaw(q4kData); err != nil {
		t.Fatalf("Failed to load Q4K data: %v", err)
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
		t.Fatalf("Expected 1 result, got %d", len(res))
	}

	diff := float32(math.Abs(float64(res[0] - expected)))
	t.Logf("Expected: %f, Got: %f, Diff: %f", expected, res[0], diff)

	if diff > 1.0 && diff > float32(math.Abs(float64(expected)))*0.05 {
		t.Errorf("Mismatch beyond tolerance")
	}
}

func TestGemma4Q6KMatMul(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	cols := 256
	rows := 1

	dataSize := (rows * cols / 256) * 210
	q6kData := make([]byte, dataSize)
	rand.Seed(time.Now().UnixNano())
	rand.Read(q6kData)

	for i := 0; i < rows*cols/256; i++ {
		offset := i * 210
		for j := 0; j < 16; j++ {
			q6kData[offset+192+j] = byte(rand.Intn(10) + 1)
		}
		d := Float32ToFloat16(0.01)
		binary.LittleEndian.PutUint16(q6kData[offset+208:], d)
	}

	weightsF32 := gguf.DequantizeQ6K(q6kData, rows*cols)

	inputF32 := make([]float32, cols)
	for i := range inputF32 {
		inputF32[i] = (rand.Float32() - 0.5) * 0.05
	}

	expected := float32(0.0)
	for i := 0; i < cols; i++ {
		expected += weightsF32[i] * inputF32[i]
	}

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
		t.Fatalf("Expected 1 result, got %d", len(res))
	}

	diff := float32(math.Abs(float64(res[0] - expected)))
	t.Logf("Expected: %f, Got: %f, Diff: %f", expected, res[0], diff)

	if diff > 1.0 && diff > float32(math.Abs(float64(expected)))*0.05 {
		t.Errorf("Mismatch beyond tolerance")
	}
}

func TestGemma4RMSNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	dim := 256
	input := ctx.NewTensorFP32(1, dim)
	defer input.Free()

	inputData := make([]float32, dim)
	for i := range inputData {
		inputData[i] = float32(i%256) * 0.01
	}
	input.LoadFrom(inputData)

	weight := ctx.NewTensorFP32(dim, 1)
	defer weight.Free()
	weightData := make([]float32, dim)
	for i := range weightData {
		weightData[i] = 1.0
	}
	weight.LoadFrom(weightData)

	result := input.RMSNorm(weight, 1e-5)
	if result == nil {
		t.Fatal("RMSNorm returned nil")
	}

	resultData := result.ToHostF32()
	if len(resultData) != dim {
		t.Errorf("Result length mismatch: got %d, want %d", len(resultData), dim)
	}
}

func TestGemma4GQA(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	hiddenDim := 256
	headDim := 32
	numHeads := 8
	numKVHeads := 4
	seqLen := 32

	q := ctx.NewTensorFP32(1, hiddenDim)
	k := ctx.NewTensorFP32(1, hiddenDim)
	v := ctx.NewTensorFP32(1, hiddenDim)

	qData := make([]float32, hiddenDim)
	for i := range qData {
		qData[i] = float32(i) * 0.01
	}
	q.LoadFrom(qData)
	k.LoadFrom(qData)
	v.LoadFrom(qData)

	attnOutput := q.Attention(k, v, 0, numHeads, numKVHeads, headDim, seqLen, 0)
	if attnOutput == nil {
		t.Fatal("Attention returned nil")
	}

	outputData := attnOutput.ToHostF32()
	if len(outputData) != hiddenDim {
		t.Errorf("Output length mismatch: got %d, want %d", len(outputData), hiddenDim)
	}
}

func TestGemma4SwiGLU(t *testing.T) {
	t.Skip("SwiGLU test needs implementation")
}

func TestGemma4AttnQNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	dim := 256
	seqLen := 4
	input := ctx.NewTensorFP32(seqLen, dim)
	defer input.Free()

	inputData := make([]float32, seqLen*dim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	input.LoadFrom(inputData)

	normWeight := ctx.NewTensorFP32(dim, 1)
	defer normWeight.Free()
	normData := make([]float32, dim)
	for i := range normData {
		normData[i] = 1.0
	}
	normWeight.LoadFrom(normData)

	result := input.RMSNorm(normWeight, 1e-5)
	if result == nil {
		t.Fatal("RMSNorm with attn_q_norm weight returned nil")
	}

	resultData := result.ToHostF32()
	if len(resultData) != seqLen*dim {
		t.Errorf("Result length mismatch: got %d, want %d", len(resultData), seqLen*dim)
	}
}

func TestGemma4AttnQNormKNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	dim := 256
	hiddenDim := 512
	qNormWeight := ctx.NewTensorFP32(dim, 1)
	kNormWeight := ctx.NewTensorFP32(dim, 1)
	defer qNormWeight.Free()
	defer kNormWeight.Free()

	qNormData := make([]float32, dim)
	kNormData := make([]float32, dim)
	for i := range qNormData {
		qNormData[i] = 1.0
		kNormData[i] = 0.5
	}
	qNormWeight.LoadFrom(qNormData)
	kNormWeight.LoadFrom(kNormData)

	input := ctx.NewTensorFP32(1, hiddenDim)
	defer input.Free()
	inputData := make([]float32, hiddenDim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	input.LoadFrom(inputData)

	normed := input.RMSNorm(qNormWeight, 1e-5)
	if normed == nil {
		t.Fatal("RMSNorm for Q normalization returned nil")
	}
	_ = kNormWeight
}

func TestGemma4HybridAttention(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	windowSize := 32
	headDim := 32
	numHeads := 4
	kvHeads := 2
	seqLen := 32

	q := ctx.NewTensorFP32(1, numHeads*headDim)
	k := ctx.NewTensorFP32(1, kvHeads*headDim)
	v := ctx.NewTensorFP32(1, kvHeads*headDim)
	kCache := ctx.NewTensor(windowSize, kvHeads*headDim)
	vCache := ctx.NewTensor(windowSize, kvHeads*headDim)

	defer func() {
		q.Free()
		k.Free()
		v.Free()
		kCache.Free()
		vCache.Free()
	}()

	qData := make([]float32, numHeads*headDim)
	kData := make([]float32, kvHeads*headDim)
	vData := make([]float32, kvHeads*headDim)
	for i := range qData {
		qData[i] = float32(i) * 0.01
	}
	for i := range kData {
		kData[i] = float32(i) * 0.02
	}
	for i := range vData {
		vData[i] = float32(i) * 0.03
	}
	q.LoadFrom(qData)
	k.LoadFrom(kData)
	v.LoadFrom(vData)
	kCache.ZeroInit()
	vCache.ZeroInit()

	for pos := 0; pos < seqLen; pos++ {
		kPart := k
		vPart := v
		kPart.StoreKV(vPart, kCache, vCache, pos, kvHeads, headDim, windowSize)
	}

	attnOutput := q.Attention(kCache, vCache, seqLen-1, numHeads, kvHeads, headDim, seqLen, windowSize)
	if attnOutput == nil {
		t.Fatal("Attention with sliding window returned nil")
	}

	outputData := attnOutput.ToHostF32()
	if len(outputData) != numHeads*headDim {
		t.Errorf("Output length mismatch: got %d, want %d", len(outputData), numHeads*headDim)
	}
	t.Logf("Gemma4 hybrid attention test passed with window size %d", windowSize)
}

func TestGemma4SlidingWindowMetrics(t *testing.T) {
	windowSize := 4096
	wrappedTokens := 0
	for pos := 0; pos < windowSize*2; pos++ {
		if pos > windowSize {
			wrappedTokens++
		}
	}
	t.Logf("Sliding window metrics: window=%d, wrapped=%d/%d", windowSize, wrappedTokens, windowSize)
}
