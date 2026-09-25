//go:build (!cuda && !metal && !tpu) || !amd64 || !cgo || (!linux && !darwin)

package device

import (
	"math"
	"testing"
)

func TestCPU_MLADecompressKV(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 2
	kvLoraRank := 8
	heads := 2
	qkNopeDim := 4
	vHeadDim := 4

	// compressedKV: [2, 8]
	compKVData := make([]float32, numTokens*kvLoraRank)
	for i := range compKVData {
		compKVData[i] = float32(i+1) * 0.1
	}
	compressedKV := ctx.NewTensorFP32(numTokens, kvLoraRank)
	_ = compressedKV.LoadFrom(compKVData)

	// wUKV: [heads * (qkNopeDim + vHeadDim), kvLoraRank] = [2 * (4 + 4), 8] = [16, 8]
	totalRows := heads * (qkNopeDim + vHeadDim)
	wData := make([]float32, totalRows*kvLoraRank)
	for i := range wData {
		wData[i] = float32(i%5+1) * 0.2
	}
	wUKV := ctx.NewTensorFP32(totalRows, kvLoraRank)
	_ = wUKV.LoadFrom(wData)

	kNope, v := ctx.MLADecompressKV(compressedKV, wUKV, numTokens, kvLoraRank, heads, qkNopeDim, vHeadDim)
	if kNope == nil || v == nil {
		t.Fatalf("expected non-nil tensors, got kNope=%v, v=%v", kNope, v)
	}

	if kNope.Rows() != numTokens || kNope.Cols() != heads*qkNopeDim {
		t.Errorf("kNope shape mismatch: expected [%d, %d], got [%d, %d]", numTokens, heads*qkNopeDim, kNope.Rows(), kNope.Cols())
	}
	if v.Rows() != numTokens || v.Cols() != heads*vHeadDim {
		t.Errorf("v shape mismatch: expected [%d, %d], got [%d, %d]", numTokens, heads*vHeadDim, v.Rows(), v.Cols())
	}

	kHost := kNope.ToHost()
	vHost := v.ToHost()
	if len(kHost) != numTokens*heads*qkNopeDim || len(vHost) != numTokens*heads*vHeadDim {
		t.Fatalf("host data length mismatch")
	}

	// Verify first element of kNope: row 0 dot product with compKVData[0..7]
	var expectedK0 float32
	for j := 0; j < kvLoraRank; j++ {
		expectedK0 += compKVData[j] * wData[j]
	}
	if math.Abs(float64(kHost[0]-expectedK0)) > 1e-4 {
		t.Errorf("kNope[0] mismatch: expected %f, got %f", expectedK0, kHost[0])
	}
}

func TestCPU_MLAProjectQuerySplitRoPE(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 2
	heads := 2
	qkNopeDim := 4
	qkRopeDim := 4
	theta := float32(10000.0)

	inHeadDim := qkNopeDim + qkRopeDim
	qAllData := make([]float32, numTokens*heads*inHeadDim)
	for i := range qAllData {
		qAllData[i] = float32(i + 1)
	}
	qAll := ctx.NewTensorFP32(numTokens, heads*inHeadDim)
	_ = qAll.LoadFrom(qAllData)

	posIdsData := []float32{0, 1}
	posIds := ctx.NewTensorFP32(numTokens, 1)
	_ = posIds.LoadFrom(posIdsData)

	qNope, qRope := ctx.MLAProjectQuerySplitRoPE(qAll, posIds, numTokens, heads, qkNopeDim, qkRopeDim, theta)
	if qNope == nil || qRope == nil {
		t.Fatalf("expected non-nil tensors, got qNope=%v, qRope=%v", qNope, qRope)
	}

	if qNope.Cols() != heads*qkNopeDim {
		t.Errorf("qNope cols mismatch: expected %d, got %d", heads*qkNopeDim, qNope.Cols())
	}
	if qRope.Cols() != heads*qkRopeDim {
		t.Errorf("qRope cols mismatch: expected %d, got %d", heads*qkRopeDim, qRope.Cols())
	}

	nopeHost := qNope.ToHost()
	ropeHost := qRope.ToHost()

	// Token 0, Head 0: pos = 0 => cos(0) = 1, sin(0) = 0
	// qRope should match original rope input elements at pos 0
	if math.Abs(float64(nopeHost[0]-qAllData[0])) > 1e-4 {
		t.Errorf("qNope[0] mismatch: expected %f, got %f", qAllData[0], nopeHost[0])
	}
	if math.Abs(float64(ropeHost[0]-qAllData[qkNopeDim])) > 1e-4 {
		t.Errorf("qRope[0] at pos 0 mismatch: expected %f, got %f", qAllData[qkNopeDim], ropeHost[0])
	}
}

func TestCPU_MLAAbsorbedQuery(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 1
	heads := 2
	qkNopeDim := 4
	kvLoraRank := 4

	qNopeData := make([]float32, numTokens*heads*qkNopeDim)
	for i := range qNopeData {
		qNopeData[i] = 1.0
	}
	qNope := ctx.NewTensorFP32(numTokens, heads*qkNopeDim)
	_ = qNope.LoadFrom(qNopeData)

	wUKData := make([]float32, heads*qkNopeDim*kvLoraRank)
	for i := range wUKData {
		wUKData[i] = 0.5
	}
	wUK := ctx.NewTensorFP32(heads*qkNopeDim, kvLoraRank)
	_ = wUK.LoadFrom(wUKData)

	qAbs := ctx.MLAAbsorbedQuery(qNope, wUK, numTokens, heads, qkNopeDim, kvLoraRank)
	if qAbs == nil {
		t.Fatalf("expected non-nil qAbs")
	}

	absHost := qAbs.ToHost()
	// Each element is sum of qkNopeDim products of 1.0 * 0.5 = 4 * 0.5 = 2.0
	for i, val := range absHost {
		if math.Abs(float64(val-2.0)) > 1e-4 {
			t.Errorf("qAbs[%d] mismatch: expected 2.0, got %f", i, val)
		}
	}
}

func TestCPU_MLAAbsorbedDecodeAttention(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 1
	seqLen := 3
	heads := 2
	kvLoraRank := 4
	qkRopeDim := 2
	vHeadDim := 4
	scale := float32(1.0 / math.Sqrt(float64(qkRopeDim+kvLoraRank)))

	qAbsData := make([]float32, numTokens*heads*kvLoraRank)
	for i := range qAbsData {
		qAbsData[i] = 0.5
	}
	qAbs := ctx.NewTensorFP32(numTokens, heads*kvLoraRank)
	_ = qAbs.LoadFrom(qAbsData)

	qRopeData := make([]float32, numTokens*heads*qkRopeDim)
	for i := range qRopeData {
		qRopeData[i] = 0.5
	}
	qRope := ctx.NewTensorFP32(numTokens, heads*qkRopeDim)
	_ = qRope.LoadFrom(qRopeData)

	kCacheData := make([]float32, seqLen*kvLoraRank)
	for i := range kCacheData {
		kCacheData[i] = 0.2
	}
	kCache := ctx.NewTensorFP32(seqLen, kvLoraRank)
	_ = kCache.LoadFrom(kCacheData)

	kRopeData := make([]float32, seqLen*qkRopeDim)
	for i := range kRopeData {
		kRopeData[i] = 0.2
	}
	kRope := ctx.NewTensorFP32(seqLen, qkRopeDim)
	_ = kRope.LoadFrom(kRopeData)

	wUVData := make([]float32, heads*vHeadDim*kvLoraRank)
	for i := range wUVData {
		wUVData[i] = 0.1
	}
	wUV := ctx.NewTensorFP32(heads*vHeadDim, kvLoraRank)
	_ = wUV.LoadFrom(wUVData)

	output := ctx.MLAAbsorbedDecodeAttention(qAbs, qRope, kCache, kRope, wUV, numTokens, seqLen, heads, kvLoraRank, qkRopeDim, vHeadDim, scale)
	if output == nil {
		t.Fatalf("expected non-nil output")
	}

	outHost := output.ToHost()
	if len(outHost) != numTokens*heads*vHeadDim {
		t.Fatalf("output length mismatch: expected %d, got %d", numTokens*heads*vHeadDim, len(outHost))
	}
	for i, v := range outHost {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("output[%d] is NaN or Inf", i)
		}
	}
}
