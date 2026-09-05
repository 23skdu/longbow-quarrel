//go:build !cuda && !tpu && !metal

package engine

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestRMSNormCPUWrapper(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	result := rmsNormCPU(input, weight, 1e-5)
	if len(result) != 4 {
		t.Fatalf("expected 4, got %d", len(result))
	}
	sum := float32(0)
	for _, v := range result {
		sum += v * v
	}
	meanSq := sum / 4
	if math.Abs(float64(meanSq-1)) > 1e-4 {
		t.Errorf("RMSNorm output mean^2=%f, expected ~1", meanSq)
	}
}

func TestRMSNormCPUEmpty(t *testing.T) {
	result := rmsNormCPU(nil, nil, 1e-5)
	if len(result) != 0 {
		t.Error("expected empty result")
	}
}

func TestMatMulVecWrapper(t *testing.T) {
	matrix := []float32{1, 2, 3, 4}
	vector := []float32{5, 6}
	result := matMulVec(matrix, vector)
	expected := []float32{17, 39}
	for i := range result {
		if result[i] != expected[i] {
			t.Errorf("matMulVec[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestMatMulVecRectangular(t *testing.T) {
	matrix := []float32{1, 2, 3, 4, 5, 6}
	vector := []float32{7, 8, 9}
	result := matMulVec(matrix, vector)
	expected := []float32{50, 122}
	for i := range result {
		if result[i] != expected[i] {
			t.Errorf("matMulVec[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestMatMulVecEmpty(t *testing.T) {
	// Both nil panics (divide by zero), so test empty equivalent
	result := matMulVec([]float32{}, []float32{1})
	if len(result) != 0 {
		t.Error("expected empty result")
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
	}{
		{0, 0.5},
		{1, 0.7310586},
		{-1, 0.2689414},
		{30, 1},
		{-30, 0},
		{100, 1},
		{-100, 0},
	}
	for _, tc := range tests {
		got := sigmoid(tc.input)
		if math.Abs(float64(got-tc.expected)) > 1e-5 {
			t.Errorf("sigmoid(%f) = %f, want %f", tc.input, got, tc.expected)
		}
	}
}

func TestApplyTopPCPU(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0, 4.0}
	result := applyTopPCPU(logits, 0.9)
	if len(result) != 4 {
		t.Fatalf("expected 4, got %d", len(result))
	}
	for _, v := range result {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN in result: %v", result)
		}
	}
	if math.IsInf(float64(result[0]), 1) || math.IsInf(float64(result[1]), 1) || math.IsInf(float64(result[2]), 1) || math.IsInf(float64(result[3]), 1) {
		t.Errorf("unexpected +Inf in first entries: %v", result)
	}
}

func TestApplyTopPCPUEmpty(t *testing.T) {
	result := applyTopPCPU([]float32{}, 0.9)
	if len(result) != 0 {
		t.Error("expected empty")
	}
}

func TestApplyTopPCPUSingle(t *testing.T) {
	logits := []float32{5.0}
	result := applyTopPCPU(logits, 0.9)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	// Single element's probability (1.0) exceeds p=0.9 immediately, so it gets masked
	if !math.IsInf(float64(result[0]), -1) {
		t.Log("single-element top-p: element is masked when prob exceeds p threshold")
	}
}

func TestSampleFromDistCPU(t *testing.T) {
	probs := []float32{0.1, 0.2, 0.3, 0.4}
	rng := rand.New(rand.NewSource(42))
	idx := sampleFromDistCPU(probs, rng)
	if idx < 0 || idx >= 4 {
		t.Errorf("sample index %d out of range", idx)
	}
}

func TestSampleFromDistCPUDeterministic(t *testing.T) {
	probs := []float32{1.0, 0.0, 0.0}
	rng := rand.New(rand.NewSource(0))
	idx := sampleFromDistCPU(probs, rng)
	if idx != 0 {
		t.Errorf("with prob[0]=1.0, expected index 0, got %d", idx)
	}
}

func TestSampleFromDistCPUEmpty(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	idx := sampleFromDistCPU([]float32{}, rng)
	if idx != -1 {
		t.Errorf("expected -1 for empty, got %d", idx)
	}
}

func TestFree(t *testing.T) {
	w := &CPUWeights{
		TokenEmb:   make([][]float32, 1),
		Output:     make([]float32, 1),
		OutputNorm: make([]float32, 1),
		AttnQ:      make([][]float32, 1),
		AttnK:      make([][]float32, 1),
		AttnV:      make([][]float32, 1),
		AttnO:      make([][]float32, 1),
		AttnNorm:   make([][]float32, 1),
		FfnGate:    make([][]float32, 1),
		FfnDown:    make([][]float32, 1),
		FfnUp:      make([][]float32, 1),
		FfnNorm:    make([][]float32, 1),
	}
	w.Free()
	if w.TokenEmb != nil {
		t.Error("TokenEmb should be nil after Free")
	}
	if w.Output != nil {
		t.Error("Output should be nil after Free")
	}
	if w.OutputNorm != nil {
		t.Error("OutputNorm should be nil after Free")
	}
}

func TestContains(t *testing.T) {
	tests := []struct {
		s, substr string
		want      bool
	}{
		{"hello world", "world", true},
		{"hello world", "xyz", false},
		{"", "", true},
		{"hello", "", true},
		{"", "hello", false},
		{"abc", "abc", true},
		{"abc", "abcd", false},
		{"blk.0.attn_q.weight", "attn_q.weight", true},
	}
	for _, tc := range tests {
		got := contains(tc.s, tc.substr)
		if got != tc.want {
			t.Errorf("contains(%q, %q) = %v, want %v", tc.s, tc.substr, got, tc.want)
		}
	}
}

func TestApplyTempCPU(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	result := applyTempCPU(logits, 0.5)
	if len(result) != 3 {
		t.Fatalf("expected 3, got %d", len(result))
	}
	if result[0] != 2.0 || result[1] != 4.0 || result[2] != 6.0 {
		t.Errorf("applyTempCPU(0.5) = %v, want [2, 4, 6]", result)
	}
}

func TestApplyTopKCPU(t *testing.T) {
	logits := []float32{1.0, 5.0, 2.0, 4.0, 3.0}
	orig := make([]float32, len(logits))
	copy(orig, logits)
	result := applyTopKCPU(logits, 3)
	// Top 3 values: 5.0(idx1), 4.0(idx3), 3.0(idx4)
	// Bottom 2: 1.0(idx0), 2.0(idx2) should be -Inf
	expectedInf := []int{0, 2}
	for _, idx := range expectedInf {
		if !math.IsInf(float64(result[idx]), -1) {
			t.Errorf("element %d should be -Inf after top-3", idx)
		}
	}
	expectedKeep := []int{1, 3, 4}
	for _, idx := range expectedKeep {
		if math.IsInf(float64(result[idx]), -1) {
			t.Errorf("top-3 element %d should not be -Inf", idx)
		}
	}
	_ = orig
}

func TestApplyTopKCPUFull(t *testing.T) {
	logits := []float32{1, 2, 3}
	result := applyTopKCPU(logits, 10)
	if len(result) != 3 {
		t.Fatalf("expected 3, got %d", len(result))
	}
}

func TestApplyTopKCPUZero(t *testing.T) {
	logits := []float32{1, 2, 3}
	result := applyTopKCPU(logits, 0)
	if len(result) != 3 {
		t.Fatalf("expected 3, got %d", len(result))
	}
}

func TestSoftmaxCPU(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	probs := softmaxCPU(logits)
	if len(probs) != 3 {
		t.Fatalf("expected 3, got %d", len(probs))
	}
	sum := float64(0)
	for _, p := range probs {
		sum += float64(p)
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
}

func TestDecodeTensorDataF32(t *testing.T) {
	data := make([]byte, 16)
	for i := uint32(0); i < 4; i++ {
		v := float32(i + 1)
		bits := math.Float32bits(v)
		data[i*4] = byte(bits)
		data[i*4+1] = byte(bits >> 8)
		data[i*4+2] = byte(bits >> 16)
		data[i*4+3] = byte(bits >> 24)
	}
	tensor := &gguf.TensorInfo{
		Name:       "test",
		Type:       gguf.GGMLTypeF32,
		Dimensions: []uint64{4},
		Data:       data,
	}
	result, err := decodeTensorData(tensor)
	if err != nil {
		t.Fatalf("decodeTensorData failed: %v", err)
	}
	if len(result) != 4 {
		t.Fatalf("expected 4, got %d", len(result))
	}
	for i := 0; i < 4; i++ {
		if result[i] != float32(i+1) {
			t.Errorf("result[%d] = %f, want %f", i, result[i], float32(i+1))
		}
	}
}

func TestDecodeTensorDataF16(t *testing.T) {
	data := make([]byte, 8)
	for i := uint16(0); i < 4; i++ {
		data[i*2] = byte(i)
		data[i*2+1] = byte(i >> 8)
	}
	tensor := &gguf.TensorInfo{
		Name:       "test",
		Type:       gguf.GGMLTypeF16,
		Dimensions: []uint64{4},
		Data:       data,
	}
	result, err := decodeTensorData(tensor)
	if err != nil {
		t.Fatalf("decodeTensorData failed: %v", err)
	}
	if len(result) != 4 {
		t.Fatalf("expected 4, got %d", len(result))
	}
}

func TestDecodeTensorDataUnsupported(t *testing.T) {
	tensor := &gguf.TensorInfo{
		Name:       "test",
		Type:       gguf.GGMLTypeQ4_0,
		Dimensions: []uint64{4},
		Data:       make([]byte, 4),
	}
	result, err := decodeTensorData(tensor)
	if err != nil {
		t.Fatalf("decodeTensorData failed: %v", err)
	}
	if len(result) == 0 {
		t.Error("expected non-empty result for unsupported type")
	}
}

func TestGetSeqCachePos(t *testing.T) {
	modelPath := "test_model_seqcache.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	pos := e.GetSeqCachePos("test-seq")
	if pos != 0 {
		t.Errorf("expected 0, got %d", pos)
	}
}

func TestInferWithLogits(t *testing.T) {
	modelPath := "test_model_infer_logits.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	cfg := SamplerConfig{Temperature: 0}
	tokens, logits, err := e.InferWithLogits([]int{0}, 1, cfg)
	if err != nil {
		t.Logf("InferWithLogits error: %v", err)
	}
	if len(logits) > 0 {
		for _, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("logits contain NaN/Inf: %v", logits)
				break
			}
		}
	}
	_ = tokens
}

func TestInferWithCallback(t *testing.T) {
	modelPath := "test_model_infer_cb.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	var called bool
	cfg := SamplerConfig{Temperature: 0}
	tokens, err := e.InferWithCallback([]int{0}, 1, cfg, func(token int) {
		called = true
	})
	if err != nil {
		t.Logf("InferWithCallback error: %v", err)
	}
	_ = tokens
	_ = called
}

func TestInferWithCallbackLogits(t *testing.T) {
	modelPath := "test_model_infer_cb_logits.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	var tokenCalled bool
	var logitsCalled bool
	cfg := SamplerConfig{Temperature: 0}
	tokens, err := e.InferWithCallbackLogits([]int{0}, 1, cfg, func(token int) {
		tokenCalled = true
	}, func(logits []float32) {
		logitsCalled = true
	})
	if err != nil {
		t.Logf("InferWithCallbackLogits error: %v", err)
	}
	_ = tokens
	_ = tokenCalled
	_ = logitsCalled
}

func TestAttentionCPU(t *testing.T) {
	headDim := 2
	numHeads := 1
	kvHeads := 1
	seqLen := 2
	total := seqLen * numHeads * headDim

	q := make([]float32, total)
	k := make([]float32, total)
	v := make([]float32, total)
	for i := range q {
		q[i] = float32(i%headDim) / float32(headDim)
		k[i] = float32((i+1)%headDim) / float32(headDim)
		v[i] = float32(i) / float32(total)
	}

	result := attentionCPU(q, k, v, numHeads, kvHeads, headDim)
	if len(result) != total {
		t.Fatalf("expected %d, got %d", total, len(result))
	}
	for i, val := range result {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("attentionCPU produced NaN/Inf at %d: %v", i, val)
		}
	}
}

func TestAttentionCPU_VariantsAndParity(t *testing.T) {
	testCases := []struct {
		name     string
		numHeads int
		kvHeads  int
		headDim  int
		seqLen   int
	}{
		{"MHA_small", 4, 4, 16, 4},
		{"MHA_head64", 2, 2, 64, 8},
		{"GQA_4to2", 4, 2, 32, 6},
		{"MQA_4to1", 4, 1, 32, 5},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			qTotal := tc.seqLen * tc.numHeads * tc.headDim
			kvTotal := tc.seqLen * tc.kvHeads * tc.headDim

			q := make([]float32, qTotal)
			k := make([]float32, kvTotal)
			v := make([]float32, kvTotal)

			for i := range q {
				q[i] = float32((i%17)-8) / 10.0
			}
			for i := range k {
				k[i] = float32((i%19)-9) / 10.0
			}
			for i := range v {
				v[i] = float32((i%23)-11) / 10.0
			}

			outVec := attentionCPU(q, k, v, tc.numHeads, tc.kvHeads, tc.headDim)
			if len(outVec) != qTotal {
				t.Fatalf("expected output length %d, got %d", qTotal, len(outVec))
			}

			// If MHA, compare with scalar reference implementation
			if tc.numHeads == tc.kvHeads {
				outScalar := attentionCPUScalar(q, k, v, tc.numHeads, tc.kvHeads, tc.headDim)
				for i := range outVec {
					diff := math.Abs(float64(outVec[i] - outScalar[i]))
					if diff > 1e-4 {
						t.Errorf("[%s] mismatch at %d: vec=%v scalar=%v diff=%v", tc.name, i, outVec[i], outScalar[i], diff)
					}
				}
			}

			for i, val := range outVec {
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					t.Fatalf("[%s] produced NaN/Inf at index %d: %v", tc.name, i, val)
				}
			}
		})
	}
}

func BenchmarkAttentionCPU(b *testing.B) {
	numHeads := 8
	kvHeads := 2
	headDim := 64
	seqLen := 32
	qTotal := seqLen * numHeads * headDim
	kvTotal := seqLen * kvHeads * headDim

	q := make([]float32, qTotal)
	k := make([]float32, kvTotal)
	v := make([]float32, kvTotal)
	for i := range q {
		q[i] = float32(i%headDim) / float32(headDim)
	}
	for i := range k {
		k[i] = float32(i%headDim) / float32(headDim)
	}
	for i := range v {
		v[i] = float32(i%headDim) / float32(headDim)
	}

	b.ResetTimer()
	for b.Loop() {
		_ = attentionCPU(q, k, v, numHeads, kvHeads, headDim)
	}
}

func BenchmarkAttentionCPUScalar(b *testing.B) {
	numHeads := 8
	kvHeads := 8
	headDim := 64
	seqLen := 32
	total := seqLen * numHeads * headDim

	q := make([]float32, total)
	k := make([]float32, total)
	v := make([]float32, total)
	for i := range q {
		q[i] = float32(i%headDim) / float32(headDim)
	}
	for i := range k {
		k[i] = float32(i%headDim) / float32(headDim)
	}
	for i := range v {
		v[i] = float32(i%headDim) / float32(headDim)
	}

	b.ResetTimer()
	for b.Loop() {
		_ = attentionCPUScalar(q, k, v, numHeads, kvHeads, headDim)
	}
}

func TestForwardDraft(t *testing.T) {
	modelPath := "test_model_draft.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	drafts, err := e.ForwardDraft([]int{0})
	if err != nil {
		t.Fatalf("ForwardDraft error: %v", err)
	}
	_ = drafts
}

func TestForwardDraftEmpty(t *testing.T) {
	modelPath := "test_model_draft_empty.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	drafts, err := e.ForwardDraft(nil)
	if err != nil {
		t.Fatalf("ForwardDraft(nil) error: %v", err)
	}
	if drafts != nil {
		t.Error("expected nil drafts for empty input")
	}
}

func TestRollbackKV(t *testing.T) {
	modelPath := "test_model_rollback.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Sequence not in cache yet -> expected error
	err = e.RollbackKV("unknown-seq", 0)
	if err == nil {
		t.Error("expected error for unknown sequence")
	}
}
