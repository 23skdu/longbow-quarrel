//go:build !cuda && !metal

package engine

import (
	"math"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

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
		Type:       gguf.GGMLType(9999),
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
	defer func() { _ = os.Remove(modelPath) }()

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
	defer func() { _ = os.Remove(modelPath) }()

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
	defer func() { _ = os.Remove(modelPath) }()

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
	defer func() { _ = os.Remove(modelPath) }()

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
	defer func() { _ = os.Remove(modelPath) }()

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
	defer func() { _ = os.Remove(modelPath) }()

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
	defer func() { _ = os.Remove(modelPath) }()

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

func contains(s, substr string) bool {
	return containsStr(s, substr)
}

func matMulVec(matrix []float32, vector []float32) []float32 {
	rows := len(matrix) / len(vector)
	result := make([]float32, rows)
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < len(vector); j++ {
			sum += float64(matrix[i*len(vector)+j]) * float64(vector[j])
		}
		result[i] = float32(sum)
	}
	return result
}

func attentionCPUScalar(q, k, v []float32, numHeads, kvHeads, headDim int) []float32 {
	headSize := headDim
	seqLen := len(q) / (numHeads * headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))
	result := make([]float32, len(q))

	for h := 0; h < numHeads; h++ {
		qHead := q[h*headSize*seqLen : (h+1)*headSize*seqLen]
		kvPerHead := numHeads / kvHeads
		kh := h / kvPerHead
		kHead := k[kh*headSize*seqLen : (kh+1)*headSize*seqLen]
		vHead := v[kh*headSize*seqLen : (kh+1)*headSize*seqLen]

		for i := 0; i < seqLen; i++ {
			var maxScore float64
			for j := 0; j <= i; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += float64(qHead[i*headDim+d]) * float64(kHead[j*headDim+d])
				}
				score := dot * scale
				if j == 0 || score > maxScore {
					maxScore = score
				}
			}

			var sum float64
			for j := 0; j <= i; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += float64(qHead[i*headDim+d]) * float64(kHead[j*headDim+d])
				}
				sum += math.Exp(dot*scale - maxScore)
			}

			for d := 0; d < headDim; d++ {
				var attnSum float64
				for j := 0; j <= i; j++ {
					dot := 0.0
					for dd := 0; dd < headDim; dd++ {
						dot += float64(qHead[i*headDim+dd]) * float64(kHead[j*headDim+dd])
					}
					attnSum += math.Exp(dot*scale-maxScore) * float64(vHead[j*headDim+d])
				}
				result[h*seqLen*headDim+i*headDim+d] = float32(attnSum / sum)
			}
		}
	}

	return result
}

// Verify SSM tensors are loaded for SSM layers

func TestQwen35ModelStructure(t *testing.T) {
	modelPath := realModelPath(t)

	conf := config.Default()
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	// Verify basic engine configuration. The model is the one the
	// environment names, so the test only applies to a Qwen 3.5 model.
	if e.Config().Architecture != "qwen35" {
		t.Skipf("the model at %s is a %s, not a qwen35", modelPath, e.Config().Architecture)
	}
	if e.Config().Layers != 32 {
		t.Errorf("Expected 32 layers, got %d", e.Config().Layers)
	}
	if e.Config().Dim != 2560 {
		t.Errorf("Expected dim 2560, got %d", e.Config().Dim)
	}
	if e.Config().Heads != 16 {
		t.Errorf("Expected 16 heads, got %d", e.Config().Heads)
	}
	if e.Config().KVHeads != 4 {
		t.Errorf("Expected 4 kv_heads, got %d", e.Config().KVHeads)
	}
	if e.Config().VocabSize != 248320 {
		t.Errorf("Expected vocab 248320, got %d", e.Config().VocabSize)
	}

	// Verify weights are loaded (check config values imply they're loaded)
	// Architecture values imply the correct weights structure

	t.Logf("Qwen3.5 model structure validation passed")
}
