package engine

import (
	"context"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

type mockEngineForSpec struct {
	tokens []int
	logits []float32
}

func (m *mockEngineForSpec) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	out := make([]int, count)
	for i := 0; i < count; i++ {
		out[i] = 100 + i
	}
	return out, nil
}

func (m *mockEngineForSpec) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	out := make([]int, count)
	for i := 0; i < count; i++ {
		out[i] = 100 + i
	}
	logits := make([]float32, 200)
	for i := range logits {
		logits[i] = 1.0
	}
	return out, logits, nil
}

func (m *mockEngineForSpec) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(int)) ([]int, error) {
	out, err := m.Infer(tokens, count, cfg)
	if err == nil && callback != nil {
		for _, t := range out {
			callback(t)
		}
	}
	return out, err
}

func (m *mockEngineForSpec) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tc func(int), lc func([]float32)) ([]int, error) {
	return m.InferWithCallback(tokens, count, cfg, tc)
}

func (m *mockEngineForSpec) Config() config.Config { return config.Config{} }
func (m *mockEngineForSpec) Close() {}
func (m *mockEngineForSpec) SwapModel(path string, cfg config.Config) error { return nil }
func (m *mockEngineForSpec) LoadAdapter(path, id string) error { return nil }
func (m *mockEngineForSpec) GetSeqCachePos(seqID string) int { return 0 }
func (m *mockEngineForSpec) RollbackKV(seqID string, newPos int) error { return nil }
func (m *mockEngineForSpec) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) { return nil, nil }

func (m *mockEngineForSpec) ForwardDraft(tokens []int) ([][]float32, error) {
	// Return uniform logits that will accept tokens
	res := make([][]float32, len(tokens))
	for i := range res {
		row := make([]float32, 200)
		for j := range row {
			row[j] = 2.0
		}
		res[i] = row
	}
	return res, nil
}

func TestAsymmetricSpeculativeEngine(t *testing.T) {
	draft := &mockEngineForSpec{}
	target := &mockEngineForSpec{}

	asym := NewAsymmetricSpeculativeEngine(target, draft)
	defer asym.Close()

	initialTokens := []int{1, 2, 3}
	cfg := SamplerConfig{Temperature: 0.7}

	var emitted []int
	out, err := asym.InferWithCallback(initialTokens, 4, cfg, func(tok int) {
		emitted = append(emitted, tok)
	})
	if err != nil {
		t.Fatalf("InferWithCallback failed: %v", err)
	}

	if len(out) != 4 {
		t.Errorf("expected 4 tokens, got %d (%v)", len(out), out)
	}
	if len(emitted) != 4 {
		t.Errorf("expected 4 callback emissions, got %d", len(emitted))
	}

	// Test dynamic draft length getter
	dLen := asym.manager.GetDynamicDraftLength()
	if dLen < 1 || dLen > 8 {
		t.Errorf("invalid dynamic draft length: %d", dLen)
	}

	// Test GenerateSpeculative with context
	seq := &Sequence{
		Tokens: []int{10, 20},
		Pos:    2,
	}
	if err := asym.manager.GenerateSpeculative(context.Background(), seq); err != nil {
		t.Fatalf("GenerateSpeculative failed: %v", err)
	}
}
