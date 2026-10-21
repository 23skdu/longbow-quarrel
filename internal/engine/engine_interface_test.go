package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// The engine interface has a lot of thin delegation methods that nothing
// exercised. They are the surface the API server, the Arrow Flight server
// and the Flight server all call, so they are worth covering.

// countingEngine records the calls that reach it.
type countingEngine struct {
	MockEngine
	token          int
	inferCalls     int
	adapterCalls   int
	swapCalls      int
	closeCalls     int
	rollbackCalls  int
	draftCalls     int
	cachePos       int
	forwardLogits  [][]float32
	forwardBatches int
}

func (e *countingEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	e.inferCalls++
	if e.token != 0 {
		return []int{e.token}, nil
	}
	return []int{7, 8, 9}, nil
}

func (e *countingEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	e.inferCalls++
	if e.token != 0 {
		return []int{e.token}, []float32{0.1, 0.9}, nil
	}
	return []int{7}, []float32{0.5, 0.5}, nil
}

func (e *countingEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(int)) ([]int, error) {
	e.inferCalls++
	out := []int{7, 8}
	for _, t := range out {
		if callback != nil {
			callback(t)
		}
	}
	return out, nil
}

func (e *countingEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	e.inferCalls++
	if tokenCallback != nil {
		tokenCallback(7)
	}
	if logitsCallback != nil {
		logitsCallback([]float32{0.5})
	}
	return []int{7}, nil
}

func (e *countingEngine) LoadAdapter(path, id string) error {
	e.adapterCalls++
	return nil
}

func (e *countingEngine) SwapModel(modelPath string, cfg config.Config) error {
	e.swapCalls++
	return nil
}

func (e *countingEngine) Close() { e.closeCalls++ }

func (e *countingEngine) GetSeqCachePos(seqID string) int { return e.cachePos }

func (e *countingEngine) RollbackKV(seqID string, newPos int) error {
	e.rollbackCalls++
	return nil
}

func (e *countingEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	e.draftCalls++
	return e.forwardLogits, nil
}

func (e *countingEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	e.forwardBatches++
	return nil, nil
}

// TestAsymmetricSpeculativeEngineInterface covers the whole Engine
// interface of the speculative wrapper.
//
// The wrapper is what the CLI and the REST API use when a draft model is
// configured, and most of its methods had no test at all.
func TestAsymmetricSpeculativeEngineInterface(t *testing.T) {
	target := &countingEngine{
		cachePos: 42,
		forwardLogits: [][]float32{
			{0.1, 0.9}, {0.1, 0.9}, {0.1, 0.9}, {0.1, 0.9},
		},
	}
	draft := &countingEngine{
		token:         1,
		forwardLogits: [][]float32{{0.1, 0.9}},
	}

	spec := NewAsymmetricSpeculativeEngineWithOptions(target, draft, 4, 1)

	if got, err := spec.Infer([]int{1}, 2, SamplerConfig{}); err != nil || len(got) == 0 {
		t.Fatalf("Infer returned %v, %v", got, err)
	}
	if target.inferCalls == 0 && draft.inferCalls == 0 {
		t.Error("Infer did not reach either engine")
	}

	if _, _, err := spec.InferWithLogits([]int{1}, 2, SamplerConfig{}); err != nil {
		t.Errorf("InferWithLogits failed: %v", err)
	}

	var tokens []int
	var logits []float32
	if _, err := spec.InferWithCallbackLogits([]int{1}, 2, SamplerConfig{},
		func(t int) { tokens = append(tokens, t) },
		func(l []float32) { logits = l }); err != nil {
		t.Errorf("InferWithCallbackLogits failed: %v", err)
	}
	if len(tokens) == 0 {
		t.Error("the token callback was never called")
	}
	if len(logits) == 0 {
		t.Error("the logits callback was never called")
	}

	// The configuration comes from the target.
	if spec.Config().MaxTokens != target.Config().MaxTokens {
		t.Error("the configuration does not come from the target engine")
	}

	// The model swap goes to the target.
	if err := spec.SwapModel("/tmp/model.gguf", config.Default()); err != nil {
		t.Errorf("SwapModel failed: %v", err)
	}
	if target.swapCalls == 0 {
		t.Error("SwapModel did not reach the target engine")
	}

	// The adapter load goes to the target.
	if err := spec.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err != nil {
		t.Errorf("LoadAdapter failed: %v", err)
	}
	if target.adapterCalls == 0 {
		t.Error("LoadAdapter did not reach the target engine")
	}

	// The cache position comes from the target.
	if got := spec.GetSeqCachePos("seq-1"); got != 42 {
		t.Errorf("the cache position is %d, want 42", got)
	}

	// The draft comes from the target as well.
	if _, err := spec.ForwardDraft([]int{1}); err != nil {
		t.Errorf("ForwardDraft failed: %v", err)
	}
	if target.draftCalls == 0 {
		t.Error("ForwardDraft did not reach the target engine")
	}

	// The rollback reaches both engines.
	if err := spec.RollbackKV("seq-1", 3); err != nil {
		t.Errorf("RollbackKV failed: %v", err)
	}
	if target.rollbackCalls == 0 || draft.rollbackCalls == 0 {
		t.Error("RollbackKV did not reach both engines")
	}

	// The batched forward goes to the target.
	if _, err := spec.ForwardBatch(&BatchDescriptor{}); err != nil {
		t.Errorf("ForwardBatch failed: %v", err)
	}
	if target.forwardBatches == 0 {
		t.Error("ForwardBatch did not reach the target engine")
	}

	// The draft length is adjustable.
	spec.SetDraftLength(6)
	if got := spec.manager.GetDynamicDraftLength(); got != 6 {
		t.Errorf("the draft length is %d, want 6", got)
	}
	// A non-positive value keeps the current one.
	spec.SetDraftLength(0)
	if got := spec.manager.GetDynamicDraftLength(); got != 6 {
		t.Errorf("the draft length is %d, want 6", got)
	}

	// Closing closes both engines.
	spec.Close()
	if target.closeCalls == 0 || draft.closeCalls == 0 {
		t.Error("Close did not close both engines")
	}
}

// TestMockEngineMethods covers the mock engine, which the API and the
// API tests use.
func TestMockEngineMethods(t *testing.T) {
	m, err := NewMockEngine("model.gguf", config.Default())
	if err != nil {
		t.Fatalf("failed to create the mock engine: %v", err)
	}

	if err := m.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err != nil {
		t.Errorf("LoadAdapter failed: %v", err)
	}
	if _, err := m.ForwardBatch(&BatchDescriptor{}); err != nil {
		t.Errorf("ForwardBatch failed: %v", err)
	}
	if err := m.RollbackKV("seq-1", 1); err != nil {
		t.Errorf("RollbackKV failed: %v", err)
	}
	if got := m.GetSeqCachePos("seq-1"); got != 0 {
		t.Errorf("the cache position is %d, want 0", got)
	}
	m.Close()
}

// TestLoRAManagerLoadAdapterErrors covers the LoRA manager's error paths.
func TestLoRAManagerLoadAdapterErrors(t *testing.T) {
	lm := NewLoRAManager()

	// A file that does not exist.
	if err := lm.LoadAdapter(nil, "/nonexistent/adapter.gguf", "adapter-1"); err == nil {
		t.Error("expected an error for a missing adapter file")
	}
}

// TestCPUEngineSwapModelErrors covers the CPU engine's model swap guard.
func TestCPUEngineSwapModelErrors(t *testing.T) {
	e := &CPUEngine{weights: &CPUWeights{}}

	// A model path that does not exist is an error, not a panic.
	if err := e.SwapModel("/nonexistent/model.gguf", config.Default()); err == nil {
		t.Error("expected an error when the model cannot be loaded")
	}
}
