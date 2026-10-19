package engine

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/apache/arrow-go/v18/arrow/flight"
)

// tensorFlightServer answers the tensor-forward operation with a result
// tensor, the way a worker that implements the sharded forward would.
type tensorFlightServer struct {
	workerFlightServer
	values []float32
}

func (s *tensorFlightServer) DoPut(stream flight.FlightService_DoPutServer) error {
	for {
		_, err := stream.Recv()
		if err != nil {
			break
		}
	}

	payload := make([]byte, len(s.values)*4)
	for i, v := range s.values {
		binary.LittleEndian.PutUint32(payload[i*4:], math.Float32bits(v))
	}

	return stream.SendMsg(&flight.PutResult{AppMetadata: payload})
}

// TestRemoteWorkerEngineForwardBatch covers the remote worker's batched
// forward against a worker that answers the tensor-forward operation.
func TestRemoteWorkerEngineForwardBatchWithAResult(t *testing.T) {
	addr := startTensorWorkerServer(t, []float32{1, 2, 3, 4})
	worker := newRemoteWorker(t, addr)

	batch := &BatchDescriptor{
		Sequences:   []*Sequence{{ID: 1, Tokens: []int{5, 6}}},
		Tokens:      []int{5, 6},
		Offsets:     []int{0},
		ContextLens: []int{0},
	}

	results, err := worker.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("ForwardBatch failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("the forward returned %d results, want 1", len(results))
	}
	if results[0] == nil {
		t.Fatal("the forward returned no tensor")
	}
	defer results[0].Free()

	data := results[0].ToHostF32()
	if len(data) == 0 {
		t.Fatal("the result tensor is empty")
	}
}

// TestRemoteWorkerEngineForwardBatchRejectsBadBatches covers the guards of
// the batched forward.
func TestRemoteWorkerEngineForwardBatchGuards(t *testing.T) {
	addr := startTensorWorkerServer(t, []float32{1})
	worker := newRemoteWorker(t, addr)

	// A nil sequence.
	if _, err := worker.ForwardBatch(&BatchDescriptor{
		Sequences: []*Sequence{nil},
	}); err == nil {
		t.Error("expected an error for a nil sequence")
	}

	// A sequence with no tokens.
	if _, err := worker.ForwardBatch(&BatchDescriptor{
		Sequences: []*Sequence{{ID: 1}},
		Tokens:    []int{},
		Offsets:   []int{0},
	}); err == nil {
		t.Error("expected an error for a sequence with no tokens")
	}
}

// TestQualityEvaluatorWithTokenizer covers the metrics that need a
// tokenizer, which are the ones the API and the benchmarks use.
func TestQualityEvaluatorWithTokenizer(t *testing.T) {
	qe := &QualityEvaluator{tokenizer: &stubTokenizer{}}

	rouge := qe.CalculateROUGE("hello world", "hello world")
	if rouge.ROUGEL_F1 <= 0 {
		t.Errorf("the ROUGE of a text against itself is %v, want a positive score", rouge.ROUGEL_F1)
	}
	if rouge.Precision <= 0 || rouge.Recall <= 0 {
		t.Errorf("the ROUGE precision and recall are %v and %v, want positive values", rouge.Precision, rouge.Recall)
	}

	// A text against an unrelated one has no overlap.
	none := qe.CalculateROUGE("hello world", "goodbye moon")
	if none.ROUGEL_F1 != 0 {
		t.Errorf("the ROUGE of unrelated texts is %v, want 0", none.ROUGEL_F1)
	}

	bleu := qe.CalculateBLEU("hello world", "hello world")
	if bleu.BLEU1 <= 0 {
		t.Errorf("the BLEU of a text against itself is %v, want a positive score", bleu.BLEU1)
	}
}

// stubTokenizer is a tokenizer that maps each word to its length.
type stubTokenizer struct{}

func (t *stubTokenizer) Encode(s string) []int {
	words := splitWords(s)
	out := make([]int, len(words))
	for i := range words {
		out[i] = len(words[i])
	}
	return out
}

func (t *stubTokenizer) Decode(tokens []int) string { return "" }

// splitWords splits a string on the spaces.
func splitWords(s string) []string {
	var out []string
	current := ""
	for _, r := range s {
		if r == ' ' {
			if current != "" {
				out = append(out, current)
				current = ""
			}
			continue
		}
		current += string(r)
	}
	if current != "" {
		out = append(out, current)
	}
	return out
}

// TestPromptWrapperConditions covers the chat template's condition
// evaluation, which decides whether a block is rendered.
func TestPromptWrapperConditions(t *testing.T) {
	cases := []struct {
		name string
		expr string
		want bool
		vars map[string]string
	}{
		{"a non-empty string is true", "name", true, map[string]string{"name": "llama"}},
		{"an empty string is false", "name", false, map[string]string{"name": ""}},
		{"a missing variable is false", "missing", false, map[string]string{}},
		{"not inverts the result", "not name", false, map[string]string{"name": "llama"}},
		{"and needs both", "a and b", true, map[string]string{"a": "1", "b": "2"}},
		{"and needs both to be true", "a and b", false, map[string]string{"a": "1"}},
		{"or needs one", "a or b", true, map[string]string{"b": "2"}},
		{"equality compares", `name == "llama"`, true, map[string]string{"name": "llama"}},
		{"equality compares the other way", `name == "llama"`, false, map[string]string{"name": "qwen"}},
		{"inequality compares", `name != "llama"`, true, map[string]string{"name": "qwen"}},
		{"is defined checks the map", "name is defined", true, map[string]string{"name": "llama"}},
		{"is defined checks a missing name", "name is defined", false, map[string]string{}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := evalCondition(tc.expr, tc.vars)
			if got != tc.want {
				t.Errorf("the condition %q evaluated to %v, want %v", tc.expr, got, tc.want)
			}
		})
	}
}

// TestSamplerAdvanced covers the advanced sampler, which applies the
// grammar, the repetition penalties and the temperature together.
func TestSamplerAdvanced(t *testing.T) {
	s := NewSampler(SamplerConfig{
		Temperature:      0.7,
		TopK:             40,
		TopP:             0.95,
		RepPenalty:       1.1,
		FrequencyPenalty: 0.2,
		PresencePenalty:  0.1,
		MinP:             0.0,
	})

	logits := make([]float32, 16)
	for i := range logits {
		logits[i] = float32(i)
	}

	token := s.SampleAdvanced(logits, []int{1, 2, 3}, false)
	if token < 0 || token >= len(logits) {
		t.Errorf("the sampler produced the out-of-range token %d", token)
	}
}

// TestCPUEngineCloseIsIdempotent checks that closing the engine twice is
// safe: the stop channel is closed once, and the sequence caches are
// released.
func TestCPUEngineCloseIsIdempotent(t *testing.T) {
	e := &CPUEngine{
		weights:     &CPUWeights{},
		stopChan:    make(chan struct{}),
		doneChan:    make(chan struct{}),
		seqKVCaches: map[string]*CPUKVCache{"seq-1": NewCPUKVCache(1)},
	}
	// The batch loop closes the done channel when it exits. There is no
	// loop here, so it stands in for one.
	close(e.doneChan)

	e.Close()

	select {
	case <-e.stopChan:
	default:
		t.Error("the stop channel was not closed")
	}
	if e.seqKVCaches != nil {
		t.Error("the sequence caches were not released")
	}

	// Closing again must not panic on the closed channel.
	e.Close()
}

// TestEngineConfigIsUsed checks that the engine reports its configuration.
func TestEngineConfigIsUsed(t *testing.T) {
	cfg := config.Default()
	cfg.Layers = 11
	e := &CPUEngine{config: cfg}
	if got := e.Config().Layers; got != 11 {
		t.Errorf("the engine reports %d layers, want 11", got)
	}
}

// TestCPUEngineGetSeqCachePos checks the cache position lookup.
func TestCPUEngineGetSeqCachePos(t *testing.T) {
	e := &CPUEngine{weights: &CPUWeights{}}
	if got := e.GetSeqCachePos("seq-1"); got != 0 {
		t.Errorf("the cache position is %d, want 0", got)
	}
}
