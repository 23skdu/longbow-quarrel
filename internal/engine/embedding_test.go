//go:build darwin && metal

package engine

import (
	"testing"
)

func TestEmbeddingDim(t *testing.T) {
	t.Skip("Requires loaded model")

	e := &Engine{
		Weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}

	dim := e.EmbeddingDim()
	if dim != 0 {
		t.Errorf("Expected dim 0 with nil TokenEmb, got %d", dim)
	}
}

func TestGetEmbedding(t *testing.T) {
	t.Skip("Requires loaded model")

	e := &Engine{
		Weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}

	_, err := e.GetEmbedding(0)
	if err == nil {
		t.Error("Expected error when TokenEmb is nil")
	}
}

func TestGetEmbeddings(t *testing.T) {
	t.Skip("Requires loaded model")

	e := &Engine{
		Weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}

	_, err := e.GetEmbeddings([]int{0, 1, 2})
	if err == nil {
		t.Error("Expected error when TokenEmb is nil")
	}
}

func TestTextToEmbedding(t *testing.T) {
	t.Skip("Requires loaded model and tokenizer")

	e := &Engine{
		Tokenizer: nil,
	}

	_, err := e.TextToEmbedding("hello")
	if err == nil {
		t.Error("Expected error when tokenizer is nil")
	}
}

func BenchmarkGetEmbedding(b *testing.B) {
	b.Skip("Requires loaded model")
}

func BenchmarkGetEmbeddings(b *testing.B) {
	b.Skip("Requires loaded model")
}

func BenchmarkTextToEmbedding(b *testing.B) {
	b.Skip("Requires loaded model")
}
