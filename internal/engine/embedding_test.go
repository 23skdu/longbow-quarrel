//go:build darwin && metal

package engine

import (
	"testing"
)

// The tests used to skip unconditionally: the assertions below them were
// dead code, because a skip stops the test before it reaches them. The
// engine is built here instead, so the assertions run.

func TestEmbeddingDim(t *testing.T) {
	e := &metalEngine{
		weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}

	dim := e.EmbeddingDim()
	if dim != 0 {
		t.Errorf("Expected dim 0 with nil TokenEmb, got %d", dim)
	}
}

func TestGetEmbedding(t *testing.T) {
	e := &metalEngine{
		weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}

	_, err := e.GetEmbedding(0)
	if err == nil {
		t.Error("Expected error when TokenEmb is nil")
	}
}

func TestGetEmbeddings(t *testing.T) {
	e := &metalEngine{
		weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}

	_, err := e.GetEmbeddings([]int{0, 1, 2})
	if err == nil {
		t.Error("Expected error when TokenEmb is nil")
	}
}

func TestTextToEmbedding(t *testing.T) {
	e := &metalEngine{
		Tokenizer: nil,
	}

	_, err := e.TextToEmbedding("hello")
	if err == nil {
		t.Error("Expected error when tokenizer is nil")
	}
}

func BenchmarkGetEmbedding(b *testing.B) {
	e := &metalEngine{
		weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}
	for i := 0; i < b.N; i++ {
		_, _ = e.GetEmbedding(0)
	}
}

func BenchmarkGetEmbeddings(b *testing.B) {
	e := &metalEngine{
		weights: &LlamaWeights{
			TokenEmb: nil,
		},
	}
	for i := 0; i < b.N; i++ {
		_, _ = e.GetEmbeddings([]int{0, 1, 2})
	}
}

func BenchmarkTextToEmbedding(b *testing.B) {
	e := &metalEngine{
		Tokenizer: nil,
	}
	for i := 0; i < b.N; i++ {
		_, _ = e.TextToEmbedding("hello")
	}
}
