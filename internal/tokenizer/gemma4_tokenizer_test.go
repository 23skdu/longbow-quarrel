package tokenizer

import (
	"fmt"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestGemma4Tokens(t *testing.T) {
	modelPath := "/home/rsd/.cache/llmfit/models/Gemma4_E2B_Abliterated_Opus_Distilled.Q8_0.gguf"
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		t.Fatalf("failed to load GGUF: %v", err)
	}
	defer f.Close()

	tok, err := NewFromGGUF(f)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}

	for _, text := range []string{"The capital of France is", " Paris.", "Paris", " Paris"} {
		fmt.Printf("Encode(%q) = %v\n", text, tok.Encode(text))
	}
}
