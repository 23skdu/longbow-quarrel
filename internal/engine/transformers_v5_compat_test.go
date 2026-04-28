package engine

import (
	"testing"
)

func TestTokenizerParityV5(t *testing.T) {
	t.Skip("Requires HuggingFace tokenizer - set HF_TOKEN to run")

}

func TestChatTemplateParity(t *testing.T) {
	t.Skip("Requires model files - set MODEL_PATH to run")

}

func TestQuantizationParity(t *testing.T) {
	t.Skip("Requires GGUF model with quantization - set MODEL_PATH to run")

}