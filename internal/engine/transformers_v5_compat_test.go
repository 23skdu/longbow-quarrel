package engine

import (
	"math"
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/simd"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func TestTokenizerParityV5(t *testing.T) {
	// Synthetic tokenizer with BPE merges and special tokens matching HF Transformers v5 format
	vocab := map[string]int{
		"<unk>":        0,
		"<s>":          1,
		"</s>":         2,
		"<|eot_id|>":   3,
		"<|im_start|>": 4,
		"<|im_end|>":   5,
		"\u2581":       6,
		"Hello":        7,
		"\u2581world":  8,
		"!":            9,
	}
	tokens := []string{"<unk>", "<s>", "</s>", "<|eot_id|>", "<|im_start|>", "<|im_end|>", "\u2581", "Hello", "\u2581world", "!"}

	tok := &tokenizer.Tokenizer{
		Tokens: tokens,
		Vocab:  vocab,
	}

	// 1. Encode simple text
	encoded := tok.Encode("Hello world")
	if len(encoded) == 0 {
		t.Fatalf("expected non-empty encoding for 'Hello world'")
	}

	// 2. Decode back
	decoded := tok.Decode(encoded)
	if !strings.Contains(decoded, "Hello") || !strings.Contains(decoded, "world") {
		t.Errorf("expected decoded text to contain 'Hello' and 'world', got %q", decoded)
	}

	// 3. Special tokens handling
	if id, ok := tok.Vocab["<|eot_id|>"]; !ok || id != 3 {
		t.Errorf("expected <|eot_id|> ID 3, got %d (ok: %v)", id, ok)
	}
}

func TestChatTemplateParity(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is 2+2?"},
		{Role: "assistant", Content: "4"},
	}

	// 1. LLaMA 3 Chat Template
	llama3Template := "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{ .User }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>"
	pwLlama3 := NewPromptWrapper()
	pwLlama3.ChatTemplate = llama3Template

	promptLlama3, err := pwLlama3.Wrap(messages)
	if err != nil {
		t.Fatalf("LLaMA 3 template wrap failed: %v", err)
	}
	if !strings.Contains(promptLlama3, "<|begin_of_text|>") ||
		!strings.Contains(promptLlama3, "You are a helpful assistant.") ||
		!strings.Contains(promptLlama3, "What is 2+2?") ||
		!strings.Contains(promptLlama3, "4<|eot_id|>") {
		t.Errorf("LLaMA 3 prompt formatting mismatch: %s", promptLlama3)
	}

	// 2. ChatML Template
	chatmlTemplate := "<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .User }}<|im_end|>\n<|im_start|>assistant\n{{ .Response }}<|im_end|>"
	pwChatML := NewPromptWrapper()
	pwChatML.ChatTemplate = chatmlTemplate

	promptChatML, err := pwChatML.Wrap(messages)
	if err != nil {
		t.Fatalf("ChatML template wrap failed: %v", err)
	}
	if !strings.Contains(promptChatML, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>") ||
		!strings.Contains(promptChatML, "<|im_start|>user\nWhat is 2+2?<|im_end|>") ||
		!strings.Contains(promptChatML, "<|im_start|>assistant\n4<|im_end|>") {
		t.Errorf("ChatML prompt formatting mismatch: %s", promptChatML)
	}
}

func TestQuantizationParity(t *testing.T) {
	dim := 128
	orig := make([]float32, dim)
	for i := range orig {
		orig[i] = float32(math.Sin(float64(i)*0.2)) * 2.5
	}

	// 1. Test TurboQuant / PolarQuant parity
	rotation := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		rotation[i*dim+i] = 1.0
	}
	q, scale, _ := simd.PolarQuantVariant(orig, rotation, dim, simd.TurboQuant8)
	if len(q) != dim {
		t.Fatalf("expected polar quant length %d, got %d", dim, len(q))
	}
	if scale <= 0 {
		t.Errorf("expected positive scale, got %v", scale)
	}

	recon := simd.DequantizeTurboQuant(q, scale, rotation, dim)
	if len(recon) != dim {
		t.Fatalf("expected dequantized length %d, got %d", dim, len(recon))
	}

	// Check reconstruction error (TurboQuant8 has high fidelity, SNR > 20 dB)
	var errSum, origSum float64
	for i := range orig {
		diff := float64(orig[i] - recon[i])
		errSum += diff * diff
		origSum += float64(orig[i] * orig[i])
	}
	if origSum > 0 {
		snr := 10 * math.Log10(origSum/errSum)
		if snr < 15.0 {
			t.Errorf("TurboQuant8 SNR too low: %f dB (expected >= 15 dB)", snr)
		}
	}

	// 2. Test Q8_0 dequantization parity
	// A Q8_0 block has 32 float16 delta + 32 int8 weights = 2 + 32 = 34 bytes
	blockQ8 := make([]byte, 34)
	// Scale = 1.0 (FP16 0x3C00 in little endian = [0x00, 0x3c])
	blockQ8[0] = 0x00
	blockQ8[1] = 0x3c
	for i := 0; i < 32; i++ {
		blockQ8[2+i] = byte(int8(i - 16))
	}
	dequant := gguf.DequantizeQ8_0(blockQ8, 32)
	if len(dequant) != 32 {
		t.Fatalf("expected 32 dequantized floats, got %d", len(dequant))
	}
	for i := 0; i < 32; i++ {
		expected := float32(int8(i - 16))
		if math.Abs(float64(dequant[i]-expected)) > 1e-3 {
			t.Errorf("Q8_0 dequant mismatch at %d: got %v, expected %v", i, dequant[i], expected)
		}
	}
}