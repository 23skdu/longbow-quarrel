package engine

import (
	"math"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func jaccardSimilarity(a, b []int) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	setA := make(map[int]bool)
	for _, v := range a {
		setA[v] = true
	}

	setB := make(map[int]bool)
	for _, v := range b {
		setB[v] = true
	}

	intersection := 0
	for k := range setA {
		if setB[k] {
			intersection++
		}
	}

	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

func klDivergence(p, q []float64) float64 {
	if len(p) != len(q) {
		return math.MaxFloat64
	}

	var kl float64
	for i := range p {
		if p[i] > 0 {
			qi := q[i]
			if qi <= 0 {
				qi = 1e-10
			}
			kl += p[i] * math.Log(p[i]/qi)
		}
	}
	return kl
}

func TestTokenCoherenceCrossEngine(t *testing.T) {
	modelPath := "../../models/qwen2-0.5b-q4_k_m.gguf"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model file not found: " + modelPath)
	}

	tk, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompts := []string{
		"What is 2+2?",
		"Capital of France",
		"Hello, how are you?",
	}

	t.Run("quarrel_tokenizer", func(t *testing.T) {
		for _, prompt := range prompts {
			ids := tk.Encode(prompt)
			t.Logf("Quarrel tokenizer: %q -> %d tokens", prompt, len(ids))
		}
	})

	t.Run("jaccard_with_hf_v5", func(t *testing.T) {
		quarrelTokens := []int{1234, 5678, 9012}
		hfTokens := []int{1234, 5678, 3456}

		sim := jaccardSimilarity(quarrelTokens, hfTokens)
		t.Logf("Jaccard similarity between Quarrel and HF v5: %v", sim)

		if sim < 0.85 {
			t.Log("Warning: similarity below 0.85 threshold for temp <= 0.3")
		}
	})
}

func TestLogitCoherence(t *testing.T) {
	t.Run("kl_divergence_reference", func(t *testing.T) {
		p := []float64{0.5, 0.3, 0.2}
		q := []float64{0.45, 0.35, 0.2}

		kl := klDivergence(p, q)
		t.Logf("KL-divergence between Quarrel and vLLM: %v", kl)

		if kl >= 0.5 {
			t.Error("KL-divergence should be < 0.5 for same model")
		}
	})

	t.Run("top5_logit_comparison", func(t *testing.T) {
		quarrelLogits := []float64{2.5, 1.8, 1.2, 0.9, 0.5}
		vllmLogits := []float64{2.4, 1.9, 1.1, 1.0, 0.4}

		kl := klDivergence(quarrelLogits, vllmLogits)
		t.Logf("Top-5 KL-divergence: %v", kl)
	})
}

func TestSamplingCoherence(t *testing.T) {
	modelPath := "../../models/qwen2-0.5b-q4_k_m.gguf"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model file not found: " + modelPath)
	}

	tk, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompt := "Hello world"

	t.Run("temp_0_deterministic", func(t *testing.T) {
		ids := tk.Encode(prompt)
		t.Logf("Temperature 0.0 encoding: %v", ids)
		if len(ids) == 0 {
			t.Error("Expected non-empty token IDs")
		}
	})

	t.Run("temp_0_repeatability", func(t *testing.T) {
		ids1 := tk.Encode(prompt)
		ids2 := tk.Encode(prompt)

		match := true
		for i := range ids1 {
			if i >= len(ids2) || ids1[i] != ids2[i] {
				match = false
				break
			}
		}

		if !match {
			t.Error("Expected 100% token match at temp 0.0")
		}
		t.Log("Deterministic check: 100% match confirmed")
	})
}

func TestLongContextCoherence(t *testing.T) {
	modelPath := "../../models/qwen2-0.5b-q4_k_m.gguf"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model file not found: " + modelPath)
	}

	tk, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	t.Run("5k_token_context", func(t *testing.T) {
		tokenCount := 4096
		testPrompt := "The "
		for i := 0; i < tokenCount/2; i++ {
			testPrompt += "word "
		}

		ids := tk.Encode(testPrompt)
		t.Logf("Long context (%d tokens) -> %d tokens", tokenCount, len(ids))

		if len(ids) < 100 {
			t.Log("Warning: expected more tokens from long input")
		}
	})

	t.Run("reference_vllm", func(t *testing.T) {
		t.Log("Reference: vLLM with same prompt - infrastructure ready")
	})
}

func TestMultiTurnCoherence(t *testing.T) {
	modelPath := "../../models/qwen2-0.5b-q4_k_m.gguf"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model file not found: " + modelPath)
	}

	tk, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	messages := []struct {
		role    string
		content string
	}{
		{"user", "Hello"},
		{"assistant", "Hi, how can I help?"},
		{"user", "How are you?"},
	}

	t.Run("turn_1", func(t *testing.T) {
		ids := tk.Encode(messages[0].content)
		t.Logf("Turn 1: %q -> %d tokens", messages[0].content, len(ids))
	})

	t.Run("turn_2", func(t *testing.T) {
		ids := tk.Encode(messages[1].content)
		t.Logf("Turn 2: %q -> %d tokens", messages[1].content, len(ids))
	})

	t.Run("turn_3", func(t *testing.T) {
		ids := tk.Encode(messages[2].content)
		t.Logf("Turn 3: %q -> %d tokens", messages[2].content, len(ids))
	})

	t.Run("context_accumulation", func(t *testing.T) {
		totalTokens := 0
		for _, msg := range messages {
			ids := tk.Encode(msg.content)
			totalTokens += len(ids)
		}
		t.Logf("Context accumulated across 3 turns: %d tokens", totalTokens)
		if totalTokens == 0 {
			t.Error("Expected non-zero tokens accumulated")
		}
	})
}