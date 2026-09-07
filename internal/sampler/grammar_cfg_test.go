package sampler

import (
	"testing"
)

func TestGrammar_Regex(t *testing.T) {
	vocab := []string{"123", "abc", "!", "456"}
	g, err := NewRegexGrammar(`^[0-9]+$`, vocab)
	if err != nil {
		t.Fatalf("failed to create regex grammar: %v", err)
	}

	logits := []float32{1.0, 2.0, 3.0, 4.0}
	if err := g.Apply(logits); err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	// "123" (index 0) and "456" (index 3) match regex ^[0-9]+$
	// "abc" (index 1) and "!" (index 2) should be masked
	if logits[0] < 0 {
		t.Errorf("expected logits[0] not masked, got %f", logits[0])
	}
	if logits[3] < 0 {
		t.Errorf("expected logits[3] not masked, got %f", logits[3])
	}
}

func TestGrammar_JSONSchemaTransitions(t *testing.T) {
	vocab := []string{"{", "}", ":", ",", `"name"`, `"Alice"`, "123", "true"}
	g := NewJSONGrammar(vocab)

	// Step 1: At root, only '{' is allowed
	logits := make([]float32, len(vocab))
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[0] < 0 {
		t.Errorf("expected '{' to be allowed at root, got %f", logits[0])
	}
	if logits[4] > -1e8 {
		t.Errorf("expected key to be masked before '{', got %f", logits[4])
	}

	// Step 2: After '{', key `"name"` or '}' is allowed
	g.Update("{")
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[4] < 0 {
		t.Errorf("expected key to be allowed after '{', got %f", logits[4])
	}
	if logits[1] < 0 {
		t.Errorf("expected '}' to be allowed for empty object, got %f", logits[1])
	}
	if logits[2] > -1e8 {
		t.Errorf("expected ':' to be masked before key, got %f", logits[2])
	}

	// Step 3: After key `"name"`, only ':' is allowed
	g.Update(`"name"`)
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[2] < 0 {
		t.Errorf("expected ':' to be allowed after key, got %f", logits[2])
	}
	if logits[4] > -1e8 {
		t.Errorf("expected key to be masked before ':', got %f", logits[4])
	}

	// Step 4: After ':', values are allowed
	g.Update(":")
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[5] < 0 {
		t.Errorf("expected string value to be allowed, got %f", logits[5])
	}
	if logits[6] < 0 {
		t.Errorf("expected number value to be allowed, got %f", logits[6])
	}

	// Step 5: After value `"Alice"`, comma ',' or '}' is allowed
	g.Update(`"Alice"`)
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[1] < 0 {
		t.Errorf("expected '}' to be allowed after value, got %f", logits[1])
	}
	if logits[3] < 0 {
		t.Errorf("expected ',' to be allowed after value, got %f", logits[3])
	}

	// Step 6: After '}', object is closed
	g.Update("}")
	if len(g.JSONState.Stack) != 0 {
		t.Errorf("expected stack to be empty after closing '}', got %d", len(g.JSONState.Stack))
	}
}
