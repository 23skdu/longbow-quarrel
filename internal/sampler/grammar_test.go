package sampler

import (
	"testing"
)

func TestGrammar_Basic(t *testing.T) {
	t.Run("Initialize", func(t *testing.T) {
		// Mock a simple grammar rule
		_ = &Grammar{Active: true}
	})
	
	t.Run("ApplyMask", func(t *testing.T) {
		g := NewJSONGrammar([]string{"{", "}", "a"})
		logits := []float32{0.1, 0.2, 0.3}
		err := g.Apply(logits)
		if err != nil {
			t.Errorf("Apply failed: %v", err)
		}
		// In JSON grammar empty stack, only '{' or '[' allowed.
		// '{' is index 0. Index 2 ('a') should be masked.
		if logits[2] > -1e8 {
			t.Errorf("expected 'a' to be masked, got %f", logits[2])
		}
	})

	t.Run("ApplyInactive", func(t *testing.T) {
		g := &Grammar{Active: false}
		logits := []float32{1.0, 2.0}
		if err := g.Apply(logits); err != nil {
			t.Errorf("Apply failed: %v", err)
		}
		if logits[0] != 1.0 || logits[1] != 2.0 {
			t.Errorf("Inactive grammar modified logits: %v", logits)
		}
	})

	t.Run("Update", func(t *testing.T) {
		g := NewJSONGrammar([]string{"{", "}", "[", "]", "foo"})
		// Initial stack empty
		g.Update("}") // Pop on empty stack
		if len(g.JSONState.Stack) != 0 {
			t.Errorf("Expected empty stack, got %v", g.JSONState.Stack)
		}

		g.Update("{")
		if len(g.JSONState.Stack) != 1 || g.JSONState.Stack[0] != '{' {
			t.Errorf("Expected stack with '{', got %v", g.JSONState.Stack)
		}

		g.Update("[foo]")
		// Should have pushed '[', then popped ']'
		if len(g.JSONState.Stack) != 1 || g.JSONState.Stack[0] != '{' {
			t.Errorf("Expected stack with '{', got %v", g.JSONState.Stack)
		}

		g.Update("}")
		if len(g.JSONState.Stack) != 0 {
			t.Errorf("Expected empty stack, got %v", g.JSONState.Stack)
		}

		// Update on uninitialized Grammar
		gNil := &Grammar{}
		gNil.Update("{")
		if len(gNil.JSONState.Stack) != 1 {
			t.Errorf("Expected initialized stack, got %v", gNil.JSONState.Stack)
		}
	})
}

