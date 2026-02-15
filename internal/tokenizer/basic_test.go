package tokenizer

import "testing"

func TestSimple(t *testing.T) {
	if 1+1 != 2 {
		t.Error("math is broken")
	}
}

func TestSplitWordsBasic(t *testing.T) {
	result := splitWords("hello world")
	if len(result) != 2 {
		t.Errorf("expected 2 words, got %d", len(result))
	}
}
