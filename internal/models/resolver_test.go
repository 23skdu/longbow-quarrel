package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModelPath_ExactPath(t *testing.T) {
	// Create a temporary test file
	tmpDir := t.TempDir()
	modelFile := filepath.Join(tmpDir, "test_model.gguf")
	if err := os.WriteFile(modelFile, []byte("GGUF"), 0644); err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}

	resolved, err := ResolveModelPath(modelFile)
	if err != nil {
		t.Fatalf("expected resolution to succeed, got: %v", err)
	}
	if resolved != modelFile {
		t.Errorf("expected %s, got %s", modelFile, resolved)
	}
}

func TestResolveModelPath_Empty(t *testing.T) {
	_, err := ResolveModelPath("")
	if err == nil {
		t.Errorf("expected error on empty string, got nil")
	}
}

func TestResolveModelPath_CacheResolution(t *testing.T) {
	// Check if the user's downloaded model resolves via short query
	targetFile := "/home/rsd/.cache/llmfit/models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf"
	if _, err := os.Stat(targetFile); err == nil {
		// Test full basename
		resolved, err := ResolveModelPath("Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf")
		if err != nil {
			t.Fatalf("failed to resolve basename: %v", err)
		}
		if resolved != targetFile {
			t.Errorf("expected %s, got %s", targetFile, resolved)
		}

		// Test substring match
		resolvedSub, err := ResolveModelPath("Huihui-Qwen3.5")
		if err != nil {
			t.Fatalf("failed to resolve substring: %v", err)
		}
		if resolvedSub != targetFile {
			t.Errorf("expected %s, got %s", targetFile, resolvedSub)
		}
	}
}
