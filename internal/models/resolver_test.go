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

func TestResolveModelPath_TildeAndNotFound(t *testing.T) {
	// Test tilde expansion for existing file in home
	home, err := os.UserHomeDir()
	if err == nil {
		tmpInHome := filepath.Join(home, ".test_quarrel_tilde.gguf")
		if err := os.WriteFile(tmpInHome, []byte("GGUF"), 0644); err == nil {
			defer os.Remove(tmpInHome)
			res, err := ResolveModelPath("~/.test_quarrel_tilde.gguf")
			if err != nil || res != tmpInHome {
				t.Errorf("Tilde resolution failed: %v, got %s", err, res)
			}
		}
	}

	// Test directory path (should not resolve since it is a dir, not a file)
	tmpDir := t.TempDir()
	if _, err := ResolveModelPath(tmpDir); err == nil {
		t.Error("Expected error when resolving a directory path")
	}

	// Test non-existent model name
	if _, err := ResolveModelPath("non_existent_model_xyz_12345"); err == nil {
		t.Error("Expected error for non-existent model")
	}

	// Test tilde expansion for non-existent file
	if _, err := ResolveModelPath("~/non_existent_model_xyz_12345.gguf"); err == nil {
		t.Error("Expected error for non-existent tilde file")
	}
}

func TestResolveModelPath_SearchDirsDirect(t *testing.T) {
	dirs := SearchDirs()
	if len(dirs) > 0 {
		testDir := dirs[0]
		_ = os.MkdirAll(testDir, 0755)
		testFile := filepath.Join(testDir, "test_direct_match.gguf")
		if err := os.WriteFile(testFile, []byte("GGUF"), 0644); err == nil {
			defer os.Remove(testFile)

			// With .gguf
			res, err := ResolveModelPath("test_direct_match.gguf")
			if err != nil || res != testFile {
				t.Errorf("Direct match failed: %v, got %s", err, res)
			}

			// Without .gguf
			resNoExt, err := ResolveModelPath("test_direct_match")
			if err != nil || resNoExt != testFile {
				t.Errorf("Direct match without ext failed: %v, got %s", err, resNoExt)
			}
		}
	}
}

func TestResolveModelPath_OllamaFallback(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", tmpDir)

	manifestDir := filepath.Join(tmpDir, "manifests", "registry.ollama.ai", "library", "testmodel")
	_ = os.MkdirAll(manifestDir, 0755)
	blobsDir := filepath.Join(tmpDir, "blobs")
	_ = os.MkdirAll(blobsDir, 0755)

	blobFile := filepath.Join(blobsDir, "sha256-abc12345")
	_ = os.WriteFile(blobFile, []byte("GGUF"), 0644)

	manifestContent := `{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc12345","size":4}]}`
	_ = os.WriteFile(filepath.Join(manifestDir, "latest"), []byte(manifestContent), 0644)

	resolved, err := ResolveModelPath("testmodel")
	if err != nil || resolved != blobFile {
		t.Errorf("Ollama resolution failed: %v, got %s", err, resolved)
	}
}



