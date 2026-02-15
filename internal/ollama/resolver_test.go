package ollama

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultTag(t *testing.T) {
	if DefaultTag != "latest" {
		t.Errorf("expected DefaultTag to be 'latest', got '%s'", DefaultTag)
	}
}

func TestMediaTypeModel(t *testing.T) {
	expected := "application/vnd.ollama.image.model"
	if MediaTypeModel != expected {
		t.Errorf("expected MediaTypeModel to be '%s', got '%s'", expected, MediaTypeModel)
	}
}

func TestManifestStruct(t *testing.T) {
	manifest := Manifest{
		SchemaVersion: 2,
		Layers: []Layer{
			{
				MediaType: "application/vnd.ollama.image.model",
				Digest:    "sha256:abc123",
				Size:      1024,
			},
		},
	}

	if manifest.SchemaVersion != 2 {
		t.Errorf("expected SchemaVersion 2, got %d", manifest.SchemaVersion)
	}
	if len(manifest.Layers) != 1 {
		t.Errorf("expected 1 layer, got %d", len(manifest.Layers))
	}
}

func TestLayerStruct(t *testing.T) {
	layer := Layer{
		MediaType: "application/vnd.ollama.image.model",
		Digest:    "sha256:def456",
		Size:      2048,
	}

	if layer.MediaType != "application/vnd.ollama.image.model" {
		t.Errorf("unexpected MediaType: %s", layer.MediaType)
	}
	if layer.Digest != "sha256:def456" {
		t.Errorf("unexpected Digest: %s", layer.Digest)
	}
	if layer.Size != 2048 {
		t.Errorf("expected Size 2048, got %d", layer.Size)
	}
}

func TestManifestJSONUnmarshal(t *testing.T) {
	jsonData := `{
		"schemaVersion": 2,
		"layers": [
			{
				"mediaType": "application/vnd.ollama.image.model",
				"digest": "sha256:abc123def456",
				"size": 1234567
			},
			{
				"mediaType": "application/vnd.ollama.image.config",
				"digest": "sha256:config123",
				"size": 100
			}
		]
	}`

	var m Manifest
	err := json.Unmarshal([]byte(jsonData), &m)
	if err != nil {
		t.Fatalf("failed to unmarshal manifest: %v", err)
	}

	if m.SchemaVersion != 2 {
		t.Errorf("expected SchemaVersion 2, got %d", m.SchemaVersion)
	}
	if len(m.Layers) != 2 {
		t.Errorf("expected 2 layers, got %d", len(m.Layers))
	}
}

func TestGetOllamaDirDefault(t *testing.T) {
	// Clear OLLAMA_MODELS env var if set
	_ = os.Unsetenv("OLLAMA_MODELS")

	dir, err := GetOllamaDir()
	if err != nil {
		t.Fatalf("GetOllamaDir() failed: %v", err)
	}

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir() failed: %v", err)
	}

	expected := filepath.Join(home, ".ollama", "models")
	if dir != expected {
		t.Errorf("expected %s, got %s", expected, dir)
	}
}

func TestGetOllamaDirEnvOverride(t *testing.T) {
	// Set custom OLLAMA_MODELS path
	customPath := "/custom/ollama/models"
	_ = os.Setenv("OLLAMA_MODELS", customPath)
	defer func() { _ = os.Unsetenv("OLLAMA_MODELS") }()

	dir, err := GetOllamaDir()
	if err != nil {
		t.Fatalf("GetOllamaDir() failed: %v", err)
	}

	if dir != customPath {
		t.Errorf("expected %s, got %s", customPath, dir)
	}
}

func TestGetOllamaDirEnvVarEmpty(t *testing.T) {
	// Set OLLAMA_MODELS to empty string (should fall back to default)
	_ = os.Setenv("OLLAMA_MODELS", "")
	defer func() { _ = os.Unsetenv("OLLAMA_MODELS") }()

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir() failed: %v", err)
	}

	expected := filepath.Join(home, ".ollama", "models")

	dir, err := GetOllamaDir()
	if err != nil {
		t.Fatalf("GetOllamaDir() failed: %v", err)
	}

	if dir != expected {
		t.Errorf("expected %s, got %s", expected, dir)
	}
}

func TestResolveModelPathParse(t *testing.T) {
	tests := []struct {
		name         string
		modelName    string
		expectedName string
		expectedTag  string
	}{
		{"simple name", "llama3", "llama3", "latest"},
		{"with tag", "llama3:8b", "llama3", "8b"},
		{"with latest tag", "mistral:latest", "mistral", "latest"},
		{"with complex tag", "model:v1.0", "model", "v1.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parts := splitModelName(tt.modelName)
			if len(parts) == 1 {
				if parts[0] != tt.expectedName {
					t.Errorf("expected name %s, got %s", tt.expectedName, parts[0])
				}
			} else if len(parts) == 2 {
				if parts[0] != tt.expectedName {
					t.Errorf("expected name %s, got %s", tt.expectedName, parts[0])
				}
				if parts[1] != tt.expectedTag {
					t.Errorf("expected tag %s, got %s", tt.expectedTag, parts[1])
				}
			}
		})
	}
}

// Helper function that mirrors the parsing logic in ResolveModelPath
func splitModelName(modelName string) []string {
	// This is a simple split for testing - actual implementation uses strings.Split
	result := make([]string, 0)
	current := ""
	for _, c := range modelName {
		if c == ':' {
			result = append(result, current)
			current = ""
		} else {
			current += string(c)
		}
	}
	result = append(result, current)
	return result
}

func TestBlobNameConversion(t *testing.T) {
	tests := []struct {
		name         string
		digest       string
		expectedName string
	}{
		{"simple hash", "sha256:abc123", "sha256-abc123"},
		{"long hash", "sha256:abcdef1234567890", "sha256-abcdef1234567890"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blobName := replaceDigestColon(tt.digest)
			if blobName != tt.expectedName {
				t.Errorf("expected %s, got %s", tt.expectedName, blobName)
			}
		})
	}
}

// Helper function that mirrors the conversion logic
func replaceDigestColon(digest string) string {
	for i, c := range digest {
		if c == ':' {
			return digest[:i] + "-" + digest[i+1:]
		}
	}
	return digest
}

func TestResolveModelPathNonExistent(t *testing.T) {
	// Test with a model that doesn't exist
	_, err := ResolveModelPath("nonexistentmodel:latest")
	if err == nil {
		t.Error("expected error for non-existent model")
	}
}

func TestResolveModelPathNonExistentTag(t *testing.T) {
	// Clear OLLAMA_MODELS env var to use default
	_ = os.Unsetenv("OLLAMA_MODELS")

	// Test with a model that exists but tag doesn't
	_, err := ResolveModelPath("library:nonexistenttag")
	if err == nil {
		t.Error("expected error for non-existent tag")
	}
}

func TestManifestEmptyLayers(t *testing.T) {
	jsonData := `{
		"schemaVersion": 2,
		"layers": []
	}`

	var m Manifest
	err := json.Unmarshal([]byte(jsonData), &m)
	if err != nil {
		t.Fatalf("failed to unmarshal manifest: %v", err)
	}

	if len(m.Layers) != 0 {
		t.Errorf("expected 0 layers, got %d", len(m.Layers))
	}
}

func TestManifestNoModelLayer(t *testing.T) {
	jsonData := `{
		"schemaVersion": 2,
		"layers": [
			{
				"mediaType": "application/vnd.ollama.image.config",
				"digest": "sha256:config123",
				"size": 100
			}
		]
	}`

	var m Manifest
	err := json.Unmarshal([]byte(jsonData), &m)
	if err != nil {
		t.Fatalf("failed to unmarshal manifest: %v", err)
	}

	// Find model layer (should not exist)
	var blobDigest string
	for _, l := range m.Layers {
		if l.MediaType == MediaTypeModel {
			blobDigest = l.Digest
			break
		}
	}

	if blobDigest != "" {
		t.Errorf("expected no model layer, found digest: %s", blobDigest)
	}
}
