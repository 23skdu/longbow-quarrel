package ollama

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	DefaultTag     = "latest"
	MediaTypeModel = "application/vnd.ollama.image.model"
)

type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	Layers        []Layer `json:"layers"`
}

type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
}

func GetOllamaDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	// Check for OLLAMA_MODELS env var
	if env := os.Getenv("OLLAMA_MODELS"); env != "" {
		return env, nil
	}

	// Default: ~/.ollama/models
	if runtime.GOOS == "windows" {
		// Not supporting windows specific logic detailed check right now
		return filepath.Join(home, ".ollama", "models"), nil
	}
	return filepath.Join(home, ".ollama", "models"), nil
}

// ResolveModelPath attempts to find the GGUF blob path for a given model name.
// modelName can be "llama3", "llama3:latest", "llama3:8b", etc.
// Assumes registry.ollama.ai/library unless specified otherwise (full image names not fully supported yet).
func ResolveModelPath(modelName string) (string, error) {
	// Parse model name
	// Short name: llama3 -> registry.ollama.ai/library/llama3/latest
	// Tagged: llama3:instruct -> registry.ollama.ai/library/llama3/instruct

	parts := strings.Split(modelName, ":")
	var name, tag string
	if len(parts) == 1 {
		name = parts[0]
		tag = DefaultTag
	} else {
		name = parts[0]
		tag = parts[1]
	}

	baseDir, err := GetOllamaDir()
	if err != nil {
		return "", err
	}

	// Construct manifest path
	// Standard install: ~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>
	// Note: User might use custom registry, but we assume default for short names
	manifestPath := filepath.Join(baseDir, "manifests", "registry.ollama.ai", "library", name, tag)

	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		// Try without library? No, ollama structure usually includes it for official models
		return "", fmt.Errorf("model manifest not found at %s", manifestPath)
	}

	// Read Manifest
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return "", err
	}

	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return "", err
	}

	// Find the model layer
	var blobDigest string
	for _, l := range m.Layers {
		if l.MediaType == MediaTypeModel {
			blobDigest = l.Digest
			break
		}
	}

	if blobDigest == "" {
		return "", fmt.Errorf("no model layer found in manifest")
	}

	// Blob path: blobs/sha256-<hash>
	// Digest is "sha256:hash" -> replace ':' with '-'
	blobName := strings.Replace(blobDigest, ":", "-", 1)
	blobPath := filepath.Join(baseDir, "blobs", blobName)

	if _, err := os.Stat(blobPath); os.IsNotExist(err) {
		return "", fmt.Errorf("model blob not found at %s", blobPath)
	}

	return blobPath, nil
}
