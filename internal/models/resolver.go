package models

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/ollama"
)

// SearchDirs returns the standard model directories checked by Quarrel in order of priority.
func SearchDirs() []string {
	var dirs []string

	home, err := os.UserHomeDir()
	if err == nil {
		dirs = append(dirs,
			filepath.Join(home, ".cache", "llmfit", "models"),
			filepath.Join(home, ".cache", "llama.cpp"),
			filepath.Join(home, ".cache", "huggingface", "hub"),
			filepath.Join(home, ".cache", "quarrel", "models"),
			filepath.Join(home, ".local", "share", "nomic.ai", "GPT4All"),
		)
	}

	// Local project search directories
	dirs = append(dirs,
		"models",
		".",
	)

	return dirs
}

// ResolveModelPath attempts to locate a model file given an absolute path, relative path,
// model name, filename, or Ollama model tag.
func ResolveModelPath(nameOrPath string) (string, error) {
	if nameOrPath == "" {
		return "", fmt.Errorf("model path or name cannot be empty")
	}

	// 1. Check if the provided path directly exists on disk
	cleanPath := filepath.Clean(nameOrPath)
	if fi, err := os.Stat(cleanPath); err == nil && !fi.IsDir() {
		return cleanPath, nil
	}

	// If path starts with ~/, expand user home directory
	if strings.HasPrefix(nameOrPath, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			expanded := filepath.Join(home, nameOrPath[2:])
			if fi, err := os.Stat(expanded); err == nil && !fi.IsDir() {
				return expanded, nil
			}
		}
	}

	// 2. Check standard search directories for exact file match
	searchDirs := SearchDirs()
	for _, dir := range searchDirs {
		candidate := filepath.Join(dir, nameOrPath)
		if fi, err := os.Stat(candidate); err == nil && !fi.IsDir() {
			return candidate, nil
		}
		// If user omitted .gguf, try adding it
		if !strings.HasSuffix(strings.ToLower(nameOrPath), ".gguf") {
			candidateGGUF := candidate + ".gguf"
			if fi, err := os.Stat(candidateGGUF); err == nil && !fi.IsDir() {
				return candidateGGUF, nil
			}
		}
	}

	// 3. Substring / Prefix match in cache directories for .gguf files
	normalizedQuery := strings.ToLower(filepath.Base(nameOrPath))
	normalizedQuery = strings.TrimSuffix(normalizedQuery, ".gguf")

	for _, dir := range searchDirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			fileName := entry.Name()
			if !strings.HasSuffix(strings.ToLower(fileName), ".gguf") {
				continue
			}
			lowerName := strings.ToLower(fileName)
			if strings.Contains(lowerName, normalizedQuery) {
				return filepath.Join(dir, fileName), nil
			}
		}
	}

	// 4. Fallback to Ollama manifest resolver
	if ollamaPath, err := ollama.ResolveModelPath(nameOrPath); err == nil {
		if fi, err := os.Stat(ollamaPath); err == nil && !fi.IsDir() {
			return ollamaPath, nil
		}
	}

	return "", fmt.Errorf("model %q not found in filesystem or standard cache directories (~/.cache/llmfit/models, ~/.cache/llama.cpp, ~/.cache/huggingface, ~/.ollama/models)", nameOrPath)
}
