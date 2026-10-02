package ollama

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// writeModelTree materialises the directory layout Ollama uses on disk:
//
//	<root>/manifests/registry.ollama.ai/library/<name>/<tag>
//	<root>/blobs/sha256-<hash>
//
// It returns the absolute path of the blob backing that model. Writing a
// manifest with no model layer (emptyManifest=true) models a manifest that
// carries only config/template layers.
func writeModelTree(t *testing.T, root, name, tag, digest string, emptyManifest bool) string {
	t.Helper()

	manifest := Manifest{SchemaVersion: 2}
	if !emptyManifest {
		manifest.Layers = append(manifest.Layers, Layer{
			MediaType: "application/vnd.ollama.image.template",
			Digest:    "sha256:template0000",
			Size:      512,
		})
		// The model layer is intentionally not first, so the resolver has to
		// scan past a non-model layer rather than taking layers[0].
		manifest.Layers = append(manifest.Layers, Layer{
			MediaType: MediaTypeModel,
			Digest:    digest,
			Size:      4096,
		})
	}

	manifestDir := filepath.Join(root, "manifests", "registry.ollama.ai", "library", name)
	if err := os.MkdirAll(manifestDir, 0o755); err != nil {
		t.Fatalf("mkdir manifest dir: %v", err)
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(manifestDir, tag), data, 0o644); err != nil { // #nosec G306 -- test fixture
		t.Fatalf("write manifest: %v", err)
	}

	blobPath := filepath.Join(root, "blobs", strings.Replace(digest, ":", "-", 1))
	if err := os.MkdirAll(filepath.Dir(blobPath), 0o755); err != nil {
		t.Fatalf("mkdir blob dir: %v", err)
	}
	if err := os.WriteFile(blobPath, []byte("GGUF"), 0o644); err != nil { // #nosec G306 -- test fixture
		t.Fatalf("write blob: %v", err)
	}
	return blobPath
}

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

// TestResolveModelPath_ResolvesBlob drives the real resolver against an
// on-disk model tree and asserts it returns the backing blob. It also pins
// the two behaviours the old tests only re-implemented in test-local
// helpers: default tag selection and the digest ":" -> "-" blob naming.
func TestResolveModelPath_ResolvesBlob(t *testing.T) {
	const digest = "sha256:abcdef0123456789"

	for _, tag := range []string{"latest", "8b", "v1.0"} {
		t.Run("tag="+tag, func(t *testing.T) {
			root := t.TempDir()
			t.Setenv("OLLAMA_MODELS", root)
			want := writeModelTree(t, root, "llama3", tag, digest, false)

			model := "llama3"
			if tag != DefaultTag {
				model = "llama3:" + tag
			}

			got, err := ResolveModelPath(model)
			if err != nil {
				t.Fatalf("ResolveModelPath(%q) failed: %v", model, err)
			}
			if got != want {
				t.Errorf("ResolveModelPath(%q) = %q, want %q", model, got, want)
			}
			if !filepath.IsAbs(got) {
				t.Errorf("expected an absolute blob path, got %q", got)
			}
		})
	}
}

// TestResolveModelPath_PrefersModelLayer guards the layer scan: the manifest
// lists a template layer before the model layer, so a resolver that took
// layers[0] would return the template blob instead.
func TestResolveModelPath_PrefersModelLayer(t *testing.T) {
	const modelDigest = "sha256:1111111111111111"

	root := t.TempDir()
	t.Setenv("OLLAMA_MODELS", root)
	modelBlob := writeModelTree(t, root, "mistral", "latest", modelDigest, false)

	// Materialise the template blob too, so returning it is a real wrong answer.
	if err := os.WriteFile(
		filepath.Join(root, "blobs", "sha256-template0000"), []byte("TEMPLATE"), 0o644); err != nil { // #nosec G306 -- test fixture
		t.Fatalf("write template blob: %v", err)
	}

	got, err := ResolveModelPath("mistral")
	if err != nil {
		t.Fatalf("ResolveModelPath failed: %v", err)
	}
	if got != modelBlob {
		t.Errorf("got %q, want the model blob %q", got, modelBlob)
	}
}

func TestResolveModelPath_Errors(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T, root string)
		model   string
		wantErr string
	}{
		{
			name:    "manifest absent",
			setup:   func(*testing.T, string) {},
			model:   "ghost:latest",
			wantErr: "model manifest not found",
		},
		{
			name: "tag absent",
			setup: func(t *testing.T, root string) {
				writeModelTree(t, root, "llama3", "latest", "sha256:aaaa", false)
			},
			model:   "llama3:missing-tag",
			wantErr: "model manifest not found",
		},
		{
			name: "manifest is not valid json",
			setup: func(t *testing.T, root string) {
				dir := filepath.Join(root, "manifests", "registry.ollama.ai", "library", "broken")
				if err := os.MkdirAll(dir, 0o755); err != nil {
					t.Fatalf("mkdir: %v", err)
				}
				if err := os.WriteFile(filepath.Join(dir, "latest"), []byte("{not json"), 0o644); err != nil { // #nosec G306 -- test fixture
					t.Fatalf("write: %v", err)
				}
			},
			model:   "broken",
			wantErr: "invalid character",
		},
		{
			name: "manifest has no model layer",
			setup: func(t *testing.T, root string) {
				writeModelTree(t, root, "empty", "latest", "sha256:bbbb", true)
			},
			model:   "empty",
			wantErr: "no model layer found",
		},
		{
			name: "blob referenced by manifest is missing",
			setup: func(t *testing.T, root string) {
				writeModelTree(t, root, "dangling", "latest", "sha256:cccc", false)
				// Keep the manifest, drop the blob.
				if err := os.Remove(filepath.Join(root, "blobs", "sha256-cccc")); err != nil {
					t.Fatalf("remove blob: %v", err)
				}
			},
			model:   "dangling",
			wantErr: "model blob not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			t.Setenv("OLLAMA_MODELS", root)
			tt.setup(t, root)

			got, err := ResolveModelPath(tt.model)
			if err == nil {
				t.Fatalf("ResolveModelPath(%q) = %q, want error containing %q", tt.model, got, tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.wantErr)
			}
			if got != "" {
				t.Errorf("expected empty path on error, got %q", got)
			}
		})
	}
}

// TestResolveModelPath_DoesNotEscapeModelRoot checks that a model name
// carrying path separators cannot walk out of the manifests tree and resolve
// to an arbitrary file on disk.
func TestResolveModelPath_DoesNotEscapeModelRoot(t *testing.T) {
	root := t.TempDir()
	t.Setenv("OLLAMA_MODELS", root)

	decoy := filepath.Join(root, "outside.manifest")
	if err := os.WriteFile(decoy, []byte(`{"schemaVersion":2,"layers":[]}`), 0o644); err != nil { // #nosec G306 -- test fixture
		t.Fatalf("write decoy: %v", err)
	}

	got, err := ResolveModelPath("../outside")
	if err == nil {
		t.Fatalf("expected traversal to be rejected, got path %q", got)
	}
	if !strings.Contains(err.Error(), "model manifest not found") {
		t.Errorf("unexpected error for traversal attempt: %v", err)
	}
}

// TestGetOllamaDir_WindowsAndUnixShareLayout documents that both branches of
// GetOllamaDir currently return the same layout; if the Windows path ever
// diverges this test will need to branch with runtime.GOOS.
func TestGetOllamaDir_WindowsAndUnixShareLayout(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", "")
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home directory available: %v", err)
	}

	dir, err := GetOllamaDir()
	if err != nil {
		t.Fatalf("GetOllamaDir() failed: %v", err)
	}
	want := filepath.Join(home, ".ollama", "models")
	if dir != want {
		t.Errorf("GOOS=%s: GetOllamaDir() = %q, want %q", runtime.GOOS, dir, want)
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
