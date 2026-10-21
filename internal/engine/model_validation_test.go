package engine

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// TestValidateModelFileRejectsACorruptTensorTable checks the load-time
// validation of the tensor table: an offset that does not describe the data
// behind it is reported with the tensor's name instead of surfacing later as
// a nil dereference or a read past the end of the data section.
func TestValidateModelFileRejectsACorruptTensorTable(t *testing.T) {
	file := &gguf.GGUFFile{
		KV: map[string]interface{}{"general.alignment": uint32(32)},
		Tensors: []*gguf.TensorInfo{
			{Name: "token_embd.weight", Type: gguf.GGMLTypeF32, Dimensions: []uint64{4, 4}, Offset: 0},
			{Name: "blk.0.attn_q.weight", Type: gguf.GGMLTypeF32, Dimensions: []uint64{4, 4}, Offset: 9999},
		},
		DataOffset: 0,
	}

	err := validateModelFile(file)
	if err == nil {
		t.Fatal("expected a corrupt tensor table to be rejected")
	}
	if !strings.Contains(err.Error(), "blk.0.attn_q.weight") {
		t.Errorf("the error does not name the corrupt tensor: %v", err)
	}
}

// TestValidateModelFileRejectsAModelWithoutEmbeddings checks that a model
// with no token_embd.weight fails the load with a message that names the
// tensor: every backend dereferences the embedding table on its first
// forward pass.
func TestValidateModelFileRejectsAModelWithoutEmbeddings(t *testing.T) {
	file := &gguf.GGUFFile{
		KV:         map[string]interface{}{"general.alignment": uint32(32)},
		Tensors:    []*gguf.TensorInfo{},
		DataOffset: 0,
	}

	err := validateModelFile(file)
	if err == nil {
		t.Fatal("expected a model without an embedding table to be rejected")
	}
	if !strings.Contains(err.Error(), "token_embd.weight") {
		t.Errorf("the error does not name the missing tensor: %v", err)
	}
}

// TestNewCPUEngineRejectsAnInvalidModelFile checks that the validation runs
// on the real load path, not only in the helper's unit tests.
func TestNewCPUEngineRejectsAnInvalidModelFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "no_embeddings.gguf")
	kv := map[string]interface{}{
		"general.architecture":   "llama",
		"general.alignment":      uint32(32),
		"llama.block_count":      uint32(2),
		"llama.embedding_length": uint32(8),
	}
	if err := gguf.WriteFile(path, kv, nil, 0); err != nil {
		t.Fatalf("failed to write the invalid model: %v", err)
	}

	_, err := NewCPUEngine(path, config.Default())
	if err == nil {
		t.Fatal("expected the load to be rejected")
	}
	if !strings.Contains(err.Error(), "token_embd.weight") {
		t.Errorf("the error does not name the missing tensor: %v", err)
	}
}
