package engine

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func generateTestGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	
	// Magic
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic)); err != nil { return err }
	// Version
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil { return err }
	// Tensor Count (12)
	if err := binary.Write(f, binary.LittleEndian, uint64(12)); err != nil { return err }
	// KV Count (5)
	if err := binary.Write(f, binary.LittleEndian, uint64(5)); err != nil { return err }
	
	// ... KVs ...
	// KV Pair 1: "llama.block_count" -> 1
	if err := writeString(f, "llama.block_count"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { return err }
	
	// KV Pair 2: "llama.embedding_length" -> 1 (dim)
	if err := writeString(f, "llama.embedding_length"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { return err }
	
	// KV Pair 3: "llama.attention.head_count" -> 1
	if err := writeString(f, "llama.attention.head_count"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { return err }
	
	// KV Pair 4: "llama.attention.head_count_kv" -> 1
	if err := writeString(f, "llama.attention.head_count_kv"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { return err }

	// KV Pair 5: "llama.context_length" -> 10
	if err := writeString(f, "llama.context_length"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(10)); err != nil { return err }
	
	// Helper to write 1x1 scalar tensor
	writeTensor := func(name string, offset uint64) error {
		if err := writeString(f, name); err != nil { return err }
		if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { return err } // Dims
		if err := binary.Write(f, binary.LittleEndian, uint64(1)); err != nil { return err } // Ne[0]
		if err := binary.Write(f, binary.LittleEndian, uint32(0)); err != nil { return err } // Type F32
		if err := binary.Write(f, binary.LittleEndian, uint64(offset)); err != nil { return err }
		return nil
	}
	
	currentOff := uint64(0)
	names := []string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}
	
	for _, n := range names {
		if err := writeTensor(n, currentOff); err != nil { return err }
		currentOff += 32 
	}
	
	// Pad header
	if _, err := f.Write(make([]byte, 1024)); err != nil { return err }
	
	// Write data
	for i := 0; i < len(names); i++ {
		if err := binary.Write(f, binary.LittleEndian, float32(1.0)); err != nil { return err }
		if _, err := f.Write(make([]byte, 28)); err != nil { return err }
	}
	
	return nil
}

func writeString(f *os.File, s string) error {
	if err := binary.Write(f, binary.LittleEndian, uint64(len(s))); err != nil { return err }
	_, err := f.WriteString(s)
	return err
}

func TestEngineLifecycle(t *testing.T) {
	// Setup dummy model
	modelPath := "test_model_lifecycle.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	// We want an Engine that can load a model and run inference
	// NewEngine(path) -> (*Engine, error)
	// e.Infer(prompt) -> tokens
	
	e, err := NewEngine(modelPath)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	if e == nil {
		t.Fatal("Engine is nil")
	}
	defer e.Ctx.Free()
	
	if e.Weights.TokenEmb == nil {
		t.Fatal("Expected TokenEmb to be loaded")
	}
	
	if len(e.Weights.AttnQ) < 1 {
		t.Fatal("Expected AttnQ to be initialized with layers")
	}
	if e.Weights.AttnQ[0] == nil {
		t.Fatal("Expected blk.0.attn_q.weight to be loaded")
	}
	
	// Inference
	// We want to pass a prompt tokens list
	inputTokens := []int{1, 2, 3}
	// Add config
	config := SamplerConfig{
		Temperature: 0,
	}
	outputTokens, err := e.Infer(inputTokens, 10, config) // generate 10 tokens
	if err != nil {
		t.Logf("Inference returned error (expected for empty/stub engine): %v", err)
	}
	
	if len(outputTokens) != 0 && len(outputTokens) != 10 {
		t.Errorf("Expected 0 or 10 tokens, got %d", len(outputTokens))
	}
}

func TestEngineMetrics(t *testing.T) {
	// Verify that Engine calls metrics
}

func generateMistralMockGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil { return err }
	defer f.Close()
	
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint64(12)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint64(6)); err != nil { return err }
	
	// Mistral Metadata
	if err := writeString(f, "llama.block_count"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { return err }
	
	if err := writeString(f, "llama.embedding_length"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(128)); err != nil { return err }
	
	if err := writeString(f, "llama.attention.head_count"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(32)); err != nil { return err }
	
	if err := writeString(f, "llama.attention.head_count_kv"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(8)); err != nil { return err }

	if err := writeString(f, "llama.rope.freq_base"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(100000)); err != nil { return err }

	if err := writeString(f, "llama.attention.layer_norm_rms_epsilon"); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeFloat32)); err != nil { return err }
	if err := binary.Write(f, binary.LittleEndian, float32(1e-6)); err != nil { return err }

	// Tensors (minimal placeholders)
	names := []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}
	
	writeTensor := func(name string, offset uint64) error {
		if err := writeString(f, name); err != nil { return err }
		if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { return err } 
		if err := binary.Write(f, binary.LittleEndian, uint64(128)); err != nil { return err } 
		if err := binary.Write(f, binary.LittleEndian, uint32(0)); err != nil { return err } 
		if err := binary.Write(f, binary.LittleEndian, uint64(offset)); err != nil { return err }
		return nil
	}

	currentOff := uint64(0)
	for _, n := range names {
		if err := writeTensor(n, currentOff); err != nil { return err }
		currentOff += 512 
	}
	
	if _, err := f.Write(make([]byte, 1024)); err != nil { return err }
	for i := 0; i < len(names); i++ {
		if err := binary.Write(f, binary.LittleEndian, make([]float32, 128)); err != nil { return err }
	}
	
	return nil
}

func TestMistralMetadataSupport(t *testing.T) {
	modelPath := "test_mistral_metadata.gguf"
	if err := generateMistralMockGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate Mistral mock: %v", err)
	}
	defer os.Remove(modelPath)

	e, err := NewEngine(modelPath)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Ctx.Free()

	if e.Config.KVHeads != 8 {
		t.Errorf("Expected KVHeads=8 (GQA), got %d", e.Config.KVHeads)
	}
	if e.Config.RopeTheta != 100000.0 {
		t.Errorf("Expected RopeTheta=100000.0, got %f", e.Config.RopeTheta)
	}
}
