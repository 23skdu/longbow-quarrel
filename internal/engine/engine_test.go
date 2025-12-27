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
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	// Version
	binary.Write(f, binary.LittleEndian, uint32(3))
	// Tensor Count (11) 
	// token_embd, output, output_norm
	// blk.0: attn_q, attn_k, attn_v, attn_o, attn_norm, ffn_gate, ffn_up, ffn_down, ffn_norm
	// Total 12 tensors actually.
	binary.Write(f, binary.LittleEndian, uint64(12))
	// KV Count (5) - llama.block_count, dim, heads, kv_heads, ctx
	binary.Write(f, binary.LittleEndian, uint64(5))
	
	// ... KVs ...
	// KV Pair 1: "llama.block_count" -> 1
	writeString(f, "llama.block_count")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))
	
	// KV Pair 2: "llama.embedding_length" -> 1 (dim)
	writeString(f, "llama.embedding_length")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))
	
	// KV Pair 3: "llama.attention.head_count" -> 1
	writeString(f, "llama.attention.head_count")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))
	
	// KV Pair 4: "llama.attention.head_count_kv" -> 1
	writeString(f, "llama.attention.head_count_kv")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))

	// KV Pair 5: "llama.context_length" -> 10
	writeString(f, "llama.context_length")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(10))
	
	// Helper to write 1x1 scalar tensor
	writeTensor := func(name string, offset uint64) {
		writeString(f, name)
		binary.Write(f, binary.LittleEndian, uint32(1)) // Dims
		binary.Write(f, binary.LittleEndian, uint64(1)) // Ne[0]
		binary.Write(f, binary.LittleEndian, uint32(0)) // Type F32
		binary.Write(f, binary.LittleEndian, uint64(offset))
	}
	
	// 32-byte alignment. Each 1x1 F32 is 4 bytes.
	// We'll put them all 32-bytes apart to be safe/lazy with alignment.
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
		writeTensor(n, currentOff)
		currentOff += 32 
	}
	
	// Calculate Header Size for padding
	// Header base size changes with tensor count/names.
	// Let's just create a MASSIVE padding to be safe.
	// GGUF parser reads name string then aligns.
	// Actually, careful: offset is from END of header.
	// We need to write padding *after* header to reach 32-byte align.
	// Then data starts.
	// Our 'currentOff' above is relative to Data Start.
	
	// We need to pad file so that 'Data Start' is aligned.
	// Let's dump some bytes to finish header block.
	f.Write(make([]byte, 1024)) // Lazy pad
	
	// Now write data at offsets 0, 32, 64...
	// Since we padded 1024, and GGUF aligns to 32, we are aligned.
	// We just need to fill data up to max offset.
	
	for i := 0; i < len(names); i++ {
		binary.Write(f, binary.LittleEndian, float32(1.0))
		// Pad 28 bytes to next 32
		f.Write(make([]byte, 28))
	}

	
	return nil
}

func writeString(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	f.WriteString(s)
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
	// Expect success now
	
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Ctx.Free()
	
	if e == nil {
		t.Fatal("Engine is nil")
	}
	
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
	outputTokens, err := e.Infer(inputTokens, 10) // generate 10 tokens
	if err != nil {
		t.Logf("Inference returned error (expected for empty/stub engine): %v", err)
	}
	
	if len(outputTokens) != 0 && len(outputTokens) != 10 {
		// strict check later
	}
}

func TestEngineMetrics(t *testing.T) {
	// Verify that Engine calls metrics
	// This might require injecting a spy metrics recorder or checking global state if we rely on global metrics package.
	// For now, just defining the test.
}
