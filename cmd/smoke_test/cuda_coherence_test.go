//go:build linux && cuda

package main

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func generateTestGGUFCUDA(path string, seqLen uint32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	_ = binary.Write(f, binary.LittleEndian, uint32(3))
	_ = binary.Write(f, binary.LittleEndian, uint64(12))
	_ = binary.Write(f, binary.LittleEndian, uint64(6))

	writeKV := func(key string, val interface{}) {
		_ = binary.Write(f, binary.LittleEndian, uint64(len(key)))
		_, _ = f.WriteString(key)
		switch v := val.(type) {
		case uint32:
			_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
			_ = binary.Write(f, binary.LittleEndian, v)
		case float32:
			_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeFloat32))
			_ = binary.Write(f, binary.LittleEndian, v)
		}
	}

	writeKV("llama.block_count", uint32(1))
	writeKV("llama.embedding_length", uint32(32))
	writeKV("llama.attention.head_count", uint32(4))
	writeKV("llama.attention.head_count_kv", uint32(4))
	writeKV("llama.context_length", seqLen)
	writeKV("llama.rope.freq_base", float32(10000.0))

	writeTensor := func(name string, rows, cols int) {
		_ = binary.Write(f, binary.LittleEndian, uint64(len(name)))
		_, _ = f.WriteString(name)
		_ = binary.Write(f, binary.LittleEndian, uint32(2))
		_ = binary.Write(f, binary.LittleEndian, uint64(cols))
		_ = binary.Write(f, binary.LittleEndian, uint64(rows))
		_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGMLTypeF32))
		offset := uint64(0)
		_ = binary.Write(f, binary.LittleEndian, offset)
	}

	tensorNames := []string{
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

	for _, n := range tensorNames {
		rows, cols := 32, 32
		if n == "token_embd.weight" || n == "output.weight" {
			rows = 100
			cols = 32
		}
		writeTensor(n, rows, cols)
	}

	_, _ = f.Write(make([]byte, 1024))

	data := make([]byte, 100*32*4)
	_, _ = f.Write(data)

	return nil
}

func TestCUDACoherenceWrapping(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping coherence test in short mode")
	}

	t.Skip("CUDA engine not implemented - this is a placeholder for CUDA coherence tests")
}

func TestCUDAMultiTokenCoherence(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping coherence test in short mode")
	}

	t.Skip("CUDA engine not implemented - this is a placeholder for CUDA coherence tests")
}

func TestCUDASelfConsistency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping coherence test in short mode")
	}

	t.Skip("CUDA engine not implemented - this is a placeholder for CUDA coherence tests")
}
