//go:build darwin && metal
// +build darwin,metal

package engine

import (
	"encoding/binary"
	"os"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestMetalEngineInternalState(t *testing.T) {
	modelPath := "test_model_internal.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	me, ok := e.(*metalEngine)
	if !ok {
		t.Fatal("Expected *metalEngine when build tag 'metal' is present")
	}

	if me.weights.TokenEmb == nil {
		t.Fatal("Expected TokenEmb to be loaded")
	}
	if len(me.weights.AttnQ) < 1 {
		t.Fatal("Expected AttnQ to be initialized")
	}
}

func TestMistralMetadataSupport(t *testing.T) {
	modelPath := "test_mistral_metadata.gguf"
	if err := generateMistralMockGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate Mistral mock: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	ctx := device.NewContext()
	defer ctx.Free()

	me := &metalEngine{
		ctx:       ctx,
		config:    conf,
		weights:   &LlamaWeights{},
		ActLogger: NewActivationLogger(),
		SeqMgr:    NewSequenceManager(),
		// Channels initialized in NewMetalEngine but manual struct init for test
		stopChan: make(chan struct{}),
		doneChan: make(chan struct{}),
	}

	err := me.loadModel(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	if me.config.KVHeads != 8 {
		t.Errorf("Expected KVHeads=8 (GQA), got %d", me.config.KVHeads)
	}
	if me.config.RopeTheta != 100000.0 {
		t.Errorf("Expected RopeTheta=100000.0, got %f", me.config.RopeTheta)
	}
}

func TestNemotronStyleLoading(t *testing.T) {
	modelPath := "test_nemotron_loading.gguf"
	f, err := os.Create(modelPath)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Magic + Version
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	binary.Write(f, binary.LittleEndian, uint32(3))
	// Tensor Count (3)
	binary.Write(f, binary.LittleEndian, uint64(3))
	// KV Count (4)
	binary.Write(f, binary.LittleEndian, uint64(4))

	// KV: llama.embedding_length = 128
	writeString(f, "llama.embedding_length")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(128))

	// KV: llama.block_count = 1
	writeString(f, "llama.block_count")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))

	// KV: llama.attention.head_count = 1
	writeString(f, "llama.attention.head_count")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))

	// KV: general.architecture = "nemotron"
	writeString(f, "general.architecture")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeString))
	writeString(f, "nemotron")

	// Tensors: nemotron.token_embd.weight, nemotron.output_norm.weight, nemotron.output.weight
	names := []string{
		"nemotron.token_embd.weight",
		"nemotron.output_norm.weight",
		"nemotron.output.weight",
	}

	for i, name := range names {
		writeString(f, name)
		binary.Write(f, binary.LittleEndian, uint32(2))             // Dims
		binary.Write(f, binary.LittleEndian, uint64(128))           // Ne[0]
		binary.Write(f, binary.LittleEndian, uint64(1))             // Ne[1]
		binary.Write(f, binary.LittleEndian, uint32(0))             // Type F32
		binary.Write(f, binary.LittleEndian, uint64(uint64(i)*512)) // Offset
	}

	f.Write(make([]byte, 1024)) // Header pad
	for i := 0; i < 3; i++ {
		binary.Write(f, binary.LittleEndian, make([]float32, 128))
	}
	f.Close()
	defer os.Remove(modelPath)

	conf := config.Default()
	ctx := device.NewContext()
	defer ctx.Free()

	me := &metalEngine{
		ctx:       ctx,
		config:    conf,
		weights:   &LlamaWeights{},
		ActLogger: NewActivationLogger(),
		SeqMgr:    NewSequenceManager(),
		stopChan:  make(chan struct{}),
		doneChan:  make(chan struct{}),
	}

	err = me.loadModel(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	if me.weights.TokenEmb == nil {
		t.Error("Expected TokenEmb to be loaded via suffix match")
	}
	if me.weights.OutputNorm == nil {
		t.Error("Expected OutputNorm to be loaded via suffix match")
	}
	if me.weights.Output == nil {
		t.Error("Expected Output to be loaded via suffix match")
	}
}
