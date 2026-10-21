//go:build darwin && metal

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

// generateMistralMockGGUF writes a small Mistral-style model: grouped
// query attention with eight key heads and a rope base of a hundred
// thousand.
//
// The helper was missing: the test that calls it never compiled, because
// nothing built the Metal test files. The Metal type check now vets the
// test files as well, which is how the gap was found.
func generateMistralMockGGUF(path string) error {
	const (
		vocab   = 8
		dim     = 64
		heads   = 8
		kvHeads = 8
		headDim = 8
		layers  = 1
		ffnDim  = 32
	)

	attnDim := heads * headDim
	kvDim := kvHeads * headDim

	zeros := func(n int) []byte { return make([]byte, n*4) }

	kv := map[string]interface{}{
		"general.architecture":                   "llama",
		"general.alignment":                      uint32(32),
		"llama.block_count":                      uint32(layers),
		"llama.embedding_length":                 uint32(dim),
		"llama.attention.head_count":             uint32(heads),
		"llama.attention.head_count_kv":          uint32(kvHeads),
		"llama.feed_forward_length":              uint32(ffnDim),
		"llama.rope.dimension_count":             uint32(headDim),
		"llama.rope.freq_base":                   float32(100000),
		"llama.context_length":                   uint32(256),
		"llama.vocab_size":                       uint32(vocab),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
		"tokenizer.ggml.tokens":                  []string{"a", "b", "c", "d", "e", "f", "g", "h"},
		"tokenizer.ggml.scores":                  []float32{0, -1, -2, -3, -4, -5, -6, -7},
		"tokenizer.ggml.token_type":              []int32{2, 1, 1, 1, 1, 1, 1, 1},
	}

	tensors := []gguf.TensorData{
		{Name: "token_embd.weight", Dimensions: []uint64{uint64(dim), uint64(vocab)}, Type: gguf.GGMLTypeF32, Data: zeros(dim * vocab)},
		{Name: "output_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: zeros(dim)},
		{Name: "output.weight", Dimensions: []uint64{uint64(dim), uint64(vocab)}, Type: gguf.GGMLTypeF32, Data: zeros(dim * vocab)},
	}

	for l := 0; l < layers; l++ {
		prefix := "blk." + string(rune('0'+l)) + "."
		tensors = append(tensors,
			gguf.TensorData{Name: prefix + "attn_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: zeros(dim)},
			gguf.TensorData{Name: prefix + "attn_q.weight", Dimensions: []uint64{uint64(dim), uint64(attnDim)}, Type: gguf.GGMLTypeF32, Data: zeros(attnDim * dim)},
			gguf.TensorData{Name: prefix + "attn_k.weight", Dimensions: []uint64{uint64(dim), uint64(kvDim)}, Type: gguf.GGMLTypeF32, Data: zeros(kvDim * dim)},
			gguf.TensorData{Name: prefix + "attn_v.weight", Dimensions: []uint64{uint64(dim), uint64(kvDim)}, Type: gguf.GGMLTypeF32, Data: zeros(kvDim * dim)},
			gguf.TensorData{Name: prefix + "attn_output.weight", Dimensions: []uint64{uint64(attnDim), uint64(dim)}, Type: gguf.GGMLTypeF32, Data: zeros(dim * attnDim)},
			gguf.TensorData{Name: prefix + "ffn_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: zeros(dim)},
			gguf.TensorData{Name: prefix + "ffn_gate.weight", Dimensions: []uint64{uint64(dim), uint64(ffnDim)}, Type: gguf.GGMLTypeF32, Data: zeros(ffnDim * dim)},
			gguf.TensorData{Name: prefix + "ffn_up.weight", Dimensions: []uint64{uint64(dim), uint64(ffnDim)}, Type: gguf.GGMLTypeF32, Data: zeros(ffnDim * dim)},
			gguf.TensorData{Name: prefix + "ffn_down.weight", Dimensions: []uint64{uint64(ffnDim), uint64(dim)}, Type: gguf.GGMLTypeF32, Data: zeros(dim * ffnDim)},
		)
	}

	return gguf.WriteFile(path, kv, tensors, 32)
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
