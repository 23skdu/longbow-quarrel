package gguf

import (
	"testing"
)

func TestMetadataAnalyzerBasic(t *testing.T) {
	file := &GGUFFile{
		KV: map[string]interface{}{
			"general.architecture":          "llama",
			"general.name":                  "test-model",
			"llama.context_length":          uint64(4096),
			"llama.hidden_size":             uint64(4096),
			"llama.attention.head_count":    uint64(32),
			"llama.attention.head_count_kv": uint64(8),
			"llama.feed_forward_length":     uint64(11008),
			"llama.expert_count":            uint64(0),
			"general.quantization_version":  uint32(1),
		},
		Tensors: []*TensorInfo{
			{
				Name:       "token_embd.weight",
				Dimensions: []uint64{32000, 4096},
				Type:       GGMLTypeF32,
				Offset:     0,
				Data:       make([]byte, 32000*4096*4),
			},
			{
				Name:       "output.weight",
				Dimensions: []uint64{4096, 32000},
				Type:       GGMLTypeF32,
				Offset:     524288000,
				Data:       make([]byte, 4096*32000*4),
			},
		},
		DataOffset: 0,
	}

	analyzer := NewMetadataAnalyzer(file)
	report, err := analyzer.Analyze()
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	if report.Architecture != "llama" {
		t.Errorf("Expected architecture 'llama', got '%s'", report.Architecture)
	}

	if report.ModelName != "test-model" {
		t.Errorf("Expected model name 'test-model', got '%s'", report.ModelName)
	}

	if report.ContextLength != 4096 {
		t.Errorf("Expected context length 4096, got %d", report.ContextLength)
	}

	if report.HiddenSize != 4096 {
		t.Errorf("Expected hidden size 4096, got %d", report.HiddenSize)
	}

	if report.AttentionHeads != 32 {
		t.Errorf("Expected attention heads 32, got %d", report.AttentionHeads)
	}

	if report.TensorCount != 2 {
		t.Errorf("Expected 2 tensors, got %d", report.TensorCount)
	}

	t.Logf("Report:\n%s", report.String())
}

func TestMetadataAnalyzerMOE(t *testing.T) {
	file := &GGUFFile{
		KV: map[string]interface{}{
			"general.architecture":            "mixtral",
			"mixtral.context_length":          uint64(32768),
			"mixtral.hidden_size":             uint64(4096),
			"mixtral.attention.head_count":    uint64(32),
			"mixtral.attention.head_count_kv": uint64(8),
			"mixtral.feed_forward_length":     uint64(14336),
			"mixtral.expert_count":            uint64(8),
			"mixtral.expert_used_top_k":       uint64(2),
			"general.quantization_version":    uint32(2),
		},
		Tensors: make([]*TensorInfo, 0),
	}

	analyzer := NewMetadataAnalyzer(file)
	report, err := analyzer.Analyze()
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	if report.Architecture != "mixtral" {
		t.Errorf("Expected architecture 'mixtral', got '%s'", report.Architecture)
	}

	if report.ExpertCount != 8 {
		t.Errorf("Expected expert count 8, got %d", report.ExpertCount)
	}

	if report.ExpertTopK != 2 {
		t.Errorf("Expected expert top-k 2, got %d", report.ExpertTopK)
	}

	if report.ContextLength != 32768 {
		t.Errorf("Expected context length 32768, got %d", report.ContextLength)
	}

	t.Logf("MOE Report:\n%s", report.String())
}

func TestValidateTensors(t *testing.T) {
	file := &GGUFFile{
		KV:         make(map[string]interface{}),
		Tensors:    make([]*TensorInfo, 0),
		DataOffset: 0,
	}

	analyzer := NewMetadataAnalyzer(file)
	issues, err := analyzer.ValidateTensors()
	if err != nil {
		t.Fatalf("ValidateTensors failed: %v", err)
	}

	if len(issues) != 0 {
		t.Errorf("Expected no issues, got: %v", issues)
	}
}

func TestFindMissingTensors(t *testing.T) {
	file := &GGUFFile{
		KV: make(map[string]interface{}),
		Tensors: []*TensorInfo{
			{
				Name: "token_embd.weight",
			},
			{
				Name: "output.weight",
			},
		},
	}

	analyzer := NewMetadataAnalyzer(file)

	required := []string{"token_embd.weight", "output.weight", "blk.0.attn.weight"}
	missing := analyzer.FindMissingTensors(required)

	if len(missing) != 1 {
		t.Errorf("Expected 1 missing tensor, got %d", len(missing))
	}

	if missing[0] != "blk.0.attn.weight" {
		t.Errorf("Expected 'blk.0.attn.weight', got '%s'", missing[0])
	}
}

func TestTensorStats(t *testing.T) {
	file := &GGUFFile{
		KV: make(map[string]interface{}),
		Tensors: []*TensorInfo{
			{
				Name:       "test.weight",
				Dimensions: []uint64{4, 4},
				Type:       GGMLTypeF32,
				Offset:     0,
				Data:       []byte{0x00, 0x00, 0xA0, 0x3F, 0x00, 0x00, 0x20, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x60, 0x40},
			},
		},
	}

	analyzer := NewMetadataAnalyzer(file)
	stats, err := analyzer.ComputeStats("test.weight")
	if err != nil {
		t.Fatalf("ComputeStats failed: %v", err)
	}

	if stats.Name != "test.weight" {
		t.Errorf("Expected name 'test.weight', got '%s'", stats.Name)
	}

	if stats.Type != "F32" {
		t.Errorf("Expected type 'F32', got '%s'", stats.Type)
	}

	if stats.ElementCount != 16 {
		t.Errorf("Expected 16 elements, got %d", stats.ElementCount)
	}

	t.Logf("Stats: %+v", stats)
}

func TestMemoryEstimation(t *testing.T) {
	file := &GGUFFile{
		KV: make(map[string]interface{}),
		Tensors: []*TensorInfo{
			{
				Name:       "test_f32",
				Dimensions: []uint64{100, 100},
				Type:       GGMLTypeF32,
				Offset:     0,
				Data:       make([]byte, 100*100*4),
			},
			{
				Name:       "test_f16",
				Dimensions: []uint64{100, 100},
				Type:       GGMLTypeF16,
				Offset:     40000,
				Data:       make([]byte, 100*100*2),
			},
		},
	}

	analyzer := NewMetadataAnalyzer(file)
	report, err := analyzer.Analyze()
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	expectedMemory := int64(100*100*4 + 100*100*2)
	if report.MemoryEstimate != expectedMemory {
		t.Errorf("Expected memory %d, got %d", expectedMemory, report.MemoryEstimate)
	}

	expectedParams := int64(20000)
	if report.TotalParameters != expectedParams {
		t.Errorf("Expected %d parameters, got %d", expectedParams, report.TotalParameters)
	}
}

func TestKVIntLookup(t *testing.T) {
	kv := map[string]interface{}{
		"key1": uint64(100),
		"key2": int64(200),
		"key3": uint32(300),
		"key4": int(400),
	}

	if getKVInt(kv, "key1") != 100 {
		t.Error("Failed to get uint64 value")
	}

	if getKVInt(kv, "key2") != 200 {
		t.Error("Failed to get int64 value")
	}

	if getKVInt(kv, "key3") != 300 {
		t.Error("Failed to get uint32 value")
	}

	if getKVInt(kv, "key4") != 400 {
		t.Error("Failed to get int value")
	}

	if getKVInt(kv, "nonexistent") != 0 {
		t.Error("Expected 0 for nonexistent key")
	}
}
