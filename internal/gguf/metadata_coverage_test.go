package gguf

import (
	"testing"
)

func TestMetadataAnalyzer_StatsCasts(t *testing.T) {
	f := &GGUFFile{
		KV: make(map[string]interface{}),
		Tensors: []*TensorInfo{
			{
				Name:       "test_f32",
				Type:       GGMLTypeF32,
				Dimensions: []uint64{1},
				Data:       []byte{0, 0, 0, 0},
			},
			{
				Name:       "test_f16",
				Type:       GGMLTypeF16,
				Dimensions: []uint64{1},
				Data:       []byte{0, 0},
			},
		},
	}
	a := NewMetadataAnalyzer(f)
	
	// Coverage for castToFloat32 and castToUint16
	_, _ = a.ComputeStats("test_f32")
	stats, _ := a.ComputeStats("test_f16")
	if stats.Name != "test_f16" {
		t.Error("failed to compute f16 stats")
	}
	
	// Coverage for ValidateTensors issues
	f.Tensors[0].Offset = 100 // mismatch
	issues, _ := a.ValidateTensors()
	if len(issues) == 0 {
		t.Error("expected validation issues for offset mismatch")
	}
	
	// Coverage for estimateMemoryUsage unknown type
	f.Tensors[0].Type = 999
	_ = a.estimateMemoryUsage()
}

func TestMetadataAnalyzer_MissingArchitecture(t *testing.T) {
	f := &GGUFFile{KV: make(map[string]interface{})}
	a := NewMetadataAnalyzer(f)
	report, _ := a.Analyze()
	if report.Architecture != "" {
		t.Error("expected empty architecture")
	}
}
