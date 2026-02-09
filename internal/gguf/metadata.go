package gguf

import (
	"fmt"
	"math"
)

type MetadataAnalyzer struct {
	file *GGUFFile
}

func NewMetadataAnalyzer(file *GGUFFile) *MetadataAnalyzer {
	return &MetadataAnalyzer{file: file}
}

type AnalysisReport struct {
	Architecture     string
	ModelName        string
	ContextLength    int
	HiddenSize       int
	AttentionHeads   int
	KVHeads          int
	IntermediateSize int
	ExpertCount      int
	ExpertTopK       int
	Quantization     string
	TotalParameters  int64
	TensorCount      int
	MemoryEstimate   int64
}

func (a *MetadataAnalyzer) Analyze() (*AnalysisReport, error) {
	report := &AnalysisReport{
		TensorCount: len(a.file.Tensors),
	}

	arch, ok := a.file.KV["general.architecture"].(string)
	if ok {
		report.Architecture = arch
	}

	name, ok := a.file.KV["general.name"].(string)
	if ok {
		report.ModelName = name
	}

	report.ContextLength = int(getKVInt(a.file.KV, report.Architecture+".context_length", "general.context_length"))
	if report.ContextLength == 0 {
		report.ContextLength = 2048
	}

	report.HiddenSize = int(getKVInt(a.file.KV, report.Architecture+".hidden_size", report.Architecture+".embedding_length"))

	report.AttentionHeads = int(getKVInt(a.file.KV, report.Architecture+".attention.head_count", ""))

	kvHeads := getKVInt(a.file.KV, report.Architecture+".attention.head_count_kv", report.Architecture+".attention.kv_head_count")
	if kvHeads == 0 {
		kvHeads = uint64(report.AttentionHeads)
	}
	report.KVHeads = int(kvHeads)

	report.IntermediateSize = int(getKVInt(a.file.KV, report.Architecture+".feed_forward_length", report.Architecture+".intermediate_size"))

	report.ExpertCount = int(getKVInt(a.file.KV, report.Architecture+".expert_count", ""))

	report.ExpertTopK = int(getKVInt(a.file.KV, report.Architecture+".expert_used_top_k", report.Architecture+".expert_top_k"))

	if quantVal, ok := a.file.KV["general.quantization_version"].(uint32); ok {
		report.Quantization = fmt.Sprintf("Q%d", quantVal*4)
	} else {
		report.Quantization = "Unknown"
	}

	var totalParams int64 = 0
	for _, t := range a.file.Tensors {
		elements := int64(1)
		for _, d := range t.Dimensions {
			elements *= int64(d)
		}
		totalParams += elements
	}
	report.TotalParameters = totalParams

	report.MemoryEstimate = a.estimateMemoryUsage()

	return report, nil
}

func (a *MetadataAnalyzer) estimateMemoryUsage() int64 {
	var totalBytes int64 = 0
	for _, t := range a.file.Tensors {
		size := t.SizeBytes()
		if size > 0 {
			totalBytes += int64(size)
		} else {
			elements := int64(1)
			for _, d := range t.Dimensions {
				elements *= int64(d)
			}
			switch t.Type {
			case GGMLTypeF32:
				totalBytes += elements * 4
			case GGMLTypeF16:
				totalBytes += elements * 2
			default:
				totalBytes += elements * 4
			}
		}
	}
	return totalBytes
}

func getKVInt(kv map[string]interface{}, keys ...string) uint64 {
	for _, key := range keys {
		if val, ok := kv[key]; ok {
			switch v := val.(type) {
			case uint64:
				return v
			case int64:
				return uint64(v)
			case uint32:
				return uint64(v)
			case int:
				return uint64(v)
			}
		}
	}
	return 0
}

func (r *AnalysisReport) String() string {
	return fmt.Sprintf(`GGUF Model Analysis Report
============================
Architecture:      %s
Model Name:       %s
Context Length:   %d
Hidden Size:     %d
Attention Heads:  %d
KV Heads:         %d
Intermediate:     %d
Expert Count:     %d
Expert Top-K:     %d
Quantization:     %s
Total Tensors:    %d
Total Parameters: %d (%.2fB)
Memory Estimate:  %.2f GB
`,
		r.Architecture,
		r.ModelName,
		r.ContextLength,
		r.HiddenSize,
		r.AttentionHeads,
		r.KVHeads,
		r.IntermediateSize,
		r.ExpertCount,
		r.ExpertTopK,
		r.Quantization,
		r.TensorCount,
		r.TotalParameters,
		float64(r.TotalParameters)/1e9,
		float64(r.MemoryEstimate)/1e9,
	)
}

func (a *MetadataAnalyzer) ValidateTensors() ([]string, error) {
	var issues []string

	expectedOffset := a.file.DataOffset
	for i, t := range a.file.Tensors {
		if t.Offset != expectedOffset {
			issues = append(issues,
				fmt.Sprintf("Tensor %d (%s): expected offset %d, got %d",
					i, t.Name, expectedOffset, t.Offset))
		}

		expectedSize := t.SizeBytes()
		if expectedSize == 0 {
			issues = append(issues,
				fmt.Sprintf("Tensor %d (%s): unknown size for type %s",
					i, t.Name, t.Type))
		}

		expectedOffset += expectedSize
	}

	return issues, nil
}

func (a *MetadataAnalyzer) FindMissingTensors(requiredLayers []string) []string {
	existing := make(map[string]bool)
	for _, t := range a.file.Tensors {
		existing[t.Name] = true
	}

	var missing []string
	for _, layer := range requiredLayers {
		if !existing[layer] {
			missing = append(missing, layer)
		}
	}

	return missing
}

type TensorStats struct {
	Name         string
	Type         string
	Dimensions   []uint64
	ElementCount uint64
	SizeBytes    uint64
	MinValue     float64
	MaxValue     float64
	MeanValue    float64
	HasNaN       bool
	HasInf       bool
}

func (a *MetadataAnalyzer) ComputeStats(tensorName string) (*TensorStats, error) {
	var tensor *TensorInfo
	for _, t := range a.file.Tensors {
		if t.Name == tensorName {
			tensor = t
			break
		}
	}

	if tensor == nil {
		return nil, fmt.Errorf("tensor %s not found", tensorName)
	}

	stats := &TensorStats{
		Name:       tensor.Name,
		Type:       tensor.Type.String(),
		Dimensions: tensor.Dimensions,
	}

	elements := uint64(1)
	for _, d := range tensor.Dimensions {
		elements *= d
	}
	stats.ElementCount = elements
	stats.SizeBytes = tensor.SizeBytes()

	switch tensor.Type {
	case GGMLTypeF32:
		data := castToFloat32(tensor.Data)
		if len(data) > 0 {
			stats.MinValue = float64(data[0])
			stats.MaxValue = float64(data[0])
			sum := float64(0)
			for _, v := range data {
				if math.IsNaN(float64(v)) {
					stats.HasNaN = true
				}
				if math.IsInf(float64(v), 0) {
					stats.HasInf = true
				}
				if float64(v) < stats.MinValue {
					stats.MinValue = float64(v)
				}
				if float64(v) > stats.MaxValue {
					stats.MaxValue = float64(v)
				}
				sum += float64(v)
			}
			stats.MeanValue = sum / float64(len(data))
		}
	case GGMLTypeF16:
		data := castToUint16(tensor.Data)
		if len(data) > 0 {
			stats.MinValue = float64(data[0])
			stats.MaxValue = float64(data[0])
			sum := float64(0)
			for _, v := range data {
				if float64(v) < stats.MinValue {
					stats.MinValue = float64(v)
				}
				if float64(v) > stats.MaxValue {
					stats.MaxValue = float64(v)
				}
				sum += float64(v)
			}
			stats.MeanValue = sum / float64(len(data))
		}
	}

	return stats, nil
}

func castToFloat32(data []byte) []float32 {
	n := len(data) / 4
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 |
			uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		result[i] = math.Float32frombits(bits)
	}
	return result
}

func castToUint16(data []byte) []uint16 {
	n := len(data) / 2
	result := make([]uint16, n)
	for i := 0; i < n; i++ {
		result[i] = uint16(data[i*2]) | uint16(data[i*2+1])<<8
	}
	return result
}
