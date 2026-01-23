package engine

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// ActivationLog stores layer-by-layer activations for debugging
type ActivationLog struct {
	Prompt      string             `json:"prompt"`
	Tokens      []int              `json:"tokens"`
	Embedding   []float32          `json:"embedding"` // First 100 values
	Layers      []LayerLog         `json:"layers"`
	FinalLogits map[string]float32 `json:"final_logits"` // String keys for JSON
}

// LayerLog captures activations for a single transformer layer
type LayerLog struct {
	Idx        int       `json:"idx"`
	QMax       float32   `json:"q_max"`
	KMax       float32   `json:"k_max"`
	VMax       float32   `json:"v_max"`
	AttnOutMax float32   `json:"attn_out_max"`
	FFNOutMax  float32   `json:"ffn_out_max"`
	QSample    []float32 `json:"q_sample"` // First 10 values
	KSample    []float32 `json:"k_sample"`
	VSample    []float32 `json:"v_sample"`

	// NaN/Inf detection
	QNaNCount    int `json:"q_nan_count"`
	QInfCount    int `json:"q_inf_count"`
	KNaNCount    int `json:"k_nan_count"`
	KInfCount    int `json:"k_inf_count"`
	VNaNCount    int `json:"v_nan_count"`
	VInfCount    int `json:"v_inf_count"`
	AttnNaNCount int `json:"attn_nan_count"`
	AttnInfCount int `json:"attn_inf_count"`
	FFNNaNCount  int `json:"ffn_nan_count"`
	FFNInfCount  int `json:"ffn_inf_count"`
}

// ActivationLogger manages activation logging during inference
type ActivationLogger struct {
	enabled bool
	log     *ActivationLog
}

// NewActivationLogger creates a new activation logger
func NewActivationLogger() *ActivationLogger {
	return &ActivationLogger{
		enabled: false,
		log:     nil,
	}
}

// Enable turns on activation logging
func (al *ActivationLogger) Enable(prompt string, tokens []int) {
	al.enabled = true
	al.log = &ActivationLog{
		Prompt:      prompt,
		Tokens:      tokens,
		Embedding:   make([]float32, 0),
		Layers:      make([]LayerLog, 0),
		FinalLogits: make(map[string]float32),
	}
}

// IsEnabled returns whether logging is active
func (al *ActivationLogger) IsEnabled() bool {
	return al.enabled
}

// LogEmbedding captures embedding output
func (al *ActivationLogger) LogEmbedding(data []float32) {
	if !al.enabled {
		return
	}
	// Store first 100 values
	limit := 100
	if len(data) < limit {
		limit = len(data)
	}
	al.log.Embedding = make([]float32, limit)
	copy(al.log.Embedding, data[:limit])
}

// LogLayer captures layer activations
func (al *ActivationLogger) LogLayer(layerIdx int, qMax, kMax, vMax, attnMax, ffnMax float32,
	qSample, kSample, vSample, qData, kData, vData, attnData, ffnData []float32) {
	if !al.enabled {
		return
	}

	layer := LayerLog{
		Idx:        layerIdx,
		QMax:       qMax,
		KMax:       kMax,
		VMax:       vMax,
		AttnOutMax: attnMax,
		FFNOutMax:  ffnMax,
		QSample:    make([]float32, len(qSample)),
		KSample:    make([]float32, len(kSample)),
		VSample:    make([]float32, len(vSample)),
	}

	copy(layer.QSample, qSample)
	copy(layer.KSample, kSample)
	copy(layer.VSample, vSample)

	// NaN/Inf detection
	if qData != nil {
		layer.QNaNCount, layer.QInfCount = countNaNInf(qData)
	}
	if kData != nil {
		layer.KNaNCount, layer.KInfCount = countNaNInf(kData)
	}
	if vData != nil {
		layer.VNaNCount, layer.VInfCount = countNaNInf(vData)
	}
	if attnData != nil {
		layer.AttnNaNCount, layer.AttnInfCount = countNaNInf(attnData)
	}
	if ffnData != nil {
		layer.FFNNaNCount, layer.FFNInfCount = countNaNInf(ffnData)
	}

	al.log.Layers = append(al.log.Layers, layer)
}

// LogLogits captures final logits for specific tokens
func (al *ActivationLogger) LogLogits(logits []float32, tokenIDs []int) {
	if !al.enabled {
		return
	}

	for _, tid := range tokenIDs {
		if tid < len(logits) {
			al.log.FinalLogits[fmt.Sprintf("%d", tid)] = logits[tid]
		}
	}
}

// SaveToFile writes the activation log to a JSON file
func (al *ActivationLogger) SaveToFile(filename string) error {
	if !al.enabled || al.log == nil {
		return fmt.Errorf("no activation log to save")
	}

	data, err := json.MarshalIndent(al.log, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	fmt.Printf("Activation log saved to: %s\n", filename)
	return nil
}

// GetSampleFromTensor extracts first N values from tensor data
func GetSampleFromTensor(data []float32, n int) []float32 {
	if len(data) < n {
		n = len(data)
	}
	sample := make([]float32, n)
	copy(sample, data[:n])
	return sample
}

// GetMaxFromTensor finds maximum absolute value in tensor
func GetMaxFromTensor(data []float32) float32 {
	maxVal := float32(0)
	for _, v := range data {
		absV := v
		if absV < 0 {
			absV = -absV
		}
		if absV > maxVal {
			maxVal = absV
		}
	}
	return maxVal
}

// countNaNInf counts NaN and Inf values in a float32 slice
func countNaNInf(data []float32) (nanCount, infCount int) {
	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nanCount++
		} else if math.IsInf(float64(v), 0) {
			infCount++
		}
	}
	return nanCount, infCount
}
