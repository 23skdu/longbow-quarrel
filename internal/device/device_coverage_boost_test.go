package device

import (
	"math"
	"testing"
)

func TestValidationFunctions(t *testing.T) {
	// ValidateInputTokens
	if _, err := ValidateInputTokens(nil, 100); err == nil {
		t.Error("expected error for empty tokens")
	}
	if _, err := ValidateInputTokens([]int{-1}, 100); err == nil {
		t.Error("expected error for negative token")
	}
	if _, err := ValidateInputTokens([]int{100}, 100); err == nil {
		t.Error("expected error for token >= vocabSize")
	}
	toks, err := ValidateInputTokens([]int{1, 2, 3}, 100)
	if err != nil || len(toks) != 3 {
		t.Errorf("unexpected error: %v", err)
	}

	// ModelConfig.Validate
	cfg := &ModelConfig{
		Dim:       64,
		HiddenDim: 128,
		Layers:    2,
		Heads:     4,
		KVHeads:   2,
		HeadDim:   16,
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("expected valid config, got: %v", err)
	}

	invalidConfigs := []*ModelConfig{
		{Dim: 0},
		{Dim: 64, Layers: 0},
		{Dim: 64, Layers: 1, Heads: 0},
		{Dim: 64, Layers: 1, Heads: 4, KVHeads: 0},
		{Dim: 64, Layers: 1, Heads: 4, KVHeads: 3}, // not divisible
		{Dim: 64, Layers: 1, Heads: 4, KVHeads: 2, HeadDim: 0},
		{Dim: 64, Layers: 1, Heads: 4, KVHeads: 2, HeadDim: 8}, // 4*8 != 64
		{Dim: 64, Layers: 1, Heads: 4, KVHeads: 2, HeadDim: 16, HiddenDim: 0},
	}
	for i, ic := range invalidConfigs {
		if err := ic.Validate(); err == nil {
			t.Errorf("expected error for invalid config %d", i)
		}
	}

	// ValidateLinearDimensions
	if err := ValidateLinearDimensions(10, 20); err == nil {
		t.Error("expected linear dimension mismatch")
	}
	if err := ValidateLinearDimensions(10, 10); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// ValidateAddDimensions
	if err := ValidateAddDimensions(2, 3, 2, 4); err == nil {
		t.Error("expected add dimension mismatch")
	}
	if err := ValidateAddDimensions(2, 3, 2, 3); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// NaN and Inf detection
	data := []float32{1.0, float32(math.NaN()), float32(math.Inf(1))}
	nanCount, infCount := CheckNumericalStability(data, "test")
	if nanCount != 1 || infCount != 1 {
		t.Errorf("CheckNumericalStability got nan=%d, inf=%d", nanCount, infCount)
	}

	info := DetectNaN(data, 5)
	if !info.HasNaN() || info.Count != 1 || !info.HasInf || info.InfCount != 1 {
		t.Errorf("DetectNaN failed: %+v", info)
	}
	if info.IsValid() {
		t.Error("expected info to not be valid")
	}
	validInfo := DetectNaN([]float32{1.0, 2.0}, 5)
	if !validInfo.IsValid() {
		t.Error("expected validInfo to be valid")
	}

	// ValidationError
	valErr := NewValidationError("op", "msg", "path")
	if valErr.Error() != "op: msg (path)" {
		t.Errorf("unexpected error string: %s", valErr.Error())
	}

	// Quant dimensions
	if err := ValidateQ4_0Dimensions(1, 31); err == nil {
		t.Error("expected error for Q4_0 with cols%32 != 0")
	}
	if err := ValidateQ4_0Dimensions(1, 32); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if err := ValidateQ4_KDimensions(1, 255); err == nil {
		t.Error("expected error for Q4_K with cols%256 != 0")
	}
	if err := ValidateQ4_KDimensions(1, 256); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if err := ValidateQ6_KDimensions(1, 255); err == nil {
		t.Error("expected error for Q6_K with cols%256 != 0")
	}
	if err := ValidateQ6_KDimensions(1, 256); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPrecomputedTables(t *testing.T) {
	rot := GetPrecomputedRotation(64)
	if len(rot) != 64*64 {
		t.Errorf("expected 64x64 rotation, got len %d", len(rot))
	}
	rotCached := GetPrecomputedRotation(64)
	if len(rotCached) != 64*64 {
		t.Errorf("expected 64x64 cached rotation, got len %d", len(rotCached))
	}

	signs := GetPrecomputedQJLSigns(64)
	if len(signs) != 64 {
		t.Errorf("expected 64 signs, got len %d", len(signs))
	}
	signsCached := GetPrecomputedQJLSigns(64)
	if len(signsCached) != 64 {
		t.Errorf("expected 64 signs, got len %d", len(signsCached))
	}
}
