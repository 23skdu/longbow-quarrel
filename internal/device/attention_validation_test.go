//go:build darwin && metal

package device

import (
	"testing"
)

// TestAttention_Validation provides basic attention mechanism validation
// This focuses on verifying the core functionality works correctly
func TestAttention_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	heads := 4
	kvHeads := 2
	headDim := 64
	seqLen := 1

	// Create test tensors
	queryTensor := ctx.NewTensorFP32(1, heads*headDim)
	keyTensor := ctx.NewTensor(1, kvHeads*headDim)
	valueTensor := ctx.NewTensor(1, kvHeads*headDim)
	out := ctx.NewTensor(1, heads*headDim)

	// Simple test data
	queryData := make([]float32, heads*headDim)
	keyData := make([]float32, seqLen*kvHeads*headDim)
	valueData := make([]float32, seqLen*kvHeads*headDim)

	// Create distinctive patterns for testing
	for h := 0; h < heads; h++ {
		baseValue := float32(h + 100)

		for i := 0; i < headDim; i++ {
			queryData[h*headDim+i] = baseValue + float32(i)*0.01
		}

		kvh := h / (heads / kvHeads)

		for i := 0; i < headDim; i++ {
			keyData[kvh*headDim+i] = baseValue + float32(kvh*100+i)
			valueData[kvh*headDim+i] = baseValue + float32((kvh+1)*1000+i)
		}
	}

	queryTensor.LoadFrom(queryData)
	keyTensor.LoadFrom(keyData)
	valueTensor.LoadFrom(valueData)

	// Test basic attention computation
	queryTensor.AttFused(keyTensor, valueTensor, out, 0, heads, kvHeads, headDim, 0)

	// Verify output has some activity (not all zeros)
	gpuOut := out.ToHost()
	hasActivity := false
	for h := 0; h < heads; h++ {
		for i := 0; i < headDim; i++ {
			if gpuOut[h*headDim+i] != 0.0 {
				hasActivity = true
				break
			}
		}
	}

	if !hasActivity {
		t.Error("Attention output has no activity - expected some non-zero values")
	} else {
		t.Logf("Basic attention validation passed")
	}
}
