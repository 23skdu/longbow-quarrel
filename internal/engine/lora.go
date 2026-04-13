package engine

import (
	"fmt"
	"strings"
	"sync"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// LoRAWeight holds the rank matrices for a single projection (e.g., blk.0.attn_q).
type LoRAWeight struct {
	A     *device.Tensor // [Rank, DimIn]
	B     *device.Tensor // [DimOut, Rank]
	Rank  int
	Alpha float32
}

// LoRAAdapter holds a collection of weights for various layers in a model.
type LoRAAdapter struct {
	ID      string
	Weights map[string]*LoRAWeight // key is layer/projection name
}

// LoRAManager handles loading and multiplying active adapters seamlessly during forward passes.
type LoRAManager struct {
	mu       sync.RWMutex
	adapters map[string]*LoRAAdapter
}

func NewLoRAManager() *LoRAManager {
	return &LoRAManager{
		adapters: make(map[string]*LoRAAdapter),
	}
}

// LoadAdapter parses LoRA weights from a GGUF file into VRAM.
func (lm *LoRAManager) LoadAdapter(ctx *device.CUDAContext, path string, id string) error {
	f, err := gguf.LoadFile(path)
	if err != nil {
		return fmt.Errorf("failed to load lora file: %w", err)
	}
	defer f.Close()

	lm.mu.Lock()
	defer lm.mu.Unlock()

	if _, exists := lm.adapters[id]; exists {
		return fmt.Errorf("adapter %s already loaded", id)
	}

	adapter := &LoRAAdapter{
		ID:      id,
		Weights: make(map[string]*LoRAWeight),
	}

	// LoRA metadata
	alpha := float32(8.0) // Default if missing
	if a, ok := f.KV["adapter.lora_alpha"].(float32); ok {
		alpha = a
	}

	// 1. Group tensors by layer
	tempA := make(map[string]*gguf.TensorInfo)
	tempB := make(map[string]*gguf.TensorInfo)

	for _, t := range f.Tensors {
		if strings.HasSuffix(t.Name, ".lora_A.weight") {
			base := strings.TrimSuffix(t.Name, ".lora_A.weight")
			tempA[base] = t
		} else if strings.HasSuffix(t.Name, ".lora_B.weight") {
			base := strings.TrimSuffix(t.Name, ".lora_B.weight")
			tempB[base] = t
		}
	}

	// 2. Load pairs
	for base, tA := range tempA {
		tB, ok := tempB[base]
		if !ok {
			continue
		}

		// Rank is the rows of A (or cols of B)
		// GGUF Dims: [DimIn, Rank] for A, [Rank, DimOut] for B
		r := int(tA.Dimensions[1])
		dimIn := int(tA.Dimensions[0])
		dimOut := int(tB.Dimensions[1])

		weight := &LoRAWeight{
			Rank:  r,
			Alpha: alpha,
		}

		// Upload A
		weight.A = ctx.NewTensorWithType(r, dimIn, device.DataTypeF16)
		if tA.Type == gguf.GGMLTypeF16 {
			_ = weight.A.LoadFrom(tA.Data)
		} else {
			// Convert F32 to F16 if needed
			// For now, assume F16
			_ = weight.A.LoadFrom(tA.Data) 
		}

		// Upload B
		weight.B = ctx.NewTensorWithType(dimOut, r, device.DataTypeF16)
		_ = weight.B.LoadFrom(tB.Data)

		adapter.Weights[base] = weight
	}

	lm.adapters[id] = adapter
	return nil
}

func (lm *LoRAManager) GetWeights(adapterID, layerName string) (*LoRAWeight, bool) {
	if adapterID == "" {
		return nil, false
	}
	lm.mu.RLock()
	defer lm.mu.RUnlock()

	adapter, ok := lm.adapters[adapterID]
	if !ok {
		return nil, false
	}
	w, ok := adapter.Weights[layerName]
	return w, ok
}
