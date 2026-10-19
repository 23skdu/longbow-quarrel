package engine

import (
	"fmt"
	"sort"
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
	Path    string                 // source GGUF path the adapter was loaded from
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
func (lm *LoRAManager) LoadAdapter(ctx *device.Context, path string, id string) error {
	f, err := gguf.LoadFile(path)
	if err != nil {
		return fmt.Errorf("failed to load lora file: %w", err)
	}
	defer func() { _ = f.Close() }()

	lm.mu.Lock()
	defer lm.mu.Unlock()

	if _, exists := lm.adapters[id]; exists {
		return fmt.Errorf("adapter %s already loaded", id)
	}

	adapter := &LoRAAdapter{
		ID:      id,
		Path:    path,
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

		// The GGUF dimensions are the number of elements in a row first,
		// so a LoRA A matrix of the logical shape [dimIn, rank] is stored
		// as [rank, dimIn], and a B matrix of [rank, dimOut] as
		// [dimOut, rank]. The dimensions used to be read the other way
		// round, so a real adapter was loaded with the rank and the
		// dimensions swapped.
		r := int(tA.Dimensions[0])      // #nosec G115 -- safe: LoRA dimensions are always small
		dimIn := int(tA.Dimensions[1])  // #nosec G115 -- safe: LoRA dimensions are always small
		dimOut := int(tB.Dimensions[0]) // #nosec G115 -- safe: LoRA dimensions are always small

		weight := &LoRAWeight{
			Rank:  r,
			Alpha: alpha,
		}

		// Upload A.
		//
		// The tensor is dequantized first: the file may hold float32
		// values, and loading those bytes into a half-precision tensor
		// read them as half-precision and produced garbage. The comment
		// in the old code said "convert F32 to F16 if needed" and then
		// did the same thing either way.
		weight.A = ctx.NewTensorWithType(r, dimIn, device.DataTypeF16)
		aValues, err := gguf.DequantizeTensor(tA)
		if err != nil {
			continue
		}
		_ = weight.A.LoadFrom(aValues)

		// Upload B.
		weight.B = ctx.NewTensorWithType(dimOut, r, device.DataTypeF16)
		bValues, err := gguf.DequantizeTensor(tB)
		if err != nil {
			continue
		}
		_ = weight.B.LoadFrom(bValues)

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

// AdapterInfo describes a loaded LoRA adapter.
type AdapterInfo struct {
	ID         string
	Path       string
	NumWeights int
}

// ListAdapters returns metadata for every currently loaded adapter, sorted
// by ID so the result is stable across calls.
func (lm *LoRAManager) ListAdapters() []AdapterInfo {
	lm.mu.RLock()
	defer lm.mu.RUnlock()

	out := make([]AdapterInfo, 0, len(lm.adapters))
	for id, a := range lm.adapters {
		info := AdapterInfo{ID: id, Path: a.Path, NumWeights: len(a.Weights)}
		out = append(out, info)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
