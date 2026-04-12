package engine

import (
	"fmt"
	"sync"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// LoRAAdapter holds the scaled rank matrices for low-rank fine-tuning.
type LoRAAdapter struct {
	ID    string
	Rank  int
	Alpha float32
	A     *device.Tensor
	B     *device.Tensor
}

// LoRAManager handles loading and multiplying active adapters seamlessly during forward passes.
type LoRAManager struct {
	mu           sync.RWMutex
	adapters     map[string]*LoRAAdapter
	activeAdapters []string
}

func NewLoRAManager() *LoRAManager {
	return &LoRAManager{
		adapters: make(map[string]*LoRAAdapter),
	}
}

// LoadAdapter parses LoRA weights into VRAM.
func (lm *LoRAManager) LoadAdapter(id string, rank int, alpha float32) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	if _, exists := lm.adapters[id]; exists {
		return fmt.Errorf("adapter %s already loaded", id)
	}

	// Implementation Stub: allocating rank matrices.
	lm.adapters[id] = &LoRAAdapter{
		ID:    id,
		Rank:  rank,
		Alpha: alpha,
	}
	return nil
}

// SetActive dynamically hot-swaps which adapters are applied to the linear layers.
func (lm *LoRAManager) SetActive(ids []string) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	lm.activeAdapters = ids
}
