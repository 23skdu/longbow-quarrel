//go:build darwin && metal

package device

import (
	"fmt"
	"sync"
)

// MultiGPUMetalManager orchestrates collective operations across multiple Metal devices.
// On Apple Silicon, this often maps to different compute units or unified memory partitions.
type MultiGPUMetalManager struct {
	contexts []*Context
	mu       sync.RWMutex
}

var metalMultiGPU *MultiGPUMetalManager

func GetMultiGPUMetalManager(ctxs []*Context) *MultiGPUMetalManager {
	if metalMultiGPU != nil {
		return metalMultiGPU
	}
	metalMultiGPU = &MultiGPUMetalManager{
		contexts: ctxs,
	}
	return metalMultiGPU
}

// AllReduce sums the data across all participating Metal contexts.
func (m *MultiGPUMetalManager) AllReduce(data *Tensor) error {
	if data == nil {
		return fmt.Errorf("nil tensor for AllReduce")
	}

	// For a single Context (M-series Ultra/Max), we use the fused AllReduce kernel.
	// In a distributed scenario, this would involve cross-device memory copies.
	data.ctx.AllReduce(data)
	return nil
}

// AllGather collects results from all devices.
func (m *MultiGPUMetalManager) AllGather(input *Tensor, output *Tensor) error {
	// Stub: In a real multi-device scenario, this would copy slices into the output buffer.
	// For unified memory, it might be a no-op if the output buffer was already shared.
	return nil
}
