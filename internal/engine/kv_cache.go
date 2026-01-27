//go:build darwin && metal

package engine

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// CacheView holds the tensors and metadata required for attention
type CacheView struct {
	K          *device.Tensor
	V          *device.Tensor
	BlockTable *device.Tensor // Optional: Paged Attention Block Table (Int32)
	BlockSize  int            // Optional: for Paged Attention
}

// KVCache abstraction allows switching between different caching strategies
type KVCache interface {
	Init(ctx *device.Context, config config.Config) error
	Update(layer, pos int, k, v *device.Tensor) error
	Get(layer int) CacheView
	Size() int
	Free()
}

// TensorKVCache is the standard contiguous tensor implementation
type TensorKVCache struct {
	ctx        *device.Context
	config     config.Config
	kvHeads    int
	headDim    int
	contextLen int
	layers     int

	// Cache tensors per layer
	kCache []*device.Tensor
	vCache []*device.Tensor

	initialized bool
}

// Init initializes the cache tensors
func (c *TensorKVCache) Init(ctx *device.Context, config config.Config) error {
	c.ctx = ctx
	c.config = config
	c.kvHeads = config.KVHeads
	c.headDim = config.HeadDim

	// Determine context length (priority: KVCacheSize config, then WindowSize, then SeqLen, default 2048)
	c.contextLen = config.KVCacheSize
	if c.contextLen == 0 {
		c.contextLen = config.WindowSize
	}
	if c.contextLen == 0 {
		c.contextLen = config.SeqLen
		// SAFETY: If SeqLen is huge (e.g. 1M for Nemotron), cap it to 8192 to prevent OOM
		// unless explicitly overridden by WindowSize or KVCacheSize.
		if c.contextLen > 8192 {
			logger.Log.Warn("Capping default context length", "original", c.contextLen, "new", 8192)
			c.contextLen = 8192
		}
	}
	if c.contextLen == 0 {
		c.contextLen = 2048 // Default fallback
	}

	c.layers = config.Layers
	if c.layers == 0 {
		return fmt.Errorf("invalid config: layers=0")
	}

	c.kCache = make([]*device.Tensor, c.layers)
	c.vCache = make([]*device.Tensor, c.layers)

	kvDim := c.kvHeads * c.headDim
	if kvDim == 0 {
		return fmt.Errorf("invalid config: kvDim=0")
	}

	// Allocate tensors for each layer
	for i := 0; i < c.layers; i++ {
		// K cache: [ContextLen, KVDim]
		k := ctx.NewTensor(c.contextLen, kvDim)
		if k == nil {
			c.Free() // Cleanup what we allocated so far
			return fmt.Errorf("failed to allocate K cache for layer %d", i)
		}
		c.kCache[i] = k

		// V cache: [ContextLen, KVDim]
		v := ctx.NewTensor(c.contextLen, kvDim)
		if v == nil {
			c.Free()
			return fmt.Errorf("failed to allocate V cache for layer %d", i)
		}
		c.vCache[i] = v
	}

	c.initialized = true

	// Record initial stats
	totalBytes := int64(c.layers * 2 * c.contextLen * kvDim * 2) // *2 for FP16 (2 bytes)
	metrics.RecordKVCacheStats(totalBytes, 0)

	return nil
}

// Update stores new K/V pairs at the specified position
func (c *TensorKVCache) Update(layer, pos int, k, v *device.Tensor) error {
	if !c.initialized {
		return fmt.Errorf("cache not initialized")
	}
	if layer < 0 || layer >= c.layers {
		return fmt.Errorf("invalid layer index: %d", layer)
	}
	if pos < 0 || pos >= c.contextLen {
		metrics.RecordKVCacheOutOfBounds(pos, c.contextLen)
		return fmt.Errorf("position out of bounds: %d (max %d)", pos, c.contextLen)
	}

	kTarget := c.kCache[layer]
	vTarget := c.vCache[layer]

	// Use tensor.StoreKV to copy data to GPU buffer
	// Note: StoreKV signature assumes it handles the update logic
	// prototype: StoreKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int)
	k.StoreKV(v, kTarget, vTarget, pos, c.kvHeads, c.headDim, c.contextLen)

	// Update used bytes metric (rough approximation: pos * dim * layers * 2 * 2bytes)
	// Theoretically we should track 'max_pos' to know how much is actually filled
	usedBytes := int64(c.layers * 2 * (pos + 1) * c.kvHeads * c.headDim * 2)
	metrics.KVCacheUsedBytes.Set(float64(usedBytes))

	return nil
}

// Get returns the K and V cache tensors for a layer
func (c *TensorKVCache) Get(layer int) CacheView {
	if !c.initialized || layer < 0 || layer >= len(c.kCache) {
		return CacheView{}
	}
	// Count a hit since we are retrieving the cache for attention
	metrics.KVCacheHits.Inc()
	return CacheView{
		K: c.kCache[layer],
		V: c.vCache[layer],
	}
}

// Size returns the configured context length
func (c *TensorKVCache) Size() int {
	return c.contextLen
}

// Free releases all GPU resources
func (c *TensorKVCache) Free() {
	if c.kCache != nil {
		for _, t := range c.kCache {
			if t != nil {
				// No Explicit Free method on Tensor in some versions, relying on Context.Free or similar?
				// Looking at metal.go, there is no specialized Free for pooled tensors usually,
				// but for large persistent buffers we might want to manually null them out
				// or if there is an explicit Free method.
				// Based on reading: device.Tensor has a Free method.
				t.Free()
			}
		}
		c.kCache = nil
	}
	if c.vCache != nil {
		for _, t := range c.vCache {
			if t != nil {
				t.Free()
			}
		}
		c.vCache = nil
	}
	c.initialized = false
}
