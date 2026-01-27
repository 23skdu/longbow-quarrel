//go:build darwin && metal

package engine

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// SlidingWindowKVCache implements a fixed-size storage that wraps around
// effectively maintaining the last N tokens where N is the window size.
type SlidingWindowKVCache struct {
	ctx        *device.Context
	config     config.Config
	kvHeads    int
	headDim    int
	windowSize int // The physical size of the buffer
	layers     int

	// Cache tensors per layer
	kCache []*device.Tensor
	vCache []*device.Tensor

	initialized bool
}

// Init initializes the sliding window cache tensors
func (c *SlidingWindowKVCache) Init(ctx *device.Context, config config.Config) error {
	c.ctx = ctx
	c.config = config
	c.kvHeads = config.KVHeads
	c.headDim = config.HeadDim

	// Use WindowSize from config.
	// If WindowSize is 0, this cache strategy degenerates to a standard cache (if SeqLen fits)
	// or fails if usage exceeds SeqLen.
	// But generally SlidingWindow cache implies we enforce a window.
	c.windowSize = config.WindowSize
	if c.windowSize == 0 {
		// Fallback to SeqLen if WindowSize not explicit, but treat as window
		c.windowSize = config.SeqLen
	}
	if c.windowSize == 0 {
		c.windowSize = 2048 // Default
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

	// Allocate tensors for each layer with size = WindowSize
	for i := 0; i < c.layers; i++ {
		k := ctx.NewTensor(c.windowSize, kvDim)
		if k == nil {
			c.Free()
			return fmt.Errorf("failed to allocate K cache for layer %d", i)
		}
		c.kCache[i] = k

		v := ctx.NewTensor(c.windowSize, kvDim)
		if v == nil {
			c.Free()
			return fmt.Errorf("failed to allocate V cache for layer %d", i)
		}
		c.vCache[i] = v
	}

	c.initialized = true

	// Record initial capacity
	totalBytes := int64(c.layers * 2 * c.windowSize * kvDim * 2)
	metrics.RecordKVCacheStats(totalBytes, 0)

	return nil
}

// Update stores new K/V pairs at the specified position.
// For SlidingWindow, pos can be > windowSize.
// Detailed mapping is handled by the metal kernel using modulo logic.
func (c *SlidingWindowKVCache) Update(layer, pos int, k, v *device.Tensor) error {
	if !c.initialized {
		return fmt.Errorf("cache not initialized")
	}
	if layer < 0 || layer >= c.layers {
		return fmt.Errorf("invalid layer index: %d", layer)
	}

	// We allow pos to grow indefinitely (conceptually).
	// But we check negative.
	if pos < 0 {
		return fmt.Errorf("negative position: %d", pos)
	}

	kTarget := c.kCache[layer]
	vTarget := c.vCache[layer]

	// k.StoreKV(v, kCache, vCache, pos, heads, headDim, windowSize)
	// The metal kernel (StoreKV_F16) uses pos % windowSize for physical storage
	k.StoreKV(v, kTarget, vTarget, pos, c.kvHeads, c.headDim, c.windowSize)

	// Metric update
	wrapped := pos >= c.windowSize
	metrics.RecordKVCacheSlidingWindow(c.windowSize, pos, wrapped)

	// Update used bytes
	// If full, it's max capacity. If filling, it's (pos+1).
	usedSlots := pos + 1
	if usedSlots > c.windowSize {
		usedSlots = c.windowSize
	}
	usedBytes := int64(c.layers * 2 * usedSlots * c.kvHeads * c.headDim * 2)
	metrics.KVCacheUsedBytes.Set(float64(usedBytes))

	if wrapped {
		metrics.KVCacheEvictions.Inc()
	}

	return nil
}

// Get returns the K and V cache tensors for a layer
func (c *SlidingWindowKVCache) Get(layer int) CacheView {
	if !c.initialized || layer < 0 || layer >= len(c.kCache) {
		return CacheView{}
	}
	metrics.KVCacheHits.Inc()
	return CacheView{
		K: c.kCache[layer],
		V: c.vCache[layer],
	}
}

// Size returns the window size
func (c *SlidingWindowKVCache) Size() int {
	return c.windowSize
}

// Free releases all GPU resources
func (c *SlidingWindowKVCache) Free() {
	if c.kCache != nil {
		for _, t := range c.kCache {
			if t != nil {
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
