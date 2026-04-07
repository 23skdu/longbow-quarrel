//go:build darwin && metal

package engine

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/logger"
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
func (c *SlidingWindowKVCache) Init(ctx *device.Context, cfg config.Config) error {
	c.ctx = ctx
	c.config = cfg
	c.kvHeads = cfg.KVHeads
	c.headDim = cfg.HeadDim

	// Priority for window size:
	// 1. Explicit config.KVCacheSize (user override)
	// 2. config.WindowSize (model metadata for SWA)
	// 3. config.SeqLen (fallback to full context)
	c.windowSize = cfg.WindowSize
	if cfg.KVCacheSize > 0 {
		c.windowSize = cfg.KVCacheSize
		logger.Log.Info("KV Cache using explicit override size", "size", c.windowSize)
	} else if c.windowSize == 0 {
		c.windowSize = cfg.SeqLen
		if c.windowSize > 8192 {
			logger.Log.Warn("Capping excessively large context length", "original", cfg.SeqLen, "capped", 8192)
			c.windowSize = 8192
		}
	}
	if c.windowSize == 0 {
		c.windowSize = 2048
	}

	c.layers = cfg.Layers
	if c.layers == 0 {
		return fmt.Errorf("invalid config: layers=0")
	}

	c.kCache = make([]*device.Tensor, c.layers)
	c.vCache = make([]*device.Tensor, c.layers)

	kvDim := c.kvHeads * c.headDim
	if kvDim == 0 {
		return fmt.Errorf("invalid config: kvDim=0")
	}

	// Determine data type
	dt := device.DataTypeF32
	switch cfg.KVCacheType {
	case config.KVCacheF16:
		dt = device.DataTypeF16
	case config.KVCacheTQ1_0:
		dt = device.DataTypeTQ1_0
	case config.KVCacheTQ2_0:
		dt = device.DataTypeTQ2_0
	}

	// Allocate tensors for each layer with size = WindowSize
	for i := 0; i < c.layers; i++ {
		var k, v *device.Tensor
		if dt == device.DataTypeTQ1_0 || dt == device.DataTypeTQ2_0 {
			// For TurboQuant KV Cache, we use headDim as the block size
			k = ctx.NewTurboTensor(c.windowSize, kvDim, dt, c.headDim, 64)
			v = ctx.NewTurboTensor(c.windowSize, kvDim, dt, c.headDim, 64)
		} else {
			k = ctx.NewTensorWithType(c.windowSize, kvDim, dt)
			v = ctx.NewTensorWithType(c.windowSize, kvDim, dt)
		}

		if k == nil {
			c.Free()
			return fmt.Errorf("failed to allocate K cache for layer %d", i)
		}
		// Zero initialize to prevent stale data from previous allocations
		k.ZeroInit()
		c.kCache[i] = k

		if v == nil {
			c.Free()
			return fmt.Errorf("failed to allocate V cache for layer %d", i)
		}
		// Zero initialize to prevent stale data from previous allocations
		v.ZeroInit()
		c.vCache[i] = v
	}

	c.initialized = true

	// Record initial capacity
	// Accurate memory estimation for TurboQuant: numBlocks * bytesPerBlock
	// FP16/FP32 case: contextLen * heads * headDim * bytesPerElement
	bytesPerElement := 4 // F32
	if dt == device.DataTypeF16 {
		bytesPerElement = 2
	}
	totalBytes := int64(c.layers * 2 * c.windowSize * kvDim * bytesPerElement)
	
	if dt == device.DataTypeTQ1_0 || dt == device.DataTypeTQ2_0 {
		qjlRows := 64 // Fixed
		bytesPerBlock := c.headDim + qjlRows + 8
		totalBytes = int64(c.layers * 2 * c.windowSize * c.kvHeads * bytesPerBlock)
	}

	// OOM Guard: Check against MaxMemory
	if device.AllocatedBytes()+totalBytes > device.MaxGPUMemory {
		logger.Log.Warn("KV Cache exceeds memory budget", "requested_mb", totalBytes/1024/1024, "available_mb", (device.MaxGPUMemory-device.AllocatedBytes())/1024/1024)
		// We could attempt to clear pool or scale down window size here
	}

	metrics.RecordKVCacheStats(totalBytes, 0)

	return nil
}

// Update stores new K/V pairs at the specified position.
// For SlidingWindow, pos can be > windowSize.
// Detailed mapping is handled by the metal kernel using modulo logic.
func (c *SlidingWindowKVCache) Update(seqID string, layer, pos int, k, v *device.Tensor) error {
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
func (c *SlidingWindowKVCache) Get(seqID string, layer int) CacheView {
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
