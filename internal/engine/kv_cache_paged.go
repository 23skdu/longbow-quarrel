//go:build darwin && metal

package engine

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// PagedKVCache implements a block-based KV cache inspired by PagedAttention.
// It manages a pool of memory blocks and a page table (BlockTable) to map logical tokens to physical blocks.
type PagedKVCache struct {
	ctx     *device.Context
	config  config.Config
	kvHeads int
	headDim int
	layers  int

	blockSize   int
	totalBlocks int

	// Memory Pools per layer
	// Shape: [TotalBlocks * BlockSize, KVHeads, HeadDim]
	kPools []*device.Tensor
	vPools []*device.Tensor

	// Block Allocation
	freeBlocks []int32
	mu         sync.Mutex

	// Block Table: Maps logical block index -> physical block index
	// We assume a single sequence for now, so one block table.
	// In a multi-sequence server, this would be a map[seqID][]int32.
	blockTableHost   []int32
	blockTableDevice *device.Tensor
	blockTableDirty  bool

	initialized bool
}

// Init initializes the paged cache
func (c *PagedKVCache) Init(ctx *device.Context, config config.Config) error {
	c.ctx = ctx
	c.config = config
	c.kvHeads = config.KVHeads
	c.headDim = config.HeadDim
	c.layers = config.Layers

	// Configurable block size
	c.blockSize = 16 // Default to 16
	if config.WindowSize > 0 && config.WindowSize%c.blockSize != 0 {
		// Adjust if needed, or ensure window fits blocks
	}

	// Total capacity
	// If WindowSize is set, we allocate enough for WindowSize.
	// If not, we allocate for SeqLen.
	capacity := config.WindowSize
	if capacity == 0 {
		capacity = config.SeqLen
	}
	if capacity == 0 {
		capacity = 4096 // Default
	}

	// Calculate number of blocks
	c.totalBlocks = (capacity + c.blockSize - 1) / c.blockSize

	// Ensure we align to block size
	capacity = c.totalBlocks * c.blockSize

	// Init Allocator
	c.freeBlocks = make([]int32, c.totalBlocks)
	for i := 0; i < c.totalBlocks; i++ {
		c.freeBlocks[i] = int32(c.totalBlocks - 1 - i) // Stack order
	}

	c.blockTableHost = make([]int32, 0, c.totalBlocks)
	// Allocate BlockTable Device Tensor
	// Size: totalBlocks (since logical can go up to totalBlocks)
	// We use Int32. Metal doesn't have Int32 tensor helper exposed nicely in what I saw,
	// but we can allocate raw bytes. 4 bytes per entry.
	// We'll treat it as a Q4K tensor? No.
	// We'll alloc raw buffer.
	// Use NewTensor with DataTypeF32 (4 bytes) but cast usage?
	// AttPaged expects `device const int *`.
	// F32 is 4 bytes. Int32 is 4 bytes.
	// We can use NewTensorFP32 and cast content.
	c.blockTableDevice = ctx.NewTensorFP32(1, c.totalBlocks)

	c.kPools = make([]*device.Tensor, c.layers)
	c.vPools = make([]*device.Tensor, c.layers)

	kvDim := c.kvHeads * c.headDim
	// poolElements := capacity * kvDim
	if kvDim == 0 {
		return fmt.Errorf("invalid config: kvDim=0")
	}

	for i := 0; i < c.layers; i++ {
		// NewTensor creates FP16 tensor
		k := ctx.NewTensor(capacity, kvDim)
		if k == nil {
			c.Free()
			return fmt.Errorf("failed to allocate K pool for layer %d", i)
		}
		c.kPools[i] = k

		v := ctx.NewTensor(capacity, kvDim)
		if v == nil {
			c.Free()
			return fmt.Errorf("failed to allocate V pool for layer %d", i)
		}
		c.vPools[i] = v
	}

	c.initialized = true

	// Initial stats
	totalBytes := int64(c.layers * 2 * capacity * kvDim * 2)
	metrics.RecordKVCacheStats(totalBytes, 0)

	return nil
}

func (c *PagedKVCache) allocateBlock() (int32, error) {
	if len(c.freeBlocks) == 0 {
		return -1, fmt.Errorf("OOM: no free blocks")
	}
	// Pop
	block := c.freeBlocks[len(c.freeBlocks)-1]
	c.freeBlocks = c.freeBlocks[:len(c.freeBlocks)-1]
	return block, nil
}

// Update stores new K/V pairs
func (c *PagedKVCache) Update(layer, pos int, k, v *device.Tensor) error {
	if !c.initialized {
		return fmt.Errorf("cache not initialized")
	}

	logicalBlockIdx := pos / c.blockSize
	blockOffset := pos % c.blockSize

	// Check if we need to allocate a new block
	// We only allocate if we are at strict new block start AND we haven't allocated it yet.
	// Or if logicalBlockIdx >= len(blockTable).

	// Note: Engine calls Update layer-by-layer.
	// We should only allocate ONCE per position (at layer 0).
	// But `Update` is called per layer.
	// So BlockTable Logic should be synchronized or shared?
	// It's shared `c.blockTableHost`.

	c.mu.Lock()
	if logicalBlockIdx >= len(c.blockTableHost) {
		// Need new block
		// Only allocate if layer == 0?
		// Or check if we already allocated for this step?
		// Since execution is sequential layer 0..N, layer 0 will hit this first.
		// Other layers will see logicalBlockIdx < len.
		// NOTE: If we re-generate (KV cache restoration), pos might go back?
		// Assuming append-only for now.

		phys, err := c.allocateBlock()
		if err != nil {
			c.mu.Unlock()
			return err
		}
		c.blockTableHost = append(c.blockTableHost, phys)
		c.blockTableDirty = true
	}
	physBlock := c.blockTableHost[logicalBlockIdx]
	c.mu.Unlock() // Unlock early

	// Calculate physical offset in the pool
	// Pool is [TotalBlocks * BlockSize, KVDim]
	// Physical Index = physBlock * BlockSize + blockOffset
	// However, `StoreKV` kernel takes `pos`.
	// We CANNOT use standard `StoreKV` easily because it uses modulo window size or linear pos.
	// Here `pos` maps to scattered `physBlock`.
	// We can cheat: Pass `PhysicalPos` to `StoreKV`?
	// `StoreKV` writes to `kCache[pos]`.
	// If current `kCache` is the *whole* pool, and we pass `physPos`, it writes to the right place!
	// `StoreKV` signature: `pos`.
	// `StoreKV` kernel treats `pos` as index into `kCache` (if window size large).
	// `metal.go`: `C.Metal_StoreKV_F16_Batch(..., kCache, ..., pos, ...)`
	// `StoreKV_F16`: `device half *dst = kCache + pos * kv_dim + ...`
	// Yes! So we just need to calculate `physPos`.

	physPos := int(physBlock)*c.blockSize + blockOffset

	// Use standard StoreKV logic but with calculated physical position
	// We treat "WindowSize" as TotalCapacity so valid range is full buffer.
	// We pass `physPos` as `pos`.

	// Note: We need to pass `physPos` to `Update`, but `Update` takes `pos` (logical).
	// But `TensorKVCache` uses `pos` logic.
	// `PagedKVCache` calculates `physPos`.

	kTarget := c.kPools[layer]
	vTarget := c.vPools[layer]

	// We treat the pool as a large contiguous buffer.
	// Passing `physPos` works.
	// WindowSize arg to StoreKV should be Capacity (TotalBlocks * BlockSize).
	capacity := c.totalBlocks * c.blockSize

	k.StoreKV(v, kTarget, vTarget, physPos, c.kvHeads, c.headDim, capacity)

	// Metrics
	// ... (Simplification: just track bytes)
	usedBytes := int64(c.layers * 2 * (pos + 1) * c.kvHeads * c.headDim * 2)
	metrics.KVCacheUsedBytes.Set(float64(usedBytes))

	return nil
}

// Get returns the K and V cache tensors for a layer
func (c *PagedKVCache) Get(layer int) CacheView {
	if !c.initialized || layer < 0 || layer >= len(c.kPools) {
		return CacheView{}
	}

	c.mu.Lock()
	if c.blockTableDirty {
		// Sync Host -> Device
		// BlockTableHost is []int32.
		// BlockTableDevice is F32 Tensor (4 bytes per element).
		// We can cast []int32 to []byte and load.
		// Go unsafe cast.
		data := unsafe.Slice((*byte)(unsafe.Pointer(&c.blockTableHost[0])), len(c.blockTableHost)*4)
		c.blockTableDevice.LoadFromRaw(data)
		c.blockTableDirty = false
	}
	c.mu.Unlock()

	metrics.KVCacheHits.Inc()
	return CacheView{
		K:          c.kPools[layer],
		V:          c.vPools[layer],
		BlockTable: c.blockTableDevice,
		BlockSize:  c.blockSize,
	}
}

func (c *PagedKVCache) Size() int {
	return c.totalBlocks * c.blockSize
}

func (c *PagedKVCache) Free() {
	if c.kPools != nil {
		for _, t := range c.kPools {
			if t != nil {
				t.Free()
			}
		}
		c.kPools = nil
	}
	if c.vPools != nil {
		for _, t := range c.vPools {
			if t != nil {
				t.Free()
			}
		}
		c.vPools = nil
	}
	if c.blockTableDevice != nil {
		c.blockTableDevice.Free()
		c.blockTableDevice = nil
	}
	c.initialized = false
}
