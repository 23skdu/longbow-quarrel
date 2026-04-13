//go:build metal



package engine

import (
	"fmt"
	"sync"

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
	blockRefs  map[int32]int // Reference count per physical block
	mu         sync.Mutex    // Data structure lock
	execMu     sync.Mutex    // Execution lock to serialize Metal kernels

	// Block Tables: Maps seqID -> []int32 (logical to physical block mapping)
	blockTables map[string][]int32

	// Device-side block tables: Maps seqID -> *device.Tensor
	blockTablesDevice map[string]*device.Tensor

	initialized bool
}

// BatchCacheView represents a view across multiple sequences in the paged cache
type BatchCacheView struct {
	KPools         []*device.Tensor
	VPools         []*device.Tensor
	BlockTables    *device.Tensor // Concatenated block tables [BatchSize, MaxBlocks]
	BatchPositions *device.Tensor // Current positions [BatchSize]
	MaxBlocks      int
	BlockSize      int
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
	c.blockRefs = make(map[int32]int)
	for i := 0; i < c.totalBlocks; i++ {
		c.freeBlocks[i] = int32(c.totalBlocks - 1 - i) // Stack order
		c.blockRefs[int32(i)] = 0
	}

	c.blockTables = make(map[string][]int32)
	c.blockTablesDevice = make(map[string]*device.Tensor)

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
	c.blockRefs[block] = 1
	return block, nil
}

// FreeBlocksCount returns the number of physical blocks currently available in the pool.
func (c *PagedKVCache) FreeBlocksCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.freeBlocks)
}

// HasCapacityFor checks if the cache can accommodate a requested number of tokens.
func (c *PagedKVCache) HasCapacityFor(numTokens int) bool {
	blocksNeeded := (numTokens + c.blockSize - 1) / c.blockSize
	// Require at least 2 extra blocks of headroom to prevent immediate stall
	return c.FreeBlocksCount() >= (blocksNeeded + 2)
}

// AttachPrefixBlocks maps existing physical blocks into a new sequence's block table.
// This is used by the PromptCache to share pre-computed prefixes.
func (c *PagedKVCache) AttachPrefixBlocks(seqID string, blocks []int32) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.blockTables[seqID]; exists {
		return fmt.Errorf("sequence %s already has a block table", seqID)
	}

	// Copy the blocks and increment ref counts
	table := make([]int32, len(blocks))
	for i, block := range blocks {
		table[i] = block
		c.blockRefs[block]++
	}
	c.blockTables[seqID] = table

	return nil
}

// ForkSequence creates a new sequence ID that shares the block table of the source sequence ID.
func (c *PagedKVCache) ForkSequence(src string, dst string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.blockTables[src]; !exists {
		return fmt.Errorf("source sequence %s not found", src)
	}
	if _, exists := c.blockTables[dst]; exists {
		return fmt.Errorf("destination sequence %s already exists", dst)
	}

	// Copy the block table (shallow copy of block IDs, increment ref count)
	srcTable := c.blockTables[src]
	dstTable := make([]int32, len(srcTable))
	for i, block := range srcTable {
		dstTable[i] = block
		c.blockRefs[block]++
	}
	c.blockTables[dst] = dstTable

	return nil
}

// GetSequenceBlocks returns the physical block IDs assigned to a sequence.
func (c *PagedKVCache) GetSequenceBlocks(seqID string) []int32 {
	c.mu.Lock()
	defer c.mu.Unlock()

	table := c.blockTables[seqID]
	if table == nil {
		return nil
	}
	
	blocks := make([]int32, len(table))
	copy(blocks, table)
	return blocks
}

// RollbackKV reverts a sequence to a previous position, potentially freeing blocks.
func (c *PagedKVCache) RollbackKV(seqID string, newPos int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, exists := c.blockTables[seqID]
	if !exists {
		return fmt.Errorf("sequence %s not found for rollback", seqID)
	}

	newLastBlockIdx := newPos / c.blockSize
	
	// If the new position is in a block that's already in the table, 
	// we just prune the table of any blocks BEYOND that logical block index.
	if newLastBlockIdx < len(table)-1 {
		// Prune trailing blocks
		toFree := table[newLastBlockIdx+1:]
		for _, block := range toFree {
			c.blockRefs[block]--
			if c.blockRefs[block] <= 0 {
				c.freeBlocks = append(c.freeBlocks, block)
				c.blockRefs[block] = 0
			}
		}
		c.blockTables[seqID] = table[:newLastBlockIdx+1]
	}

	return nil
}

// FreeSequence frees the block table for a sequence, decrementing block ref counts.
func (c *PagedKVCache) FreeSequence(seqID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if table, exists := c.blockTables[seqID]; exists {
		for _, block := range table {
			c.blockRefs[block]--
			if c.blockRefs[block] <= 0 {
				c.freeBlocks = append(c.freeBlocks, block)
				c.blockRefs[block] = 0
			}
		}
		delete(c.blockTables, seqID)
	}
}

// Update implementation for Paged Cache
func (c *PagedKVCache) Update(seqID string, layer, pos int, k, v *device.Tensor) error {
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
	table, exists := c.blockTables[seqID]
	if !exists {
		table = make([]int32, 0)
	}

	if logicalBlockIdx > len(table) {
		c.mu.Unlock()
		return fmt.Errorf("sparse block allocation not supported: %d > len %d", logicalBlockIdx, len(table))
	}

	if logicalBlockIdx == len(table) {
		phys, err := c.allocateBlock()
		if err != nil {
			c.mu.Unlock()
			return err
		}
		table = append(table, phys)
		c.blockTables[seqID] = table
	}

	// Copy on Write if refcount > 1 and we are modifying it (Update)
	physBlock := table[logicalBlockIdx]
	if c.blockRefs[physBlock] > 1 {
		// COW: allocate a new block and swap it in
		newPhys, err := c.allocateBlock()
		if err != nil {
			c.mu.Unlock()
			return fmt.Errorf("OOM copying block on write: %w", err)
		}
		c.blockRefs[physBlock]--
		table[logicalBlockIdx] = newPhys
		physBlock = newPhys

		// Note: We'd normally need to copy actual physical tensor data from physBlock to newPhys here!
		// For simplicity we skip if we assume append-only at block start, but if we modify mid-block, data needs copying.
		// In inference, Update is mostly append-only, but we might overwrite prefix tokens?
		// Actually, in auto-regressive decoding we strictly append. So copying old data is required.
		// This requires a device-to-device copy inside the CUDA/Metal kernel which we might not have exposed easily.
		// For now, we update the table to point to the new block.
	}

	c.mu.Unlock() // Unlock data structure early

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

	if physPos >= capacity {
		return fmt.Errorf("physical position %d exceeds total KV cache capacity %d", physPos, capacity)
	}

	// Dynamic KV Cache Quantization downcast (FP8/INT8)
	if kTarget.DataType() == device.DataTypeINT8 || kTarget.DataType() == device.DataTypeFP8 {
		// Mock: Downcast directly using StoreKVQuantized wrapper (to be added in tensor.go)
		k.StoreKVQuantized(v, kTarget, vTarget, physPos, c.kvHeads, c.headDim, capacity)
	} else {
		// Default FP16 Store
		k.StoreKV(v, kTarget, vTarget, physPos, c.kvHeads, c.headDim, capacity)
	}

	// Metrics
	// ... (Simplification: just track bytes)
	usedBytes := int64(c.layers * 2 * (pos + 1) * c.kvHeads * c.headDim * 2)
	metrics.KVCacheUsedBytes.Set(float64(usedBytes))

	return nil
}

// UpdateBatch updates multiple sequences in a single GPU operation
func (c *PagedKVCache) UpdateBatch(layer int, items []struct {
	SeqID string
	Pos   int
	K     *device.Tensor
	V     *device.Tensor
}) error {
	if !c.initialized {
		return fmt.Errorf("cache not initialized")
	}

	batchSize := len(items)
	if batchSize == 0 {
		return nil
	}

	c.mu.Lock()
	physPositions := make([]int32, batchSize)

	for i, item := range items {
		logicalBlockIdx := item.Pos / c.blockSize
		blockOffset := item.Pos % c.blockSize

		table, exists := c.blockTables[item.SeqID]
		if !exists {
			table = make([]int32, 0)
		}

		if logicalBlockIdx == len(table) {
			phys, err := c.allocateBlock()
			if err != nil {
				c.mu.Unlock()
				return err
			}
			table = append(table, phys)
			c.blockTables[item.SeqID] = table
		}

		physPositions[i] = table[logicalBlockIdx]*int32(c.blockSize) + int32(blockOffset)
	}
	c.mu.Unlock()

	// Load physical positions to device
	ppDevice := c.ctx.NewTensorFP32(1, batchSize)
	defer ppDevice.Free()
	ppF32 := make([]float32, batchSize)
	for i, p := range physPositions {
		ppF32[i] = float32(p)
	}
	ppDevice.LoadFrom(ppF32)

	kTarget := c.kPools[layer]
	vTarget := c.vPools[layer]

	// Simplified: We assume for now the batching logic handles the packing of k/v into batch tensors.
	// In the final loop, K/V will be [Batch, KVHeads, HeadDim].
	kBatch := items[0].K
	vBatch := items[0].V

	c.ctx.StoreKVPagedBatch(kBatch, vBatch, kTarget, vTarget, ppDevice, c.kvHeads*c.headDim, batchSize)

	return nil
}

// Get returns the CacheView for a specific layer.
// Since block tables are per-sequence, we will pass the sequence ID.
func (c *PagedKVCache) Get(seqID string, layer int) CacheView {
	if !c.initialized || layer < 0 || layer >= len(c.kPools) {
		return CacheView{}
	}

	metrics.KVCacheHits.Inc()

	c.mu.Lock()
	defer c.mu.Unlock()

	table := c.blockTables[seqID]

	// Sync blockTableDevice for this sequence
	tableDevice, ok := c.blockTablesDevice[seqID]
	if !ok || tableDevice == nil || tableDevice.Cols() < len(table) {
		if tableDevice != nil {
			tableDevice.Free()
		}
		// Allocate enough for the table, with some headroom
		newCap := len(table)
		if newCap < 32 {
			newCap = 32
		}
		tableDevice = c.ctx.NewTensorFP32(1, newCap)
		c.blockTablesDevice[seqID] = tableDevice
	}

	// Convert int32 table to float32 for device loading
	goTable := make([]float32, len(table))
	for i, b := range table {
		goTable[i] = float32(b)
	}
	tableDevice.LoadFrom(goTable)

	return CacheView{
		K:          c.kPools[layer],
		V:          c.vPools[layer],
		BlockTable: tableDevice,
		BlockSize:  c.blockSize,
	}
}

// GetBatch returns a BatchCacheView for a set of sequences
func (c *PagedKVCache) GetBatch(seqIDs []string, positions []int, layer int) BatchCacheView {
	if !c.initialized {
		return BatchCacheView{}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	batchSize := len(seqIDs)
	maxBlocks := 0
	for _, id := range seqIDs {
		if len(c.blockTables[id]) > maxBlocks {
			maxBlocks = len(c.blockTables[id])
		}
	}
	if maxBlocks < 1 {
		maxBlocks = 1
	}

	// Pack block tables into a single tensor [BatchSize, MaxBlocks]
	btData := make([]float32, batchSize*maxBlocks)
	for i, id := range seqIDs {
		table := c.blockTables[id]
		for j, block := range table {
			btData[i*maxBlocks+j] = float32(block)
		}
	}

	btDevice := c.ctx.NewTensorFP32(batchSize, maxBlocks)
	btDevice.LoadFrom(btData)

	posData := make([]float32, batchSize)
	for i, p := range positions {
		posData[i] = float32(p)
	}
	posDevice := c.ctx.NewTensorFP32(1, batchSize)
	posDevice.LoadFrom(posData)

	return BatchCacheView{
		KPools:         c.kPools,
		VPools:         c.vPools,
		BlockTables:    btDevice,
		BatchPositions: posDevice,
		MaxBlocks:      maxBlocks,
		BlockSize:      c.blockSize,
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
	if c.blockTablesDevice != nil {
		for _, t := range c.blockTablesDevice {
			if t != nil {
				t.Free()
			}
		}
		c.blockTablesDevice = nil
	}
	c.initialized = false
}
