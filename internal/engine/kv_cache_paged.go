package engine

import (
	"encoding/binary"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/mathx"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/simd"
)

// PagedKVCache implements a block-based KV cache inspired by PagedAttention.
// It manages a pool of memory blocks and a page table (BlockTable) to map logical tokens to physical blocks.
type PagedKVCache struct {
	ctx       *device.Context
	config    config.Config
	kvHeads   int
	headDim   int
	layers    int
	Precision device.DataType

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

	// Dirty flags: Maps seqID -> true if block table needs to be reloaded to device
	dirty map[string]bool

	// TurboQuant matrices for KV cache compression
	tqRotation *device.Tensor
	tqQJL      *device.Tensor
	qjlRows    int

	// Dynamic Block Scales for quantized KV cache (FP8 / Q8_0)
	kScales []*device.Tensor
	vScales []*device.Tensor

	// StreamingLLM attention sink and sliding window configuration per sequence
	streamingParams map[string]StreamingLLMParams
	seqMaxPos       map[string]int

	// Dynamic KV Cache Tiering & Memory Defragmentation
	offloadedSeqs map[string]*OffloadedSequence
	lastAccess    map[string]time.Time

	initialized bool
}

// kvCacheDataType maps the configured KV cache type onto the element type
// the device tensors are allocated with. F16 is the default: it is what the
// GPU kernels read and write, and the F32 pool is only correct on the CPU.
func kvCacheDataType(t config.KVCacheType) device.DataType {
	switch t {
	case config.KVCacheF32:
		return device.DataTypeF32
	case config.KVCacheTQ1_0:
		return device.DataTypeTQ1_0
	case config.KVCacheTQ2_0:
		return device.DataTypeTQ2_0
	case config.KVCacheFP8:
		return device.DataTypeFP8
	case config.KVCacheQ8_0:
		return device.DataTypeQ8_0
	case config.KVCacheF16:
		return device.DataTypeF16
	default:
		return device.DataTypeF16
	}
}

// kvCacheTypeName is the name of a KV cache type as it is spelled on the
// command line.
func kvCacheTypeName(t config.KVCacheType) string {
	return t.String()
}

// ParseKVCacheType maps the -kv-cache-type flag onto a KV cache type.
func ParseKVCacheType(name string) (config.KVCacheType, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "f16":
		return config.KVCacheF16, nil
	case "f32":
		return config.KVCacheF32, nil
	case "tq1_0", "tq1":
		return config.KVCacheTQ1_0, nil
	case "tq2_0", "tq2":
		return config.KVCacheTQ2_0, nil
	case "fp8":
		return config.KVCacheFP8, nil
	case "q8_0", "q8":
		return config.KVCacheQ8_0, nil
	default:
		return config.KVCacheF16, fmt.Errorf("unknown KV cache type %q: expected one of "+
			"f32, f16, tq1_0, tq2_0, fp8, q8_0", name)
	}
}

// OffloadedBlockData holds layer-wise tensor data for an offloaded physical block.
type OffloadedBlockData struct {
	Layer int
	K     []float32
	V     []float32
}

// OffloadedSequence preserves an inactive sequence's KV state in host RAM.
type OffloadedSequence struct {
	SeqID       string
	BlockData   map[int]OffloadedBlockData
	NumBlocks   int
	OffloadedAt time.Time
}

// StreamingLLMParams defines attention sink and sliding window configuration for StreamingLLM.
type StreamingLLMParams struct {
	Enabled      bool
	SinkBlocks   int // Number of initial physical blocks preserved as attention sinks (default 4)
	WindowBlocks int // Maximum number of rolling window blocks
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

	// The precision of the cache is selected with -kv-cache-type. It used
	// to be left at the zero value, so the FP8 and Q8_0 pools and the
	// quantized attention kernel could never be reached from a shipped
	// binary.
	c.Precision = kvCacheDataType(config.KVCacheType)

	// Not every precision is implemented on every backend. The CPU
	// reference attention reads the pool as float32; the CUDA kernels
	// read the F16 pool, or a quantized pool through the quantized
	// attention kernel. Asking for the wrong one is an error, not a
	// silent reinterpretation of the bytes.
	switch c.Precision {
	case device.DataTypeFP8, device.DataTypeQ8_0:
		if !ctx.SupportsQuantizedKVCache() {
			return fmt.Errorf("the %s KV cache is only implemented on the CUDA backend",
				kvCacheTypeName(config.KVCacheType))
		}
	case device.DataTypeF32:
		if ctx.DeviceID() >= 0 {
			return fmt.Errorf("the F32 KV cache is only implemented on the CPU backend; " +
				"the GPU attention kernels read the F16 pool")
		}
	}

	// Configurable block size
	c.blockSize = 16 // Default to 16
	if config.BlockSize > 0 {
		c.blockSize = config.BlockSize
	}
	// Total capacity
	capacity := config.KVCacheSize
	if capacity == 0 {
		capacity = config.WindowSize
	}
	if capacity == 0 {
		capacity = config.SeqLen
	}
	if capacity == 0 || capacity > 4096 {
		if config.KVCacheSize > 0 {
			capacity = config.KVCacheSize
		} else {
			capacity = 4096
		}
	}

	// Calculate number of blocks
	if config.TotalBlocks > 0 {
		c.totalBlocks = config.TotalBlocks
	} else {
		c.totalBlocks = (capacity + c.blockSize - 1) / c.blockSize
	}

	// Ensure we align to block size
	capacity = c.totalBlocks * c.blockSize

	// Init Allocator
	c.freeBlocks = make([]int32, c.totalBlocks)
	c.blockRefs = make(map[int32]int)
	for i := 0; i < c.totalBlocks; i++ {
		c.freeBlocks[i] = int32(c.totalBlocks - 1 - i) // #nosec G115 -- safe: totalBlocks is bounded by memory
		c.blockRefs[int32(i)] = 0
	}

	c.blockTables = make(map[string][]int32)
	c.blockTablesDevice = make(map[string]*device.Tensor)
	c.dirty = make(map[string]bool)
	c.streamingParams = make(map[string]StreamingLLMParams)
	c.seqMaxPos = make(map[string]int)
	c.offloadedSeqs = make(map[string]*OffloadedSequence)
	c.lastAccess = make(map[string]time.Time)

	c.kPools = make([]*device.Tensor, c.layers)
	c.vPools = make([]*device.Tensor, c.layers)

	c.qjlRows = 32 // Default for many models
	if ctx.TQQJL != nil && ctx.TQQJL.Rows() > 0 {
		c.qjlRows = ctx.TQQJL.Rows()
	}

	kvDim := c.kvHeads * c.headDim
	if c.Precision == device.DataTypeTQ1_0 || c.Precision == device.DataTypeTQ2_0 {
		// TurboQuant Block Structure: [headDim int8][qjlRows int8][8 bytes metadata]
		// Each token slot stores c.kvHeads blocks
		tqBlockSize := c.headDim + c.qjlRows + 8
		kvDim = tqBlockSize * c.kvHeads
	}

	allocLayers := c.layers
	if config.NumGPULayers > 0 && config.NumGPULayers < c.layers && ctx.DeviceID() >= 0 {
		allocLayers = config.NumGPULayers
	}

	for i := 0; i < allocLayers; i++ {
		var k, v *device.Tensor
		switch c.Precision {
		case device.DataTypeTQ1_0, device.DataTypeTQ2_0, device.DataTypeQ8_0:
			k = ctx.NewTensorWithType(capacity, kvDim, device.DataTypeINT8)
			v = ctx.NewTensorWithType(capacity, kvDim, device.DataTypeINT8)
		case device.DataTypeFP8:
			k = ctx.NewTensorWithType(capacity, kvDim, device.DataTypeFP8)
			v = ctx.NewTensorWithType(capacity, kvDim, device.DataTypeFP8)
		default:
			// The pool has to be allocated with the element type the
			// backend's store and attention kernels actually use: float32
			// on the CPU, packed half-precision values on the GPU
			// backends. Declaring it float32 everywhere allocated twice
			// the memory the cache needed on a GPU and forced every
			// host-side read to decode the halves by hand.
			k = ctx.NewPagedKVPool(capacity, kvDim)
			v = ctx.NewPagedKVPool(capacity, kvDim)
		}

		if k == nil || v == nil {
			c.Free()
			return fmt.Errorf("failed to allocate pools for layer %d", i)
		}
		c.kPools[i] = k
		c.vPools[i] = v
	}

	// The quantization scales live on the device: the quantized store
	// writes them and the quantized attention reads them, so a host
	// round-trip per step would be both slow and, in the decode loop,
	// an allocation.
	c.kScales = make([]*device.Tensor, allocLayers)
	c.vScales = make([]*device.Tensor, allocLayers)
	for i := 0; i < allocLayers; i++ {
		kScale := ctx.NewTensorFP32(1, capacity)
		vScale := ctx.NewTensorFP32(1, capacity)
		if kScale == nil || vScale == nil {
			c.Free()
			return fmt.Errorf("failed to allocate the quantization scales for layer %d", i)
		}
		// A scale of one is the identity: it is what an unwritten slot
		// decodes to.
		ones := make([]float32, capacity)
		for j := range ones {
			ones[j] = 1.0
		}
		_ = kScale.LoadFrom(ones)
		_ = vScale.LoadFrom(ones)
		c.kScales[i] = kScale
		c.vScales[i] = vScale
	}

	// Initialize TurboQuant matrices from device context or fallbacks if available
	if c.Precision == device.DataTypeTQ1_0 || c.Precision == device.DataTypeTQ2_0 {
		if ctx.TQRotation != nil && ctx.TQQJL != nil {
			c.tqRotation = ctx.TQRotation
			c.tqQJL = ctx.TQQJL
		} else {
			rotData := device.GetPrecomputedRotation(c.headDim)
			qjlData := device.GetPrecomputedQJLSigns(c.qjlRows * c.headDim)
			rot := ctx.NewTensorFP32(c.headDim, c.headDim)
			if rot != nil {
				_ = rot.LoadFrom(rotData)
				c.tqRotation = rot
			}
			qjl := ctx.NewTensorFP32(c.qjlRows, c.headDim)
			if qjl != nil {
				_ = qjl.LoadFrom(qjlData)
				c.tqQJL = qjl
			}
		}
		logger.Log.Debug("PagedKVCache initialized with TurboQuant matrices", "head_dim", c.headDim, "qjl_rows", c.qjlRows)
	}

	c.initialized = true
	return nil
}

// StoreKVPagedBatch updates the cache pools with new K/V projections.
func (c *PagedKVCache) StoreKVPagedBatch(layer int, k, v, physicalPositions *device.Tensor) {
	c.execMu.Lock()
	defer c.execMu.Unlock()

	kPool := c.kPools[layer]
	vPool := c.vPools[layer]

	// The number of tokens, not the number of sequences: the store walks
	// the tokens and their physical positions.
	numTokens := k.Rows()
	if numTokens == 0 {
		numTokens = v.Rows()
	}
	if numTokens == 0 {
		numTokens = physicalPositions.Cols()
	}

	switch c.Precision {
	case device.DataTypeTQ1_0, device.DataTypeTQ2_0:
		c.encodeKVTurboQuant(k, v, kPool, vPool, physicalPositions)
	case device.DataTypeFP8, device.DataTypeQ8_0:
		// The quantized pool holds one byte per element and the scale
		// of every slot is written next to the codes.
		kScales, vScales := c.scaleTensors(layer)
		if err := c.ctx.StoreKVPagedBatchQuantized(k, v, kPool, vPool,
			kScales, vScales, physicalPositions, c.kvHeads*c.headDim, numTokens,
			c.Precision == device.DataTypeFP8); err != nil {
			logger.Log.Error("failed to store the quantized KV cache", "layer", layer, "error", err)
		}
	default:
		c.ctx.StoreKVPagedBatch(k, v, kPool, vPool, physicalPositions, c.kvHeads*c.headDim, numTokens)
	}
}

// scaleTensors returns the K and V scale tensors of a layer, or nil when
// the layer has none.
func (c *PagedKVCache) scaleTensors(layer int) (kScales, vScales *device.Tensor) {
	if layer < 0 || layer >= len(c.kScales) {
		return nil, nil
	}
	return c.kScales[layer], c.vScales[layer]
}

// encodeKVTurboQuant encodes K/V tensors to TurboQuant format for paged storage.
func (c *PagedKVCache) encodeKVTurboQuant(k, v, kCache, vCache *device.Tensor, physicalPositions *device.Tensor) {
	if c.tqRotation == nil || c.tqQJL == nil {
		logger.Log.Error("TurboQuant matrices not initialized for paged KV cache")
		return
	}

	startTime := time.Now()

	blockSize := c.headDim
	qjlRows := c.qjlRows
	if qjlRows == 0 {
		qjlRows = 32
	}

	bits := 2
	if c.Precision == device.DataTypeTQ2_0 {
		bits = 4
	}

	kHost := k.ToHostF32()
	vHost := v.ToHostF32()
	posHost := physicalPositions.ToHostF32()
	rotData := c.tqRotation.ToHostF32()
	qjlData := c.tqQJL.ToHostF32()

	if len(qjlData) > 0 && blockSize > 0 {
		availRows := len(qjlData) / blockSize
		if qjlRows > availRows {
			qjlRows = availRows
		}
	}
	bytesPerBlock := blockSize + qjlRows + 8

	numTokens := len(posHost)
	if numTokens == 0 && k.Rows() > 0 {
		numTokens = k.Rows()
	}

	bytesPerToken := c.kvHeads * bytesPerBlock
	kRaw := kCache.RawData()
	vRaw := vCache.RawData()

	for b := 0; b < numTokens; b++ {
		pPos := 0
		if b < len(posHost) {
			pPos = int(posHost[b])
		}
		dstTokenStart := pPos * bytesPerToken

		for h := 0; h < c.kvHeads; h++ {
			srcOffset := b*(c.kvHeads*c.headDim) + h*c.headDim
			if srcOffset+c.headDim > len(kHost) || srcOffset+c.headDim > len(vHost) {
				continue
			}

			kHead := kHost[srcOffset : srcOffset+c.headDim]
			vHead := vHost[srcOffset : srcOffset+c.headDim]

			dstHeadStart := dstTokenStart + h*bytesPerBlock
			if dstHeadStart+bytesPerBlock > len(kRaw) || dstHeadStart+bytesPerBlock > len(vRaw) {
				continue
			}

			// PolarQuant + QJL for K
			qK, sK, resK := simd.PolarQuantSIMD(kHead, rotData, blockSize, bits)
			qjK, sjK := simd.QJLTransformSIMD(resK, qjlData, qjlRows, blockSize)

			kDst := kRaw[dstHeadStart : dstHeadStart+bytesPerBlock]
			for i, val := range qK {
				kDst[i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			for i, val := range qjK {
				kDst[blockSize+i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			binary.LittleEndian.PutUint32(kDst[blockSize+qjlRows:blockSize+qjlRows+4], mathx.Float32bits(sK))
			binary.LittleEndian.PutUint32(kDst[blockSize+qjlRows+4:blockSize+qjlRows+8], mathx.Float32bits(sjK))

			// PolarQuant + QJL for V
			qV, sV, resV := simd.PolarQuantSIMD(vHead, rotData, blockSize, bits)
			qjV, sjV := simd.QJLTransformSIMD(resV, qjlData, qjlRows, blockSize)

			vDst := vRaw[dstHeadStart : dstHeadStart+bytesPerBlock]
			for i, val := range qV {
				vDst[i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			for i, val := range qjV {
				vDst[blockSize+i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			binary.LittleEndian.PutUint32(vDst[blockSize+qjlRows:blockSize+qjlRows+4], mathx.Float32bits(sV))
			binary.LittleEndian.PutUint32(vDst[blockSize+qjlRows+4:blockSize+qjlRows+8], mathx.Float32bits(sjV))
		}
	}

	_ = kCache.LoadFrom(kRaw)
	_ = vCache.LoadFrom(vRaw)

	compressionRatio := float64(c.headDim*4) / float64(bytesPerBlock)
	durationSec := time.Since(startTime).Seconds()
	metrics.RecordTurboQuantCompression("paged", compressionRatio)
	metrics.RecordTurboQuantLatency(durationSec)
	metrics.RecordTurboQuantBatch(compressionRatio, durationSec)
}

// SetStreamingLLM configures attention sinks and rolling sliding window for a sequence.
func (c *PagedKVCache) SetStreamingLLM(seqID string, sinkBlocks, windowBlocks int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.streamingParams == nil {
		c.streamingParams = make(map[string]StreamingLLMParams)
	}
	if sinkBlocks <= 0 {
		sinkBlocks = 4 // Default 4 blocks (64 tokens)
	}
	if windowBlocks <= 0 {
		windowBlocks = c.totalBlocks - sinkBlocks
		if windowBlocks <= 0 {
			windowBlocks = 16
		}
	}
	c.streamingParams[seqID] = StreamingLLMParams{
		Enabled:      true,
		SinkBlocks:   sinkBlocks,
		WindowBlocks: windowBlocks,
	}
}

// GetStreamingLLM returns the StreamingLLM parameters for a sequence.
func (c *PagedKVCache) GetStreamingLLM(seqID string) (StreamingLLMParams, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.streamingParams == nil {
		return StreamingLLMParams{}, false
	}
	params, exists := c.streamingParams[seqID]
	return params, exists
}

// Allocate reserves blocks for a sequence to accommodate numTokens.
func (c *PagedKVCache) Allocate(seqID string, numTokens int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	params, isStreaming := c.streamingParams[seqID]
	maxBlocks := c.totalBlocks
	if isStreaming && params.Enabled {
		maxBlocks = params.SinkBlocks + params.WindowBlocks
	}

	if existing, exists := c.blockTables[seqID]; exists {
		// Calculate how many MORE blocks we need
		currentBlocks := len(existing)
		blocksNeeded := (numTokens + c.blockSize - 1) / c.blockSize
		if isStreaming && params.Enabled && blocksNeeded > maxBlocks {
			blocksNeeded = maxBlocks
		}
		additional := blocksNeeded - currentBlocks
		if additional <= 0 {
			return nil // Already has enough
		}

		for i := 0; i < additional; i++ {
			if isStreaming && params.Enabled && len(existing) >= maxBlocks {
				if len(existing) > params.SinkBlocks {
					evictBlock := existing[params.SinkBlocks]
					c.blockRefs[evictBlock]--
					if c.blockRefs[evictBlock] <= 0 {
						c.freeBlocks = append(c.freeBlocks, evictBlock)
						c.blockRefs[evictBlock] = 0
					}
					copy(existing[params.SinkBlocks:], existing[params.SinkBlocks+1:])
					existing = existing[:len(existing)-1]
				}
			}
			phys, err := c.allocateBlock()
			if err != nil {
				return err
			}
			existing = append(existing, phys)
		}
		c.blockTables[seqID] = existing
		return nil
	}

	blocksNeeded := (numTokens + c.blockSize - 1) / c.blockSize
	if isStreaming && params.Enabled && blocksNeeded > maxBlocks {
		blocksNeeded = maxBlocks
	}
	table := make([]int32, 0, blocksNeeded)
	for i := 0; i < blocksNeeded; i++ {
		phys, err := c.allocateBlock()
		if err != nil {
			// Rollback allocation
			for _, b := range table {
				c.blockRefs[b]--
				if c.blockRefs[b] <= 0 {
					c.freeBlocks = append(c.freeBlocks, b)
					c.blockRefs[b] = 0
				}
			}
			return err
		}
		table = append(table, phys)
	}
	c.blockTables[seqID] = table
	return nil
}

// GetPhysicalPositions returns the physical memory offsets for a range of logical tokens.
func (c *PagedKVCache) GetPhysicalPositions(seqID string, startPos, numTokens int) ([]int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, exists := c.blockTables[seqID]
	if !exists {
		return nil, fmt.Errorf("sequence %s not found", seqID)
	}

	params, isStreaming := c.streamingParams[seqID]
	positions := make([]int, numTokens)
	sinkTokens := 0
	if isStreaming && params.Enabled {
		sinkTokens = params.SinkBlocks * c.blockSize
	}

	for i := 0; i < numTokens; i++ {
		logicalPos := startPos + i
		var blockIdx int
		var offset int

		if isStreaming && params.Enabled && logicalPos >= sinkTokens {
			totalSeen := c.seqMaxPos[seqID]
			windowTokens := params.WindowBlocks * c.blockSize
			if totalSeen > sinkTokens+windowTokens {
				windowStart := totalSeen - windowTokens + 1
				if logicalPos < windowStart {
					blockIdx = 0
					offset = 0
				} else {
					rel := logicalPos - windowStart
					blockIdx = params.SinkBlocks + (rel / c.blockSize)
					if blockIdx >= len(table) {
						blockIdx = len(table) - 1
					}
					offset = logicalPos % c.blockSize
				}
			} else {
				blockIdx = logicalPos / c.blockSize
				if blockIdx >= len(table) {
					blockIdx = len(table) - 1
				}
				offset = logicalPos % c.blockSize
			}
		} else {
			blockIdx = logicalPos / c.blockSize
			offset = logicalPos % c.blockSize
			if blockIdx >= len(table) {
				return nil, fmt.Errorf("logical position %d exceeds allocated blocks", logicalPos)
			}
		}

		positions[i] = int(table[blockIdx])*c.blockSize + offset
	}
	return positions, nil
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

// FreeBlock returns a physical block to the free pool.
func (c *PagedKVCache) FreeBlock(block int32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.blockRefs[block] > 0 {
		c.blockRefs[block]--
	}
	if c.blockRefs[block] == 0 {
		c.freeBlocks = append(c.freeBlocks, block)
	}
}

func (c *PagedKVCache) copyBlockData(layer int, srcBlock, dstBlock int32) error {
	if srcBlock == dstBlock {
		return nil
	}

	if layer < 0 || layer >= len(c.kPools) || c.kPools[layer] == nil {
		return nil
	}

	kPool := c.kPools[layer]
	srcStart := int(srcBlock) * c.blockSize
	dstStart := int(dstBlock) * c.blockSize
	cols := kPool.Cols()

	data := kPool.ToHostF32()
	if data == nil {
		return nil
	}

	for row := 0; row < c.blockSize; row++ {
		for col := 0; col < cols; col++ {
			data[(dstStart+row)*cols+col] = data[(srcStart+row)*cols+col]
		}
	}

	if err := c.kPools[layer].LoadFrom(data); err != nil {
		return err
	}

	if layer < len(c.vPools) && c.vPools[layer] != nil {
		vPool := c.vPools[layer]
		vData := vPool.ToHostF32()
		if vData != nil {
			for row := 0; row < c.blockSize; row++ {
				for col := 0; col < cols; col++ {
					vData[(dstStart+row)*cols+col] = vData[(srcStart+row)*cols+col]
				}
			}
			if err := vPool.LoadFrom(vData); err != nil {
				return err
			}
		}
	}

	return nil
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
		delete(c.streamingParams, seqID)
		delete(c.seqMaxPos, seqID)
	}
	delete(c.offloadedSeqs, seqID)
	delete(c.lastAccess, seqID)
}

// Update implementation for Paged Cache
func (c *PagedKVCache) Update(seqID string, layer, pos int, k, v *device.Tensor) error {
	if !c.initialized {
		return fmt.Errorf("cache not initialized")
	}

	blockOffset := pos % c.blockSize

	c.mu.Lock()
	table, exists := c.blockTables[seqID]
	if !exists {
		table = make([]int32, 0)
	}

	params, isStreaming := c.streamingParams[seqID]
	maxBlocks := c.totalBlocks
	if isStreaming && params.Enabled {
		maxBlocks = params.SinkBlocks + params.WindowBlocks
	}

	var physBlock int32

	if isStreaming && params.Enabled && pos >= params.SinkBlocks*c.blockSize {
		sinkTokens := params.SinkBlocks * c.blockSize
		windowTokens := params.WindowBlocks * c.blockSize

		if len(table) < maxBlocks {
			logicalBlockIdx := pos / c.blockSize
			if logicalBlockIdx == len(table) && layer == 0 {
				phys, err := c.allocateBlock()
				if err != nil {
					c.mu.Unlock()
					return err
				}
				table = append(table, phys)
				c.blockTables[seqID] = table
			}
			if len(table) == 0 {
				c.mu.Unlock()
				return fmt.Errorf("no blocks allocated for sequence %s", seqID)
			}
			physBlock = table[len(table)-1]
		} else {
			// Sliding window ring buffer eviction
			if pos >= sinkTokens+windowTokens && (pos%c.blockSize == 0) && layer == 0 && pos > c.seqMaxPos[seqID] {
				recycledBlock := table[params.SinkBlocks]
				copy(table[params.SinkBlocks:], table[params.SinkBlocks+1:])
				table[len(table)-1] = recycledBlock
				c.blockTables[seqID] = table
			}
			physBlock = table[len(table)-1]
		}
		if pos > c.seqMaxPos[seqID] {
			c.seqMaxPos[seqID] = pos
		}
	} else {
		logicalBlockIdx := pos / c.blockSize
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
		physBlock = table[logicalBlockIdx]
	}

	// Copy on Write if refcount > 1 and we are modifying it (Update)
	oldPhys := physBlock
	if c.blockRefs[physBlock] > 1 {
		// COW: allocate a new block and swap it in
		newPhys, err := c.allocateBlock()
		if err != nil {
			c.mu.Unlock()
			return fmt.Errorf("OOM copying block on write: %w", err)
		}
		c.blockRefs[physBlock]--
		for idx, b := range table {
			if b == oldPhys {
				table[idx] = newPhys
				break
			}
		}
		physBlock = newPhys

		if err := c.copyBlockData(layer, oldPhys, physBlock); err != nil {
			c.mu.Unlock()
			return fmt.Errorf("failed to copy block data: %w", err)
		}
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

	if layer < 0 || layer >= len(c.kPools) {
		return fmt.Errorf("layer %d out of bounds (max %d)", layer, len(c.kPools)-1)
	}

	kTarget := c.kPools[layer]
	vTarget := c.vPools[layer]
	if kTarget == nil || vTarget == nil {
		return nil
	}

	// We treat the pool as a large contiguous buffer.
	// Passing `physPos` works.
	// WindowSize arg to StoreKV should be Capacity (TotalBlocks * BlockSize).
	capacity := c.totalBlocks * c.blockSize

	if physPos >= capacity {
		return fmt.Errorf("physical position %d exceeds total KV cache capacity %d", physPos, capacity)
	}

	// Dynamic KV Cache Quantization downcast (FP8/INT8/Q8_0)
	switch c.Precision {
	case device.DataTypeFP8, device.DataTypeQ8_0:
		kScale, vScale := c.scaleTensors(layer)
		if err := c.ctx.StoreKVPagedSlotQuantized(k, v, kTarget, vTarget, kScale, vScale, physPos,
			c.kvHeads*c.headDim, c.Precision == device.DataTypeFP8); err != nil {
			return err
		}
		precisionStr := "FP8"
		compRatio := 4.0
		if c.Precision == device.DataTypeQ8_0 {
			precisionStr = "Q8_0"
			compRatio = 3.8
		}
		metrics.RecordKVCacheQuantization(precisionStr, 1, compRatio)
	default:
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

		physPositions[i] = int32(int64(table[logicalBlockIdx])*int64(c.blockSize) + int64(blockOffset)) // #nosec G115 -- safe: block positions are bounded
	}
	c.mu.Unlock()

	// Load physical positions to device
	ppDevice := c.ctx.NewTensorFP32(1, batchSize)
	defer ppDevice.Free()
	ppF32 := make([]float32, batchSize)
	for i, p := range physPositions {
		ppF32[i] = float32(p)
	}
	if err := ppDevice.LoadFrom(ppF32); err != nil {
		return err
	}

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

	// Convert int32 table to float32 for device loading only if dirty
	if c.dirty[seqID] {
		goTable := make([]float32, len(table))
		for i, b := range table {
			goTable[i] = float32(b)
		}
		_ = tableDevice.LoadFrom(goTable)
		c.dirty[seqID] = false
	}

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
	_ = btDevice.LoadFrom(btData)

	posData := make([]float32, batchSize)
	for i, p := range positions {
		posData[i] = float32(p)
	}
	posDevice := c.ctx.NewTensorFP32(1, batchSize)
	_ = posDevice.LoadFrom(posData)

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

// GetScaleTensors returns the device tensors that hold the quantization
// scales of a layer, one scale per pool slot.
//
// They are only meaningful for a quantized cache: the quantized store
// writes them and the quantized attention reads them.
func (c *PagedKVCache) GetScaleTensors(layer int) (kScales, vScales *device.Tensor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if layer >= 0 && layer < len(c.kScales) {
		return c.kScales[layer], c.vScales[layer]
	}
	return nil, nil
}

// OffloadSequence transfers physical KV blocks of an inactive sequence from device memory into host RAM.
func (c *PagedKVCache) OffloadSequence(seqID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	table, exists := c.blockTables[seqID]
	if !exists {
		return fmt.Errorf("sequence %s not found in paged cache", seqID)
	}

	if len(table) == 0 {
		delete(c.blockTables, seqID)
		return nil
	}

	// Capture block data for each layer
	blockData := make(map[int]OffloadedBlockData)
	for blockIdx, block := range table {
		for l := 0; l < len(c.kPools); l++ {
			if c.kPools[l] == nil {
				continue
			}
			cols := c.kPools[l].Cols()
			totalFloats := c.blockSize * cols

			kData := make([]float32, totalFloats)
			vData := make([]float32, totalFloats)

			srcStart := int(block) * c.blockSize * cols
			kHost := c.kPools[l].ToHostF32()
			if kHost != nil && srcStart+totalFloats <= len(kHost) {
				copy(kData, kHost[srcStart:srcStart+totalFloats])
			}
			if l < len(c.vPools) && c.vPools[l] != nil {
				vHost := c.vPools[l].ToHostF32()
				if vHost != nil && srcStart+totalFloats <= len(vHost) {
					copy(vData, vHost[srcStart:srcStart+totalFloats])
				}
			}

			key := blockIdx*1000 + l
			blockData[key] = OffloadedBlockData{
				Layer: l,
				K:     kData,
				V:     vData,
			}
		}

		// Decrement ref count and free if uniquely owned
		c.blockRefs[block]--
		if c.blockRefs[block] <= 0 {
			c.freeBlocks = append(c.freeBlocks, block)
			c.blockRefs[block] = 0
		}
	}

	if c.offloadedSeqs == nil {
		c.offloadedSeqs = make(map[string]*OffloadedSequence)
	}
	c.offloadedSeqs[seqID] = &OffloadedSequence{
		SeqID:       seqID,
		BlockData:   blockData,
		NumBlocks:   len(table),
		OffloadedAt: time.Now(),
	}

	delete(c.blockTables, seqID)
	c.dirty[seqID] = true
	logger.Log.Info("Offloaded sequence KV blocks to host memory", "seq_id", seqID, "blocks", len(table))
	return nil
}

// RestoreSequence reinstates offloaded host KV blocks back into active device pools.
func (c *PagedKVCache) RestoreSequence(seqID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.offloadedSeqs == nil {
		return fmt.Errorf("sequence %s not found in offloaded store", seqID)
	}
	offloaded, exists := c.offloadedSeqs[seqID]
	if !exists {
		return fmt.Errorf("sequence %s not found in offloaded store", seqID)
	}

	if len(c.freeBlocks) < offloaded.NumBlocks {
		return fmt.Errorf("insufficient free blocks to restore sequence %s: need %d, have %d", seqID, offloaded.NumBlocks, len(c.freeBlocks))
	}

	newTable := make([]int32, offloaded.NumBlocks)
	for i := 0; i < offloaded.NumBlocks; i++ {
		phys, err := c.allocateBlock()
		if err != nil {
			// Rollback allocation
			for j := 0; j < i; j++ {
				c.blockRefs[newTable[j]]--
				if c.blockRefs[newTable[j]] <= 0 {
					c.freeBlocks = append(c.freeBlocks, newTable[j])
					c.blockRefs[newTable[j]] = 0
				}
			}
			return err
		}
		newTable[i] = phys

		// Restore data for each layer
		for l := 0; l < len(c.kPools); l++ {
			if c.kPools[l] == nil {
				continue
			}
			key := i*1000 + l
			data, ok := offloaded.BlockData[key]
			if !ok {
				continue
			}

			cols := c.kPools[l].Cols()
			totalFloats := c.blockSize * cols
			dstStart := int(phys) * c.blockSize * cols

			kHost := c.kPools[l].ToHostF32()
			if kHost != nil && dstStart+totalFloats <= len(kHost) {
				copy(kHost[dstStart:dstStart+totalFloats], data.K)
				_ = c.kPools[l].LoadFrom(kHost)
			}

			if l < len(c.vPools) && c.vPools[l] != nil {
				vHost := c.vPools[l].ToHostF32()
				if vHost != nil && dstStart+totalFloats <= len(vHost) {
					copy(vHost[dstStart:dstStart+totalFloats], data.V)
					_ = c.vPools[l].LoadFrom(vHost)
				}
			}
		}
	}

	c.blockTables[seqID] = newTable
	delete(c.offloadedSeqs, seqID)
	c.dirty[seqID] = true
	if c.lastAccess == nil {
		c.lastAccess = make(map[string]time.Time)
	}
	c.lastAccess[seqID] = time.Now()
	logger.Log.Info("Restored sequence KV blocks from host memory", "seq_id", seqID, "blocks", len(newTable))
	return nil
}

// IsOffloaded returns whether a sequence's KV blocks are currently offloaded to host RAM.
func (c *PagedKVCache) IsOffloaded(seqID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.offloadedSeqs == nil {
		return false
	}
	_, ok := c.offloadedSeqs[seqID]
	return ok
}

// OffloadLRU finds the least-recently-accessed sequence and offloads its KV blocks to host RAM.
func (c *PagedKVCache) OffloadLRU() (string, error) {
	c.mu.Lock()
	if len(c.blockTables) == 0 {
		c.mu.Unlock()
		return "", nil
	}

	// Find sequence with oldest lastAccess
	var oldestSeq string
	var oldestTime time.Time
	first := true
	for seqID := range c.blockTables {
		t, ok := c.lastAccess[seqID]
		if !ok {
			t = time.Time{}
		}
		if first || t.Before(oldestTime) {
			oldestTime = t
			oldestSeq = seqID
			first = false
		}
	}
	c.mu.Unlock()

	if oldestSeq != "" {
		err := c.OffloadSequence(oldestSeq)
		return oldestSeq, err
	}
	return "", nil
}

// Defragment compacts active physical blocks into lower indices to reduce fragmentation.
func (c *PagedKVCache) Defragment() (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.execMu.Lock()
	defer c.execMu.Unlock()

	if !c.initialized || len(c.blockTables) == 0 {
		return 0, nil
	}

	// 1. Identify which blocks are currently in use
	usedMap := make(map[int32]bool)
	for _, table := range c.blockTables {
		for _, b := range table {
			usedMap[b] = true
		}
	}

	// 2. Build sorted lists of free and used blocks
	var freeIndices []int32
	var usedIndices []int32
	for i := 0; i < c.totalBlocks; i++ {
		b := int32(i)
		if usedMap[b] {
			usedIndices = append(usedIndices, b)
		} else {
			freeIndices = append(freeIndices, b)
		}
	}

	compacted := 0
	freePtr := 0
	usedPtr := len(usedIndices) - 1

	// Remap mapping: oldBlock -> newBlock
	remap := make(map[int32]int32)

	// Move highest used block into lowest free block if free < used
	for freePtr < len(freeIndices) && usedPtr >= 0 {
		freeB := freeIndices[freePtr]
		usedB := usedIndices[usedPtr]
		if freeB >= usedB {
			break
		}

		// Copy data from usedB to freeB across all layers
		for l := 0; l < len(c.kPools); l++ {
			if err := c.copyBlockData(l, usedB, freeB); err != nil {
				return compacted, err
			}
		}

		remap[usedB] = freeB
		c.blockRefs[freeB] = c.blockRefs[usedB]
		c.blockRefs[usedB] = 0

		compacted++
		freePtr++
		usedPtr--
	}

	if compacted > 0 {
		// Update all block tables with remapped block IDs
		for seqID, table := range c.blockTables {
			for i, b := range table {
				if newB, ok := remap[b]; ok {
					table[i] = newB
				}
			}
			c.dirty[seqID] = true
		}

		// Rebuild c.freeBlocks cleanly
		c.freeBlocks = make([]int32, 0, len(freeIndices))
		for i := c.totalBlocks - 1; i >= 0; i-- {
			b := int32(i)
			if c.blockRefs[b] == 0 {
				c.freeBlocks = append(c.freeBlocks, b)
			}
		}
		logger.Log.Info("Paged KV cache defragmentation completed", "compacted_blocks", compacted, "free_blocks", len(c.freeBlocks))
	}

	return compacted, nil
}

// RegisterMemoryGovernor binds the paged KV cache to the proactive memory governor.
func (c *PagedKVCache) RegisterMemoryGovernor(gov *device.MemoryGovernor) {
	if gov == nil {
		gov = device.GetMemoryGovernor()
	}
	gov.RegisterDefragHandler(func() {
		_, _ = c.Defragment()
	})
	gov.RegisterOffloadHandler(func() {
		_, _ = c.OffloadLRU()
	})
}
