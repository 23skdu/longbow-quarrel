//go:build darwin && metal

package device

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework Accelerate
#include "metal_bridge.h"
#include <stdlib.h>
void* Metal_AutoreleasePoolPush();
void Metal_AutoreleasePoolPop(void* pool);
void* Metal_NewHeap(MetalWrapperRef ctx, long long size);
MetalBufferRef Metal_NewBufferFromHeap(void* heap, long long size);
void Metal_FreeHeap(void* heap);
void Metal_LinearQ8_0_F16(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale);
void Metal_LinearQ8_0_F32(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale);
void Metal_RMSNormLinear_Q6K_F16(MetalWrapperRef ctx, MetalBufferRef input,
                                 int offIn, MetalBufferRef normWeight,
                                 int offNormWeight, MetalBufferRef weight,
                                 int offWeight, MetalBufferRef result,
                                 int offRes, int M, int N, int K, float eps,
                                 float scale, int batchSize);
void Metal_SwiGLULinear_Q6K_F16(MetalWrapperRef ctx, MetalBufferRef gateIn,
                                int offGate, MetalBufferRef upIn, int offUp,
                                MetalBufferRef weight, int offWeight,
                                MetalBufferRef result, int offRes, int M, int N,
                                int K, float scale);
void Metal_RMSNormQKV_Q6K_F16(MetalWrapperRef ctx, MetalBufferRef input,
                              int offIn, MetalBufferRef normWeight,
                              int offNormWeight, MetalBufferRef qWeight,
                              int offQW, MetalBufferRef kWeight, int offKW,
                              MetalBufferRef vWeight, int offVW,
                              MetalBufferRef qOut, int offQO,
                              MetalBufferRef kOut, int offKO,
                              MetalBufferRef vOut, int offVO, int dimIn,
                              int qDim, int kvDim, float eps, float scale,
                              int batchSize);
void Metal_MOE_ExpertGateUpSwiGLU(MetalWrapperRef ctx, MetalBufferRef input,
                                  int offInput, MetalBufferRef gateWeight,
                                  int offGate, MetalBufferRef upWeight,
                                  int offUp, MetalBufferRef expertIndices,
                                  int offIndices, MetalBufferRef expertWeights,
                                  int offWeights, MetalBufferRef output,
                                  int offOutput, int batchSize, int dim,
                                  int hiddenDim, int topK);
void Metal_TurboQuant_PolarQuant(MetalWrapperRef ctx, MetalBufferRef input,
                                  int offInput, MetalBufferRef rotationMatrix,
                                  int offRot, MetalBufferRef quantized,
                                  int offQuant, MetalBufferRef scaleOut,
                                  int offScale, MetalBufferRef residual,
                                  int offRes, int n, int numBlocks, int bits);
void Metal_TurboQuant_QJLTransform(MetalWrapperRef ctx, MetalBufferRef residual,
                                    int offRes, MetalBufferRef signMatrix,
                                    int offSign, MetalBufferRef quantized,
                                    int offQuant, MetalBufferRef scaleOut,
                                    int offScale, int rows, int cols, int numBlocks);
void Metal_TurboQuant_Encode(MetalWrapperRef ctx, MetalBufferRef input,
                              int offInput, MetalBufferRef rotationMatrix,
                              int offRot, MetalBufferRef qjlMatrix,
                              int offQJL, MetalBufferRef output,
                              int offOut, MetalBufferRef scaleOut,
                              int offScale, MetalBufferRef qjlScaleOut,
                              int offQJLScale, int blockSize, int qjlRows, int numBlocks, int bits);
// Buffer Management
MetalBufferRef Metal_Alloc(MetalWrapperRef ctx, long long size);
void Metal_FreeBuffer(MetalWrapperRef ctx, MetalBufferRef buf);
void Metal_CopyToDevice(MetalBufferRef buf, int offset, const void *data, int size);
void Metal_CopyToHost(MetalBufferRef buf, int offset, void *data, int size);
void *Metal_GetBufferContents(MetalBufferRef buf);
void Metal_ZeroBuffer(MetalBufferRef buf, int offset, int size);
void Metal_ZeroBufferGPU(MetalWrapperRef ctx, MetalBufferRef buf, int offset, int size);

void Metal_Copy_F32_F16(MetalWrapperRef ctx, MetalBufferRef src, int oS, MetalBufferRef dst, int oD, int n);
void Metal_Copy_F16_F32(MetalWrapperRef ctx, MetalBufferRef src, int oS, MetalBufferRef dst, int oD, int n);
void Metal_Copy_F32(MetalWrapperRef ctx, MetalBufferRef src, int oS, MetalBufferRef dst, int oD, int n);

void Metal_TurboQuant_Decode(MetalWrapperRef ctx, MetalBufferRef input,
                              int offInput, MetalBufferRef rotationMatrix,
                              int offRot, MetalBufferRef qjlMatrix,
                              int offQJL, MetalBufferRef output,
                              int offOut, MetalBufferRef scaleIn,
                              int offScale, int blockSize, int qjlRows, int numBlocks);

void Metal_FlashAttention2_F16(MetalWrapperRef ctx, MetalBufferRef q, MetalBufferRef k_cache, MetalBufferRef v_cache, MetalBufferRef output, int num_heads, int kv_heads, int headDim, MetalBufferRef seq_lens, int block_size, MetalBufferRef block_table, int max_blocks_per_seq, MetalBufferRef token_to_seq, int batchSize);
void Metal_Linear_LoRA_Add_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn, MetalBufferRef A, int offA, MetalBufferRef B, int offB, MetalBufferRef output, int offOut, int M, int N, int K, int R, float scale);
void Metal_Vision_Patch_Embed_F32(MetalWrapperRef ctx, MetalBufferRef pixels, int offPixels, MetalBufferRef weights, int offW, MetalBufferRef output, int offOut, int patchSize, int visionDim, int numPatchesX);
void Metal_AllReduce_F16(MetalWrapperRef ctx, MetalBufferRef data, int offset, int count);
*/
import "C"
import (
	_ "embed"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

var allocatedBytes int64

func traceAlloc(ptr *Tensor, delta int64, label string) {
	newVal := atomic.AddInt64(&allocatedBytes, delta)
	metrics.RecordGPUMemory(newVal)
}

func AllocatedBytes() int64 {
	return atomic.LoadInt64(&allocatedBytes)
}

var MaxGPUMemory int64 = DefaultMaxMemoryMetal

//go:embed kernels.metal
var kernelsSource string

//go:embed kernel_flash_attention.metal
var flashKernelsSource string

// Context holds the Metal connection and tensor pool
type Context struct {
	ref    C.MetalWrapperRef
	mu     sync.Mutex
	pool   map[string][]*Tensor // pool by size key "RxCxType"
	ExecMu sync.Mutex           // Execution lock for Metal command encoding

	// TurboQuant Global Matrices
	TQRotation *Tensor
	TQQJL      *Tensor

	// Performance Counters (Hotpath)
	ArrowBytesProcessed atomic.Int64
	device              int
}

func (c *Context) DeviceID() int {
	return c.device
}

type LoRAWeight struct {
	A     *Tensor
	B     *Tensor
	Scale float32
}

func NewContext() *Context {
	combinedSrc := kernelsSource + "\n" + flashKernelsSource
	cSrc := C.CString(combinedSrc)
	defer C.free(unsafe.Pointer(cSrc))

	ref := C.Metal_Init(cSrc)
	if ref == nil {
		panic("Failed to initialize Metal backend")
	}

	return &Context{
		ref:  ref,
		pool: make(map[string][]*Tensor),
	}
}

func (c *Context) Free() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ref != nil {
		c.ClearPoolInternal() // Internal version without lock
		C.Metal_Free(c.ref)
		c.ref = nil
	}
}

func (c *Context) ClearPoolInternal() {
	if c.ref == nil {
		return
	}
	for key, tensors := range c.pool {
		for _, t := range tensors {
			if t.buf != nil {
				runtime.SetFinalizer(t, nil)
				C.Metal_FreeBuffer(c.ref, t.buf)
				traceAlloc(t, -int64(t.sizeBytes), "ClearPool")
				metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
				t.buf = nil // Prevent double free
			}
		}
		delete(c.pool, key)
	}
}

// ClearPool releases all pooled tensors to free up GPU memory.
func (c *Context) ClearPool() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ClearPoolInternal()
}

// Tensor wraps a Metal buffer. Always FP16 for this engine.
type Tensor struct {
	ctx       *Context
	rows      int
	cols      int
	sizeBytes int
	buf       C.MetalBufferRef
	heap      unsafe.Pointer // Track if this buffer is part of a heap
	Offset    int            // Offset in bytes from buf start
	dataType  DataType       // 0=F16, 1=Q4K, 2=Q3K
	blockSize int
	qjlRows   int
}

func (t *Tensor) SizeBytes() int {
	return t.sizeBytes
}

func (t *Tensor) Rows() int { return t.rows }
func (t *Tensor) Cols() int { return t.cols }

func (t *Tensor) Data() []float32 {
	return t.ToHostF32()
}

func (t *Tensor) RawData() []byte {
	// For Metal on Apple Silicon, memory is unified/shared.
	// We can get a pointer directly and wrap it in a slice for Zero-Copy Arrow transport.
	if t.buf != nil {
		ptr := C.Metal_GetBufferContents(t.buf)
		if ptr == nil {
			return nil
		}
		// Correct size: sizeBytes is already established in NewTensor/Load
		return unsafe.Slice((*byte)(ptr), t.sizeBytes)
	}
	return nil
}

// NewQ3KTensor creates a tensor with Q3_K quantization layout (110 bytes per 256 weights)
// Returns error if dimensions are invalid
func (c *Context) NewQ3KTensor(rows, cols int) (*Tensor, error) {
	numElements := rows * cols
	if numElements%256 != 0 {
		return nil, NewValidationError("NewQ3KTensor",
			fmt.Sprintf("Q3_K tensor size must be divisible by 256, got %d", numElements),
			"tensor_dims")
	}
	numBlocks := numElements / 256
	sizeBytes := numBlocks * 110

	if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
		c.ClearPool()
		if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
			return nil, fmt.Errorf("Metal_Alloc: Exceeded memory budget of %d bytes (requested %d)", MaxGPUMemory, sizeBytes)
		}
	}

	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		return nil, fmt.Errorf("Metal_Alloc returned nil for %d bytes", sizeBytes)
	}

	t := &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: int(sizeBytes),
		buf:       buf,
		dataType:  DataTypeQ3K,
	}
	traceAlloc(t, int64(sizeBytes), "NewQ3KTensor")
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))

	runtime.SetFinalizer(t, func(ft *Tensor) {
		if ft.buf != nil {
			C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
			traceAlloc(ft, -int64(ft.sizeBytes), "FinalizerQ3K")
			metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
		}
	})

	return t, nil
}

// NewQ4KTensor creates a tensor with Q4_K quantization layout (144 bytes per 256 weights)
// Returns error if dimensions are invalid
func (c *Context) NewQ4KTensor(rows, cols int) (*Tensor, error) {
	numElements := rows * cols
	if numElements%256 != 0 {
		return nil, NewValidationError("NewQ4KTensor",
			fmt.Sprintf("Q4_K tensor size must be divisible by 256, got %d", numElements),
			"tensor_dims")
	}
	numBlocks := numElements / 256
	sizeBytes := numBlocks * 144

	if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
		c.ClearPool()
		if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
			return nil, fmt.Errorf("Metal_Alloc: Exceeded memory budget of %d bytes (requested %d)", MaxGPUMemory, sizeBytes)
		}
	}

	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		return nil, fmt.Errorf("Metal_Alloc returned nil for %d bytes", sizeBytes)
	}

	t := &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: int(sizeBytes),
		buf:       buf,
		dataType:  DataTypeQ4K,
	}
	traceAlloc(t, int64(sizeBytes), "NewQ4KTensor")
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))

	runtime.SetFinalizer(t, func(ft *Tensor) {
		if ft.buf != nil && ft.ctx != nil {
			ft.ctx.mu.Lock()
			defer ft.ctx.mu.Unlock()
			if ft.ctx.ref != nil {
				C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
				traceAlloc(ft, -int64(ft.sizeBytes), "FinalizerQ4K")
				metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
			}
		}
	})

	return t, nil
}

// NewQ8_0Tensor creates a tensor with Q8_0 quantization layout (34 bytes per 32 weights)
func (c *Context) NewQ8_0Tensor(rows, cols int) (*Tensor, error) {
	numElements := rows * cols
	if numElements%32 != 0 {
		return nil, NewValidationError("NewQ8_0Tensor",
			fmt.Sprintf("Q8_0 tensor size must be divisible by 32, got %d", numElements),
			"tensor_dims")
	}
	numBlocks := numElements / 32
	sizeBytes := numBlocks * 34

	if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
		c.ClearPool()
		if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
			return nil, fmt.Errorf("Metal_Alloc: Exceeded memory budget of %d bytes (requested %d)", MaxGPUMemory, sizeBytes)
		}
	}

	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		return nil, fmt.Errorf("Metal_Alloc returned nil for %d bytes", sizeBytes)
	}

	t := &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: int(sizeBytes),
		buf:       buf,
		dataType:  DataTypeQ8_0,
	}
	traceAlloc(t, int64(sizeBytes), "NewQ8_0Tensor")
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))

	runtime.SetFinalizer(t, func(ft *Tensor) {
		if ft.buf != nil && ft.ctx != nil {
			ft.ctx.mu.Lock()
			defer ft.ctx.mu.Unlock()
			if ft.ctx.ref != nil {
				C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
				traceAlloc(ft, -int64(ft.sizeBytes), "FinalizerQ8_0")
				metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
			}
		}
	})

	return t, nil
}

// NewQ6KTensor creates a tensor with Q6_K quantization layout (210 bytes per 256 weights)
// Returns error if dimensions are invalid
func (c *Context) NewQ6KTensor(rows, cols int) (*Tensor, error) {
	numElements := rows * cols
	if numElements%256 != 0 {
		return nil, NewValidationError("NewQ6KTensor",
			fmt.Sprintf("Q6_K tensor size must be divisible by 256, got %d", numElements),
			"tensor_dims")
	}
	numBlocks := numElements / 256
	sizeBytes := numBlocks * 210

	if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
		c.ClearPool()
		if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
			return nil, fmt.Errorf("Metal_Alloc: Exceeded memory budget of %d bytes (requested %d)", MaxGPUMemory, sizeBytes)
		}
	}

	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		return nil, fmt.Errorf("Metal_Alloc returned nil for %d bytes", sizeBytes)
	}

	t := &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: sizeBytes,
		buf:       buf,
		dataType:  DataTypeQ6K,
	}
	traceAlloc(t, int64(sizeBytes), "NewQ6KTensor")
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))

	runtime.SetFinalizer(t, func(ft *Tensor) {
		if ft.buf != nil && ft.ctx != nil {
			ft.ctx.mu.Lock()
			defer ft.ctx.mu.Unlock()
			if ft.ctx.ref != nil {
				C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
				traceAlloc(ft, -int64(ft.sizeBytes), "FinalizerQ6K")
				metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
			}
		}
	})

	return t, nil
}

// NewTensor creates a standard F16 tensor
func (c *Context) newTensorInternal(rows, cols int) *Tensor {
	sizeBytes := rows * cols * 2 // FP16
	if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
		c.ClearPool() // Attempt to free some space
		if atomic.LoadInt64(&allocatedBytes)+int64(sizeBytes) > MaxGPUMemory {
			panic(fmt.Sprintf("Metal_Alloc: Exceeded memory budget of %d bytes", MaxGPUMemory))
		}
	}
	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		panic("Metal_Alloc returned nil!")
	}
	t := &Tensor{
		ctx:       c,
		buf:       buf,
		sizeBytes: sizeBytes,
		rows:      rows,
		cols:      cols,
		dataType:  DataTypeF16,
	}
	traceAlloc(t, int64(sizeBytes), "NewTensor")
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
	runtime.SetFinalizer(t, func(ft *Tensor) {
		if ft.buf != nil && ft.ctx != nil {
			ft.ctx.mu.Lock()
			defer ft.ctx.mu.Unlock()
			if ft.ctx.ref != nil {
				C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
				traceAlloc(ft, -int64(ft.sizeBytes), "Finalizer")
				metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
			}
		}
	})
	return t
}

func (c *Context) NewTensor(rows, cols int) *Tensor {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	return c.newTensorInternal(rows, cols)
}

// Free explicitly releases the Metal buffer.
// Use this for large tensors in tight loops to avoid OOM due to lazy GC finalizers.
func (t *Tensor) BufferID() uintptr {
	if t == nil {
		return 0
	}
	return uintptr(t.buf)
}

func (t *Tensor) Free() {
	if t == nil {
		return
	}
	if t.buf == nil {
		return
	}
	// Clear finalizer first to prevent double free
	runtime.SetFinalizer(t, nil)
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.freeInternal()
}

func (t *Tensor) freeInternal() {
	if t.buf == nil {
		return
	}
	C.Metal_FreeBuffer(t.ctx.ref, t.buf)

	// Only track memory if it's NOT a heap-backed buffer
	// (Heap memory is tracked when the heap itself is allocated/freed)
	if t.heap == nil {
		traceAlloc(t, -int64(t.sizeBytes), "Free")
	}
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
	t.buf = nil // Mark as freed
}

// NewTensorFP32 creates a standard F32 tensor
// NewTensorFP32Pooled creates or reuses a pooled tensor for intermediate float32 results.
func (c *Context) newTensorFP32PooledInternal(rows, cols int) *Tensor {
	key := fmt.Sprintf("%dx%dx%d", rows, cols, DataTypeF32)
	c.mu.Lock()
	if tensors, ok := c.pool[key]; ok && len(tensors) > 0 {
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		c.mu.Unlock()
		return t
	}
	c.mu.Unlock()
	return c.newTensorFP32Internal(rows, cols)
}

func (c *Context) NewTensorFP32Pooled(rows, cols int) *Tensor {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	return c.newTensorFP32PooledInternal(rows, cols)
}

func (c *Context) newTensorFP32Internal(rows, cols int) *Tensor {
	sizeBytes := rows * cols * 4
	if sizeBytes == 0 {
		return &Tensor{ctx: c, sizeBytes: 0, rows: rows, cols: cols, dataType: DataTypeF32}
	}
	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		panic("Metal_Alloc returned nil!")
	}
	t := &Tensor{ctx: c, buf: buf, sizeBytes: sizeBytes, rows: rows, cols: cols, dataType: DataTypeF32}
	traceAlloc(t, int64(sizeBytes), "NewTensorFP32")
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
	runtime.SetFinalizer(t, func(ft *Tensor) {
		if ft.buf != nil && ft.ctx != nil {
			ft.ctx.mu.Lock()
			defer ft.ctx.mu.Unlock()
			if ft.ctx.ref != nil {
				C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
				traceAlloc(ft, -int64(ft.sizeBytes), "FinalizerFP32")
				metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
			}
		}
	})
	return t
}

func (c *Context) NewTensorFP32(rows, cols int) *Tensor {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	return c.newTensorFP32Internal(rows, cols)
}

func (c *Context) NewTensorFromData(rows, cols int, dt DataType, data []byte) (*Tensor, error) {
	t := c.NewTensorWithType(rows, cols, dt)
	if len(data) != t.sizeBytes {
		return nil, NewValidationError("NewTensorFromData",
			fmt.Sprintf("data size mismatch: expected %d, got %d", t.sizeBytes, len(data)),
			"data_size")
	}
	if len(data) > 0 {
		C.Metal_CopyToDevice(t.buf, C.int(0), unsafe.Pointer(&data[0]), C.int(len(data)))
	}
	return t, nil
}

func (c *Context) NewTurboTensor(rows, cols int, dt DataType, blockSize, qjlRows int) *Tensor {
	numElements := rows * cols
	numBlocks := numElements / blockSize
	if numElements%blockSize != 0 {
		numBlocks++
	}
	bytesPerBlock := blockSize + qjlRows + 8
	sizeBytes := numBlocks * bytesPerBlock

	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		return nil
	}

	t := &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: sizeBytes,
		buf:       buf,
		dataType:  dt,
		blockSize: blockSize,
		qjlRows:   qjlRows,
	}
	traceAlloc(t, int64(sizeBytes), "NewTurboTensor")
	return t
}

func (c *Context) NewTensorWithType(rows, cols int, dt DataType) *Tensor {
	sb := rows * cols * 2
	switch dt {
	case DataTypeF32:
		sb = rows * cols * 4
	case DataTypeQ6K:
		numElements := rows * cols
		numBlocks := numElements / 256
		sb = numBlocks * 210
	case DataTypeQ4K:
		numElements := rows * cols
		numBlocks := numElements / 256
		sb = numBlocks * 144
	case DataTypeQ8_0:
		numElements := rows * cols
		if numElements%32 != 0 {
			panic(fmt.Sprintf("Q8_0 tensor size %d not divisible by 32", numElements))
		}
		numBlocks := numElements / 32
		sb = numBlocks * 34
	case DataTypeQ4_0:
		numElements := rows * cols
		if numElements%32 != 0 {
			panic(fmt.Sprintf("Q4_0 tensor size %d not divisible by 32", numElements))
		}
		numBlocks := numElements / 32
		sb = numBlocks * 18
	case DataTypeIQ4_NL, DataTypeMXFP4:
		numElements := rows * cols
		numBlocks := numElements / 32
		sb = numBlocks * 18
	case DataTypeTQ1_0, DataTypeTQ2_0:
		return c.NewTurboTensor(rows, cols, dt, 128, 64)
	}

	if atomic.LoadInt64(&allocatedBytes)+int64(sb) > MaxGPUMemory {
		c.ClearPool()
		if atomic.LoadInt64(&allocatedBytes)+int64(sb) > MaxGPUMemory {
			panic(fmt.Sprintf("Metal_Alloc: Exceeded memory budget of %d bytes (requested %d)", MaxGPUMemory, sb))
		}
	}

	buf := C.Metal_Alloc(c.ref, C.longlong(sb))
	if buf == nil {
		panic(fmt.Sprintf("Metal_Alloc returned nil for %d bytes", sb))
	}

	t := &Tensor{ctx: c, rows: rows, cols: cols, sizeBytes: sb, buf: buf, dataType: dt}
	traceAlloc(t, int64(sb), "NewTensorWithType")
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))

	runtime.SetFinalizer(t, func(ft *Tensor) {
		if ft.buf != nil {
			ft.ctx.mu.Lock()
			defer ft.ctx.mu.Unlock()
			if ft.ctx.ref != nil {
				C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
				traceAlloc(ft, -int64(ft.sizeBytes), "FinalizerWithType")
				metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
			}
		}
	})
	return t
}

func (c *Context) TurboQuantEncode(input *Tensor, rotationMatrix *Tensor, qjlMatrix *Tensor, output *Tensor, scaleOut *Tensor, qjlScaleOut *Tensor, blockSize, qjlRows, bits int) {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	numBlocks := input.rows * input.cols / blockSize
	C.Metal_TurboQuant_Encode(c.ref, input.buf, C.int(input.Offset), rotationMatrix.buf, C.int(rotationMatrix.Offset),
		qjlMatrix.buf, C.int(qjlMatrix.Offset), output.buf, C.int(output.Offset),
		scaleOut.buf, C.int(scaleOut.Offset), qjlScaleOut.buf, C.int(qjlScaleOut.Offset),
		C.int(blockSize), C.int(qjlRows), C.int(numBlocks), C.int(bits))
}

func (c *Context) TurboQuantDecode(input *Tensor, rotationMatrix *Tensor, qjlMatrix *Tensor, output *Tensor, scaleIn *Tensor, blockSize, qjlRows int) {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	numBlocks := output.rows * output.cols / blockSize
	C.Metal_TurboQuant_Decode(c.ref, input.buf, C.int(input.Offset), rotationMatrix.buf, C.int(rotationMatrix.Offset),
		qjlMatrix.buf, C.int(qjlMatrix.Offset), output.buf, C.int(output.Offset),
		scaleIn.buf, C.int(scaleIn.Offset),
		C.int(blockSize), C.int(qjlRows), C.int(numBlocks))
}

func (c *Context) LinearLoRAAdd(input, A, B, output *Tensor, scale float32) {
	if A == nil || B == nil {
		return
	}
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()

	// M = input rows, N = output cols, K = input cols, R = rank
	M := input.rows
	N := output.cols
	K := input.cols
	R := A.rows // A is [Rank, DimIn]
	C.Metal_Linear_LoRA_Add_F16(c.ref, input.buf, C.int(input.Offset), A.buf, C.int(A.Offset), B.buf, C.int(B.Offset), output.buf, C.int(output.Offset), C.int(M), C.int(N), C.int(K), C.int(R), C.float(scale))
}

func (c *Context) VisionPatchEmbed(pixels *Tensor, weights *Tensor, output *Tensor, patchSize, visionDim, numPatchesX int) {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	C.Metal_Vision_Patch_Embed_F32(c.ref, pixels.buf, C.int(pixels.Offset), weights.buf, C.int(weights.Offset), output.buf, C.int(output.Offset), C.int(patchSize), C.int(visionDim), C.int(numPatchesX))
}

func (c *Context) AllReduce(data *Tensor) {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	count := data.rows * data.cols
	C.Metal_AllReduce_F16(c.ref, data.buf, C.int(data.Offset), C.int(count))
}

// NewTensorPooled attempts to reuse tensor from pool (defaults to F16)
func (c *Context) newTensorPooledInternal(rows, cols int) *Tensor {
	return c.newTensorPooledWithTypeInternal(rows, cols, DataTypeF16)
}

func (c *Context) newTensorPooledWithTypeInternal(rows, cols int, dt DataType) *Tensor {
	key := fmt.Sprintf("%dx%dx%d", rows, cols, dt)
	c.mu.Lock()
	if tensors, ok := c.pool[key]; ok && len(tensors) > 0 {
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		c.mu.Unlock()
		return t
	}
	c.mu.Unlock()
	return c.NewTensorWithType(rows, cols, dt)
}

func (c *Context) NewTensorPooledWithType(rows, cols int, dt DataType) *Tensor {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	return c.newTensorPooledWithTypeInternal(rows, cols, dt)
}

func (c *Context) NewTensorPooled(rows, cols int) *Tensor {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	return c.newTensorPooledWithTypeInternal(rows, cols, DataTypeF16)
}

// ReturnToPool returns tensor to pool for reuse.
// Note: This does NOT free the Metal memory, just prevents GC from reaping it.
func (t *Tensor) ReturnToPool() {
	key := fmt.Sprintf("%dx%dx%d", t.rows, t.cols, t.dataType)

	t.ctx.mu.Lock()
	t.ctx.pool[key] = append(t.ctx.pool[key], t)
	t.ctx.mu.Unlock()
}

func (t *Tensor) LoadFrom(data interface{}) error {
	switch d := data.(type) {
	case []float32:
		if len(d) != t.rows*t.cols {
			return fmt.Errorf("LoadFrom: size mismatch: expected %d, got %d", t.rows*t.cols, len(d))
		}
		t.LoadFromF32(d)
		return nil
	case []byte:
		return t.LoadFromRaw(d)
	default:
		return fmt.Errorf("unsupported data type for LoadFrom: %T", data)
	}
}

func (t *Tensor) LoadFromF32(data []float32) error {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	if len(data) != t.rows*t.cols {
		return NewValidationError("LoadFrom",
			fmt.Sprintf("data size %d does not match tensor size %d",
				len(data), t.rows*t.cols),
			"tensor_data")
	}

	if len(data) == 0 {
		return nil
	}

	if t.dataType == DataTypeF32 {
		C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(len(data)*4))
		return nil
	}

	if t.dataType == DataTypeQ6K {
		q6k := quantizeQ6K(data)
		C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&q6k[0]), C.int(len(q6k)))
		return nil
	}

	// Convert to FP16
	f16 := make([]uint16, len(data))
	for i, v := range data {
		f16[i] = Float32ToFloat16(v)
	}

	C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&f16[0]), C.int(t.sizeBytes))
	return nil
}

func (t *Tensor) LoadRaw(data []byte) error {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	if len(data) > t.sizeBytes {
		return NewValidationError("LoadRaw",
			fmt.Sprintf("raw data size %d exceeds tensor buffer size %d", len(data), t.sizeBytes),
			"tensor_data")
	}
	C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(len(data)))
	return nil
}

func quantizeQ6K(data []float32) []byte {
	numBlocks := len(data) / 256
	out := make([]byte, numBlocks*210)
	for b := 0; b < numBlocks; b++ {
		blockData := data[b*256 : (b+1)*256]
		off := b * 210
		maxAbs := float32(0)
		for _, v := range blockData {
			if a := float32(math.Abs(float64(v))); a > maxAbs {
				maxAbs = a
			}
		}
		d := maxAbs / 31.0
		if d == 0 {
			d = 1.0
		}
		d16 := Float32ToFloat16(d)
		out[off+208] = byte(d16 & 0xFF)
		out[off+209] = byte(d16 >> 8)
		for i := 0; i < 16; i++ {
			out[off+192+i] = 1
		}
		for i := 0; i < 128; i++ {
			v0 := blockData[i*2]
			v1 := blockData[i*2+1]
			q0 := int(math.Round(float64(v0/d))) + 32
			q1 := int(math.Round(float64(v1/d))) + 32
			if q0 < 0 {
				q0 = 0
			} else if q0 > 63 {
				q0 = 63
			}
			if q1 < 0 {
				q1 = 0
			} else if q1 > 63 {
				q1 = 63
			}
			out[off+i] = byte(q0&0xF) | byte((q1&0xF)<<4)
		}
		for i := 0; i < 256; i++ {
			v := blockData[i]
			q := int(math.Round(float64(v/d))) + 32
			if q < 0 {
				q = 0
			} else if q > 63 {
				q = 63
			}
			out[off+128+i/4] |= byte(((q >> 4) & 3) << ((i % 4) * 2))
		}
	}
	return out
}

// LoadFromBytes copies raw bytes to the buffer (for Q4K data, etc.)
func (t *Tensor) LoadFromBytes(data []byte) {
	C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(len(data)))
}

func (t *Tensor) Probe(name string, n int) {
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		logger.Log.Debug("probe buffer is nil", "name", name)
		return
	}

	f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)

	f32Data := make([]float32, n)
	for i := 0; i < n && i < len(f16Slice); i++ {
		f32Data[i] = Float16ToFloat32(f16Slice[i])
	}

	logger.Log.Debug("probe data", "name", name, "len", len(f16Slice), "data", f32Data)
}

// GetBufferContents returns unsafe pointer to buffer for diagnostics
func (t *Tensor) GetBufferContents() unsafe.Pointer {
	t.ctx.Synchronize()
	return C.Metal_GetBufferContents(t.buf)
}

// DataType returns the tensor's data type
func (t *Tensor) DataType() DataType {
	return t.dataType
}

func (t *Tensor) Context() *Context {
	return t.ctx
}

func (t *Tensor) ScanNaNs(name string) int {
	if t.dataType == DataTypeQ4K {
		return 0
	}
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		return 0
	}
	f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)
	nanCount := 0
	infCount := 0
	for _, v := range f16Slice {
		// F16: exp=31 (0x1F) means NaN or Inf
		// NaN: exp=31, mant!=0
		// Inf: exp=31, mant==0
		exp := (v >> 10) & 0x1F
		mant := v & 0x3FF
		if exp == 0x1F {
			if mant != 0 {
				nanCount++
			} else {
				infCount++
			}
		}
	}
	if nanCount > 0 || infCount > 0 {
		metrics.RecordNumericalInstability(name, nanCount, infCount)
	}
	return nanCount + infCount
}

func (t *Tensor) ScanMax(name string) (float32, ActivationStats) {
	data := t.ToHostF32()
	var maxVal float32 = 0.0
	var minVal float32 = 0.0
	var sum float32 = 0.0
	var sumSq float64 = 0.0
	var zeros int = 0
	var nans int = 0
	var infs int = 0

	if len(data) > 0 {
		minVal = data[0]
		maxVal = data[0]
	}

	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nans++
			continue
		}
		if math.IsInf(float64(v), 0) {
			infs++
			continue
		}
		if v == 0 {
			zeros++
		}
		if v > maxVal {
			maxVal = v
		}
		if v < minVal {
			minVal = v
		}
		sum += v
		sumSq += float64(v) * float64(v)
	}

	mean := sum / float32(len(data))
	rms := float32(math.Sqrt(sumSq / float64(len(data))))

	stats := ActivationStats{
		Max:   maxVal,
		Min:   minVal,
		Mean:  mean,
		RMS:   rms,
		Zeros: zeros,
		NaNs:  nans,
		Infs:  infs,
	}

	sampleSize := 16
	if len(data) < sampleSize {
		sampleSize = len(data)
	}
	stats.Sample = make([]float32, sampleSize)
	copy(stats.Sample, data[:sampleSize])

	fmt.Printf("[%s] Min: %.4f Max: %.4f Mean: %.4f RMS: %.4f Zeros: %d/%d NaNs: %d Infs: %d\n", name, minVal, maxVal, mean, rms, zeros, len(data), nans, infs)
	if len(data) >= 10 {
		fmt.Printf("[%s] Sample: %v\n", name, data[:10])
	} else {
		fmt.Printf("[%s] Sample: %v\n", name, data)
	}
	return maxVal, stats
}

func (t *Tensor) ScanQ4KScales(name string) float32 {
	// DEBUG removed for performance
	return 0.0
}

func (t *Tensor) LoadQ4KFrom(raw []byte) {
	// Debug checks removed
	t.LoadFromRaw(raw)
}

// LoadFromRaw copies raw bytes directly to the GPU buffer.
// The caller must ensure the data is in the correct format (FP16 usually) and size.
func (t *Tensor) LoadFromRaw(data []byte) error {
	if len(data) != t.sizeBytes {
		return NewValidationError("LoadFromRaw",
			fmt.Sprintf("raw data size %d does not match tensor size %d", len(data), t.sizeBytes),
			"tensor_data")
	}
	if len(data) == 0 {
		return nil
	}
	C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(len(data)))
	return nil
}

func (t *Tensor) BufRef() unsafe.Pointer {
	return (unsafe.Pointer)(t.buf)
}

func (t *Tensor) ToHost() []float32 {
	if err := t.ctx.WaitWithTimeout(10 * time.Second); err != nil {
		panic(fmt.Sprintf("ToHost failed: %v", err))
	}

	if t.dataType == DataTypeF32 {
		f32 := make([]float32, t.rows*t.cols)
		C.Metal_CopyToHost(t.buf, C.int(t.Offset), unsafe.Pointer(&f32[0]), C.int(t.sizeBytes))
		return f32
	}

	if t.dataType == DataTypeQ6K {
		rawBytes := make([]byte, t.sizeBytes)
		C.Metal_CopyToHost(t.buf, C.int(t.Offset), unsafe.Pointer(&rawBytes[0]), C.int(t.sizeBytes))
		return gguf.DequantizeQ6K(rawBytes, t.rows*t.cols)
	}

	f16 := make([]uint16, t.rows*t.cols)
	C.Metal_CopyToHost(t.buf, C.int(t.Offset), unsafe.Pointer(&f16[0]), C.int(t.sizeBytes))

	f32 := make([]float32, len(f16))
	for i, v := range f16 {
		f32[i] = Float16ToFloat32(v)
	}
	return f32
}

func (t *Tensor) ToHostBytes() []byte {
	if err := t.ctx.WaitWithTimeout(10 * time.Second); err != nil {
		panic(fmt.Sprintf("ToHostBytes failed: %v", err))
	}
	// Copy raw bytes
	out := make([]byte, t.sizeBytes)
	C.Metal_CopyToHost(t.buf, C.int(t.Offset), unsafe.Pointer(&out[0]), C.int(t.sizeBytes))
	return out
}

// ZeroInit initializes tensor buffer with zeros
func (t *Tensor) ZeroInit() {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	C.Metal_ZeroBufferGPU(t.ctx.ref, t.buf, C.int(t.Offset), C.int(t.sizeBytes))
}

func (c *Context) Synchronize() {
	C.Metal_Synchronize(c.ref)
}

// WaitWithTimeout wait for GPU to complete with a timeout to prevent system lockup.
func (c *Context) WaitWithTimeout(timeout time.Duration) error {
	done := make(chan struct{})
	go func() {
		c.Synchronize()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-time.After(timeout):
		return fmt.Errorf("GPU synchronization timed out after %v", timeout)
	}
}

func (t *Tensor) ScaleBy(val float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, t.cols)
	C.Metal_Scale_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.float(val), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

// Operations

// MatMul performs matrix multiplication C = A * B
func (t *Tensor) matMulInternal(b *Tensor) *Tensor {
	M := b.rows
	N := t.rows
	K := t.cols
	t0 := time.Now()
	switch t.dataType {
	case DataTypeQ4K:
		c := t.ctx.newTensorInternal(N, M)
		C.Metal_ZeroBufferGPU(t.ctx.ref, c.buf, C.int(0), C.int(c.sizeBytes))
		C.Metal_MatMul_Q4K_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.bool(false), b.buf, C.int(b.Offset), C.bool(false), c.buf, C.int(c.Offset), C.int(M), C.int(N), C.int(K), C.float(1.0))
		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	case DataTypeQ3K:
		c := t.ctx.newTensorInternal(N, M)
		C.Metal_ZeroBufferGPU(t.ctx.ref, c.buf, C.int(0), C.int(c.sizeBytes))
		C.Metal_MatMul_Q3K_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.bool(false), b.buf, C.int(b.Offset), C.bool(false), c.buf, C.int(c.Offset), C.int(M), C.int(N), C.int(K), C.float(1.0))
		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	case DataTypeQ6K:
		c := t.ctx.newTensorInternal(N, M)
		C.Metal_ZeroBufferGPU(t.ctx.ref, c.buf, C.int(0), C.int(c.sizeBytes))
		C.Metal_MatMul_Q6K_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.bool(false), b.buf, C.int(b.Offset), C.bool(false), c.buf, C.int(c.Offset), C.int(M), C.int(N), C.int(K), C.float(1.0))
		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	}
	c := t.ctx.newTensorInternal(N, M)
	C.Metal_MatMul_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.bool(false), b.buf, C.int(b.Offset), C.bool(false), c.buf, C.int(c.Offset), C.int(M), C.int(N), C.int(K))
	metrics.RecordKernelDuration("MatMul", time.Since(t0))
	return c
}

func (t *Tensor) MatMul(b *Tensor) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	return t.matMulInternal(b)
}

// Linear performs t * weight^T
// t: [M, K], weight: [N, K] -> result: [M, N]
// Returns error if dimensions are incompatible
func (t *Tensor) Linear(weight *Tensor) (*Tensor, error) {
	if err := ValidateLinearDimensions(t.cols, weight.cols); err != nil {
		return nil, err
	}
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t0 := time.Now()
	res := t.ctx.newTensorPooledInternal(t.rows, weight.rows) // [M, N]
	if weight.dataType == DataTypeQ4K {
		C.Metal_MatMul_Q4K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), false, t.buf, C.int(t.Offset), false, res.buf, C.int(res.Offset),
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(1.0))
	} else {
		C.Metal_BatchedMatMul_F16(t.ctx.ref,
			t.buf, C.int(t.Offset), C.int(t.cols*2), false,
			weight.buf, C.int(weight.Offset), C.int(weight.cols*2), true,
			res.buf, C.int(res.Offset), C.int(weight.rows*2),
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols), 1)
	}
	metrics.RecordKernelDuration("Linear", time.Since(t0))
	return res, nil
}

// LinearInto performs Linear using existing output tensor (scratch buffer)
// Returns error if dimensions are incompatible
func (t *Tensor) linearIntoInternal(weight *Tensor, out *Tensor, scale float32) {
	if weight == nil || out == nil {
		return
	}
	if t.dataType == DataTypeF32 {
		t.linearF32IntoInternal(weight, out, scale)
		return
	}
	switch weight.dataType {
	case DataTypeQ3K:
		C.Metal_MatMul_Q3K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), C.bool(false), t.buf, C.int(t.Offset), C.bool(false), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
	case DataTypeQ4K:
		C.Metal_MatMul_Q4K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), C.bool(false), t.buf, C.int(t.Offset), C.bool(false), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
	case DataTypeQ6K:
		C.Metal_MatMul_Q6K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), C.bool(false), t.buf, C.int(t.Offset), C.bool(false), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
	case DataTypeQ4_0:
		C.Metal_LinearQ4_0_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
	case DataTypeQ8_0:
		C.Metal_LinearQ8_0_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
	default:
		C.Metal_MatMul_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), C.bool(false), t.buf, C.int(t.Offset), C.bool(false), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	}
}

func (t *Tensor) LinearInto(weight *Tensor, out *Tensor, scale float32) error {
	if t.rows != out.rows || weight.rows != out.cols {
		return NewValidationError("LinearInto",
			fmt.Sprintf("dimension mismatch: [%d,%d] * [%d,%d] -> [%d,%d]",
				t.rows, t.cols, weight.rows, weight.cols, out.rows, out.cols),
			"linear_dims")
	}
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.linearIntoInternal(weight, out, scale)
	return nil
}

// RunQ3K_Explicit for testing only
func (c *Context) RunQ3K_Explicit(w, in, out *Tensor) {
	c.ExecMu.Lock()
	defer c.ExecMu.Unlock()
	C.Metal_MatMul_Q3K_F16(c.ref, w.buf, C.int(w.Offset), C.bool(false), in.buf, C.int(in.Offset), C.bool(false), out.buf, C.int(out.Offset), C.int(1), C.int(w.rows), C.int(w.cols), C.float(1.0))
}

func (t *Tensor) RMSNorm(weight *Tensor, eps float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, t.cols)
	C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), res.buf, C.int(res.Offset), C.int(t.rows), C.int(t.cols), C.float(eps))
	return res
}

// RMSNormLinear performs fused RMSNorm + Linear in single kernel
// Eliminates intermediate buffer allocation
func (t *Tensor) RMSNormLinear(normWeight, weight *Tensor, eps float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, weight.rows)
	C.Metal_RMSNormLinear_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		normWeight.buf, C.int(normWeight.Offset),
		weight.buf, C.int(weight.Offset), res.buf, C.int(res.Offset),
		C.int(t.cols), C.int(weight.rows), C.float(eps), C.int(t.rows))
	return res
}

// RMSNormLinearQ4K performs fused RMSNorm + Linear (Q4_K)
func (t *Tensor) RMSNormLinearQ4K(normWeight, weight *Tensor, eps float32, scale float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, weight.rows)
	t.rmsNormLinearIntoQ4KInternal(normWeight, weight, res, eps, scale)
	return res
}

// RMSNormLinearIntoQ4K performs fused RMSNorm + Linear (Q4_K) into existing destination
func (t *Tensor) rmsNormLinearIntoQ4KInternal(normWeight, weight, out *Tensor, eps float32, scale float32) {
	C.Metal_RMSNormLinear_Q4K_F16(t.ctx.ref, t.buf, C.int(t.Offset), normWeight.buf, C.int(normWeight.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(eps), C.float(scale), C.int(t.rows))
}

func (t *Tensor) RMSNormLinearIntoQ4K(normWeight, weight, out *Tensor, eps float32, scale float32) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.rmsNormLinearIntoQ4KInternal(normWeight, weight, out, eps, scale)
}

// SwiGLULinearQ4K performs fused SwiGLU + Linear (Q4_K)
func (t *Tensor) SwiGLULinearQ4K(up, weight *Tensor, scale float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, weight.rows)
	t.swiGLULinearIntoQ4KInternal(up, weight, res, scale)
	return res
}

// SwiGLULinearIntoQ4K performs fused SwiGLU + Linear (Q4_K) into existing destination
func (t *Tensor) swiGLULinearIntoQ4KInternal(up, weight, out *Tensor, scale float32) {
	C.Metal_SwiGLULinear_Q4K_F16(t.ctx.ref, t.buf, C.int(t.Offset), up.buf, C.int(up.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
}

func (t *Tensor) SwiGLULinearIntoQ4K(up, weight, out *Tensor, scale float32) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.swiGLULinearIntoQ4KInternal(up, weight, out, scale)
}

// RMSNormLinearQ6K performs fused RMSNorm + Linear (Q6_K)
func (t *Tensor) RMSNormLinearQ6K(normWeight, weight *Tensor, eps float32, scale float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, weight.rows)
	t.rmsNormLinearIntoQ6KInternal(normWeight, weight, res, eps, scale)
	return res
}

// RMSNormLinearIntoQ6K performs fused RMSNorm + Linear (Q6_K) into existing destination
func (t *Tensor) rmsNormLinearIntoQ6KInternal(normWeight, weight, out *Tensor, eps float32, scale float32) {
	C.Metal_RMSNormLinear_Q6K_F16(t.ctx.ref, t.buf, C.int(t.Offset), normWeight.buf, C.int(normWeight.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(eps), C.float(scale), C.int(t.rows))
}

func (t *Tensor) RMSNormLinearIntoQ6K(normWeight, weight, out *Tensor, eps float32, scale float32) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.rmsNormLinearIntoQ6KInternal(normWeight, weight, out, eps, scale)
}

// SwiGLULinearQ6K performs fused SwiGLU + Linear (Q6_K)
func (t *Tensor) SwiGLULinearQ6K(up, weight *Tensor, scale float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, weight.rows)
	t.swiGLULinearIntoQ6KInternal(up, weight, res, scale)
	return res
}

// SwiGLULinearIntoQ6K performs fused SwiGLU + Linear (Q6_K) into existing destination
func (t *Tensor) swiGLULinearIntoQ6KInternal(up, weight, out *Tensor, scale float32) {
	C.Metal_SwiGLULinear_Q6K_F16(t.ctx.ref, t.buf, C.int(t.Offset), up.buf, C.int(up.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
}

func (t *Tensor) SwiGLULinearIntoQ6K(up, weight, out *Tensor, scale float32) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.swiGLULinearIntoQ6KInternal(up, weight, out, scale)
}

// MambaConv1d performs 1D Causal Convolution
// Input t: [Batch, Dim] (Current token)
// Weight: [Dim, KernelSize]
// Bias: [Dim]
// State: [Dim, KernelSize] (Ring buffer history)
// Output: [Batch, Dim]
func (t *Tensor) MambaConv1d(weight, bias, state, out *Tensor) {
	// Assumes Batch=1 for now as per kernels
	C.Metal_MambaConv1d_F16(t.ctx.ref,
		t.buf, C.int(t.Offset),
		weight.buf, C.int(weight.Offset),
		bias.buf, C.int(bias.Offset),
		state.buf, C.int(state.Offset),
		out.buf, C.int(out.Offset),
		C.int(t.cols), C.int(weight.cols)) // weight.cols should be kernel_size
}

// RMSNormQKV performs fused RMSNorm + QKV Linear projections
func (t *Tensor) rmsNormQKVInternal(normWeight, wQ, wK, wV *Tensor, eps float32) (*Tensor, *Tensor, *Tensor) {
	q := t.ctx.newTensorPooledInternal(t.rows, wQ.rows)
	k := t.ctx.newTensorPooledInternal(t.rows, wK.rows)
	v := t.ctx.newTensorPooledInternal(t.rows, wV.rows)
	C.Metal_RMSNormQKV_F16(t.ctx.ref, t.buf, C.int(t.Offset), normWeight.buf, C.int(normWeight.Offset), wQ.buf, C.int(wQ.Offset), wK.buf, C.int(wK.Offset), wV.buf, C.int(wV.Offset), q.buf, C.int(q.Offset), k.buf, C.int(k.Offset), v.buf, C.int(v.Offset), C.int(t.cols), C.int(wQ.rows), C.int(wK.rows), C.float(eps), C.int(t.rows))
	return q, k, v
}

func (t *Tensor) RMSNormQKV(normWeight, wQ, wK, wV *Tensor, eps float32) (*Tensor, *Tensor, *Tensor) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	return t.rmsNormQKVInternal(normWeight, wQ, wK, wV, eps)
}

// FusedFFN performs one entire FFN block: RMSNorm + Gate/Up Linear + SwiGLU + Down Linear
func (t *Tensor) fusedFFNInternal(normWeight, wGate, wUp, wDown *Tensor, eps float32) *Tensor {
	res := t.ctx.newTensorPooledInternal(t.rows, t.cols)
	C.Metal_FusedFFN_F16(t.ctx.ref, t.buf, C.int(t.Offset), normWeight.buf, C.int(normWeight.Offset), wGate.buf, C.int(wGate.Offset), wUp.buf, C.int(wUp.Offset), wDown.buf, C.int(wDown.Offset), res.buf, C.int(res.Offset), C.int(t.cols), C.int(wGate.rows), C.float(eps), C.int(t.rows))
	return res
}

func (t *Tensor) FusedFFN(normWeight, wGate, wUp, wDown *Tensor, eps float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	return t.fusedFFNInternal(normWeight, wGate, wUp, wDown, eps)
}

// LayerScratch holds pre-allocated buffers for a layer operation to avoid alloc overhead
type LayerScratch struct {
	QPart, KPart, VPart *Tensor
	AttOut, ResAtt      *Tensor
	Scores              *Tensor
	Normed              *Tensor // F16

	// FFN Intermediates (FP32)
	NormedFFN, GatePart, UpPart, SwiOut, ResFFN *Tensor // NormedFFN is F16.
	NormedFFN_F32, ResFFN_F32                   *Tensor // [Batch, Dim] FP32 (New)

	// Gemma4 Q/K Normalization buffers
	QNormed *Tensor // [Batch, QDim] - for normalized Q after q_norm
	KNormed *Tensor // [Batch, KDim] - for normalized K after k_norm

	// Logits (FP32)
	Logits *Tensor // [1, VocabSize]

	heap unsafe.Pointer // Reference to Metal Heap
	size int            // Total size of heap for accounting
}

// NewTensorFromBuffer creates a tensor sharing existing buffer at offset
func (c *Context) NewTensorFromBuffer(buf C.MetalBufferRef, offset, rows, cols int, dataType DataType) *Tensor {
	// Size check could be added if we knew buffer size
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: rows * cols * 2, // Approx
		buf:       buf,
		Offset:    offset,
		dataType:  dataType,
	}
}

// AutoreleasePoolPush pushes a new pool
func (c *Context) AutoreleasePoolPush() unsafe.Pointer {
	return C.Metal_AutoreleasePoolPush()
}

// AutoreleasePoolPop pops the pool
func (c *Context) AutoreleasePoolPop(pool unsafe.Pointer) {
	C.Metal_AutoreleasePoolPop(pool)
}

// NewHeap allocates a Metal Heap
func (c *Context) NewHeap(size int) unsafe.Pointer {
	heap := C.Metal_NewHeap(c.ref, C.longlong(size))
	if heap != nil {
		traceAlloc(nil, int64(size), "NewHeap")
	}
	return heap
}

// NewBufferFromHeap allocates from heap
func (c *Context) NewBufferFromHeap(heap unsafe.Pointer, size, rows, cols int, dt DataType) *Tensor {
	buf := C.Metal_NewBufferFromHeap(heap, C.longlong(size))
	if buf == nil {
		return nil
	}
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: size,
		buf:       buf,
		heap:      heap,
		Offset:    0,
		dataType:  dt,
	}
}

func (c *Context) NewLayerScratch(batch, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize, qNormDim, kNormDim int) *LayerScratch {
	s := &LayerScratch{}

	// Align to 4096
	align := func(n int) int {
		return (n + 4095) & ^4095
	}

	szNormed := align(batch * dim * 2)
	szQPart := align(batch * dim * 2)
	szKPart := align(batch * kvHeads * headDim * 2)
	szVPart := align(batch * kvHeads * headDim * 2)
	szAttOut := align(batch * dim * 2)
	szResAtt := align(batch * dim * 2)

	szNormedFFN := align(batch * dim * 2)
	szNormedFFN_F32 := align(batch * dim * 4) // FP32
	szResFFN := align(batch * dim * 2)
	szResFFN_F32 := align(batch * dim * 4) // FP32

	// Gemma4 Q/K norm buffers (max dimensions: 512 for full attention)
	szQNormed := align(batch * qNormDim * 2)
	szKNormed := align(batch * kNormDim * 2)

	szScores := align(heads * seqLen * 4)
	if szScores < align(32768*4) {
		szScores = align(32768 * 4)
	}

	szGate := align(batch * hiddenDim * 4)
	szUp := align(batch * hiddenDim * 4)
	szSwiOut := align(batch * hiddenDim * 4)

	szLogits := align(1 * vocabSize * 4) // F32 Logits

	total := szNormed + szQPart + szKPart + szVPart + szAttOut + szResAtt +
		szNormedFFN + szNormedFFN_F32 + szResFFN + szResFFN_F32 + szScores + szGate + szUp + szSwiOut + szLogits +
		szQNormed + szKNormed

	heap := c.NewHeap(total)
	if heap == nil {
		panic("Heap Alloc failed")
	}

	newT := func(sz, r, cols int, dt DataType) *Tensor {
		return c.NewBufferFromHeap(heap, sz, r, cols, dt)
	}

	// We need to store heap ref to free it?
	s.heap = heap // Store heap pointer
	s.size = total

	s.Normed = newT(szNormed, batch, dim, DataTypeF16)
	s.QPart = newT(szQPart, batch, dim, DataTypeF16)
	s.KPart = newT(szKPart, batch, kvHeads*headDim, DataTypeF16)
	s.VPart = newT(szVPart, batch, kvHeads*headDim, DataTypeF16)
	s.AttOut = newT(szAttOut, batch, dim, DataTypeF16)
	s.ResAtt = newT(szResAtt, batch, dim, DataTypeF16)

	s.NormedFFN = newT(szNormedFFN, batch, dim, DataTypeF16)
	s.NormedFFN_F32 = newT(szNormedFFN_F32, batch, dim, DataTypeF32)
	s.ResFFN = newT(szResFFN, batch, dim, DataTypeF16)
	s.ResFFN_F32 = newT(szResFFN_F32, batch, dim, DataTypeF32)

	s.Scores = newT(szScores, 1, szScores/4, DataTypeF32)

	s.GatePart = newT(szGate, batch, hiddenDim, DataTypeF32)
	s.UpPart = newT(szUp, batch, hiddenDim, DataTypeF32)
	s.SwiOut = newT(szSwiOut, batch, hiddenDim, DataTypeF32)

	// Gemma4 Q/K norm buffers
	s.QNormed = newT(szQNormed, batch, qNormDim, DataTypeF16)
	s.KNormed = newT(szKNormed, batch, kNormDim, DataTypeF16)

	// Logits must be F32 to preserve precision during accumulation and sampling
	s.Logits = newT(szLogits, 1, vocabSize, DataTypeF32)

	return s
}

// Free releases all buffers
func (s *LayerScratch) Free() {
	if s.heap != nil {
		// Heap memory is managed by the device allocator; tensors are freed individually below.
	}
	if s.QPart != nil {
		s.QPart.Free()
	}
	if s.KPart != nil {
		s.KPart.Free()
	}
	if s.VPart != nil {
		s.VPart.Free()
	}
	if s.AttOut != nil {
		s.AttOut.Free()
	}
	if s.ResAtt != nil {
		s.ResAtt.Free()
	}
	if s.Scores != nil {
		s.Scores.Free()
	}
	if s.Normed != nil {
		s.Normed.Free()
	}
	if s.NormedFFN != nil {
		s.NormedFFN.Free()
	}
	if s.NormedFFN_F32 != nil {
		s.NormedFFN_F32.Free()
	}
	if s.ResFFN_F32 != nil {
		s.ResFFN_F32.Free()
	}
	if s.GatePart != nil {
		s.GatePart.Free()
	}
	if s.UpPart != nil {
		s.UpPart.Free()
	}
	if s.SwiOut != nil {
		s.SwiOut.Free()
	}
	if s.ResFFN != nil {
		s.ResFFN.Free()
	}
	if s.Logits != nil {
		s.Logits.Free()
	}
	if s.QNormed != nil {
		s.QNormed.Free()
	}
	if s.KNormed != nil {
		s.KNormed.Free()
	}

	if s.heap != nil {
		traceAlloc(nil, -int64(s.size), "FreeLayerScratchHeap")
		C.Metal_FreeHeap(s.heap)
		s.heap = nil
	}
}

func (t *Tensor) Layer(layerIdx int, attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache *Tensor,
	scratch *LayerScratch,
	traceTracker interface {
		RecordLayer(layerName string, layerIdx int, stats ActivationStats)
		IsEnabled() bool
	},
	pos, heads, kvHeads, headDim int, ropeTheta, eps float32, hiddenDim, ctxLen, windowSize int, globalScale float32, debug bool, precisionMode int,
	blockTable *Tensor, blockSize int, kvStore func(k, v *Tensor),
	gemma4QNorm, gemma4KNorm *Tensor,
	gemma4Config config.Gemma4Config,
	loraQ, loraK, loraV, loraO *LoRAWeight) {

	// Use scratch buffers instead of allocating
	normed := scratch.Normed

	// 1. RMSNorm (Batched)
	t0_rmsnorm1 := time.Now()
	if t.dataType == DataTypeF32 {
		t.rmsNormFP32ToF16IntoInternal(attnNorm, eps, normed)
	} else {
		C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, C.int(t.Offset), attnNorm.buf, C.int(attnNorm.Offset),
			normed.buf, C.int(normed.Offset), C.int(t.rows), C.int(t.cols), C.float(eps))
	}
	metrics.RecordKernelDuration("Layer_RMSNorm1", time.Since(t0_rmsnorm1))

	// 2. QKV Projections (Batched)
	qPart := scratch.QPart
	kPart := scratch.KPart
	vPart := scratch.VPart

	// Guard against nil weights
	if q == nil || k == nil || v == nil || attnNorm == nil {
		return
	}

	if q.dataType == DataTypeQ4K && k.dataType == DataTypeQ4K && v.dataType == DataTypeQ4K && attnNorm.buf != nil {
		t0_qkv := time.Now()
		C.Metal_RMSNormQKV_Q4K_F16(t.ctx.ref, t.buf, C.int(t.Offset), attnNorm.buf, C.int(attnNorm.Offset),
			q.buf, C.int(q.Offset), k.buf, C.int(k.Offset), v.buf, C.int(v.Offset),
			qPart.buf, C.int(qPart.Offset), kPart.buf, C.int(kPart.Offset), vPart.buf, C.int(vPart.Offset),
			C.int(t.cols), C.int(q.rows), C.int(k.rows), C.float(eps), C.float(globalScale), C.int(t.rows))
		metrics.RecordKernelDuration("Layer_QKV_Fused_Q4K", time.Since(t0_qkv))
	} else if q.dataType == DataTypeQ6K && k.dataType == DataTypeQ6K && v.dataType == DataTypeQ6K && attnNorm.buf != nil {
		t0_qkv := time.Now()
		C.Metal_RMSNormQKV_Q6K_F16(t.ctx.ref, t.buf, C.int(t.Offset), attnNorm.buf, C.int(attnNorm.Offset),
			q.buf, C.int(q.Offset), k.buf, C.int(k.Offset), v.buf, C.int(v.Offset),
			qPart.buf, C.int(qPart.Offset), kPart.buf, C.int(kPart.Offset), vPart.buf, C.int(vPart.Offset),
			C.int(t.cols), C.int(q.rows), C.int(k.rows), C.float(eps), C.float(globalScale), C.int(t.rows))
		metrics.RecordKernelDuration("Layer_QKV_Fused_Q6K", time.Since(t0_qkv))
	} else {
		qInput := normed
		kInput := normed
		qInput.linearIntoInternal(q, qPart, globalScale)
		if loraQ != nil {
			t.ctx.LinearLoRAAdd(qInput, loraQ.A, loraQ.B, qPart, loraQ.Scale)
		}
		kInput.linearIntoInternal(k, kPart, globalScale)
		if loraK != nil {
			t.ctx.LinearLoRAAdd(kInput, loraK.A, loraK.B, kPart, loraK.Scale)
		}
		normed.linearIntoInternal(v, vPart, globalScale)
		if loraV != nil {
			t.ctx.LinearLoRAAdd(normed, loraV.A, loraV.B, vPart, loraV.Scale)
		}
	}

	// Gemma4 Q/K normalization is applied here if enabled (currently disabled)

	// 3. RoPE (Batched)
	t0_rope := time.Now()
	qPart.ropeInternal(pos, heads, headDim, t.rows, ropeTheta)
	kPart.ropeInternal(pos, kvHeads, headDim, t.rows, ropeTheta)
	metrics.RecordKernelDuration("Layer_RoPE", time.Since(t0_rope))

	// 4. Store K/V (Batched)
	t0_storekv := time.Now()
	if kvStore != nil {
		kvStore(kPart, vPart)
	} else {
		kPart.storeKVInternal(vPart, kCache, vCache, pos, kvHeads, headDim, windowSize)
	}
	metrics.RecordKernelDuration("Layer_StoreKV", time.Since(t0_storekv))

	// 5. Attention (Fused or Paged)
	attOut := scratch.AttOut
	qStride := heads * headDim * 2
	t0_attn := time.Now()

	// Gemma4 hybrid attention: use sliding window (512) for most layers, full context for layer 5, 11, etc.
	attnWindowSize := windowSize
	if gemma4Config.IsGemma4 {
		if gemma4Config.IsSlidingWindowLayer {
			attnWindowSize = 512 // Sliding window for layers 0-4, 6-10, etc.
		} else {
			attnWindowSize = 65536 // Full attention for layers 5, 11, 17, etc. (use large value)
		}
	}

	for i := 0; i < t.rows; i++ {
		p := pos + i
		offQ := i * qStride
		offAtt := i * qStride
		maxCtxLen := kCache.Cols()
		if maxCtxLen == 0 {
			maxCtxLen = p + 1
		}

		if blockTable != nil {
			// Optimized FlashAttention-2 for Paged Cache
			// For single token case, create a temporary position tensor
			maxBlocks := blockTable.Cols()
			pPos := t.ctx.NewTensorFP32(1, 1)
			pPos.LoadFrom([]float32{float32(p)})
			pSeq := t.ctx.NewTensorFP32(1, 1)
			pSeq.LoadFrom([]float32{0})
			
			t.ctx.FlashAttention2(qPart.Slice(i, 1), kCache, vCache, attOut.Slice(i, 1), pPos, heads, kvHeads, headDim, blockSize, blockTable, maxBlocks, pSeq, 1)
			
			pPos.Free()
			pSeq.Free()
		} else {
			C.Metal_AttFused_F16(t.ctx.ref, qPart.buf, C.int(qPart.Offset+offQ),
				kCache.buf, C.int(kCache.Offset), vCache.buf, C.int(vCache.Offset),
				attOut.buf, C.int(attOut.Offset+offAtt),
				C.int(p), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(attnWindowSize), C.int(maxCtxLen))
		}
	}
	metrics.RecordKernelDuration("Layer_Attention", time.Since(t0_attn))

	// 6. Attention Output Projection
	resAtt := scratch.ResAtt
	t0_attn_out := time.Now()
	attOut.LinearInto(o, resAtt, globalScale)
	if loraO != nil {
		t.ctx.LinearLoRAAdd(attOut, loraO.A, loraO.B, resAtt, loraO.Scale)
	}
	metrics.RecordKernelDuration("Layer_AttnOut", time.Since(t0_attn_out))

	// 7. Residual Add 1
	if t.dataType == DataTypeF32 {
		t.AddMixedInPlace(resAtt)
	} else {
		C.Metal_Add_F16(t.ctx.ref, t.buf, C.int(t.Offset), resAtt.buf, C.int(resAtt.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
	}

	// 8. FFN Block
	if ffnUp == nil {
		return
	}

	useF32FFN := precisionMode == 2
	useMixedPrecisionFFN := precisionMode == 3

	if useF32FFN || useMixedPrecisionFFN {
		normedFFN := scratch.NormedFFN_F32
		t0_rmsnorm2 := time.Now()
		if t.dataType == DataTypeF32 {
			t.RMSNormFP32_Into(ffnNorm, eps, normedFFN)
		} else if useMixedPrecisionFFN {
			tCopy := t.ToF32()
			tCopy.RMSNormFP32_Into(ffnNorm, eps, normedFFN)
			tCopy.ReturnToPool()
		} else {
			t.RMSNormFP32_ToF16_Into(ffnNorm, eps, normedFFN)
		}
		metrics.RecordKernelDuration("Layer_RMSNorm2", time.Since(t0_rmsnorm2))

		gatePart := scratch.GatePart
		upPart := scratch.UpPart
		normedFFN.LinearF32_Into(ffnGate, gatePart, globalScale)
		normedFFN.LinearF32_Into(ffnUp, upPart, globalScale)

		swiOut := scratch.SwiOut
		gatePart.SwiGLU_FP32_Into(upPart, swiOut)

		resFFN := scratch.ResFFN_F32
		swiOut.LinearF32_Into(ffnDown, resFFN, globalScale)

		if t.dataType == DataTypeF32 {
			t.AddInPlace(resFFN)
		} else if useMixedPrecisionFFN {
			resFFNF16 := t.ctx.NewTensorPooled(t.rows, t.cols)
			resFFN.CopyToF16_Into(resFFNF16)
			C.Metal_Add_F16(t.ctx.ref, t.buf, C.int(t.Offset), resFFNF16.buf, C.int(resFFNF16.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
			resFFNF16.ReturnToPool()
		}
	} else {
		normedFFN := scratch.NormedFFN
		t0_rmsnorm2 := time.Now()
		if t.dataType == DataTypeF32 {
			t.RMSNormFP32_ToF16_Into(ffnNorm, eps, normedFFN)
		} else {
			C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, C.int(t.Offset), ffnNorm.buf, C.int(ffnNorm.Offset),
				normedFFN.buf, C.int(normedFFN.Offset), C.int(t.rows), C.int(t.cols), C.float(eps))
		}
		metrics.RecordKernelDuration("Layer_RMSNorm2", time.Since(t0_rmsnorm2))

		var gatePart *Tensor
		if ffnGate != nil {
			gatePart = t.ctx.NewTensorPooled(t.rows, ffnGate.rows)
			normedFFN.LinearInto(ffnGate, gatePart, globalScale)
		}

		upPart := t.ctx.NewTensorPooled(t.rows, ffnUp.rows)
		normedFFN.LinearInto(ffnUp, upPart, globalScale)

		resFFN := scratch.ResFFN
		if ffnGate != nil {
			switch ffnDown.dataType {
			case DataTypeQ4K:
				gatePart.SwiGLULinearIntoQ4K(upPart, ffnDown, resFFN, globalScale)
			case DataTypeQ6K:
				gatePart.SwiGLULinearIntoQ6K(upPart, ffnDown, resFFN, globalScale)
			default:
				swiOut, _ := upPart.SwiGLU(gatePart)
				swiOut.LinearInto(ffnDown, resFFN, globalScale)
				swiOut.ReturnToPool()
			}
			gatePart.ReturnToPool()
		} else {
			// Non-GLU Path (e.g. standard SiLU/GELU FFN)
			// Apply SiLU to upPart directly (in-place)
			upPart.SiLUInPlace()
			upPart.LinearInto(ffnDown, resFFN, globalScale)
		}
		upPart.ReturnToPool()

		if t.dataType == DataTypeF32 {
			t.AddMixedInPlace(resFFN)
		} else {
			C.Metal_Add_F16(t.ctx.ref, t.buf, C.int(t.Offset), resFFN.buf, C.int(resFFN.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
		}
	}
}

// LayerBatch executes a full transformer layer for a batch of sequences
func (t *Tensor) LayerBatch(layerIdx int, attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache *Tensor,
	scratch *LayerScratch,
	tokenPositions, tokenToSeqMapping, blockTables *Tensor, maxBlocksPerSeq int,
	heads, kvHeads, headDim int, ropeTheta, eps float32, hiddenDim, blockSize, numTokens int, globalScale float32,
	kvStoreBatch func(k, v *Tensor)) {

	if attnNorm == nil || q == nil || k == nil || v == nil || o == nil {
		return
	}

	normed := scratch.Normed

	// 1. RMSNorm (Batched)
	t0_rmsnorm1 := time.Now()
	C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, C.int(t.Offset), attnNorm.buf, C.int(attnNorm.Offset),
		normed.buf, C.int(normed.Offset), C.int(numTokens), C.int(t.cols), C.float(eps))
	metrics.RecordKernelDuration("LayerBatch_RMSNorm1", time.Since(t0_rmsnorm1))

	// 2. QKV Projections (Batched)
	qPart := scratch.QPart
	kPart := scratch.KPart
	vPart := scratch.VPart

	if q.dataType == DataTypeQ4K && k.dataType == DataTypeQ4K && v.dataType == DataTypeQ4K {
		C.Metal_RMSNormQKV_Q4K_F16(t.ctx.ref, t.buf, C.int(t.Offset), attnNorm.buf, C.int(attnNorm.Offset),
			q.buf, C.int(q.Offset), k.buf, C.int(k.Offset), v.buf, C.int(v.Offset),
			qPart.buf, C.int(qPart.Offset), kPart.buf, C.int(kPart.Offset), vPart.buf, C.int(vPart.Offset),
			C.int(t.cols), C.int(q.rows), C.int(k.rows), C.float(eps), C.float(globalScale), C.int(numTokens))
	} else if q.dataType == DataTypeQ6K && k.dataType == DataTypeQ6K && v.dataType == DataTypeQ6K {
		C.Metal_RMSNormQKV_Q6K_F16(t.ctx.ref, t.buf, C.int(t.Offset), attnNorm.buf, C.int(attnNorm.Offset),
			q.buf, C.int(q.Offset), k.buf, C.int(k.Offset), v.buf, C.int(v.Offset),
			qPart.buf, C.int(qPart.Offset), kPart.buf, C.int(kPart.Offset), vPart.buf, C.int(vPart.Offset),
			C.int(t.cols), C.int(q.rows), C.int(k.rows), C.float(eps), C.float(globalScale), C.int(numTokens))
	} else {
		normed.linearIntoInternal(q, qPart, globalScale)
		normed.linearIntoInternal(k, kPart, globalScale)
		normed.linearIntoInternal(v, vPart, globalScale)
	}

	// 3. RoPE (Batched Ragged)
	// We pass the token-specific positions directly to RoPE
	qPart.ropeInternal(tokenPositions, heads, headDim, numTokens, ropeTheta)
	kPart.ropeInternal(tokenPositions, kvHeads, headDim, numTokens, ropeTheta)

	// 4. Store K/V (Batched Paged)
	if kvStoreBatch != nil {
		kvStoreBatch(kPart, vPart)
	}

	// 5. Attention (Batched Paged Ragged)
	attOut := scratch.AttOut
	t.ctx.AttentionPagedBatch(qPart, kCache, vCache, attOut, tokenPositions, blockTables, maxBlocksPerSeq, heads, kvHeads, headDim, blockSize, tokenToSeqMapping, numTokens)

	// 6. Projections & Residual 1
	resAtt := scratch.ResAtt
	attOut.linearIntoInternal(o, resAtt, globalScale)
	t.AddInPlace(resAtt)

	// 7. FFN Part (Batched)
	normedFFN := scratch.NormedFFN
	C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, C.int(t.Offset), ffnNorm.buf, C.int(ffnNorm.Offset),
		normedFFN.buf, C.int(normedFFN.Offset), C.int(numTokens), C.int(t.cols), C.float(eps))

	gatePart := scratch.GatePart
	upPart := scratch.UpPart
	normedFFN.linearIntoInternal(ffnGate, gatePart, globalScale)
	normedFFN.linearIntoInternal(ffnUp, upPart, globalScale)

	// SwiGLU Fused
	C.Metal_SwiGLU_F16(t.ctx.ref, upPart.buf, C.int(upPart.Offset), gatePart.buf, C.int(gatePart.Offset),
		scratch.SwiOut.buf, C.int(scratch.SwiOut.Offset), C.int(numTokens), C.int(hiddenDim))

	resFFN := scratch.ResFFN
	scratch.SwiOut.linearIntoInternal(ffnDown, resFFN, globalScale)
	t.AddInPlace(resFFN)
}

func (t *Tensor) ropeInternal(pos interface{}, heads, headDim, seqLen int, ropeTheta float32) {
	switch p := pos.(type) {
	case int:
		C.Metal_RoPE_F16(t.ctx.ref, t.buf, C.int(t.Offset), 1, C.int(seqLen), C.int(heads), C.int(headDim), C.int(p), C.float(ropeTheta))
	case *Tensor:
		C.Metal_RoPE_Ragged_F16(t.ctx.ref, t.buf, C.int(t.Offset), p.buf, C.int(seqLen), C.int(heads), C.int(headDim), C.float(ropeTheta))
	}
}

func (t *Tensor) RoPE(posOffset, headDim, numHeads, seqLen int, ropeTheta float32) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.ropeInternal(posOffset, numHeads, headDim, seqLen, ropeTheta)
}

func (t *Tensor) swiGLUInternal(gate *Tensor) (*Tensor, error) {
	interSize := t.cols
	res := t.ctx.newTensorPooledInternal(t.rows, interSize)
	C.Metal_SwiGLU_F16(t.ctx.ref, t.buf, C.int(t.Offset), gate.buf, C.int(gate.Offset), res.buf, C.int(res.Offset), C.int(t.rows), C.int(interSize))
	return res, nil
}

func (t *Tensor) SwiGLU(gate *Tensor) (*Tensor, error) {
	if t.rows != gate.rows || t.cols != gate.cols {
		return nil, NewValidationError("SwiGLU", fmt.Sprintf("dimension mismatch: t[%d,%d] != gate[%d,%d]", t.rows, t.cols, gate.rows, gate.cols), "swiglu_dims")
	}
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	return t.swiGLUInternal(gate)
}

func (t *Tensor) Softmax() {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.softmaxInternal()
}

func (t *Tensor) softmaxInternal() {
	C.Metal_Softmax_F16(t.ctx.ref, t.buf, C.int(t.Offset), t.buf, C.int(t.Offset), C.int(t.rows), C.int(t.cols))
}

// FP32 FFN Methods for Small Models (SmolLM2, TinyLlama)

// LinearToFP32 performs weight × FP16 input → FP32 output
// Used for output head (Q6K * F16 -> F32) or small models
func (t *Tensor) LinearToFP32_Into(weight *Tensor, out *Tensor) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	logger.Log.Debug("LinearToFP32_Into", "t_rows", t.rows, "t_cols", t.cols, "t_type", t.dataType,
		"weight_rows", weight.rows, "weight_cols", weight.cols, "weight_type", weight.dataType,
		"out_rows", out.rows, "out_cols", out.cols, "out_type", out.dataType)
	switch weight.dataType {
	case DataTypeQ6K:
		// Output head logic: F16 input * Q6K weight -> F32 output
		C.Metal_LinearQ6K_F16_F32(t.ctx.ref, weight.buf, C.int(weight.Offset),
			t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), 1.0)
	case DataTypeQ4K:
		// Q4K -> F32 (Output Head)
		C.Metal_LinearQ4K_F16_F32(t.ctx.ref, weight.buf, C.int(weight.Offset),
			t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), 1.0)
	case DataTypeQ4_0:
		// Q4_0 -> F32 (Output Head)
		C.Metal_LinearQ4_0_F32(t.ctx.ref, weight.buf, C.int(weight.Offset),
			t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), 1.0)
	default:
		// Default: F16 weight * F16 input -> F32 output
		C.Metal_LinearF16ToF32(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	}
}

// LinearToFP32 performs FP16 weight × FP16 input → FP32 output
// Used for Gate/Up projections in FP32 FFN path
func (t *Tensor) LinearToFP32(weight *Tensor) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	out := t.ctx.newTensorFP32PooledInternal(t.rows, weight.rows)
	C.Metal_LinearF16ToF32(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
		C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	return out
}

// RMSNormFP32_Into performs RMSNorm (FP32 -> FP32)
// SiLUInPlace performs in-place element-wise SiLU activation
func (t *Tensor) SiLUInPlace() {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	C.Metal_SiLU_F16(t.ctx.ref, t.buf, C.int(t.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
}

func (t *Tensor) RMSNormFP32_Into(weight *Tensor, eps float32, out *Tensor) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	C.Metal_RMSNorm_F32(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))
}

// LinearF32_Into performs Linear into F32 output
// Used for Output Layer (Logits)
func (t *Tensor) linearF32IntoInternal(weight *Tensor, out *Tensor, scale float32) {
	switch weight.dataType {
	case DataTypeQ4K:
		if out.dataType == DataTypeF16 {
			C.Metal_MatMul_Q4K_F32_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
		} else {
			C.Metal_MatMul_Q4K_F32(t.ctx.ref, weight.buf, C.int(weight.Offset), 0, t.buf, C.int(t.Offset), 0, out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
		}
	case DataTypeQ6K:
		if out.dataType == DataTypeF16 {
			C.Metal_MatMul_Q6K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), false, t.buf, C.int(t.Offset), false, out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
		} else {
			C.Metal_MatMul_Q6K_F32(t.ctx.ref, weight.buf, C.int(weight.Offset), 0, t.buf, C.int(t.Offset), 0, out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
		}
	case DataTypeQ4_0:
		if out.dataType == DataTypeF16 {
			C.Metal_LinearQ4_0_F16(t.ctx.ref, weight.buf, C.int(weight.Offset),
				t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
				C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
		} else {
			C.Metal_LinearQ4_0_F32(t.ctx.ref, weight.buf, C.int(weight.Offset),
				t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
				C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
		}
	case DataTypeF16:
		C.Metal_MatMul_F16_F32_F32(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	case DataTypeQ8_0:
		C.Metal_LinearQ8_0_F32(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
	}
}

func (t *Tensor) LinearF32_Into(weight *Tensor, out *Tensor, scale float32) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.linearF32IntoInternal(weight, out, scale)
}

// SwiGLU_FP32 performs SwiGLU with FP32 inputs and outputs
func (gate *Tensor) SwiGLU_FP32_Into(up *Tensor, out *Tensor) {
	C.Metal_SwiGLU_F32(gate.ctx.ref, gate.buf, C.int(gate.Offset), up.buf, C.int(up.Offset), out.buf, C.int(out.Offset), C.int(gate.rows), C.int(gate.cols))
}

func (gate *Tensor) SwiGLU_FP32(up *Tensor) (*Tensor, error) {
	if gate.rows != up.rows || gate.cols != up.cols {
		return nil, NewValidationError("SwiGLU_FP32", fmt.Sprintf("dimension mismatch: gate[%d,%d] != up[%d,%d]", gate.rows, gate.cols, up.rows, up.cols), "swiglu_dims")
	}
	if gate.dataType != DataTypeF32 || up.dataType != DataTypeF32 {
		return nil, NewValidationError("SwiGLU_FP32", fmt.Sprintf("requires FP32 inputs, got gate=%v, up=%v", gate.dataType, up.dataType), "datatype")
	}
	gate.ctx.ExecMu.Lock()
	defer gate.ctx.ExecMu.Unlock()
	res := gate.ctx.newTensorFP32PooledInternal(gate.rows, gate.cols)
	C.Metal_SwiGLU_F32(gate.ctx.ref, gate.buf, C.int(gate.Offset), up.buf, C.int(up.Offset), res.buf, C.int(res.Offset), C.int(gate.rows), C.int(gate.cols))
	return res, nil
}

// LinearFromFP32 performs FP16 weight × FP32 input → FP16 output
// Used for Down projection in FP32 FFN path
func (t *Tensor) LinearFromFP32(weight *Tensor) (*Tensor, error) {
	if t.dataType != DataTypeF32 {
		return nil, NewValidationError("LinearFromFP32",
			fmt.Sprintf("requires FP32 input, got %v", t.dataType),
			"datatype")
	}
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	out := t.ctx.newTensorPooledInternal(t.rows, weight.rows)
	C.Metal_LinearF32ToF16(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
		C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	return out, nil
}

func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	if err := ValidateAddDimensions(t.rows, t.cols, other.rows, other.cols); err != nil {
		return nil, err
	}
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, t.cols)
	C.Metal_Add_F16(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res, nil
}

// AddInPlace performs t += other (FP32)
func (t *Tensor) AddInPlace(other *Tensor) error {
	if err := ValidateAddDimensions(t.rows, t.cols, other.rows, other.cols); err != nil {
		return err
	}
	if t.dataType != DataTypeF32 || other.dataType != DataTypeF32 {
		return NewValidationError("AddInPlace", "requires FP32 inputs", "datatype")
	}
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	C.Metal_Add_F32(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
	return nil
}

func (t *Tensor) EmbeddingLookup(row int, scale float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(1, t.cols)
	switch t.dataType {
	case DataTypeQ4K:
		// Use optimized Q4K embedding kernel for better performance
		C.Metal_Embedding_Q4K_Optimized(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols), C.float(scale))
	case DataTypeQ6K:
		// Q6K embedding - use FP16 kernel after dequantization
		C.Metal_Embedding_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols))
	case DataTypeQ4_0:
		C.Metal_EmbeddingQ4_0_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols))
	case DataTypeQ8_0:
		// Q8_0 embedding - use FP16 kernel after dequantization
		C.Metal_Embedding_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols))
	default:
		C.Metal_Embedding_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols))
	}
	return res
}

func (t *Tensor) storeKVInternal(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		// TurboQuant StoreKV
		bits := 2
		if kCache.dataType == DataTypeTQ2_0 {
			bits = 4
		}

		// For TurboQuant KV Cache, we use headDim as the blockSize
		blockSize := headDim
		qjlRows := 64 // Consistent with CPU device logic
		// numBlocks := heads // Each head is one or more blocks?
		// Actually, if headDim is blockSize, then each head is 1 block.

		if t.ctx.TQRotation == nil || t.ctx.TQQJL == nil {
			// Fallback to F16 if matrices are missing
			C.Metal_StoreKV_F16(t.ctx.ref, t.buf, C.int(t.Offset), v.buf, C.int(v.Offset), kCache.buf, C.int(kCache.Offset), vCache.buf, C.int(vCache.Offset), C.int(pos), C.int(heads), C.int(headDim), C.int(windowSize))
			return
		}

		// K Cache encoding
		// We need to calculate the physical offset in the circular buffer
		// The kernel handles modulo windowSize internally for non-TurboQuant.
		// For TurboQuant, we should probably pass the physical index too.
		physicalPos := pos
		if windowSize > 0 {
			physicalPos = pos % windowSize
		}

		// Each head is encoded separately
		// Note: The TurboQuant Meta kernels might handle multiple blocks.
		// 1 head = headDim elements.

		// K Encode
		C.Metal_TurboQuant_Encode(t.ctx.ref, t.buf, C.int(t.Offset),
			t.ctx.TQRotation.buf, C.int(t.ctx.TQRotation.Offset),
			t.ctx.TQQJL.buf, C.int(t.ctx.TQQJL.Offset),
			kCache.buf, C.int(kCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)),
			kCache.buf, C.int(kCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)+heads*(blockSize+qjlRows)), // scale offset
			kCache.buf, C.int(kCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)+heads*(blockSize+qjlRows)+4), // qjlScale offset
			C.int(blockSize), C.int(qjlRows), C.int(heads), C.int(bits))

		// V Encode
		C.Metal_TurboQuant_Encode(t.ctx.ref, v.buf, C.int(v.Offset),
			t.ctx.TQRotation.buf, C.int(t.ctx.TQRotation.Offset),
			t.ctx.TQQJL.buf, C.int(t.ctx.TQQJL.Offset),
			vCache.buf, C.int(vCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)),
			vCache.buf, C.int(vCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)+heads*(blockSize+qjlRows)), // scale offset
			vCache.buf, C.int(vCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)+heads*(blockSize+qjlRows)+4), // qjlScale offset
			C.int(blockSize), C.int(qjlRows), C.int(heads), C.int(bits))

		return
	}
	C.Metal_StoreKV_F16(t.ctx.ref, t.buf, C.int(t.Offset), v.buf, C.int(v.Offset), kCache.buf, C.int(kCache.Offset), vCache.buf, C.int(vCache.Offset), C.int(pos), C.int(heads), C.int(headDim), C.int(windowSize))
}

func (t *Tensor) StoreKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.storeKVInternal(v, kCache, vCache, pos, heads, headDim, windowSize)
}

func (t *Tensor) StoreKVQuantized(v, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	// For now, fall back to StoreKV to satisfy interface/build
	t.StoreKV(v, kCache, vCache, pos, heads, headDim, windowSize)
}

func (t *Tensor) FetchKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()

	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		if t.ctx.TQRotation == nil {
			// For TurboQuant, rotation is mandatory for decoding.
			// If missing, we skip or could implement a zero-fallback, but that would produce garbage.
			return
		}
		blockSize := headDim
		qjlRows := 64
		physicalPos := pos
		if windowSize > 0 {
			physicalPos = pos % windowSize
		}

		// K Decode
		C.Metal_TurboQuant_Decode(t.ctx.ref, kCache.buf, C.int(kCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)),
			t.ctx.TQRotation.buf, C.int(t.ctx.TQRotation.Offset),
			t.ctx.TQQJL.buf, C.int(t.ctx.TQQJL.Offset),
			t.buf, C.int(t.Offset),
			kCache.buf, C.int(kCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)+heads*(blockSize+qjlRows)), // scale offset
			C.int(blockSize), C.int(qjlRows), C.int(heads))

		// V Decode
		C.Metal_TurboQuant_Decode(t.ctx.ref, vCache.buf, C.int(vCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)),
			t.ctx.TQRotation.buf, C.int(t.ctx.TQRotation.Offset),
			t.ctx.TQQJL.buf, C.int(t.ctx.TQQJL.Offset),
			v.buf, C.int(v.Offset),
			vCache.buf, C.int(vCache.Offset+physicalPos*heads*(blockSize+qjlRows+8)+heads*(blockSize+qjlRows)), // scale offset
			C.int(blockSize), C.int(qjlRows), C.int(heads))
		return
	}

	// Standard F16 copy
	count := heads * headDim
	C.Metal_Copy_F16(t.ctx.ref, kCache.buf, C.int(kCache.Offset+pos*count*2), t.buf, C.int(t.Offset), C.int(count))
	C.Metal_Copy_F16(t.ctx.ref, vCache.buf, C.int(vCache.Offset+pos*count*2), v.buf, C.int(v.Offset), C.int(count))
}

func (t *Tensor) PrepareTQQuery(rotationMatrix, qjlMatrix *Tensor, headDim, qjlRows, numHeads int) (*Tensor, *Tensor) {
	qPrime := t.ctx.newTensorInternal(numHeads, headDim)
	qDoublePrime := t.ctx.newTensorInternal(numHeads, qjlRows)

	C.Metal_Prepare_TQ_Query(t.ctx.ref, t.buf, C.int(t.Offset),
		rotationMatrix.buf, C.int(rotationMatrix.Offset),
		qjlMatrix.buf, C.int(qjlMatrix.Offset),
		qPrime.buf, C.int(qPrime.Offset),
		qDoublePrime.buf, C.int(qDoublePrime.Offset),
		C.int(headDim), C.int(qjlRows), C.int(numHeads))

	return qPrime, qDoublePrime
}

func (t *Tensor) attentionTQInternal(kCache, vCache *Tensor, pos, numHeads, kvHeads, headDim, ctxLen, windowSize int) *Tensor {
	qjlRows := 64 // Standard TQ config

	// Step 1: Prepare Query
	qPrime, qDoublePrime := t.PrepareTQQuery(t.ctx.TQRotation, t.ctx.TQQJL, headDim, qjlRows, numHeads)
	defer qPrime.freeInternal()
	defer qDoublePrime.freeInternal()

	// Step 2: Fused Scoring
	scoresDim := numHeads * ctxLen
	if scoresDim < 32768 {
		scoresDim = 32768
	}
	scores := t.ctx.newTensorFP32PooledInternal(1, scoresDim)
	defer scores.ReturnToPool()

	smScale := 1.0 / float32(math.Sqrt(float64(headDim)))

	C.Metal_Attention_TQ_Scores_F16(t.ctx.ref,
		qPrime.buf, C.int(qPrime.Offset),
		qDoublePrime.buf, C.int(qDoublePrime.Offset),
		kCache.buf, C.int(kCache.Offset),
		scores.buf, C.int(scores.Offset),
		C.int(headDim), C.int(qjlRows), C.int(pos),
		C.int(numHeads), C.int(kvHeads), C.float(smScale))

	scores.softmaxInternal()

	res := t.ctx.newTensorPooledInternal(1, numHeads*headDim)
	C.Metal_Attention_TQ_Values_F16(t.ctx.ref,
		scores.buf, C.int(scores.Offset),
		vCache.buf, C.int(vCache.Offset),
		res.buf, C.int(res.Offset),
		C.int(headDim), C.int(qjlRows), C.int(pos),
		C.int(numHeads), C.int(kvHeads))

	return res
}

func (t *Tensor) attentionInternal(kCache, vCache *Tensor, pos, numHeads, kvHeads, headDim, ctxLen, windowSize int) *Tensor {
	res := t.ctx.newTensorPooledInternal(1, numHeads*headDim)
	scoresDim := numHeads * ctxLen
	if scoresDim < 32768 {
		scoresDim = 32768
	}
	scores := t.ctx.newTensorFP32PooledInternal(1, scoresDim)
	defer scores.ReturnToPool()
	C.Metal_Attention_F16(t.ctx.ref, t.buf, C.int(t.Offset), kCache.buf, C.int(kCache.Offset), vCache.buf, C.int(vCache.Offset), res.buf, C.int(res.Offset), scores.buf, C.int(scores.Offset), C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim), C.int(ctxLen), C.int(windowSize))
	return res
}

func (t *Tensor) Attention(kCache, vCache *Tensor, pos, numHeads, kvHeads, headDim, ctxLen, windowSize int) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()

	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		return t.attentionTQInternal(kCache, vCache, pos, numHeads, kvHeads, headDim, ctxLen, windowSize)
	}

	return t.attentionInternal(kCache, vCache, pos, numHeads, kvHeads, headDim, ctxLen, windowSize)
}

// AttFused performs fused attention (Score + Softmax + Value Aggregation)
// Returns error if output tensor dimensions are invalid for the attention computation
func (t *Tensor) AttFused(kCache, vCache *Tensor, out *Tensor, pos, numHeads, kvHeads, headDim, windowSize int) error {
	expectedOutRows := 1
	expectedOutCols := numHeads * headDim

	if out.Rows() != expectedOutRows || out.Cols() != expectedOutCols {
		return NewValidationError("AttFused",
			fmt.Sprintf("output tensor dimensions [%d,%d] do not match expected [%d,%d]",
				out.Rows(), out.Cols(), expectedOutRows, expectedOutCols),
			"attention_output")
	}

	if numHeads%kvHeads != 0 {
		return NewValidationError("AttFused",
			fmt.Sprintf("numHeads (%d) must be divisible by kvHeads (%d)", numHeads, kvHeads),
			"gqa_config")
	}

	if pos < 0 {
		return NewValidationError("AttFused",
			fmt.Sprintf("invalid position: %d (must be non-negative)", pos),
			"position")
	}

	maxCtxLen := kCache.Cols()
	if maxCtxLen == 0 {
		maxCtxLen = pos + 1
	}

	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	C.Metal_AttFused_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		kCache.buf, C.int(kCache.Offset),
		vCache.buf, C.int(vCache.Offset),
		out.buf, C.int(out.Offset),
		C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim), C.int(windowSize), C.int(maxCtxLen))
	return nil
}

// FP32 Operations

func (t *Tensor) RMSNormFP32_ToF16(weight *Tensor, eps float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, t.cols) // Result is F16
	t.rmsNormFP32ToF16IntoInternal(weight, eps, res)
	return res
}

func (t *Tensor) rmsNormFP32ToF16IntoInternal(weight *Tensor, eps float32, out *Tensor) {
	C.Metal_RMSNorm_F32_F16(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset), C.int(t.rows), C.int(t.cols), C.float(eps))
}

func (t *Tensor) RMSNormFP32_ToF16_Into(weight *Tensor, eps float32, out *Tensor) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.rmsNormFP32ToF16IntoInternal(weight, eps, out)
}

func (t *Tensor) RMSNormFP32(weight *Tensor, eps float32) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorFP32Internal(t.rows, t.cols)
	// Input t is F32, Weight is F16
	C.Metal_RMSNorm_F32(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), res.buf, C.int(res.Offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))
	return res
}

func (t *Tensor) LinearIntoFP32(weight *Tensor, out *Tensor, scale float32) {
	t.LinearF32_Into(weight, out, scale)
}

func (t *Tensor) addMixedInPlaceInternal(other *Tensor) {
	C.Metal_Add_F32_F16(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
}

func (t *Tensor) AddMixedInPlace(other *Tensor) error {
	if err := ValidateAddDimensions(t.rows, t.cols, other.rows, other.cols); err != nil {
		return err
	}
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.addMixedInPlaceInternal(other)
	return nil
}

func (t *Tensor) ToF32() *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorFP32PooledInternal(t.rows, t.cols)
	C.Metal_Copy_F16_F32(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) AddFP32(other *Tensor) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorFP32Internal(t.rows, t.cols)
	C.Metal_Add_F32(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) SwiGLUFP32(gate *Tensor) *Tensor {
	// t is up (F32), gate is gate (F32)
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorFP32Internal(t.rows, t.cols)
	C.Metal_SwiGLU_F32(t.ctx.ref, t.buf, C.int(t.Offset), gate.buf, C.int(gate.Offset), res.buf, C.int(res.Offset), C.int(t.rows), C.int(t.cols))
	return res
}

func (t *Tensor) SwiGLU_F32_InPlace(gate *Tensor) {
	C.Metal_SwiGLU_F32(t.ctx.ref, t.buf, C.int(t.Offset), gate.buf, C.int(gate.Offset), t.buf, C.int(t.Offset), C.int(t.rows), C.int(t.cols))
}

func (t *Tensor) CopyToF32() *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorFP32Internal(t.rows, t.cols)
	C.Metal_Copy_F16_F32(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) ScaleInPlace(val float32) {
	C.Metal_Scale_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.float(val), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
}

func (t *Tensor) CopyToF16() *Tensor {
	res := t.ctx.NewTensor(t.rows, t.cols)
	C.Metal_Copy_F32_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) CopyToF16_Into(dest *Tensor) error {
	if t.rows != dest.rows || t.cols != dest.cols {
		return NewValidationError("CopyToF16_Into",
			fmt.Sprintf("dimension mismatch: src[%d,%d] != dest[%d,%d]",
				t.rows, t.cols, dest.rows, dest.cols),
			"copy_dims")
	}
	C.Metal_Copy_F32_F16(t.ctx.ref, t.buf, C.int(t.Offset), dest.buf, C.int(dest.Offset), C.int(t.rows*t.cols))
	return nil
}

func (t *Tensor) CopyF32Into(dest *Tensor) error {
	if t.rows != dest.rows || t.cols != dest.cols {
		return NewValidationError("CopyF32Into",
			fmt.Sprintf("dimension mismatch: src[%d,%d] != dest[%d,%d]",
				t.rows, t.cols, dest.rows, dest.cols),
			"copy_dims")
	}
	C.Metal_Copy_F32(t.ctx.ref, t.buf, C.int(t.Offset), dest.buf, C.int(dest.Offset), C.int(t.rows*t.cols))
	return nil
}

func (t *Tensor) ToF32InPlace(res *Tensor) {
	C.Metal_Copy_F16_F32(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
}

// LoadBuffer loads raw bytes into a tensor's buffer
func (c *Context) LoadBuffer(t *Tensor, data []byte) {
	C.Metal_CopyToDevice(t.buf, C.int(0), unsafe.Pointer(&data[0]), C.int(len(data)))
}

// Test helper methods

// LoadFromF32 loads F32 data into tensor (converts to F16 if needed)
func (t *Tensor) LoadFromF32(data []float32) {
	t.LoadFrom(data)
}

// ToHostF32 retrieves tensor data as F32 (converts from F16 if needed)
func (t *Tensor) ToHostF32() []float32 {
	if t.dataType == DataTypeF32 {
		return t.ToHost()
	}
	// Convert F16 to F32
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		return make([]float32, t.rows*t.cols)
	}
	f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)
	f32Data := make([]float32, t.rows*t.cols)
	for i, v := range f16Slice {
		f32Data[i] = Float16ToFloat32(v)
	}
	return f32Data
}

// AttentionScores computes Q·K^T scores with scaling
func (t *Tensor) AttentionScores(kCache *Tensor, scores *Tensor, pos, numHeads, kvHeads, headDim, stride, windowSize int) {
	// DEBUG: Trace args
	fmt.Printf("[Metal] AttentionScores: pos=%d heads=%d kvheads=%d dim=%d stride=%d win=%d\n", pos, numHeads, kvHeads, headDim, stride, windowSize)

	C.Metal_AttScores_F16(
		t.ctx.ref,
		t.buf, C.int(t.Offset),
		kCache.buf, C.int(kCache.Offset),
		scores.buf, C.int(scores.Offset),
		C.int(pos),
		C.int(numHeads),
		C.int(kvHeads),
		C.int(headDim),
		C.int(stride),
		C.int(windowSize),
	)
}

// NewTensorF32 creates a new F32 tensor (alias for NewTensorFP32)
func (c *Context) NewTensorF32(rows, cols int) *Tensor {
	return c.NewTensorFP32(rows, cols)
}

// DebugRoPEFreq runs the debug kernel to compute RoPE Frequencies
func (t *Tensor) DebugRoPEFreq(headDim int, theta float32, pos int) {
	C.Metal_DebugRoPEFreq(t.ctx.ref, t.buf, C.int(headDim), C.float(theta), C.int(pos))
}

// DebugDot computes dot product using debug kernel
func (t *Tensor) DebugDot(b *Tensor, output *Tensor, dim int) {
	C.Metal_DebugDot(t.ctx.ref, t.buf, b.buf, output.buf, C.int(dim))
}

// StoreKV stores K and V projections into their respective caches
// AttSoftmax performs attention softmax [Heads, Stride]
func (t *Tensor) attSoftmaxInternal(pos, heads, stride int) {
	C.Metal_AttSoftmax_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.int(pos), C.int(heads), C.int(stride))
}

func (t *Tensor) AttSoftmax(pos, heads, stride int) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.attSoftmaxInternal(pos, heads, stride)
}

// AttValues performs attention value aggregation
func (t *Tensor) attValuesInternal(vCache, out *Tensor, pos, numHeads, kvHeads, headDim, stride, windowSize int) {
	C.Metal_AttValues_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		vCache.buf, C.int(vCache.Offset),
		out.buf, C.int(out.Offset),
		C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim), C.int(stride), C.int(windowSize))
}

func (t *Tensor) AttValues(vCache, out *Tensor, pos, numHeads, kvHeads, headDim, stride, windowSize int) {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	t.attValuesInternal(vCache, out, pos, numHeads, kvHeads, headDim, stride, windowSize)
}

// CopyFromAt copies src tensor into this tensor at specified row offset
func (t *Tensor) CopyFromAt(src *Tensor, destRow int) {
	off := destRow * t.cols * 2 // F16 bytes
	if t.dataType == DataTypeF32 {
		off = destRow * t.cols * 4
	}
	C.Metal_Copy_F16(t.ctx.ref, src.buf, C.int(src.Offset), t.buf, C.int(t.Offset+off), C.int(src.rows*src.cols))
}

// EmbeddingLookupBatch performs embedding lookup for multiple tokens at once
func (t *Tensor) EmbeddingLookupBatch(tokens []int, scale float32) *Tensor {
	if t == nil {
		return nil
	}
	ctx := t.ctx
	batchSize := len(tokens)
	output := ctx.NewTensorWithType(batchSize, t.cols, DataTypeF16)
	for i, token := range tokens {
		emb := t.EmbeddingLookup(token, scale)
		output.CopyFromAt(emb, i)
		emb.Free()
	}
	return output
}

// AttPaged performs paged attention
func (t *Tensor) attPagedInternal(kCache, vCache, blockTable *Tensor, output *Tensor, pos, nh, kh, hd, blockSize int) {
	C.Metal_AttPaged_F16(
		t.ctx.ref,
		t.buf, C.int(t.Offset),
		kCache.buf, C.int(kCache.Offset),
		vCache.buf, C.int(vCache.Offset),
		output.buf, C.int(output.Offset),
		blockTable.buf, C.int(blockTable.Offset),
		C.int(pos),
		C.int(nh),
		C.int(kh),
		C.int(hd),
		C.int(blockSize),
		C.int(4096), // maxCtxLen (used for internal limit)
	)
}

// SiLU performs element-wise SiLU activation
func (t *Tensor) SiLU() *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_SiLU_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

// Slice extracts a sub-tensor of numCols from totalCols starting at startCol
func (t *Tensor) Slice(startCol, numCols int) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, numCols)
	C.Metal_Slice_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(startCol), C.int(numCols), C.int(t.cols), C.int(t.rows))
	return res
}

// MambaScan executes a single SSM scan step
func (t *Tensor) MambaScan(h, A, B, ssmC, D, dt *Tensor, dState int) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, t.cols)
	C.Metal_MambaScan_F16(t.ctx.ref, t.buf, C.int(t.Offset), h.buf, C.int(h.Offset), A.buf, C.int(A.Offset), B.buf, C.int(B.Offset), ssmC.buf, C.int(ssmC.Offset), D.buf, C.int(D.Offset), dt.buf, C.int(dt.Offset), res.buf, C.int(res.Offset), C.int(t.cols), C.int(dState))
	return res
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	t.ctx.ExecMu.Lock()
	defer t.ctx.ExecMu.Unlock()
	res := t.ctx.newTensorPooledInternal(t.rows, t.cols)
	C.Metal_Mul_F16(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

// ============================================================================
// MOE (Mixture of Experts) Operations
// ============================================================================

// MOERouterLogits computes routing logits for MOE layer
// input: [batch_size, dim]
// gateWeight: [num_experts, dim]
// Returns: [batch_size, num_experts] logits
func (ctx *Context) MOERouterLogits(input, gateWeight *Tensor) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	numExperts := gateWeight.Rows()
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	logits := ctx.newTensorFP32Internal(batchSize, numExperts)
	C.Metal_MOE_RouterLogits(ctx.ref,
		input.buf, C.int(input.Offset),
		gateWeight.buf, C.int(gateWeight.Offset),
		logits.buf, C.int(logits.Offset),
		C.int(batchSize), C.int(dim), C.int(numExperts))
	return logits
}

// MOETopKSelection selects top-k experts per token and computes softmax weights
// logits: [batch_size, num_experts]
// Returns: expertIndices [batch_size, top_k] (int32), expertWeights [batch_size, top_k] (float32)
func (ctx *Context) MOETopKSelection(logits *Tensor, topK int) (*Tensor, *Tensor) {
	batchSize := logits.Rows()
	numExperts := logits.Cols()
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	expertIndices := ctx.newTensorFP32Internal(batchSize, topK) // Will store int32
	expertWeights := ctx.newTensorFP32Internal(batchSize, topK)

	C.Metal_MOE_TopKSelection(ctx.ref,
		logits.buf, C.int(logits.Offset),
		expertIndices.buf, C.int(expertIndices.Offset),
		expertWeights.buf, C.int(expertWeights.Offset),
		C.int(batchSize), C.int(numExperts), C.int(topK))

	return expertIndices, expertWeights
}

// MOEExpertForward applies selected experts to input with weighted mixing
// input: [batch_size, dim]
// expertWeight: [hidden_dim * num_experts, dim] (flattened 3D)
// expertIndices: [batch_size, top_k] (int32)
// expertWeights: [batch_size, top_k] (float32)
// Returns: [batch_size, hidden_dim]
func (ctx *Context) MOEExpertForward(input, expertWeight, expertIndices, expertWeights *Tensor, hiddenDim int) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	topK := expertIndices.Cols()
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	output := ctx.newTensorPooledInternal(batchSize, hiddenDim)
	C.Metal_MOE_ExpertForward(ctx.ref, input.buf, C.int(input.Offset), expertWeight.buf, C.int(expertWeight.Offset), expertIndices.buf, C.int(expertIndices.Offset), expertWeights.buf, C.int(expertWeights.Offset), output.buf, C.int(output.Offset), C.int(batchSize), C.int(dim), C.int(hiddenDim), C.int(topK))
	return output
}

// MOEExpertGateUpSwiGLU applies fused gate, up and SwiGLU forward pass for multiple experts
func (ctx *Context) MOEExpertGateUpSwiGLU(input, gateWeight, upWeight, expertIndices, expertWeights *Tensor, hiddenDim int) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	topK := expertIndices.Cols()
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	output := ctx.newTensorPooledInternal(batchSize, hiddenDim)
	C.Metal_MOE_ExpertGateUpSwiGLU(ctx.ref, input.buf, C.int(input.Offset), gateWeight.buf, C.int(gateWeight.Offset), upWeight.buf, C.int(upWeight.Offset), expertIndices.buf, C.int(expertIndices.Offset), expertWeights.buf, C.int(expertWeights.Offset), output.buf, C.int(output.Offset), C.int(batchSize), C.int(dim), C.int(hiddenDim), C.int(topK))
	return output
}

// ============================================================================
// TurboQuant Operations (GPU Accelerated) - Requires additional CGO bindings
// ============================================================================

/*
func (ctx *Context) TurboQuantQJLTransform(residual, signMatrix *Tensor, rows, cols int) (quantized *Tensor, scale float32) {
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	quantized = ctx.newTensorInt8Internal(rows, 1)
	scaleTensor := ctx.newTensorFP32Internal(1, 1)
	C.Metal_TurboQuant_QJLTransform(ctx.ref, residual.buf, C.int(residual.Offset), signMatrix.buf, C.int(signMatrix.Offset), quantized.buf, C.int(quantized.Offset), scaleTensor.buf, C.int(scaleTensor.Offset), C.int(rows), C.int(cols))
	scaleData := scaleTensor.Data()
	if len(scaleData) > 0 {
		scale = scaleData[0]
	}
	return quantized, scale
}

func (ctx *Context) TurboQuantEncode(input, rotationMatrix, qjlMatrix *Tensor, blockSize, qjlRows, bits int) (output *Tensor, scale, qjlScale float32) {
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	outputSize := blockSize + qjlRows
	output = ctx.newTensorInt8Internal(outputSize, 1)
	scaleTensor := ctx.newTensorFP32Internal(1, 1)
	qjlScaleTensor := ctx.newTensorFP32Internal(1, 1)
	C.Metal_TurboQuant_Encode(ctx.ref, input.buf, C.int(input.Offset), rotationMatrix.buf, C.int(rotationMatrix.Offset), qjlMatrix.buf, C.int(qjlMatrix.Offset), output.buf, C.int(output.Offset), scaleTensor.buf, C.int(scaleTensor.Offset), qjlScaleTensor.buf, C.int(qjlScaleTensor.Offset), C.int(blockSize), C.int(qjlRows), C.int(bits))
	scaleData := scaleTensor.Data()
	qjlScaleData := qjlScaleTensor.Data()
	if len(scaleData) > 0 {
		scale = scaleData[0]
	}
	if len(qjlScaleData) > 0 {
		qjlScale = qjlScaleData[0]
	}
	return output, scale, qjlScale
}

func (ctx *Context) TurboQuantDecode(input, rotationMatrix, scaleIn *Tensor, blockSize, qjlRows int) *Tensor {
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	output := ctx.newTensorFP32Internal(blockSize, 1)
	C.Metal_TurboQuant_Decode(ctx.ref, input.buf, C.int(input.Offset), rotationMatrix.buf, C.int(rotationMatrix.Offset), output.buf, C.int(output.Offset), scaleIn.buf, C.int(scaleIn.Offset), C.int(blockSize), C.int(qjlRows))
	return output
}
*/

// FlashAttention2 executes the memory-fused FlashAttention-2 kernel for paged KV cache
func (ctx *Context) FlashAttention2(q, kCache, vCache, output *Tensor, seqLens *Tensor, numHeads, kvHeads, headDim, blockSize int, blockTable *Tensor, maxBlocksPerSeq int, tokenToSeq *Tensor, batchSize int) {
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	C.Metal_FlashAttention2_F16(ctx.ref, q.buf, kCache.buf, vCache.buf, output.buf,
		C.int(numHeads), C.int(kvHeads), C.int(headDim), seqLens.buf, C.int(blockSize),
		blockTable.buf, C.int(maxBlocksPerSeq), tokenToSeq.buf, C.int(batchSize))
}

// AttentionPagedBatch performs paged attention across a batch of sequences
func (ctx *Context) AttentionPagedBatch(q, kCache, vCache, output, tokenPositions, blockTables *Tensor, maxBlocksPerSeq, heads, kvHeads, headDim, blockSize int, tokenToSeq *Tensor, batchSize int) {
	ctx.FlashAttention2(q, kCache, vCache, output, tokenPositions, heads, kvHeads, headDim, blockSize, blockTables, maxBlocksPerSeq, tokenToSeq, batchSize)
}

// StoreKVPagedBatch stores K and V projections for a batch of sequences into their respective physical blocks
func (ctx *Context) StoreKVPagedBatch(k, v, kCache, vCache, physicalPositions *Tensor, kvDim, batchSize int) {
	ctx.ExecMu.Lock()
	defer ctx.ExecMu.Unlock()
	C.Metal_StoreKVPagedBatch_F16(ctx.ref, k.buf, C.int(k.Offset), v.buf, C.int(v.Offset), kCache.buf, C.int(kCache.Offset), vCache.buf, C.int(vCache.Offset), physicalPositions.buf, C.int(physicalPositions.Offset), C.int(kvDim), C.int(batchSize))
}

