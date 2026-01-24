//go:build darwin && metal

package device

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework Accelerate
#include "metal_bridge.h"
#include <stdlib.h>
void* Metal_AutoreleasePoolPush();
void Metal_AutoreleasePoolPop(void* pool);
void* Metal_NewHeap(MetalContextRef ctx, long long size);
MetalBufferRef Metal_NewBufferFromHeap(void* heap, long long size);
void Metal_FreeHeap(void* heap);
*/
import "C"
import (
	_ "embed"
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

var allocatedBytes int64

// MaxGPUMemory is a soft limit for total allocations (default: 8GB)
var MaxGPUMemory int64 = 8 * 1024 * 1024 * 1024

//go:embed kernels.metal
var kernelsSource string

// Context holds the Metal connection and tensor pool
type Context struct {
	ref  C.MetalContextRef
	mu   sync.Mutex
	pool map[string][]*Tensor // pool by size key "RxCxType"
}

func NewContext() *Context {
	cSrc := C.CString(kernelsSource)
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
	if c.ref != nil {
		c.ClearPool()
		C.Metal_Free(c.ref)
		c.ref = nil
	}
}

// ClearPool releases all pooled tensors to free up GPU memory.
func (c *Context) ClearPool() {
	c.mu.Lock()
	defer c.mu.Unlock()

	for key, tensors := range c.pool {
		for _, t := range tensors {
			C.Metal_FreeBuffer(c.ref, t.buf)
			atomic.AddInt64(&allocatedBytes, -int64(t.sizeBytes))
			metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
			t.buf = nil // Prevent double free
		}
		delete(c.pool, key)
	}
}

type DataType int

const (
	DataTypeF16  DataType = 0
	DataTypeQ4K  DataType = 1
	DataTypeQ4_0 DataType = 2
	DataTypeQ3K  DataType = 5
	DataTypeF32  DataType = 3
	DataTypeQ6K  DataType = 4
)

// Tensor wraps a Metal buffer. Always FP16 for this engine.
type Tensor struct {
	ctx       *Context
	rows      int
	cols      int
	sizeBytes int
	buf       C.MetalBufferRef
	Offset    int      // Offset in bytes from buf start
	dataType  DataType // 0=F16, 1=Q4K, 2=Q3K
}

func (t *Tensor) Rows() int { return t.rows }
func (t *Tensor) Cols() int { return t.cols }

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

	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: int(sizeBytes),
		buf:       buf,
		dataType:  DataTypeQ3K,
	}, nil
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

	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: int(sizeBytes),
		buf:       buf,
		dataType:  DataTypeQ4K,
	}, nil
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
	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: sizeBytes,
		buf:       buf,
		dataType:  DataTypeQ6K,
	}, nil
}

// NewTensor creates a standard F16 tensor
func (c *Context) NewTensor(rows, cols int) *Tensor {
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

	atomic.AddInt64(&allocatedBytes, int64(sizeBytes))
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))

	runtime.SetFinalizer(t, func(ft *Tensor) {
		C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
		atomic.AddInt64(&allocatedBytes, -int64(ft.sizeBytes))
		metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
	})

	return t
}

// Free explicitly releases the Metal buffer.
// Use this for large tensors in tight loops to avoid OOM due to lazy GC finalizers.
func (t *Tensor) Free() {
	if t.buf != nil {
		// Clear finalizer first to prevent double free
		runtime.SetFinalizer(t, nil)
		C.Metal_FreeBuffer(t.ctx.ref, t.buf)
		atomic.AddInt64(&allocatedBytes, -int64(t.sizeBytes))
		metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
		t.buf = nil // Mark as freed
	}
}

// NewTensorFP32 creates a standard F32 tensor
// NewTensorFP32Pooled creates or reuses a pooled tensor for intermediate float32 results.
func (c *Context) NewTensorFP32Pooled(rows, cols int) *Tensor {
	key := fmt.Sprintf("%dx%dx%d", rows, cols, DataTypeF32)

	c.mu.Lock()
	if tensors, ok := c.pool[key]; ok && len(tensors) > 0 {
		// fmt.Printf("DEBUG_POOL: Reuse F32 %s\n", key)
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		c.mu.Unlock()
		C.Metal_ZeroBufferGPU(c.ref, t.buf, C.int(0), C.int(t.sizeBytes))
		return t
	}
	c.mu.Unlock()

	// fmt.Printf("DEBUG_POOL: Alloc F32 %s\n", key)
	return c.NewTensorFP32(rows, cols)
}

func (c *Context) NewTensorFP32(rows, cols int) *Tensor {
	// ... (unchanged)
	sizeBytes := rows * cols * 4
	buf := C.Metal_Alloc(c.ref, C.longlong(sizeBytes))
	if buf == nil {
		panic("Metal_Alloc returned nil!")
	}
	t := &Tensor{ctx: c, buf: buf, sizeBytes: sizeBytes, rows: rows, cols: cols, dataType: DataTypeF32}
	atomic.AddInt64(&allocatedBytes, int64(sizeBytes))
	metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
	runtime.SetFinalizer(t, func(ft *Tensor) {
		C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
		atomic.AddInt64(&allocatedBytes, -int64(ft.sizeBytes))
		metrics.RecordGPUMemory(atomic.LoadInt64(&allocatedBytes))
	})
	return t
}

func (c *Context) NewTensorWithType(rows, cols int, dt DataType) *Tensor {
	// ... (unchanged - just for context for the next block if needed, but I can target specific ranges)
	sb := rows * cols * 2
	if dt == DataTypeF32 {
		sb = rows * cols * 4
	} else if dt == DataTypeQ6K {
		numElements := rows * cols
		numBlocks := numElements / 256
		sb = numBlocks * 210
		// ...
	} else if dt == DataTypeQ4K {
		numElements := rows * cols
		// Validation? Usually blocked by 256.
		numBlocks := numElements / 256
		sb = numBlocks * 144
	} else if dt == DataTypeQ4_0 {
		numElements := rows * cols
		if numElements%32 != 0 {
			// Pad if needed? Or Panic?
			// GGUF Usually padded.
			//Panic for now.
			panic(fmt.Sprintf("Q4_0 tensor size %d not divisible by 32", numElements))
		}
		numBlocks := numElements / 32
		sb = numBlocks * 18
	}
	buf := C.Metal_Alloc(c.ref, C.longlong(sb))
	t := &Tensor{ctx: c, rows: rows, cols: cols, sizeBytes: sb, buf: buf, dataType: dt}
	atomic.AddInt64(&allocatedBytes, int64(sb))
	runtime.SetFinalizer(t, func(ft *Tensor) {
		C.Metal_FreeBuffer(ft.ctx.ref, ft.buf)
		atomic.AddInt64(&allocatedBytes, -int64(ft.sizeBytes))
	})
	return t
}

// NewTensorPooled attempts to reuse tensor from pool (defaults to F16)
func (c *Context) NewTensorPooled(rows, cols int) *Tensor {
	key := fmt.Sprintf("%dx%dx%d", rows, cols, DataTypeF16)

	c.mu.Lock()
	if tensors, ok := c.pool[key]; ok && len(tensors) > 0 {
		// Pop from pool
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		c.mu.Unlock()
		C.Metal_ZeroBufferGPU(c.ref, t.buf, C.int(0), C.int(t.sizeBytes))
		return t
	}
	c.mu.Unlock()

	// Fallback to new allocation
	return c.NewTensor(rows, cols)
}

// ReturnToPool returns tensor to pool for reuse.
// Note: This does NOT free the Metal memory, just prevents GC from reaping it.
func (t *Tensor) ReturnToPool() {
	key := fmt.Sprintf("%dx%dx%d", t.rows, t.cols, t.dataType)

	t.ctx.mu.Lock()
	t.ctx.pool[key] = append(t.ctx.pool[key], t)
	t.ctx.mu.Unlock()
}

func (t *Tensor) LoadFrom(data []float32) error {
	if len(data) != t.rows*t.cols {
		return NewValidationError("LoadFrom",
			fmt.Sprintf("data size %d does not match tensor size %d",
				len(data), t.rows*t.cols),
			"tensor_data")
	}

	if t.dataType == DataTypeF32 {
		// Copy directly as F32
		C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(len(data)*4))
		return nil
	}

	if t.dataType == DataTypeF32 {
		C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(t.sizeBytes))
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
	if len(data) > t.sizeBytes {
		return NewValidationError("LoadRaw",
			fmt.Sprintf("raw data size %d exceeds tensor buffer size %d", len(data), t.sizeBytes),
			"tensor_data")
	}
	C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(len(data)))
	return nil
}

// LoadFromBytes copies raw bytes to the buffer (for Q4K data, etc.)
func (t *Tensor) LoadFromBytes(data []byte) {
	C.Metal_CopyToDevice(t.buf, C.int(t.Offset), unsafe.Pointer(&data[0]), C.int(len(data)))
}

func (t *Tensor) Probe(name string, n int) {
	t.ctx.Synchronize()
	// data := t.ToHost()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		fmt.Printf("DEBUG_PROBE: %s -> BUFFER IS NIL!\n", name)
		return
	}

	// Access as uint16 slice
	f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)

	// Convert first n to float32
	f32Data := make([]float32, n)
	for i := 0; i < n && i < len(f16Slice); i++ {
		f32Data[i] = Float16ToFloat32(f16Slice[i])
	}

	fmt.Printf("DEBUG_PROBE: %s [%d]: %v\n", name, len(f16Slice), f32Data)
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
	if len(data) >= 32 {
		fmt.Printf("[%s] Sample: %v\n", name, data[:32])
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
	C.Metal_ZeroBuffer(t.buf, C.int(t.Offset), C.int(t.sizeBytes))
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
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_Scale_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.float(val), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

// Operations

// MatMul performs matrix multiplication C = A * B
func (t *Tensor) MatMul(b *Tensor) *Tensor {
	// A=t: [N x K] (Weights)
	// B=b: [M x K] (Input)
	M := b.rows
	N := t.rows
	K := t.cols

	// If t is Q4_K, dispatch specialized kernel
	t0 := time.Now()
	if t.dataType == DataTypeQ4K {
		c := t.ctx.NewTensor(N, M)
		c.ZeroInit()
		C.Metal_MatMul_Q4K_F16(t.ctx.ref,
			t.buf, C.int(t.Offset), C.bool(false),
			b.buf, C.int(b.Offset), C.bool(false),
			c.buf, C.int(c.Offset),
			C.int(M), C.int(N), C.int(K), C.float(1.0))

		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	} else if t.dataType == DataTypeQ3K {
		c := t.ctx.NewTensor(N, M)
		c.ZeroInit()
		C.Metal_MatMul_Q3K_F16(t.ctx.ref,
			t.buf, C.int(t.Offset), C.bool(false),
			b.buf, C.int(b.Offset), C.bool(false),
			c.buf, C.int(c.Offset),
			C.int(M), C.int(N), C.int(K))

		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	} else if t.dataType == DataTypeQ6K {
		c := t.ctx.NewTensor(N, M)
		c.ZeroInit()
		C.Metal_MatMul_Q6K_F16(t.ctx.ref,
			t.buf, C.int(t.Offset), C.bool(false),
			b.buf, C.int(b.Offset), C.bool(false),
			c.buf, C.int(c.Offset),
			C.int(M), C.int(N), C.int(K), C.float(1.0))

		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	}

	c := t.ctx.NewTensor(N, M)
	C.Metal_MatMul_F16(t.ctx.ref,
		t.buf, C.int(t.Offset), C.bool(false),
		b.buf, C.int(b.Offset), C.bool(false),
		c.buf, C.int(c.Offset),
		C.int(M), C.int(N), C.int(K))
	metrics.RecordKernelDuration("MatMul", time.Since(t0))
	return c
}

// Linear performs t * weight^T
// t: [M, K], weight: [N, K] -> result: [M, N]
// Returns error if dimensions are incompatible
func (t *Tensor) Linear(weight *Tensor) (*Tensor, error) {
	if err := ValidateLinearDimensions(t.cols, weight.cols); err != nil {
		return nil, err
	}
	t0 := time.Now()
	res := t.ctx.NewTensorPooled(t.rows, weight.rows) // [M, N]

	// MatMul(A, B^T)
	// Swap arguments: A=weight, B=t.
	// We want Weight (Matrix) as buffer 0 (primary stride source)
	// and Input (Vector) as buffer 1 (broadcast source).

	if weight.dataType == DataTypeQ4K {
		C.Metal_MatMul_Q4K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), false, t.buf, C.int(t.Offset), false, res.buf, C.int(res.Offset),
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(1.0))
	} else {
		// BatchedMatMul_F16 using MPS
		M := t.rows
		N := weight.rows
		K := weight.cols
		sA := K * 2
		sB := K * 2
		sC := N * 2

		C.Metal_BatchedMatMul_F16(t.ctx.ref,
			t.buf, C.int(t.Offset), C.int(sA), false,
			weight.buf, C.int(weight.Offset), C.int(sB), true,
			res.buf, C.int(res.Offset), C.int(sC),
			C.int(M), C.int(N), C.int(K), 1)
	}
	metrics.RecordKernelDuration("Linear", time.Since(t0))
	return res, nil
}

// LinearInto performs Linear using existing output tensor (scratch buffer)
// Returns error if dimensions are incompatible
func (t *Tensor) LinearInto(weight *Tensor, out *Tensor, scale float32) error {
	if t.rows != out.rows || weight.rows != out.cols {
		return NewValidationError("LinearInto",
			fmt.Sprintf("dimension mismatch: [%d,%d] * [%d,%d] -> [%d,%d]",
				t.rows, t.cols, weight.rows, weight.cols, out.rows, out.cols),
			"linear_dims")
	}

	// DEBUG: Output Layer Check
	// if weight.rows > 30000 && weight.cols > 1000 {
	// 	fmt.Printf("DEBUG: LinearInto Output Layer. Dims: [%d, %d]. Type: %d. Scale: %f\n", weight.rows, weight.cols, weight.dataType, scale)
	// }

	if t.dataType == DataTypeF32 {
		t.LinearF32_Into(weight, out, scale)
		return nil
	}

	// Use MatMul Kernel
	if weight.dataType == DataTypeQ4K {
		// Q4K weights * F16 input -> F16 output
		// Swap N, K args to match kernel dim_in (K), dim_out (N)
		C.Metal_MatMul_Q4K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), C.bool(false), t.buf, C.int(t.Offset), C.bool(false), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
	} else if weight.dataType == DataTypeQ6K {
		// Q6K weights * F16 input -> F16 output
		// Swap N, K args
		C.Metal_MatMul_Q6K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), C.bool(false), t.buf, C.int(t.Offset), C.bool(false), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
	} else if weight.dataType == DataTypeQ4_0 {
		// Q4_0 Support
		// Pass K, N (swapped) to match kernel dim_in, dim_out
		C.Metal_LinearQ4_0_F16(t.ctx.ref, weight.buf, C.int(weight.Offset),
			t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
	} else {
		// Fallback F16
		C.Metal_MatMul_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), C.bool(false), t.buf, C.int(t.Offset), C.bool(false), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	}
	return nil
}

// RunQ3K_Explicit for testing only
func (c *Context) RunQ3K_Explicit(w, in, out *Tensor) {
	C.Metal_MatMul_Q3K_F16(c.ref,
		w.buf, C.int(w.Offset), C.bool(false),
		in.buf, C.int(in.Offset), C.bool(false),
		out.buf, C.int(out.Offset),
		C.int(1), C.int(w.rows), C.int(w.cols))
}

func (t *Tensor) RMSNorm(weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), res.buf, C.int(res.Offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))
	return res
}

// RMSNormLinear performs fused RMSNorm + Linear in single kernel
// Eliminates intermediate buffer allocation
func (t *Tensor) RMSNormLinear(normWeight, weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, weight.rows)
	C.Metal_RMSNormLinear_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		normWeight.buf, C.int(normWeight.Offset),
		weight.buf, C.int(weight.Offset), res.buf, C.int(res.Offset),
		C.int(t.cols), C.int(weight.rows), C.float(eps))
	return res
}

// RMSNormLinearQ4K performs fused RMSNorm + Linear (Q4_K)
func (t *Tensor) RMSNormLinearQ4K(normWeight, weight *Tensor, eps float32, scale float32) *Tensor {
	M := t.rows
	N := weight.rows
	res := t.ctx.NewTensorPooled(M, N)
	t.RMSNormLinearIntoQ4K(normWeight, weight, res, eps, scale)
	return res
}

// RMSNormLinearIntoQ4K performs fused RMSNorm + Linear (Q4_K) into existing destination
func (t *Tensor) RMSNormLinearIntoQ4K(normWeight, weight, out *Tensor, eps float32, scale float32) {
	M := t.rows
	N := weight.rows
	K := weight.cols
	C.Metal_RMSNormLinear_Q4K_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		normWeight.buf, C.int(normWeight.Offset),
		weight.buf, C.int(weight.Offset),
		out.buf, C.int(out.Offset),
		C.int(M), C.int(N), C.int(K), C.float(eps), C.float(scale))
}

// SwiGLULinearQ4K performs fused SwiGLU + Linear (Q4_K)
func (t *Tensor) SwiGLULinearQ4K(up, weight *Tensor, scale float32) *Tensor {
	M := t.rows
	N := weight.rows
	res := t.ctx.NewTensorPooled(M, N)
	t.SwiGLULinearIntoQ4K(up, weight, res, scale)
	return res
}

// SwiGLULinearIntoQ4K performs fused SwiGLU + Linear (Q4_K) into existing destination
func (t *Tensor) SwiGLULinearIntoQ4K(up, weight, out *Tensor, scale float32) {
	M := t.rows
	N := weight.rows
	K := weight.cols
	C.Metal_SwiGLULinear_Q4K_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		up.buf, C.int(up.Offset),
		weight.buf, C.int(weight.Offset),
		out.buf, C.int(out.Offset),
		C.int(M), C.int(N), C.int(K), C.float(scale))
}

// RMSNormQKV performs fused RMSNorm + QKV Linear projections
func (t *Tensor) RMSNormQKV(normWeight, wQ, wK, wV *Tensor, eps float32) (*Tensor, *Tensor, *Tensor) {
	q := t.ctx.NewTensorPooled(t.rows, wQ.rows)
	k := t.ctx.NewTensorPooled(t.rows, wK.rows)
	v := t.ctx.NewTensorPooled(t.rows, wV.rows)
	C.Metal_RMSNormQKV_F16(t.ctx.ref, t.buf, C.int(t.Offset), normWeight.buf, C.int(normWeight.Offset),
		wQ.buf, C.int(wQ.Offset), wK.buf, C.int(wK.Offset), wV.buf, C.int(wV.Offset),
		q.buf, C.int(q.Offset), k.buf, C.int(k.Offset), v.buf, C.int(v.Offset),
		C.int(t.cols), C.int(wQ.rows), C.int(wK.rows), C.float(eps))
	return q, k, v
}

// FusedFFN performs one entire FFN block: RMSNorm + Gate/Up Linear + SwiGLU + Down Linear
func (t *Tensor) FusedFFN(normWeight, wGate, wUp, wDown *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_FusedFFN_F16(t.ctx.ref, t.buf, C.int(t.Offset), normWeight.buf, C.int(normWeight.Offset),
		wGate.buf, C.int(wGate.Offset), wUp.buf, C.int(wUp.Offset), wDown.buf, C.int(wDown.Offset), res.buf, C.int(res.Offset),
		C.int(t.cols), C.int(wGate.rows), C.float(eps))
	return res
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

	// Logits (FP32)
	Logits *Tensor // [1, VocabSize]

	heap unsafe.Pointer // Reference to Metal Heap
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
	return C.Metal_NewHeap(c.ref, C.longlong(size))
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
		Offset:    0,
		dataType:  dt,
	}
}

func (c *Context) NewLayerScratch(batch, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize int) *LayerScratch {
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

	szScores := align(heads * seqLen * 4)
	if szScores < align(32768*4) {
		szScores = align(32768 * 4)
	}

	szGate := align(batch * hiddenDim * 4)
	szUp := align(batch * hiddenDim * 4)
	szSwiOut := align(batch * hiddenDim * 4)

	szLogits := align(1 * vocabSize * 4) // F32 Logits

	total := szNormed + szQPart + szKPart + szVPart + szAttOut + szResAtt +
		szNormedFFN + szNormedFFN_F32 + szResFFN + szResFFN_F32 + szScores + szGate + szUp + szSwiOut + szLogits

	fmt.Printf("Alloc Heap: %d bytes\n", total)
	heap := C.Metal_NewHeap(c.ref, C.longlong(total))
	if heap == nil {
		panic("Heap Alloc failed")
	}

	newT := func(sz, r, cols int, dt DataType) *Tensor {
		buf := C.Metal_NewBufferFromHeap(heap, C.longlong(sz))
		if buf == nil {
			panic("Buffer from Heap failed")
		}
		// Manually retain? Metal_NewBufferFromHeap does manual retain (in backend).
		// Tensor needs Free() to release manual retain.
		sb := r * cols * 2
		if dt == DataTypeF32 {
			sb = r * cols * 4
		}
		return &Tensor{
			ctx:       c,
			rows:      r,
			cols:      cols,
			sizeBytes: sb,
			buf:       buf,
			Offset:    0,
			dataType:  dt,
			// We need a way to release Heap?
			// The Tensor does NOT own the Heap.
			// The Heap leaks if not released.
			// We should attach Heap to LayerScratch to free it?
			// Or let it leak (1 object).
		}
	}
	// We need to store heap ref to free it?
	s.heap = heap // Add field to struct

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

	// Logits must be F32 to preserve precision during accumulation and sampling
	s.Logits = newT(szLogits, 1, vocabSize, DataTypeF32)

	return s
}

// Free releases all buffers
func (s *LayerScratch) Free() {
	if s.heap != nil {
		// All tensors (except maybe Scores?) are backed by this heap.
		// Releasing the Heap invalidates them.
		// However, we manually retained buffers?
		// Metal_NewBufferFromHeap retains the buffer.
		// So buffers are valid until we release them.
		// If we release Heap, do buffers die?
		// "Buffers maintain a strong reference to the Heap".
		// So we must release Buffers FIRST.
		// Then Heap.
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

	if s.heap != nil {
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
	pos, heads, kvHeads, headDim int, ropeTheta, eps float32, hiddenDim, ctxLen, windowSize int, globalScale float32, debug bool, precisionMode int) {

	// Probes for debugging activations
	probe := func(tag string, t *Tensor) {
		if debug && traceTracker.IsEnabled() {
			_, stats := t.ScanMax(fmt.Sprintf("[Layer %d %s]", layerIdx, tag))
			traceTracker.RecordLayer(tag, layerIdx, stats)
		}
	}

	probe("Input", t)

	// Use scratch buffers instead of allocating
	normed := scratch.Normed
	// Norm into scratch
	// RMSNormFP32_ToF16 usually allocates.
	// We need RMSNormFP32_ToF16_Into(dest).
	t0_rmsnorm1 := time.Now()
	if t.dataType == DataTypeF32 {
		t.RMSNormFP32_ToF16_Into(attnNorm, eps, normed)
	} else {
		// t.RMSNorm(attnNorm, eps) // Allocates. Need Into.
		// For now assume FP32 residual
	}
	metrics.RecordKernelDuration("Layer_RMSNorm1", time.Since(t0_rmsnorm1))
	probe("Norm-1", normed)

	// 2. QKV Projections (Using Scratch)
	qPart := scratch.QPart
	kPart := scratch.KPart
	vPart := scratch.VPart

	doLin := func(w, dst *Tensor, tag string) {
		t0_lin := time.Now()
		if w.dataType == DataTypeQ4K && t.dataType == DataTypeF32 {
			t.RMSNormLinearIntoQ4K(attnNorm, w, dst, eps, globalScale)
		} else {
			normed.LinearInto(w, dst, globalScale)
		}
		metrics.RecordKernelDuration("Layer_Linear_"+tag, time.Since(t0_lin))
		probe(tag, dst)
	}

	doLin(q, qPart, "Q-Proj")
	doLin(k, kPart, "K-Proj")
	doLin(v, vPart, "V-Proj")

	// 3. RoPE & 4. Store K/V & 5. Attention (Serial Loop for Prefill/Batch)
	// Our RoPE, Attention and StoreKV kernels currently only support single-token processing.
	// We must loop over t.rows.

	// Create outputs
	attOut := scratch.AttOut

	resAtt := scratch.ResAtt // [Batch, 4096] - Output of O projection

	kvStride := kvHeads * headDim * 2 // bytes (F16)
	qStride := heads * headDim * 2    // bytes (F16)

	for i := 0; i < t.rows; i++ { // For prefill, t.rows > 1
		p := pos + i

		// Offsets
		offQ := i * qStride

		offK := i * kvStride
		offV := i * kvStride

		offAtt := i * qStride

		// 3. RoPE (In-place on qPart/kPart row i)

		// Process Q (Heads)
		t0_rope_q := time.Now()
		C.Metal_RoPE_F16(t.ctx.ref, qPart.buf, C.int(qPart.Offset+offQ), 1, 1, C.int(heads), C.int(headDim), C.int(p), C.float(ropeTheta))
		metrics.RecordKernelDuration("Layer_RoPE_Q", time.Since(t0_rope_q))
		// Process K (KVHeads)
		t0_rope_k := time.Now()
		C.Metal_RoPE_F16(t.ctx.ref, kPart.buf, C.int(kPart.Offset+offK), 1, 1, C.int(kvHeads), C.int(headDim), C.int(p), C.float(ropeTheta))
		metrics.RecordKernelDuration("Layer_RoPE_K", time.Since(t0_rope_k))

		// 4. Store K/V (Must append to KVCache at current pos)
		t0_storekv := time.Now()
		C.Metal_StoreKV_F16(t.ctx.ref, kPart.buf, C.int(kPart.Offset+offK), vPart.buf, C.int(vPart.Offset+offV),
			kCache.buf, C.int(kCache.Offset), vCache.buf, C.int(vCache.Offset),
			C.int(p), C.int(kvHeads), C.int(headDim), C.int(windowSize))
		metrics.RecordKernelDuration("Layer_StoreKV", time.Since(t0_storekv))

		if p < 2 {

		}

		// 5. Attention

		// 5. Attention (Split Debug Path)
		// We need scratch buffers for Scores and Probs.
		// Since we don't have dedicated scratch in Tensor struct yet, we can use a temporary buffer or repurpose.
		// However, to keep it simple and clean, let's just create them temporarily using pool if performance is not key.
		// Or assume 'scratch' has them? It does not.
		// Let's implement the split call IF we had the buffers.
		//
		// Wait, 'att_fused' is much better.
		// If we want to debug, we can use the Fused kernel but maybe print inside the shader? No printf in Metal.

		// To debug Scores, we MUST use split kernels.
		// Let's use `Metal_AttScores_F16` -> `Metal_Softmax_F16` -> `Metal_AttValues_F16`.
		// We need an intermediate buffer for scores: [Heads, MaxSeqLen].
		// MaxSeqLen=4096. Heads=32. Size = 32*4096*2 = 256KB. Small.

		// scores := t.ctx.NewTensorPooled(heads, ctxLen) // [Heads, CtxLen]
		// But we need to handle the loop.

		// Force Split Path for Debug
		if false {
			// Fused Path (Current)
			maxCtxLen := kCache.Cols()
			if maxCtxLen == 0 {
				maxCtxLen = p + 1
			}
			C.Metal_AttFused_F16(t.ctx.ref, qPart.buf, C.int(qPart.Offset+offQ),
				kCache.buf, C.int(kCache.Offset),
				vCache.buf, C.int(vCache.Offset),
				attOut.buf, C.int(attOut.Offset+offAtt),
				C.int(p), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(windowSize), C.int(maxCtxLen))
		} else {
			// Split Path
			scores := scratch.Scores

			// DEBUG: Check heads
			// if i == 0 {
			// 	fmt.Printf("[Metal Layer] C.Metal_AttScores_F16 Call: p=%d heads=%d kvHeads=%d dim=%d stride=%d\n", p, heads, kvHeads, headDim, ctxLen)
			// }

			t0_scores := time.Now()
			C.Metal_AttScores_F16(t.ctx.ref, qPart.buf, C.int(qPart.Offset+offQ),
				kCache.buf, C.int(kCache.Offset),
				scores.buf, C.int(scores.Offset), // Dst
				C.int(p), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(ctxLen), C.int(windowSize))
			metrics.RecordKernelDuration("Layer_AttScores", time.Since(t0_scores))

			// DEBUG: Dump Scores (Pre-Softmax)
			if layerIdx == 0 && i == 0 && p < 5 {
				t.ctx.Synchronize()
				rawScores := scores.ToHostF32() // F16 -> F32
				// scores is [Heads, CtxLen]
				// Print Head 0, first p+1 values
				fmt.Printf("DEBUG: Layer 0, Head 0, Pos %d Scores: %v\n", p, rawScores[:p+2])
			}

			t0_softmax := time.Now()
			C.Metal_AttSoftmax_F16(t.ctx.ref, scores.buf, C.int(scores.Offset),
				C.int(p), C.int(heads), C.int(ctxLen))
			metrics.RecordKernelDuration("Layer_Softmax", time.Since(t0_softmax))

			t0_attvalues := time.Now()
			C.Metal_AttValues_F16(t.ctx.ref, scores.buf, C.int(scores.Offset),
				vCache.buf, C.int(vCache.Offset),
				attOut.buf, C.int(attOut.Offset+offAtt),
				C.int(p), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(ctxLen), C.int(windowSize))
			metrics.RecordKernelDuration("Layer_AttValues", time.Since(t0_attvalues))
		}
	}

	// 6. Attention Output Projection
	t0_attn_out := time.Now()
	scratch.AttOut.LinearInto(o, resAtt, globalScale)
	metrics.RecordKernelDuration("Layer_AttnOut", time.Since(t0_attn_out))
	probe("Att-Final", resAtt)
	probe("Att-Out", scratch.AttOut)

	// 7. Residual Add 1
	t0_add1 := time.Now()
	if t.dataType == DataTypeF32 {
		t.AddMixedInPlace(resAtt)
	} else {
		t1, _ := t.Add(resAtt)
		C.Metal_Copy_F16(t.ctx.ref, t1.buf, C.int(t1.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
		t1.ReturnToPool()
	}
	metrics.RecordKernelDuration("Layer_Add1", time.Since(t0_add1))
	probe("Residual-1-Add", t)

	// --- No Cleanup needed for Scratch ---

	// --- FFN Block ---

	// Precision mode: 0=Auto, 1=FP16, 2=F32FFN (small models), 3=Mixed (large models)
	useF32FFN := precisionMode == 2
	useMixedPrecisionFFN := precisionMode == 3

	// 8. FFN Block

	if useF32FFN || useMixedPrecisionFFN {
		// FP32 or Mixed Precision FFN Path
		// 8. FFN Norm (F32 -> F32)
		normedFFN := scratch.NormedFFN_F32
		t0_rmsnorm2 := time.Now()
		if t.dataType == DataTypeF32 {
			t.RMSNorm_F32_Into(ffnNorm, eps, normedFFN)
		} else if useMixedPrecisionFFN {
			// For mixed precision, convert to F32 for FFN computations
			tCopy := t.ToF32()
			tCopy.RMSNorm_F32_Into(ffnNorm, eps, normedFFN)
			tCopy.ReturnToPool()
		} else {
			t.RMSNormFP32_ToF16_Into(ffnNorm, eps, normedFFN)
		}
		metrics.RecordKernelDuration("Layer_RMSNorm2", time.Since(t0_rmsnorm2))
		probe("FFN-Norm", normedFFN)

		// 9. Gate/Up Projections -> FP32
		gatePart := scratch.GatePart
		upPart := scratch.UpPart

		t0_ffn_gate := time.Now()
		normedFFN.LinearF32_Into(ffnGate, gatePart, globalScale)
		metrics.RecordKernelDuration("Layer_FFN_Gate", time.Since(t0_ffn_gate))
		t0_ffn_up := time.Now()
		normedFFN.LinearF32_Into(ffnUp, upPart, globalScale)
		metrics.RecordKernelDuration("Layer_FFN_Up", time.Since(t0_ffn_up))
		probe("FFN-Gate-Proj", gatePart)
		probe("FFN-Up-Proj", upPart)

		// 10. SwiGLU F32
		swiOut := scratch.SwiOut
		t0_swiglu := time.Now()
		gatePart.SwiGLU_FP32_Into(upPart, swiOut)
		metrics.RecordKernelDuration("Layer_SwiGLU", time.Since(t0_swiglu))
		probe("FFN-SwiGLU", swiOut)

		// 11. Down Projection (FP32 -> FP32)
		resFFN := scratch.ResFFN_F32
		t0_ffn_down := time.Now()
		swiOut.LinearF32_Into(ffnDown, resFFN, globalScale) // Handles Q4K
		metrics.RecordKernelDuration("Layer_FFN_Down", time.Since(t0_ffn_down))
		probe("FFN-Down-Proj", resFFN)

		// 12. Residual Add 2
		t0_add2 := time.Now()
		if t.dataType == DataTypeF32 {
			t.AddInPlace(resFFN)
		} else if useMixedPrecisionFFN {
			// Convert back to F16 for residual add
			resFFNF16 := t.ctx.NewTensorPooled(t.rows, t.cols)
			resFFN.CopyToF16_Into(resFFNF16)
			t1, _ := t.Add(resFFNF16)
			C.Metal_Copy_F16(t.ctx.ref, t1.buf, C.int(t1.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
			t1.ReturnToPool()
			resFFNF16.ReturnToPool()
		} else {
			// Fallback (Should not happen in useF32FFN path)
			t2, _ := t.Add(resFFN) // Mixed
			C.Metal_Copy_F16(t.ctx.ref, t2.buf, C.int(t2.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
			t2.ReturnToPool()
		}
		metrics.RecordKernelDuration("Layer_Add2", time.Since(t0_add2))
		probe("Residual-2-Add", t)

		// Debug FFN Output
		probe("FFN-Final", resFFN)
		probe("Gate", gatePart)
		probe("Up", upPart)
	} else {
		// FP16 FFN Path (Original)
		normedFFN := scratch.NormedFFN
		t0_rmsnorm2 := time.Now()
		t.RMSNormFP32_ToF16_Into(ffnNorm, eps, normedFFN)
		metrics.RecordKernelDuration("Layer_RMSNorm2", time.Since(t0_rmsnorm2))
		probe("FFN-Norm-FP16", normedFFN)

		// 9. FFN Gate/Up
		// For FP16 path, we need new F16 tensors for gate/up parts.
		// scratch.GatePart and scratch.UpPart are assumed to be FP32 for the 'if' block.
		// Actually scratch has GatePart and UpPart but they are FP32.
		// Let's just use NewTensorPooled for safety and ReturnToPool.

		// Debug: check for nil pointers and invalid dimensions
		if t == nil {
			fmt.Printf("DEBUG: t is nil\n")
			return
		} else if t.ctx == nil {
			fmt.Printf("DEBUG: t.ctx is nil\n")
			return
		} else if ffnGate == nil {
			fmt.Printf("DEBUG: ffnGate is nil\n")
			return
		} else if ffnUp == nil {
			fmt.Printf("DEBUG: ffnUp is nil\n")
			return
		} else if t.rows <= 0 || ffnGate.rows <= 0 || ffnUp.rows <= 0 {
			fmt.Printf("DEBUG: Invalid tensor dimensions - t.rows=%d, ffnGate.rows=%d, ffnUp.rows=%d\n",
				t.rows, ffnGate.rows, ffnUp.rows)
			return
		}

		// Additional debug: check Metal context validity
		if t.ctx.ref == nil {
			fmt.Printf("DEBUG: t.ctx.ref is nil - Metal context not initialized properly\n")
			return
		}

		// Debug allocation sizes
		gateSize := t.rows * ffnGate.rows * 2 // FP16 = 2 bytes
		upSize := t.rows * ffnUp.rows * 2
		fmt.Printf("DEBUG: Attempting FFN tensor allocation - Gate: %dx%d (%d bytes), Up: %dx%d (%d bytes)\n",
			t.rows, ffnGate.rows, gateSize, t.rows, ffnUp.rows, upSize)
		fmt.Printf("DEBUG: Current allocated bytes: %d, Max GPU memory: %d\n",
			atomic.LoadInt64(&allocatedBytes), MaxGPUMemory)

		gatePartF16_P := t.ctx.NewTensorPooled(t.rows, ffnGate.rows)
		upPartF16_P := t.ctx.NewTensorPooled(t.rows, ffnUp.rows)

		t0_ffn_gate := time.Now()
		if false && ffnGate.dataType == DataTypeQ4K {
			t.RMSNormLinearIntoQ4K(ffnNorm, ffnGate, gatePartF16_P, eps, globalScale)
			t.RMSNormLinearIntoQ4K(ffnNorm, ffnUp, upPartF16_P, eps, globalScale)
		} else {
			normedFFN.LinearInto(ffnGate, gatePartF16_P, globalScale)
			normedFFN.LinearInto(ffnUp, upPartF16_P, globalScale)
		}
		metrics.RecordKernelDuration("Layer_FFN_GateUp", time.Since(t0_ffn_gate))
		probe("FFN-Gate-Proj-FP16", gatePartF16_P)
		probe("FFN-Up-Proj-FP16", upPartF16_P)

		if layerIdx == 0 {
			normedFFN.ScanMax("Layer 0 FFN-Input (Normed)")
			gatePartF16_P.ScanMax("Layer 0 FFN-Gate")
			upPartF16_P.ScanMax("Layer 0 FFN-Up")
		}

		// 10. SwiGLU + 11. Down Projection
		resFFN := scratch.ResFFN // F16 scratch
		t0_swiglu_down := time.Now()
		if false && ffnDown.dataType == DataTypeQ4K {
			gatePartF16_P.SwiGLULinearIntoQ4K(upPartF16_P, ffnDown, resFFN, globalScale)
		} else {
			swiOut, _ := upPartF16_P.SwiGLU(gatePartF16_P)
			swiOut.LinearInto(ffnDown, resFFN, globalScale)
			swiOut.ReturnToPool()
		}
		metrics.RecordKernelDuration("Layer_SwiGLU_Down", time.Since(t0_swiglu_down))
		probe("FFN-Down-Proj-FP16", resFFN)

		gatePartF16_P.ReturnToPool()
		upPartF16_P.ReturnToPool()

		// 12. Residual Add 2
		t0_add2 := time.Now()
		if t.dataType == DataTypeF32 {
			t.AddMixedInPlace(resFFN)
		} else {
			t2, _ := t.Add(resFFN)
			C.Metal_Copy_F16(t.ctx.ref, t2.buf, C.int(t2.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
			t2.ReturnToPool()
		}
		metrics.RecordKernelDuration("Layer_Add2", time.Since(t0_add2))
		probe("Residual-2-Add-FP16", t)

		// NaN Detection at layer output
		t.ctx.Synchronize()
		data := t.ToHost()
		nanInfo := DetectNaN(data, 10)
		if nanInfo.HasNaN() {
			log.Printf("[LAYER_DEBUG] NaN detected at layer output layerIdx=%d: count=%d, positions=%v",
				layerIdx, nanInfo.Count, nanInfo.Positions)
		}

		// Debug FFN Output (Correct Location)
		if layerIdx == 0 || layerIdx == 5 {
			resFFN.ScanMax(fmt.Sprintf("Layer %d FFN-Final", layerIdx))
		}

		// --- Final Cleanup ---
		normedFFN.ReturnToPool()
		resFFN.ReturnToPool()
		// t2 returned if used
		// t2 returned if used
	}
}

// Correct RoPE implementation using arguments expected by Kernel
func (t *Tensor) RoPE(posOffset, headDim, numHeads, seqLen int, ropeTheta float32) {
	C.Metal_RoPE_F16(t.ctx.ref, t.buf, C.int(t.Offset), 1, C.int(seqLen), C.int(numHeads), C.int(headDim), C.int(posOffset), C.float(ropeTheta))
}

func (t *Tensor) SwiGLU(gate *Tensor) (*Tensor, error) {
	if t.rows != gate.rows || t.cols != gate.cols {
		return nil, NewValidationError("SwiGLU",
			fmt.Sprintf("dimension mismatch: t[%d,%d] != gate[%d,%d]",
				t.rows, t.cols, gate.rows, gate.cols),
			"swiglu_dims")
	}
	interSize := t.cols
	res := t.ctx.NewTensorPooled(t.rows, interSize)

	C.Metal_SwiGLU_F16(t.ctx.ref, t.buf, C.int(t.Offset), gate.buf, C.int(gate.Offset), res.buf, C.int(res.Offset), C.int(t.rows), C.int(interSize))
	return res, nil
}

func (t *Tensor) Softmax() {
	C.Metal_Softmax_F16(t.ctx.ref, t.buf, C.int(t.Offset), t.buf, C.int(t.Offset), C.int(t.rows), C.int(t.cols))
}

// FP32 FFN Methods for Small Models (SmolLM2, TinyLlama)

// LinearToFP32 performs weight × FP16 input → FP32 output
// Used for output head (Q6K * F16 -> F32) or small models
func (t *Tensor) LinearToFP32_Into(weight *Tensor, out *Tensor) {
	if weight.dataType == DataTypeQ6K {
		// Output head logic: F16 input * Q6K weight -> F32 output
		// weight shape: [vocabSize, dim], input shape: [batch, dim], output shape: [batch, vocabSize]
		// dimIn = weight.cols (input dimension), dimOut = weight.rows (vocab size)
		C.Metal_LinearQ6K_F16_F32(t.ctx.ref, weight.buf, C.int(weight.Offset),
			t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), 1.0)
	} else if weight.dataType == DataTypeQ4_0 {
		// Q4_0 -> F32 (Output Head)
		// weight shape: [vocabSize, dim], input shape: [batch, dim], output shape: [batch, vocabSize]
		// dimIn = weight.cols (input dimension), dimOut = weight.rows (vocab size)
		C.Metal_LinearQ4_0_F32(t.ctx.ref, weight.buf, C.int(weight.Offset),
			t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.cols), C.int(weight.rows), 1.0)
	} else {
		// Default: F16 weight * F16 input -> F32 output
		C.Metal_LinearF16ToF32(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	}
}

// LinearToFP32 performs FP16 weight × FP16 input → FP32 output
// Used for Gate/Up projections in FP32 FFN path
func (t *Tensor) LinearToFP32(weight *Tensor) *Tensor {
	out := t.ctx.NewTensorFP32Pooled(t.rows, weight.rows)
	C.Metal_LinearF16ToF32(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
		C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	return out
}

// RMSNorm_F32_Into performs RMSNorm (FP32 -> FP32)
func (t *Tensor) RMSNorm_F32_Into(weight *Tensor, eps float32, out *Tensor) {
	C.Metal_RMSNorm_F32(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))
}

// LinearF32_Into performs Linear into F32 output
// Used for Output Layer (Logits)
func (t *Tensor) LinearF32_Into(weight *Tensor, out *Tensor, scale float32) {
	if weight.dataType == DataTypeQ4K {
		if out.dataType == DataTypeF16 {
			// Swap N, K for Q4K Logic
			C.Metal_MatMul_Q4K_F32_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
				C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
		} else {
			C.Metal_MatMul_Q4K_F32(t.ctx.ref, weight.buf, C.int(weight.Offset), 0, t.buf, C.int(t.Offset), 0, out.buf, C.int(out.Offset),
				C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
		}
	} else if weight.dataType == DataTypeQ6K {
		// Native Q6K
		if out.dataType == DataTypeF16 {
			C.Metal_MatMul_Q6K_F16(t.ctx.ref, weight.buf, C.int(weight.Offset), false, t.buf, C.int(t.Offset), false, out.buf, C.int(out.Offset),
				C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
		} else {
			// F32 output. Use Metal_MatMul_Q6K_F32 (FP32 In/Out)
			// Metal_MatMul_Q6K_F32 args: M, N, K.
			// Kernel expects dim_in = K, dim_out = N.
			// Obj-C maps K -> dim_in, N -> dim_out.
			// So pass K=weight.cols, N=weight.rows.
			C.Metal_MatMul_Q6K_F32(t.ctx.ref, weight.buf, C.int(weight.Offset), 0,
				t.buf, C.int(t.Offset), 0,
				out.buf, C.int(out.Offset),
				C.int(t.rows), C.int(weight.rows), C.int(weight.cols), C.float(scale))
		}
	} else if weight.dataType == DataTypeQ4_0 {
		// Q4_0 Support
		// Q4_0 Linear Kernel expects: dim_in (K), dim_out (N)
		// weight.rows is N (Output), weight.cols is K (Input)
		// So pass K, N
		if out.dataType == DataTypeF16 {
			C.Metal_LinearQ4_0_F16(t.ctx.ref, weight.buf, C.int(weight.Offset),
				t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
				C.int(t.rows), C.int(weight.cols), C.int(weight.rows), C.float(scale))
		} else {
			panic("Q4_0 to non-F16 output not implemented in LinearInto")
		}
	} else {
		// F16 weights * FP32 input -> FP32 output
		// Use Metal_MatMul_F16_F32_F32 (Kernel linear_f16_f32) checks float* input
		C.Metal_MatMul_F16_F32_F32(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	}
}

// SwiGLU_FP32 performs SwiGLU with FP32 inputs and outputs
func (gate *Tensor) SwiGLU_FP32_Into(up *Tensor, out *Tensor) {
	C.Metal_SwiGLU_F32(gate.ctx.ref, gate.buf, C.int(gate.Offset), up.buf, C.int(up.Offset), out.buf, C.int(out.Offset), C.int(gate.rows), C.int(gate.cols))
}

// SwiGLU_FP32 performs SwiGLU with FP32 inputs and outputs
// gate and up must both be FP32 tensors
func (gate *Tensor) SwiGLU_FP32(up *Tensor) (*Tensor, error) {
	if gate.rows != up.rows || gate.cols != up.cols {
		return nil, NewValidationError("SwiGLU_FP32",
			fmt.Sprintf("dimension mismatch: gate[%d,%d] != up[%d,%d]",
				gate.rows, gate.cols, up.rows, up.cols),
			"swiglu_dims")
	}
	if gate.dataType != DataTypeF32 || up.dataType != DataTypeF32 {
		return nil, NewValidationError("SwiGLU_FP32",
			fmt.Sprintf("requires FP32 inputs, got gate=%v, up=%v", gate.dataType, up.dataType),
			"datatype")
	}

	res := gate.ctx.NewTensorFP32Pooled(gate.rows, gate.cols)

	// Use existing swiglu_f32 kernel
	C.Metal_SwiGLU_F32(gate.ctx.ref, gate.buf, C.int(gate.Offset), up.buf, C.int(up.Offset), res.buf, C.int(res.Offset),
		C.int(gate.rows), C.int(gate.cols))
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
	out := t.ctx.NewTensorPooled(t.rows, weight.rows)
	C.Metal_LinearF32ToF16(t.ctx.ref, weight.buf, C.int(weight.Offset), t.buf, C.int(t.Offset), out.buf, C.int(out.Offset),
		C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	return out, nil
}

func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	if err := ValidateAddDimensions(t.rows, t.cols, other.rows, other.cols); err != nil {
		return nil, err
	}
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
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
	// a = a + b
	// Metal_Add_F32(a, offA, b, offB, res, offRes, count)
	C.Metal_Add_F32(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
	return nil
}

func (t *Tensor) EmbeddingLookup(row int, scale float32) *Tensor {
	res := t.ctx.NewTensorPooled(1, t.cols)
	if t.dataType == DataTypeQ4K {
		// Use optimized Q4K embedding kernel for better performance
		C.Metal_Embedding_Q4K_Optimized(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols), C.float(scale))
	} else if t.dataType == DataTypeQ4_0 {
		C.Metal_EmbeddingQ4_0_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols))
	} else {
		C.Metal_Embedding_F16(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(row), C.int(t.cols))
	}
	return res
}

func (t *Tensor) StoreKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	C.Metal_StoreKV_F16(t.ctx.ref, t.buf, C.int(t.Offset), v.buf, C.int(v.Offset), kCache.buf, C.int(kCache.Offset), vCache.buf, C.int(vCache.Offset), C.int(pos), C.int(heads), C.int(headDim), C.int(windowSize))
}

func (t *Tensor) Attention(kCache, vCache *Tensor, pos, numHeads, kvHeads, headDim, ctxLen, windowSize int) *Tensor {
	res := t.ctx.NewTensorPooled(1, numHeads*headDim)
	scoresDim := numHeads * ctxLen
	if scoresDim < 32768 {
		scoresDim = 32768
	}
	scores := t.ctx.NewTensorFP32Pooled(1, scoresDim)

	C.Metal_Attention_F16(t.ctx.ref, t.buf, C.int(t.Offset), kCache.buf, C.int(kCache.Offset),
		vCache.buf, C.int(vCache.Offset), res.buf, C.int(res.Offset),
		scores.buf, C.int(scores.Offset),
		C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim), C.int(ctxLen), C.int(windowSize))
	return res
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

	C.Metal_AttFused_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		kCache.buf, C.int(kCache.Offset),
		vCache.buf, C.int(vCache.Offset),
		out.buf, C.int(out.Offset),
		C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim), C.int(windowSize), C.int(maxCtxLen))
	return nil
}

// FP32 Operations

func (t *Tensor) RMSNormFP32_ToF16(weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols) // Result is F16
	C.Metal_RMSNorm_F32_F16(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), res.buf, C.int(res.Offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))
	// DEBUG

	return res
}

func (t *Tensor) RMSNormFP32_ToF16_Into(weight *Tensor, eps float32, out *Tensor) {
	C.Metal_RMSNorm_F32_F16(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), out.buf, C.int(out.Offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))

}

func (t *Tensor) RMSNormFP32(weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
	// Input t is F32, Weight is F16
	C.Metal_RMSNorm_F32(t.ctx.ref, t.buf, C.int(t.Offset), weight.buf, C.int(weight.Offset), res.buf, C.int(res.Offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))
	return res
}

func (t *Tensor) LinearIntoFP32(weight *Tensor, out *Tensor, scale float32) {
	t.LinearF32_Into(weight, out, scale)
}

func (t *Tensor) AddMixedInPlace(other *Tensor) error {
	if err := ValidateAddDimensions(t.rows, t.cols, other.rows, other.cols); err != nil {
		return err
	}
	// t (F32) += other (F16)
	C.Metal_Add_F32_F16(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), t.buf, C.int(t.Offset), C.int(t.rows*t.cols))
	return nil
}

func (t *Tensor) ToF32() *Tensor {
	res := t.ctx.NewTensorFP32Pooled(t.rows, t.cols)
	C.Metal_Copy_F16_F32(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) AddFP32(other *Tensor) *Tensor {
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
	C.Metal_Add_F32(t.ctx.ref, t.buf, C.int(t.Offset), other.buf, C.int(other.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) SwiGLUFP32(gate *Tensor) *Tensor {
	// t is up (F32), gate is gate (F32)
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
	C.Metal_SwiGLU_F32(t.ctx.ref, t.buf, C.int(t.Offset), gate.buf, C.int(gate.Offset), res.buf, C.int(res.Offset), C.int(t.rows), C.int(t.cols))
	return res
}

func (t *Tensor) SwiGLU_F32_InPlace(gate *Tensor) {
	C.Metal_SwiGLU_F32(t.ctx.ref, t.buf, C.int(t.Offset), gate.buf, C.int(gate.Offset), t.buf, C.int(t.Offset), C.int(t.rows), C.int(t.cols))
}

func (t *Tensor) CopyToF32() *Tensor {
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
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

func (t *Tensor) ToF32InPlace(res *Tensor) {
	C.Metal_Copy_F16_F32(t.ctx.ref, t.buf, C.int(t.Offset), res.buf, C.int(res.Offset), C.int(t.rows*t.cols))
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
func (t *Tensor) AttSoftmax(pos, heads, stride int) {
	C.Metal_AttSoftmax_F16(t.ctx.ref, t.buf, C.int(t.Offset), C.int(pos), C.int(heads), C.int(stride))
}

// AttValues performs attention value aggregation
func (t *Tensor) AttValues(vCache *Tensor, out *Tensor, pos, numHeads, kvHeads, headDim, stride, windowSize int) {
	C.Metal_AttValues_F16(t.ctx.ref, t.buf, C.int(t.Offset),
		vCache.buf, C.int(vCache.Offset),
		out.buf, C.int(out.Offset),
		C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim), C.int(stride), C.int(windowSize))
}
