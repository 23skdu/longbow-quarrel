//go:build darwin && metal

package device

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework Accelerate
#cgo CFLAGS: -fobjc-arc
#include "metal_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
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
		ref: ref,
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

const (
	DataTypeF16 = 0
	DataTypeQ4K = 1
	DataTypeQ3K = 2
	DataTypeF32 = 3
	DataTypeQ6K = 4
)

// Tensor wraps a Metal buffer. Always FP16 for this engine.
type Tensor struct {
	ctx       *Context
	rows      int
	cols      int
	sizeBytes int
	buf       C.MetalBufferRef
	dataType  int // 0=F16, 1=Q4K, 2=Q3K
}

func (t *Tensor) Rows() int { return t.rows }
func (t *Tensor) Cols() int { return t.cols }

// NewQ3KTensor creates a tensor with Q3_K quantization layout (110 bytes per 256 weights)
func (c *Context) NewQ3KTensor(rows, cols int) *Tensor {
	numElements := rows * cols
	if numElements%256 != 0 {
		panic("Q3_K tensor size must be divisible by 256")
	}
	numBlocks := numElements / 256
	sizeBytes := numBlocks * 110

	buf := C.Metal_Alloc(c.ref, C.int(sizeBytes))
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: int(sizeBytes),
		buf:       buf,
		dataType:  DataTypeQ3K,
	}
}

// NewQ4KTensor creates a tensor with Q4_K quantization layout (144 bytes per 256 weights)
func (c *Context) NewQ4KTensor(rows, cols int) *Tensor {
	numElements := rows * cols
	if numElements%256 != 0 {
		panic("Q4_K tensor size must be divisible by 256")
	}
	numBlocks := numElements / 256
	sizeBytes := numBlocks * 144

	buf := C.Metal_Alloc(c.ref, C.int(sizeBytes))
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: int(sizeBytes),
		buf:       buf,
		dataType:  DataTypeQ4K,
	}
}

// NewQ6KTensor creates a tensor with Q6_K quantization layout (210 bytes per 256 weights)
func (c *Context) NewQ6KTensor(rows, cols int) *Tensor {
	numElements := rows * cols
	if numElements%256 != 0 {
		panic("Q6_K tensor size must be divisible by 256")
	}
	numBlocks := numElements / 256
	sizeBytes := numBlocks * 210
	buf := C.Metal_Alloc(c.ref, C.int(sizeBytes))
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: sizeBytes,
		buf:       buf,
		dataType:  DataTypeQ6K,
	}
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

	buf := C.Metal_Alloc(c.ref, C.int(sizeBytes))
	if buf == nil {
		panic("Metal_Alloc returned nil!")
	}
	
	t := &Tensor{
		ctx: c,
		buf: buf,
		sizeBytes: sizeBytes,
		rows: rows,
		cols: cols,
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
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		c.mu.Unlock()
		return t
	}
	c.mu.Unlock()
	
	return c.NewTensorFP32(rows, cols)
}

func (c *Context) NewTensorFP32(rows, cols int) *Tensor {
	sizeBytes := rows * cols * 4 // FP32
	buf := C.Metal_Alloc(c.ref, C.int(sizeBytes))
	if buf == nil {
		panic("Metal_Alloc returned nil!")
	}
	
	t := &Tensor{
		ctx: c,
		buf: buf,
		sizeBytes: sizeBytes,
		rows: rows,
		cols: cols,
		dataType:  DataTypeF32,
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

// NewTensorPooled attempts to reuse tensor from pool (defaults to F16)
func (c *Context) NewTensorPooled(rows, cols int) *Tensor {
	key := fmt.Sprintf("%dx%dx%d", rows, cols, DataTypeF16)
	
	c.mu.Lock()
	if tensors, ok := c.pool[key]; ok && len(tensors) > 0 {
		// Pop from pool
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		c.mu.Unlock()
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

func (t *Tensor) LoadFrom(data []float32) {
	if len(data) != t.rows*t.cols {
		panic("Data size mismatch")
	}
	
	if t.dataType == DataTypeF32 {
		// Copy directly as F32
		C.Metal_CopyToDevice(t.buf, 0, unsafe.Pointer(&data[0]), C.int(len(data)*4))
		return
	}

	// Convert to FP16
	f16 := make([]uint16, len(data))
	for i, v := range data {
		f16[i] = Float32ToFloat16(v)
	}
	C.Metal_CopyToDevice(t.buf, 0, unsafe.Pointer(&f16[0]), C.int(t.sizeBytes))
}

func (t *Tensor) LoadRaw(data []byte) {
	if len(data) > t.sizeBytes {
		panic("Raw data size exceeds tensor buffer")
	}
	C.Metal_CopyToDevice(t.buf, 0, unsafe.Pointer(&data[0]), C.int(len(data)))
}

// LoadFromBytes copies raw bytes to the buffer (for Q4K data, etc.)
func (t *Tensor) LoadFromBytes(data []byte) {
	C.Metal_CopyToDevice(t.buf, 0, unsafe.Pointer(&data[0]), C.int(len(data)))
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
func (t *Tensor) DataType() int {
	return t.dataType
}

func (t *Tensor) ScanNaNs(name string) int {
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil { return 0 }
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
		fmt.Printf("DEBUG_SCAN: %s has %d NaNs and %d Infs!\n", name, nanCount, infCount)
	} else {
		fmt.Printf("DEBUG_SCAN: %s is OCD Clean.\n", name)
	}
	return nanCount + infCount
}

func (t *Tensor) ScanMax(name string) float32 {
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		return 0
	}
	var maxVal float32 = 0
	if t.dataType == DataTypeF32 {
		f32Slice := unsafe.Slice((*float32)(ptr), t.rows*t.cols)
		for _, v := range f32Slice {
			abs := v
			if abs < 0 {
				abs = -abs
			}
			if abs > maxVal {
				maxVal = abs
			}
		}
	} else {
		f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)
		for _, v := range f16Slice {
			f := Float16ToFloat32(v)
			abs := f
			if abs < 0 {
				abs = -abs
			}
			if abs > maxVal {
				maxVal = abs
			}
		}
	}
	fmt.Printf("DEBUG_MAX: %s Max Value: %f\n", name, maxVal)
	return maxVal
}

func (t *Tensor) ScanQ4KScales(name string) float32 {
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil { return 0 }
	// Unsafe cast to byte slice to handle stride
	// Total bytes?
	// Q4K size logic: (elements / 256) * 144?
	// But t.sizeBytes is available.
	data := unsafe.Slice((*byte)(ptr), t.sizeBytes)
	
	numBlocks := (t.rows * t.cols) / 256
	var maxScale float32 = 0
	
	for i := 0; i < numBlocks; i++ {
		offset := i * 144
		if offset+2 > len(data) { break }
		// d is bytes 0-1 (FP16)
		block := data[offset : offset+2]
		dbits := binary.LittleEndian.Uint16(block)
		d := Float16ToFloat32(dbits)
		if d > maxScale { maxScale = d }
		if i == 0 {
			// Print first block details for debugging
			fmt.Printf("DEBUG_Q4K_HEX: %s Block 0 dbits=0x%04x d=%f\n", name, dbits, d)
		}
	}
	fmt.Printf("DEBUG_SCALES: %s Max Scale (d): %f\n", name, maxScale)
	return maxScale
}

func (t *Tensor) ScanMean(name string) float32 {
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil { return 0 }
	f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)
	var sum float64 = 0
	for _, v := range f16Slice {
		sum += float64(Float16ToFloat32(v))
	}
	mean := float32(sum / float64(len(f16Slice)))
	fmt.Printf("DEBUG_MEAN: %s Mean Value: %f\n", name, mean)
	return mean
}

// ScanScores prints the first N raw values (handles F16/F32)
func (t *Tensor) ScanScores(name string) {
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil { return }
	
	count := 32
	if t.rows*t.cols < count { count = t.rows*t.cols }
	
	fmt.Printf("DEBUG_SCORES: %s [Head 0, 0-%d]: ", name, count)
	
	if t.dataType == DataTypeF16 || t.dataType == DataTypeQ4K || t.dataType == DataTypeQ3K { // Q4K/Q3K stored as F16 in debug? No, but let's assume F16
		f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)
		for i := 0; i < count; i++ {
			f := Float16ToFloat32(f16Slice[i])
			fmt.Printf("%.4f ", f)
		}
	} else {
		// F32
		f32Slice := unsafe.Slice((*float32)(ptr), t.rows*t.cols)
		for i := 0; i < count; i++ {
			fmt.Printf("%.4f ", f32Slice[i])
		}
	}
	fmt.Println()
}

func (t *Tensor) ScanAbsMax(name string) float32 {
	t.ctx.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil { return 0 }
	f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)
	var maxVal float32 = 0
	for _, v := range f16Slice {
		f := Float16ToFloat32(v)
		if f < 0 { f = -f }
		if f > maxVal { maxVal = f }
	}
	fmt.Printf("DEBUG_ABSMAX: %s Max Abs Value: %f\n", name, maxVal)
	return maxVal
}

// LoadFromRaw copies raw bytes directly to the GPU buffer.
// The caller must ensure the data is in the correct format (FP16 usually) and size.
func (t *Tensor) LoadFromRaw(data []byte) {
	if len(data) != t.sizeBytes {
		panic("Raw data size mismatch")
	}
	if len(data) == 0 {
		return
	}
	C.Metal_CopyToDevice(t.buf, 0, unsafe.Pointer(&data[0]), C.int(len(data)))
}

func (t *Tensor) ToHost() []float32 {
	if err := t.ctx.WaitWithTimeout(10 * time.Second); err != nil {
		panic(fmt.Sprintf("ToHost failed: %v", err))
	}
	
	if t.dataType == DataTypeF32 {
		f32 := make([]float32, t.rows*t.cols)
		C.Metal_CopyToHost(t.buf, 0, unsafe.Pointer(&f32[0]), C.int(t.sizeBytes))
		return f32
	}

	f16 := make([]uint16, t.rows*t.cols)
	C.Metal_CopyToHost(t.buf, 0, unsafe.Pointer(&f16[0]), C.int(t.sizeBytes))
	
	f32 := make([]float32, len(f16))
	for i, v := range f16 {
		f32[i] = Float16ToFloat32(v)
	}
	return f32
}

// ZeroInit initializes tensor buffer with zeros
func (t *Tensor) ZeroInit() {
	C.Metal_ZeroBuffer(t.buf, 0, C.int(t.sizeBytes))
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

func (t *Tensor) Scale(val float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_Scale_F16(t.ctx.ref, t.buf, 0, C.float(val), res.buf, 0, C.int(t.rows*t.cols))
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
			t.buf, 0, C.bool(false),
			b.buf, 0, C.bool(false),
			c.buf, 0,
			C.int(M), C.int(N), C.int(K));
		
		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	} else if t.dataType == DataTypeQ3K {
		c := t.ctx.NewTensor(N, M)
		c.ZeroInit()
		C.Metal_MatMul_Q3K_F16(t.ctx.ref,
			t.buf, 0, C.bool(false),
			b.buf, 0, C.bool(false),
			c.buf, 0,
			C.int(M), C.int(N), C.int(K));
		
		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	} else if t.dataType == DataTypeQ6K {
		c := t.ctx.NewTensor(N, M)
		c.ZeroInit()
		C.Metal_MatMul_Q6K_F16(t.ctx.ref,
			t.buf, 0, C.bool(false),
			b.buf, 0, C.bool(false),
			c.buf, 0,
			C.int(M), C.int(N), C.int(K));
		
		metrics.RecordKernelDuration("MatMul", time.Since(t0))
		return c
	}

	c := t.ctx.NewTensor(N, M) 
	C.Metal_MatMul_F16(t.ctx.ref,
		t.buf, 0, C.bool(false),
		b.buf, 0, C.bool(false),
		c.buf, 0,
		C.int(M), C.int(N), C.int(K))
	metrics.RecordKernelDuration("MatMul", time.Since(t0))
	return c
}

// Linear performs t * weight^T
// t: [M, K], weight: [N, K] -> result: [M, N]
func (t *Tensor) Linear(weight *Tensor) *Tensor {
	// t.cols (K) must match weight.cols (K)
	if t.cols != weight.cols {
		panic(fmt.Sprintf("Linear dim mismatch: input cols %d != weight cols %d", t.cols, weight.cols))
	}
	t0 := time.Now()
	res := t.ctx.NewTensorPooled(t.rows, weight.rows) // [M, N]
	
	// MatMul(A, B^T)
	// Swap arguments: A=weight, B=t.
	// We want Weight (Matrix) as buffer 0 (primary stride source)
	// and Input (Vector) as buffer 1 (broadcast source).
	
	if weight.dataType == DataTypeQ4K {
		C.Metal_MatMul_Q4K_F16(t.ctx.ref, weight.buf, 0, false, t.buf, 0, false, res.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	} else {
		// BatchedMatMul_F16 using MPS
		M := t.rows
		N := weight.rows
		K := weight.cols
		sA := K * 2
		sB := K * 2
		sC := N * 2
		
		C.Metal_BatchedMatMul_F16(t.ctx.ref, 
			t.buf, 0, C.int(sA), false,
			weight.buf, 0, C.int(sB), true,
			res.buf, 0, C.int(sC),
			C.int(M), C.int(N), C.int(K), 1)
	}
	metrics.RecordKernelDuration("Linear", time.Since(t0))
	return res
}

// LinearInto performs Linear using existing output tensor (scratch buffer)
func (t *Tensor) LinearInto(weight *Tensor, out *Tensor) {
	// t.cols (K) must match weight.cols (K)
	if t.cols != weight.cols {
		panic(fmt.Sprintf("LinearInto dim mismatch: input cols %d != weight cols %d", t.cols, weight.cols))
	}
	// Check/Resize out tensor? Assuming caller managed size.
	// out should be [Rows, Weight.Rows] (usually [1, Dim]).
	
	if weight.dataType == DataTypeQ4K {
		C.Metal_MatMul_Q4K_F16(t.ctx.ref, weight.buf, 0, false, t.buf, 0, false, out.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	} else if weight.dataType == DataTypeQ3K {
		C.Metal_MatMul_Q3K_F16(t.ctx.ref, weight.buf, 0, false, t.buf, 0, false, out.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	} else if weight.dataType == DataTypeQ6K {
		C.Metal_MatMul_Q6K_F16(t.ctx.ref, weight.buf, 0, false, t.buf, 0, false, out.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	} else {
		// Custom F16 kernel: expects (weight, input, output, dim_in, dim_out, batch)
		// weight.buf at Index 0, t.buf at Index 1
		C.Metal_MatMul_F16(t.ctx.ref, weight.buf, 0, true, t.buf, 0, false, out.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	}
}

// RunQ3K_Explicit for testing only
func (c *Context) RunQ3K_Explicit(w, in, out *Tensor) {
	C.Metal_MatMul_Q3K_F16(c.ref,
		w.buf, 0, C.bool(false),
		in.buf, 0, C.bool(false),
		out.buf, 0,
		C.int(1), C.int(w.rows), C.int(w.cols))
}

func (t *Tensor) RMSNorm(weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, 0, weight.buf, 0, res.buf, 0, 
		C.int(t.rows), C.int(t.cols), C.float(eps))
	return res
}

// RMSNormLinear performs fused RMSNorm + Linear in single kernel
// Eliminates intermediate buffer allocation
func (t *Tensor) RMSNormLinear(normWeight, weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, weight.rows)
	C.Metal_RMSNormLinear_F16(t.ctx.ref, t.buf, 0, 
		normWeight.buf, 0, weight.buf, 0, res.buf, 0,
		C.int(t.rows), C.int(weight.rows), C.float(eps))
	return res
}

// RMSNormQKV performs fused RMSNorm + QKV Linear projections
func (t *Tensor) RMSNormQKV(normWeight, wQ, wK, wV *Tensor, eps float32) (*Tensor, *Tensor, *Tensor) {
	q := t.ctx.NewTensorPooled(t.rows, wQ.rows)
	k := t.ctx.NewTensorPooled(t.rows, wK.rows)
	v := t.ctx.NewTensorPooled(t.rows, wV.rows)
	C.Metal_RMSNormQKV_F16(t.ctx.ref, t.buf, 0, normWeight.buf, 0,
		wQ.buf, 0, wK.buf, 0, wV.buf, 0,
		q.buf, 0, k.buf, 0, v.buf, 0,
		C.int(t.cols), C.int(wQ.rows), C.int(wK.rows), C.float(eps))
	return q, k, v
}

// FusedFFN performs one entire FFN block: RMSNorm + Gate/Up Linear + SwiGLU + Down Linear
func (t *Tensor) FusedFFN(normWeight, wGate, wUp, wDown *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_FusedFFN_F16(t.ctx.ref, t.buf, 0, normWeight.buf, 0,
		wGate.buf, 0, wUp.buf, 0, wDown.buf, 0, res.buf, 0,
		C.int(t.cols), C.int(wGate.rows), C.float(eps))
	return res
}


func (t *Tensor) Layer(attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache, s1, s2, s3, s4 *Tensor, pos, heads, kvHeads, headDim int, ropeTheta, eps float32, hiddenDim, ctxLen int) {
	// Sequential Implementation to support Mixed Precision (Q4_K + F16)
	// s1, s2, s3, s4 are scratch buffers.
	
	// Sequential Implementation with Pooled Buffers
	// s1..s4 inputs are ignored to avoid sizing issues.

	// 1. Attention Norm
	// attnNorm.ScanAbsMax("Attn Norm Weight")
	normed := t.RMSNorm(attnNorm, eps)
	t.ctx.Synchronize()
	
	// normed.ScanScores("Attn Norm Out")
		
	// 2. QKV Projections (NO POOLING)
	qPart := t.ctx.NewTensor(t.rows, q.rows) // [1, 4096]
	kPart := t.ctx.NewTensor(t.rows, k.rows) // [1, 1024]
	vPart := t.ctx.NewTensor(t.rows, v.rows) // [1, 1024]
	
	normed.LinearInto(v, vPart)
	t.ctx.Synchronize()
	// qPart.ScanScores("Q Part Out")
	// kPart.ScanAbsMax("K Part Out AbsMax")
	
	// 3. RoPE & 4. Store K/V & 5. Attention (Serial Loop for Prefill/Batch)
	// Our RoPE, Attention and StoreKV kernels currently only support single-token processing.
	// We must loop over t.rows.
	
	
	// Create outputs
	attOut := t.ctx.NewTensor(t.rows, t.cols) // [Batch, 4096]
	
	// scores buffer: Max needed for one token attention [Heads, CtxLen]
	// allocate once and reuse. MUST BE FP32 (4 bytes).
	scoresDim := heads * ctxLen 
	if scoresDim < 32768 { scoresDim = 32768 }
	scores := t.ctx.NewTensorFP32Pooled(1, scoresDim)
	resAtt := t.ctx.NewTensor(t.rows, o.rows) // [Batch, 4096] - Output of O projection
	
	kvStride := kvHeads * headDim * 2 // bytes (F16)
	qStride := heads * headDim * 2    // bytes (F16)
	attStride := heads * headDim * 2  // bytes (F16)
	
	for i := 0; i < t.rows; i++ {
		p := pos + i
		
		// Offsets
		offQ := i * qStride
		offK := i * kvStride
		offV := i * kvStride
		offAtt := i * attStride
		
		// 3. RoPE (In-place on qPart/kPart row i)
		C.Metal_RoPE_F16(t.ctx.ref, qPart.buf, C.int(offQ), 1, C.int(1), C.int(heads), C.int(headDim), C.int(p), C.float(ropeTheta))
		C.Metal_RoPE_F16(t.ctx.ref, kPart.buf, C.int(offK), 1, C.int(1), C.int(kvHeads), C.int(headDim), C.int(p), C.float(ropeTheta))
		
		// 4. Store K/V
		C.Metal_StoreKV_F16(t.ctx.ref, kPart.buf, C.int(offK), vPart.buf, C.int(offV), 
			kCache.buf, vCache.buf, 
			C.int(p), C.int(kvHeads), C.int(headDim))
		t.ctx.Synchronize()
		

		if p < 1024 {
			C.Metal_AttFused_F16(t.ctx.ref, qPart.buf, C.int(offQ),
				kCache.buf, vCache.buf, attOut.buf, C.int(offAtt),
				C.int(p), C.int(heads), C.int(kvHeads), C.int(headDim))
		} else {
			// 5a. Scores
			C.Metal_AttScores_F16(t.ctx.ref, qPart.buf, C.int(offQ), kCache.buf, 
				scores.buf, 0,
				C.int(p), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(ctxLen))
			t.ctx.Synchronize()
			
			// 5b. Softmax
			C.Metal_AttSoftmax_F16(t.ctx.ref, scores.buf, 0, C.int(p), C.int(heads), C.int(ctxLen))
			t.ctx.Synchronize()

			// 5c. Values
			C.Metal_AttValues_F16(t.ctx.ref, scores.buf, 0, vCache.buf, attOut.buf, C.int(offAtt),
				C.int(p), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(ctxLen))
		}
		t.ctx.Synchronize()
	}
	t.ctx.Synchronize()
		
	// 6. Attention Output Projection
	attOut.LinearInto(o, resAtt)
	t.ctx.Synchronize()
	
	
	// 7. Residual Add 1
	t1 := t.Add(resAtt)
	
	C.Metal_Copy_F16(t.ctx.ref, t1.buf, 0, t.buf, 0, C.int(t.rows*t.cols))
	
	// --- Cleanup Attn Intermediates ---
	qPart.Free()
	kPart.Free()
	vPart.Free()
	normed.Free()
	attOut.Free()
	scores.Free()
	resAtt.ReturnToPool()
	t1.ReturnToPool()
	
	// --- FFN Block ---
	
	// Use FP32 FFN for small models (dim < 1024) to prevent activation explosion
	useF32FFN := t.cols < 1024
	
	// 8. FFN Norm
	normedFFN := t.RMSNorm(ffnNorm, eps)
	
	
	if useF32FFN {
		// FP32 FFN Path for Small Models (SmolLM2, TinyLlama)
		// Maintains FP32 precision through FFN to prevent explosion
		
		// 9. Gate/Up Projections → FP32
		gatePart := normedFFN.LinearToFP32(ffnGate)
		upPart := normedFFN.LinearToFP32(ffnUp)
		t.ctx.Synchronize()
		
		// 10. SwiGLU (FP32)
		swiOut := gatePart.SwiGLU_FP32(upPart)
		
		// 11. Down Projection (FP32 → FP16)
		resFFN := swiOut.LinearFromFP32(ffnDown)
		t.ctx.Synchronize()
		
		// Cleanup FP32 intermediates
		gatePart.ReturnToPool()
		upPart.ReturnToPool()
		swiOut.ReturnToPool()
		
		// 12. Residual Add 2
		t2 := t.Add(resFFN)
		
		C.Metal_Copy_F16(t.ctx.ref, t2.buf, 0, t.buf, 0, C.int(t.rows*t.cols))
		
		normedFFN.ReturnToPool()
		resFFN.ReturnToPool()
		t2.ReturnToPool()
	} else {
		// Standard FP16 FFN Path for Large Models (Mistral, Llama)
		
		// 9. FFN Gate/Up
		gatePart := t.ctx.NewTensorPooled(t.rows, ffnGate.rows)
		upPart := t.ctx.NewTensorPooled(t.rows, ffnUp.rows)
		normedFFN.LinearInto(ffnGate, gatePart)
		normedFFN.LinearInto(ffnUp, upPart)
		t.ctx.Synchronize()
		
		// 10. SwiGLU
		swiOut := upPart.SwiGLU(gatePart)
		
		// 11. Down Projection
		resFFN := t.ctx.NewTensorPooled(t.rows, ffnDown.rows)
		swiOut.LinearInto(ffnDown, resFFN)
		t.ctx.Synchronize()
		
		// 12. Residual Add 2
		t2 := t.Add(resFFN)
		
		C.Metal_Copy_F16(t.ctx.ref, t2.buf, 0, t.buf, 0, C.int(t.rows*t.cols))
		
		// --- Final Cleanup ---
		normedFFN.ReturnToPool()
		gatePart.ReturnToPool()
		upPart.ReturnToPool()
		swiOut.ReturnToPool()
		resFFN.ReturnToPool()
		t2.ReturnToPool()
	}
}

// Correct RoPE implementation using arguments expected by Kernel
func (t *Tensor) RoPE(posOffset, headDim, numHeads, seqLen int, ropeTheta float32) {
	C.Metal_RoPE_F16(t.ctx.ref, t.buf, 0, 1, C.int(seqLen), C.int(numHeads), C.int(headDim), C.int(posOffset), C.float(ropeTheta))
}

func (t *Tensor) SwiGLU(gate *Tensor) *Tensor {
	// Input t is 'val' (up projection). 'gate' is gate projection.
	// Both must be same size [Rows, InterSize].
	if t.rows != gate.rows || t.cols != gate.cols {
		panic("SwiGLU dim mismatch")
	}
	interSize := t.cols
	res := t.ctx.NewTensorPooled(t.rows, interSize)
	
	C.Metal_SwiGLU_F16(t.ctx.ref, t.buf, 0, gate.buf, 0, res.buf, 0, C.int(t.rows), C.int(interSize))
	return res
}

func (t *Tensor) Softmax() {
	C.Metal_Softmax_F16(t.ctx.ref, t.buf, 0, t.buf, 0, C.int(t.rows), C.int(t.cols))
}

// FP32 FFN Methods for Small Models (SmolLM2, TinyLlama)

// LinearToFP32 performs FP16 weight × FP16 input → FP32 output
// Used for Gate/Up projections in FP32 FFN path
func (t *Tensor) LinearToFP32(weight *Tensor) *Tensor {
	out := t.ctx.NewTensorFP32Pooled(t.rows, weight.rows)
	C.Metal_LinearF16ToF32(t.ctx.ref, weight.buf, 0, t.buf, 0, out.buf, 0,
		C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	return out
}

// SwiGLU_FP32 performs SwiGLU with FP32 inputs and outputs
// gate and up must both be FP32 tensors
func (gate *Tensor) SwiGLU_FP32(up *Tensor) *Tensor {
	if gate.rows != up.rows || gate.cols != up.cols {
		panic("SwiGLU_FP32 dim mismatch")
	}
	if gate.dataType != DataTypeF32 || up.dataType != DataTypeF32 {
		panic("SwiGLU_FP32 requires FP32 inputs")
	}
	
	res := gate.ctx.NewTensorFP32Pooled(gate.rows, gate.cols)
	
	// Use existing swiglu_f32 kernel
	C.Metal_SwiGLU_F32(gate.ctx.ref, gate.buf, 0, up.buf, 0, res.buf, 0,
		C.int(gate.rows), C.int(gate.cols))
	return res
}

// LinearFromFP32 performs FP16 weight × FP32 input → FP16 output
// Used for Down projection in FP32 FFN path
func (t *Tensor) LinearFromFP32(weight *Tensor) *Tensor {
	if t.dataType != DataTypeF32 {
		panic("LinearFromFP32 requires FP32 input")
	}
	out := t.ctx.NewTensorPooled(t.rows, weight.rows)
	C.Metal_LinearF32ToF16(t.ctx.ref, weight.buf, 0, t.buf, 0, out.buf, 0,
		C.int(t.rows), C.int(t.cols), C.int(weight.rows))
	return out
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	if t.rows != other.rows || t.cols != other.cols {
		panic("Add dim mismatch")
	}
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_Add_F16(t.ctx.ref, t.buf, 0, other.buf, 0, res.buf, 0, C.int(t.rows*t.cols))
	return res
}


func (t *Tensor) EmbeddingLookup(row int) *Tensor {
	res := t.ctx.NewTensorPooled(1, t.cols)
	C.Metal_Embedding_F16(t.ctx.ref, t.buf, 0, res.buf, 0, C.int(row), C.int(t.cols))
	return res
}

func (t *Tensor) StoreKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim int) {
	C.Metal_StoreKV_F16(t.ctx.ref, t.buf, 0, v.buf, 0, kCache.buf, vCache.buf, C.int(pos), C.int(heads), C.int(headDim))
}

func (t *Tensor) Attention(kCache, vCache *Tensor, pos, numHeads, kvHeads, headDim, ctxLen int) *Tensor {
	res := t.ctx.NewTensorPooled(1, numHeads*headDim)
	scoresDim := numHeads * ctxLen
	if scoresDim < 32768 { scoresDim = 32768 }
	scores := t.ctx.NewTensorPooled(1, scoresDim)
	
	C.Metal_Attention_F16(t.ctx.ref, t.buf, 0, kCache.buf, vCache.buf, res.buf, 0,
		scores.buf, 0,
		C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim), C.int(ctxLen))
	return res
}

// FP32 Operations

func (t *Tensor) RMSNormFP32(weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
	// Input t is F32, Weight is F16
	C.Metal_RMSNorm_F32(t.ctx.ref, t.buf, 0, weight.buf, 0, res.buf, 0, 
		C.int(t.rows), C.int(t.cols), C.float(eps))
	return res
}

func (t *Tensor) LinearIntoFP32(weight *Tensor, out *Tensor) {
	if t.cols != weight.cols {
		panic(fmt.Sprintf("LinearIntoFP32 mismatch: %d != %d", t.cols, weight.cols))
	}
	
	// Guardrail: Kernel uses float4/half4, so dim_in must be multiple of 4
	if t.cols%4 != 0 {
		panic(fmt.Sprintf("LinearIntoFP32: row dimension %d must be multiple of 4", t.cols))
	}
	
	if weight.dataType == DataTypeQ4K {
		C.Metal_MatMul_Q4K_F32(t.ctx.ref, weight.buf, 0, 0, t.buf, 0, 0, out.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	} else if weight.dataType == DataTypeF16 {
		C.Metal_MatMul_F16_F32(t.ctx.ref, weight.buf, 0, t.buf, 0, out.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	} else {
		panic(fmt.Sprintf("LinearIntoFP32: unsupported weight data type %d", weight.dataType))
	}
}

func (t *Tensor) AddFP32(other *Tensor) *Tensor {
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
	C.Metal_Add_F32(t.ctx.ref, t.buf, 0, other.buf, 0, res.buf, 0, C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) SwiGLUFP32(gate *Tensor) *Tensor {
	// t is up (F32), gate is gate (F32)
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
	C.Metal_SwiGLU_F32(t.ctx.ref, t.buf, 0, gate.buf, 0, res.buf, 0, C.int(t.rows), C.int(t.cols))
	return res
}

func (t *Tensor) CopyToF32() *Tensor {
	res := t.ctx.NewTensorFP32(t.rows, t.cols)
	C.Metal_Copy_F16_F32(t.ctx.ref, t.buf, 0, res.buf, 0, C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) ScaleInPlace(val float32) {
	C.Metal_Scale_F16(t.ctx.ref, t.buf, 0, C.float(val), t.buf, 0, C.int(t.rows*t.cols))
}


func (t *Tensor) CopyToF16() *Tensor {
	res := t.ctx.NewTensor(t.rows, t.cols)
	C.Metal_Copy_F32_F16(t.ctx.ref, t.buf, 0, res.buf, 0, C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) CopyToF16_Into(dest *Tensor) {
	C.Metal_Copy_F32_F16(t.ctx.ref, t.buf, 0, dest.buf, 0, C.int(t.rows*t.cols))
}

