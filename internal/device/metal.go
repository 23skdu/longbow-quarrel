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
	_ "embed"
	"fmt"
	"time"

	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

var allocatedBytes int64

//go:embed kernels.metal
var kernelsSource string

// Context holds the Metal connection and tensor pool
type Context struct {
	ref C.MetalContextRef
	pool map[string][]*Tensor // pool by size key "RxC"
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
		C.Metal_Free(c.ref)
		c.ref = nil
	}
}

// Tensor wraps a Metal buffer. Always FP16 for this engine.
type Tensor struct {
	ctx      *Context
	buf      C.MetalBufferRef
	sizeBytes int
	rows, cols int
}

func (t *Tensor) Rows() int { return t.rows }
func (t *Tensor) Cols() int { return t.cols }

func (c *Context) NewTensor(rows, cols int) *Tensor {
	sizeBytes := rows * cols * 2 // FP16
	buf := C.Metal_Alloc(c.ref, C.int(sizeBytes))
	
	t := &Tensor{
		ctx: c,
		buf: buf,
		sizeBytes: sizeBytes,
		rows: rows,
		cols: cols,
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

// NewTensorPooled attempts to reuse tensor from pool
func (c *Context) NewTensorPooled(rows, cols int) *Tensor {
	key := fmt.Sprintf("%dx%d", rows, cols)
	if tensors, ok := c.pool[key]; ok && len(tensors) > 0 {
		// Pop from pool
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		return t
	}
	// Fallback to new allocation
	return c.NewTensor(rows, cols)
}

// ReturnToPool returns tensor to pool for reuse
func (t *Tensor) ReturnToPool() {
	key := fmt.Sprintf("%dx%d", t.rows, t.cols)
	t.ctx.pool[key] = append(t.ctx.pool[key], t)
}

func (t *Tensor) LoadFrom(data []float32) {
	if len(data) != t.rows*t.cols {
		panic("Data size mismatch")
	}
	// Convert to FP16
	f16 := make([]uint16, len(data))
	for i, v := range data {
		f16[i] = Float32ToFloat16(v)
	}
	C.Metal_CopyToDevice(t.buf, 0, unsafe.Pointer(&f16[0]), C.int(t.sizeBytes))
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
	t.ctx.Synchronize()
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

// Operations

func (t *Tensor) MatMul(other *Tensor) *Tensor {
	// Simple M x K * K x N = M x N
	if t.cols != other.rows {
		panic("MatMul dim mismatch")
	}
	t0 := time.Now()
	res := t.ctx.NewTensorPooled(t.rows, other.cols)
	C.Metal_MatMul_F16(t.ctx.ref, t.buf, 0, false, other.buf, 0, false, res.buf, 0, 
		C.int(t.rows), C.int(other.cols), C.int(t.cols))
	metrics.RecordKernelDuration("MatMul", time.Since(t0))
	return res
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
	// Metal_MatMul_F16(..., transA=false, ..., transB=true, ...)
	C.Metal_MatMul_F16(t.ctx.ref, t.buf, 0, false, weight.buf, 0, true, res.buf, 0,
		C.int(t.rows), C.int(weight.rows), C.int(t.cols))
		
	metrics.RecordKernelDuration("Linear", time.Since(t0))
	return res
}

func (t *Tensor) RMSNorm(weight *Tensor, eps float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, 0, weight.buf, 0, res.buf, 0, 
		C.int(t.rows), C.int(t.cols), C.float(eps))
	return res
}

// RMSNormLinear performs fused RMSNorm + Linear in single kernel
// Eliminates intermediate buffer allocation
func (t *Tensor) RMSNormLinear(normWeight, linearWeight *Tensor, eps float32) *Tensor {
	// normWeight: [inDim], linearWeight: [outDim, inDim]
	res := t.ctx.NewTensorPooled(t.rows, linearWeight.rows)
	C.Metal_RMSNormLinear_F16(t.ctx.ref, t.buf, 0, normWeight.buf, 0, linearWeight.buf, 0, res.buf, 0,
		C.int(t.cols), C.int(linearWeight.rows), C.float(eps))
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
	C.Metal_Layer_F16(t.ctx.ref, t.buf,
		attnNorm.buf, q.buf, k.buf, v.buf, o.buf,
		ffnNorm.buf, ffnGate.buf, ffnUp.buf, ffnDown.buf,
		kCache.buf, vCache.buf, s1.buf, s2.buf, s3.buf, s4.buf,
		C.int(pos), C.int(heads), C.int(kvHeads),
		C.int(headDim), C.int(hiddenDim), C.float(eps), C.float(ropeTheta), C.int(ctxLen))
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

func (t *Tensor) Add(other *Tensor) *Tensor {
	if t.rows != other.rows || t.cols != other.cols {
		panic("Add dim mismatch")
	}
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	C.Metal_Add_F16(t.ctx.ref, t.buf, 0, other.buf, 0, res.buf, 0, C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) Scale(val float32) *Tensor {
	res := t.ctx.NewTensorPooled(t.rows, t.cols)
	// Float32ToFloat16 is in utils.go
	v16 := Float32ToFloat16(val)
	C.Metal_Scale_F16(t.ctx.ref, t.buf, 0, C.uint16_t(v16), res.buf, 0, C.int(t.rows*t.cols))
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

func (t *Tensor) Attention(kCache, vCache *Tensor, pos, numHeads, kvHeads, headDim int) *Tensor {
	res := t.ctx.NewTensorPooled(1, numHeads*headDim)
	C.Metal_Attention_F16(t.ctx.ref, t.buf, 0, kCache.buf, vCache.buf, res.buf, 0,
		C.int(pos), C.int(numHeads), C.int(kvHeads), C.int(headDim))
	return res
}
