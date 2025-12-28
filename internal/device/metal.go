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
	"sync/atomic"
	"time"
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

const (
	DataTypeF16 = 0
	DataTypeQ4K = 1
	DataTypeQ3K = 2
	DataTypeF32 = 3
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

// NewTensor creates a standard F16 tensor
func (c *Context) NewTensor(rows, cols int) *Tensor {
	sizeBytes := rows * cols * 2 // FP16
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

// NewTensorFP32 creates a standard F32 tensor
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

func (t *Tensor) LoadRaw(data []byte) {
	if len(data) > t.sizeBytes {
		panic("Raw data size exceeds tensor buffer")
	}
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
	if ptr == nil { return 0 }
	f16Slice := unsafe.Slice((*uint16)(ptr), t.rows*t.cols)
	var maxVal float32 = 0
	for _, v := range f16Slice {
		f := Float16ToFloat32(v)
		abs := f
		if abs < 0 { abs = -abs }
		if abs > maxVal { maxVal = abs }
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
	var maxD float32 = 0
	
	for i := 0; i < numBlocks; i++ {
		offset := i * 144
		if offset+2 > len(data) { break }
		dRaw := binary.LittleEndian.Uint16(data[offset : offset+2])
		d := Float16ToFloat32(dRaw)
		if d > maxD { maxD = d }
	}
	fmt.Printf("DEBUG_SCALES: %s Max Scale (d): %f\n", name, maxD)
	return maxD
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

// MatMul performs matrix multiplication C = A * B
func (t *Tensor) MatMul(b *Tensor) *Tensor {
	// A=t: [N x K] (Weights)
	// B=b: [M x K] (Input)
	M := b.rows
	N := t.rows
	K := t.cols
	
	// If t is Q4_K, dispatch specialized kernel
	if t.dataType == DataTypeQ4K {
		c := t.ctx.NewTensor(N, M)
		c.ZeroInit()
		C.Metal_MatMul_Q4K_F16(t.ctx.ref,
			t.buf, 0, C.bool(false),
			b.buf, 0, C.bool(false),
			c.buf, 0,
			C.int(M), C.int(N), C.int(K));
			
		return c
	}

	c := t.ctx.NewTensor(N, M) 
	C.Metal_MatMul_F16(t.ctx.ref,
		t.buf, 0, C.bool(false),
		b.buf, 0, C.bool(false),
		c.buf, 0,
		C.int(M), C.int(N), C.int(K))
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
		C.Metal_MatMul_F16(t.ctx.ref, weight.buf, 0, false, t.buf, 0, false, res.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
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
	} else {
		C.Metal_MatMul_F16(t.ctx.ref, weight.buf, 0, false, t.buf, 0, false, out.buf, 0,
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
	normed := t.RMSNorm(attnNorm, eps)
	t.ctx.Synchronize()
		
	// 2. QKV Projections (NO POOLING)
	qPart := t.ctx.NewTensor(t.rows, q.rows) // [1, 4096]
	kPart := t.ctx.NewTensor(t.rows, k.rows) // [1, 1024]
	vPart := t.ctx.NewTensor(t.rows, v.rows) // [1, 1024]
	
	normed.LinearInto(q, qPart)
	normed.LinearInto(k, kPart)
	normed.LinearInto(v, vPart)
	t.ctx.Synchronize()
	
	// 3. RoPE
	qPart.RoPE(pos, headDim, heads, 1, ropeTheta)
	kPart.RoPE(pos, headDim, kvHeads, 1, ropeTheta)
	t.ctx.Synchronize()
	
	// 4. Store K/V
	C.Metal_StoreKV_F16(t.ctx.ref, kPart.buf, 0, vPart.buf, 0, kCache.buf, vCache.buf, 
		C.int(pos), C.int(kvHeads), C.int(headDim))
	t.ctx.Synchronize()
		
	// 5. Attention
	attOut := t.ctx.NewTensor(t.rows, t.cols) // [1, 4096]
	// Scores buffer for [Heads, SeqLen]. 32 Heads * ContextLen.
	// For inference, we only need [32, cachePos+1] scores.
	// We can allocate max size [32, ctxLen]. Or dynamic.
	// 4096 dim is usually enough for 32*128. But Context can be 32k.
	// Let's allocate based on KV Heas/Pos.
	// Pos is dynamic. Let's use a large scratch or dynamic.
	// Max ctxLen usually passed.
	scoresDim := heads * ctxLen 
	if scoresDim < 32768 { scoresDim = 32768 } // Min size
	scores := t.ctx.NewTensor(1, scoresDim) 
	
	C.Metal_Attention_F16(t.ctx.ref, qPart.buf, 0, kCache.buf, vCache.buf, attOut.buf, 0,
		scores.buf, 0,
		C.int(pos), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(ctxLen))
	t.ctx.Synchronize()
		
	// 6. Attention Output Projection
	resAtt := t.ctx.NewTensor(t.rows, o.rows) // [1, 4096]
	attOut.LinearInto(o, resAtt)
	t.ctx.Synchronize()
	
	// 7. Residual Add 1
	C.Metal_Add_F16(t.ctx.ref, t.buf, 0, resAtt.buf, 0, t.buf, 0, C.int(t.rows*t.cols))
	t.ctx.Synchronize()
	
	// --- FFN Block ---
	
	// Check if we should use FP32 (if weights are Q4K)
	// We check ffnGate (usually first one used).
	useFP32 := (ffnGate.dataType == DataTypeQ4K && ffnUp.dataType == DataTypeQ4K && ffnDown.dataType == DataTypeQ4K)
	
	var resFFN *Tensor
	
	if useFP32 {
		// FP32 Path
		
		// 8. FFN Norm (Input t (F16) -> Normed (F32))
		// We use RMSNormFP32 which takes F32 input? No. RMSNormFP32 takes F32 input from my implementation.
		// Wait, t is F16. I need to convert t to F32 first?
		// My RMSNormFP32 implementation: Input t is F32.
		// So convert t to F32.
		t_f32 := t.CopyToF32()
		normedFFN_f32 := t_f32.RMSNormFP32(ffnNorm, eps)
		t.ctx.Synchronize()
		
		// 9. FFN Gate/Up
		gatePart_f32 := t.ctx.NewTensorFP32(t.rows, ffnGate.rows)
		upPart_f32 := t.ctx.NewTensorFP32(t.rows, ffnUp.rows)
		
		normedFFN_f32.LinearIntoFP32(ffnGate, gatePart_f32)
		normedFFN_f32.LinearIntoFP32(ffnUp, upPart_f32)
		t.ctx.Synchronize()
		
		// 10. SwiGLU (F32)
		// Result written to gatePart_f32
		// My SwiGLUFP32 returns new tensor. Modification required?
		// Metal_SwiGLU_F32 uses 3 buffers: inV, inG, out.
		// In my LinearIntoFP32, I used separate buffers.
		// The SwiGLUFP32 wrapper creates new tensor.
		// Let's use it.
		// But wait, SwiGLU usually modifies in place or writes to gate?
		// In previous F16: C.Metal_SwiGLU_F16(..., gatePart.buf, 0, gatePart.buf, 0, ...) -> In place?
		// Arguments: buffer(0)=iG, buffer(1)=iV, buffer(2)=out.
		// If iG == out, it's in place.
		// SwiGLUFP32 implementation: returns new tensor.
		swigluOut_f32 := upPart_f32.SwiGLUFP32(gatePart_f32)
		t.ctx.Synchronize()
		
		// 11. FFN Down Projection
		resFFN_f32 := t.ctx.NewTensorFP32(t.rows, ffnDown.rows)
		swigluOut_f32.LinearIntoFP32(ffnDown, resFFN_f32)
		t.ctx.Synchronize()
		
		// 12. Residual Add 2 (t_f32 + resFFN_f32 -> final_f32)
		final_f32 := t_f32.AddFP32(resFFN_f32)
		t.ctx.Synchronize()
		
		// Copy back to t (F16) direct from F32
		final_f32.CopyToF16_Into(t)
	} else {
		// F16 Path (Original)
		// 8. FFN Norm
		normedFFN := t.ctx.NewTensor(t.rows, t.cols)
		C.Metal_RMSNorm_F16(t.ctx.ref, t.buf, 0, ffnNorm.buf, 0, normedFFN.buf, 0, 
			C.int(t.rows), C.int(t.cols), C.float(eps))
		t.ctx.Synchronize()
			
		// 9. FFN Gate/Up
		gatePart := t.ctx.NewTensor(t.rows, ffnGate.rows) // [1, 14336]
		upPart := t.ctx.NewTensor(t.rows, ffnUp.rows)     // [1, 14336]
		normedFFN.LinearInto(ffnGate, gatePart)
		normedFFN.LinearInto(ffnUp, upPart)
		t.ctx.Synchronize()
		
		// 10. SwiGLU
		// Args: metal_backend (iV, iG, o). (Up, Gate, Out).
		// Result is written to gatePart.
		C.Metal_SwiGLU_F16(t.ctx.ref, upPart.buf, 0, gatePart.buf, 0, gatePart.buf, 0, C.int(1), C.int(ffnGate.rows))
		t.ctx.Synchronize()
		
		// 11. FFN Down Projection
		// Input is gatePart (SwiGLU output)
		resFFN = t.ctx.NewTensor(t.rows, ffnDown.rows) // [1, 4096]
		gatePart.LinearInto(ffnDown, resFFN)
		t.ctx.Synchronize()
		
		// 12. Residual Add 2
		C.Metal_Add_F16(t.ctx.ref, t.buf, 0, resFFN.buf, 0, t.buf, 0, C.int(t.rows*t.cols))
		t.ctx.Synchronize()
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
	// t is F32 input, out is F32 output
	// weight can be Q4K (likely) or F16.
	
	if weight.dataType == DataTypeQ4K {
		C.Metal_MatMul_Q4K_F32(t.ctx.ref, weight.buf, 0, false, t.buf, 0, false, out.buf, 0,
			C.int(t.rows), C.int(weight.rows), C.int(weight.cols))
	} else {
		// Fallback for F16 weights with F32 input: Not implemented yet!
		// For now panic or use Q4K path if compatible? No.
		// We need Metal_MatMul_F16_F32.
		panic("LinearIntoFP32: only Q4K weights supported currently")
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

func (t *Tensor) CopyToF16() *Tensor {
	res := t.ctx.NewTensor(t.rows, t.cols)
	C.Metal_Copy_F32_F16(t.ctx.ref, t.buf, 0, res.buf, 0, C.int(t.rows*t.cols))
	return res
}

func (t *Tensor) CopyToF16_Into(dest *Tensor) {
	C.Metal_Copy_F32_F16(t.ctx.ref, t.buf, 0, dest.buf, 0, C.int(t.rows*t.cols))
}
