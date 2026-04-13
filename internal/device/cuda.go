//go:build linux && cuda

package device

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR} -lcuda_kernels -lcublas -lcudnn -lcudart
#cgo linux,amd64 CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
    CUDA_DTYPE_F16 = 0,
    CUDA_DTYPE_F32 = 1,
    CUDA_DTYPE_Q8_0 = 2,
    CUDA_DTYPE_Q4_0 = 3,
    CUDA_DTYPE_Q4_K = 4,
    CUDA_DTYPE_Q6_K = 5
} CUDADataType;

extern void cudaProfilerStart();
extern void cudaProfilerStop();
extern cudaError_t cudaEventCreate(cudaEvent_t *event);
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

extern void cudaDequantQ8_0(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ8_0ToBF16(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ4_K(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ4_KToBF16(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ6_K(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ6_KToBF16(cudaStream_t stream, void* src, void* dst, int numElements);

// Basic math kernels
extern void cudaAdd(float* a, float* b, float* out, int size, cudaStream_t stream);
extern void cudaRMSNorm(float* input, float* weight, float* output, int rows, int cols, float eps, cudaStream_t stream);
extern void cudaSwiGLU(float* gate, float* up, float* output, int size, cudaStream_t stream);

// Fused kernel exports
extern void cudaFusedAttention(cudaStream_t stream, const void* q, const void* k, const void* v, void* output, const void* kCache, const void* vCache, int batch, int heads, int seqLen, int kvSeqLen, int headDim, float scale, int useCache, int windowSize);
extern void cudaFlashFusedAttention(cudaStream_t stream, const void* q, const void* k, const void* v, void* output, int batch, int heads, int seqLen, int kvSeqLen, int headDim, float scale, int windowSize);
extern void cudaFusedRoPE(cudaStream_t stream, void* tensor, const int* posIds, int batch, int heads, int seqLen, int headDim, float theta);
extern void cudaFusedMLP(cudaStream_t stream, const void* input, const void* gateWeight, const void* upWeight, const void* downWeight, void* output, int batch, int dim, int hiddenDim);
extern void cudaFusedRMSNormAdd(cudaStream_t stream, const void* input, const void* hidden, const void* weight, void* output, int batch, int dim, float eps);
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"sync"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// DataType and Float16 utils are provided by utils.go

type CUDAContext struct {
	Ctx    C.cudaStream_t
	Cublas C.cublasHandle_t
	pool   *tensorPool
}

type Tensor struct {
	devPtr  unsafe.Pointer
	rows    int
	cols    int
	dtype   DataType
	ctx     *CUDAContext
	pooled  bool
}

type tensorPool struct {
	mu     sync.Mutex
	free   map[int][]*Tensor
}

func NewCUDAContext() (*CUDAContext, error) {
	var stream C.cudaStream_t
	if err := C.cudaStreamCreate(&stream); err != 0 {
		return nil, fmt.Errorf("cudaStreamCreate failed: %v", err)
	}

	var handle C.cublasHandle_t
	if err := C.cublasCreate(&handle); err != 0 {
		return nil, fmt.Errorf("cublasCreate failed: %v", err)
	}
	C.cublasSetStream(handle, stream)

	return &CUDAContext{
		Ctx:    stream,
		Cublas: handle,
		pool: &tensorPool{
			free: make(map[int][]*Tensor),
		},
	}, nil
}

func (ctx *CUDAContext) Free() {
	if ctx.Ctx != nil {
		C.cudaStreamDestroy(ctx.Ctx)
	}
	if ctx.Cublas != nil {
		C.cublasDestroy(ctx.Cublas)
	}
}

func (ctx *CUDAContext) NewTensor(rows, cols int, dtype DataType) (*Tensor, error) {
	size := rows * cols
	var bytes int
	switch dtype {
	case DataTypeF16, DataTypeQ4K, DataTypeQ4_0, DataTypeQ6K, DataTypeQ8_0:
		// GPU weights are typically F16 after dequantization for math kernels
		bytes = size * 2
	case DataTypeF32:
		bytes = size * 4
	default:
		bytes = size * 4 // Fallback
	}

	var ptr unsafe.Pointer
	if err := C.cudaMalloc(&ptr, C.size_t(bytes)); err != 0 {
		return nil, fmt.Errorf("cudaMalloc failed: %v", err)
	}

	return &Tensor{
		devPtr: ptr,
		rows:   rows,
		cols:   cols,
		dtype:  dtype,
		ctx:    ctx,
	}, nil
}

func (ctx *CUDAContext) NewTensorPooled(rows, cols int) *Tensor {
	size := rows * cols
	ctx.pool.mu.Lock()
	defer ctx.pool.mu.Unlock()

	if lp, ok := ctx.pool.free[size]; ok && len(lp) > 0 {
		t := lp[len(lp)-1]
		ctx.pool.free[size] = lp[:len(lp)-1]
		return t
	}

	t, _ := ctx.NewTensor(rows, cols, DataTypeF16)
	t.pooled = true
	return t
}

func (t *Tensor) ReturnToPool() {
	if !t.pooled {
		return
	}
	size := t.rows * t.cols
	t.ctx.pool.mu.Lock()
	t.ctx.pool.free[size] = append(t.ctx.pool.free[size], t)
	t.ctx.pool.mu.Unlock()
}

func (t *Tensor) Free() {
	if t.devPtr != nil && !t.pooled {
		C.cudaFree(t.devPtr)
		t.devPtr = nil
	}
}

func (t *Tensor) Rows() int { return t.rows }
func (t *Tensor) Cols() int { return t.cols }
func (t *Tensor) Data() unsafe.Pointer { return t.devPtr }

func (t *Tensor) LoadFrom(data interface{}) error {
	var src unsafe.Pointer
	var bytes int

	switch d := data.(type) {
	case []float32:
		src = unsafe.Pointer(&d[0])
		bytes = len(d) * 4
		if t.dtype == DataTypeF16 {
			return fmt.Errorf("direct load of []float32 to F16 tensor not supported")
		}
	case []uint16:
		src = unsafe.Pointer(&d[0])
		bytes = len(d) * 2
	case []byte:
		src = unsafe.Pointer(&d[0])
		bytes = len(d)
	default:
		return fmt.Errorf("unsupported data type for LoadFrom: %T", data)
	}

	if err := C.cudaMemcpyAsync(t.devPtr, src, C.size_t(bytes), C.cudaMemcpyHostToDevice, t.ctx.Ctx); err != 0 {
		return fmt.Errorf("cudaMemcpyAsync failed: %v", err)
	}
	return nil
}

func (t *Tensor) ToHost() []float32 {
	return t.ToHostF32()
}

func (t *Tensor) ToHostF32() []float32 {
	size := t.rows * t.cols
	result := make([]float32, size)

	if t.dtype == DataTypeF16 {
		hostF16 := make([]uint16, size)
		C.cudaMemcpy(unsafe.Pointer(&hostF16[0]), t.devPtr, C.size_t(size*2), C.cudaMemcpyDeviceToHost)
		for i, v := range hostF16 {
			result[i] = Float16ToFloat32(v)
		}
	} else {
		C.cudaMemcpy(unsafe.Pointer(&result[0]), t.devPtr, C.size_t(size*4), C.cudaMemcpyDeviceToHost)
	}

	return result
}

// Math Kernels

func (ctx *CUDAContext) RMSNorm(input, weight, output *Tensor, rows, cols int, eps float32) {
	C.cudaRMSNorm((*C.float)(input.devPtr), (*C.float)(weight.devPtr), (*C.float)(output.devPtr), C.int(rows), C.int(cols), C.float(eps), ctx.Ctx)
}

func (ctx *CUDAContext) Add(a, b, out *Tensor, size int) {
	C.cudaAdd((*C.float)(a.devPtr), (*C.float)(b.devPtr), (*C.float)(out.devPtr), C.int(size), ctx.Ctx)
}

func (ctx *CUDAContext) MatmulF16(a, b *Tensor) (*Tensor, error) {
	m := a.rows
	k := a.cols
	n := b.rows
	out := ctx.NewTensorPooled(m, n)
	alpha := C.float(1.0)
	beta := C.float(0.0)
	res := C.cublasGemmEx(ctx.Cublas,
		C.CUBLAS_OP_T, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		unsafe.Pointer(&alpha),
		b.devPtr, C.CUDA_R_16F, C.int(k),
		a.devPtr, C.CUDA_R_16F, C.int(k),
		unsafe.Pointer(&beta),
		out.devPtr, C.CUDA_R_16F, C.int(n),
		C.CUBLAS_COMPUTE_32F,
		C.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
	if res != 0 {
		out.ReturnToPool()
		return nil, fmt.Errorf("cublasGemmEx failed: %v", res)
	}
	return out, nil
}

func (ctx *CUDAContext) FusedRoPE(tensor *Tensor, posIds []int, batch, heads, seqLen, headDim int, theta float32) {
	var dPosPtr unsafe.Pointer
	C.cudaMalloc(&dPosPtr, C.size_t(len(posIds)*4))
	C.cudaMemcpy(dPosPtr, unsafe.Pointer(&posIds[0]), C.size_t(len(posIds)*4), C.cudaMemcpyHostToDevice)
	C.cudaFusedRoPE(ctx.Ctx, tensor.devPtr, (*C.int)(dPosPtr), C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim), C.float(theta))
	C.cudaFree(dPosPtr)
}

func (ctx *CUDAContext) FusedAttention(q, k, v, output, kCache, vCache *Tensor, batch, heads, seqLen, kvSeqLen, headDim int, scale float32, useCache, windowSize int) {
	C.cudaFusedAttention(ctx.Ctx, q.devPtr, k.devPtr, v.devPtr, output.devPtr, kCache.devPtr, vCache.devPtr, C.int(batch), C.int(heads), C.int(seqLen), C.int(kvSeqLen), C.int(headDim), C.float(scale), C.int(useCache), C.int(windowSize))
}

func (ctx *CUDAContext) FusedMLP(input, gateW, upW, downW, output *Tensor, batch, dim, hiddenDim int) {
	C.cudaFusedMLP(ctx.Ctx, input.devPtr, gateW.devPtr, upW.devPtr, downW.devPtr, output.devPtr, C.int(batch), C.int(dim), C.int(hiddenDim))
}

func (ctx *CUDAContext) Synchronize() {
	C.cudaStreamSynchronize(ctx.Ctx)
}

// CUDAModel and Weight Loading

type weight struct {
	devPtr unsafe.Pointer
	rows   int
	cols   int
	dtype  DataType
	ctx    *CUDAContext
}

type CUDAModel struct {
	Ctx     *CUDAContext
	Weights map[string]*weight
	KCache  []*Tensor
	VCache  []*Tensor
	mu      sync.RWMutex
}

func (ctx *CUDAContext) NewCUDAModel(f *gguf.GGUFFile, preDequantize bool, kvCacheSize int) (*CUDAModel, error) {
	m := &CUDAModel{
		Ctx:     ctx,
		Weights: make(map[string]*weight),
	}

	for name, tensor := range f.Tensors {
		rows := int(tensor.Dimensions[0])
		cols := 1
		if len(tensor.Dimensions) > 1 {
			cols = int(tensor.Dimensions[1])
		}
		numElements := rows * cols
		var dPtr unsafe.Pointer
		C.cudaMalloc(&dPtr, C.size_t(numElements*2))
		var hostFP16 []uint16
		switch tensor.Type {
		case gguf.GGMLTypeF32:
			f32Data := make([]float32, numElements)
			for i := 0; i < numElements; i++ {
				f32Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.HostData[i*4:]))
			}
			hostFP16 = Float32SliceToFloat16(f32Data)
		case gguf.GGMLTypeQ8_0:
			f32Data := gguf.DequantizeQ8_0(tensor.HostData, numElements)
			hostFP16 = Float32SliceToFloat16(f32Data)
		case gguf.GGMLTypeQ4_K:
			f32Data := gguf.DequantizeQ4K(tensor.HostData, numElements)
			hostFP16 = Float32SliceToFloat16(f32Data)
		case gguf.GGMLTypeQ6_K:
			f32Data := gguf.DequantizeQ6K(tensor.HostData, numElements)
			hostFP16 = Float32SliceToFloat16(f32Data)
		default:
			C.cudaFree(dPtr)
			return nil, fmt.Errorf("unsupported quantization: %v", tensor.Type)
		}
		C.cudaMemcpy(dPtr, unsafe.Pointer(&hostFP16[0]), C.size_t(numElements*2), C.cudaMemcpyHostToDevice)
		m.Weights[name] = &weight{
			devPtr: dPtr,
			rows:   rows,
			cols:   cols,
			dtype:  DataTypeF16,
			ctx:    ctx,
		}
	}

	layers := 0
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		layers = int(v)
	}
	heads := 32
	if v, ok := f.KV["llama.attention.head_count"].(uint32); ok {
		heads = int(v)
	}
	dim := 2048
	if v, ok := f.KV["llama.embedding_length"].(uint32); ok {
		dim = int(v)
	}
	headDim := dim / heads

	m.KCache = make([]*Tensor, layers)
	m.VCache = make([]*Tensor, layers)
	for i := 0; i < layers; i++ {
		m.KCache[i], _ = ctx.NewTensor(kvCacheSize*heads, headDim, DataTypeF16)
		m.VCache[i], _ = ctx.NewTensor(kvCacheSize*heads, headDim, DataTypeF16)
	}
	return m, nil
}

func (m *CUDAModel) Free() {
	for _, w := range m.Weights {
		C.cudaFree(w.devPtr)
	}
	for _, c := range m.KCache {
		c.Free()
	}
	for _, c := range m.VCache {
		c.Free()
	}
}

func (m *CUDAModel) GetWeightTensor(name string) (*Tensor, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	w, ok := m.Weights[name]
	if !ok {
		return nil, false
	}
	return &Tensor{
		devPtr: w.devPtr,
		rows:   w.rows,
		cols:   w.cols,
		dtype:  w.dtype,
		ctx:    m.Ctx,
	}, true
}

func (m *CUDAModel) GetEmbeddingTensor(token int) (*Tensor, error) {
	embWeight, ok := m.GetWeightTensor("token_embd.weight")
	if !ok {
		return nil, fmt.Errorf("embedding weight not found")
	}
	tokenEmb := m.Ctx.NewTensorPooled(1, embWeight.rows)
	offset := uintptr(token) * uintptr(embWeight.rows) * 2
	srcPtr := unsafe.Pointer(uintptr(embWeight.devPtr) + offset)
	C.cudaMemcpy(tokenEmb.devPtr, srcPtr, C.size_t(embWeight.rows*2), C.cudaMemcpyDeviceToDevice)
	return tokenEmb, nil
}

func (m *CUDAModel) GetKCache(layer int) *Tensor {
	if layer < 0 || layer >= len(m.KCache) {
		return nil
	}
	return m.KCache[layer]
}

func (m *CUDAModel) GetVCache(layer int) *Tensor {
	if layer < 0 || layer >= len(m.VCache) {
		return nil
	}
	return m.VCache[layer]
}

func CUDAAllocatedBytes() int64 {
	var free, total C.size_t
	C.cudaMemGetInfo(&free, &total)
	return int64(total - free)
}

type LayerScratch struct {
	Normed *Tensor
	Attn   *Tensor
	Gate   *Tensor
	Up     *Tensor
	Down   *Tensor
}

func (ctx *CUDAContext) NewLayerScratch(batch, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize, qNormDim, kNormDim int) *LayerScratch {
	normed, _ := ctx.NewTensor(batch, dim, DataTypeF16)
	attn, _ := ctx.NewTensor(batch*heads, headDim, DataTypeF16)
	gate, _ := ctx.NewTensor(batch, hiddenDim, DataTypeF16)
	up, _ := ctx.NewTensor(batch, hiddenDim, DataTypeF16)
	down, _ := ctx.NewTensor(batch, dim, DataTypeF16)
	return &LayerScratch{
		Normed: normed,
		Attn:   attn,
		Gate:   gate,
		Up:     up,
		Down:   down,
	}
}

func (s *LayerScratch) Free() {
	s.Normed.Free()
	s.Attn.Free()
	s.Gate.Free()
	s.Up.Free()
	s.Down.Free()
}

func (ctx *CUDAContext) CopyF16(src, dst *Tensor) {
	C.cudaMemcpyAsync(dst.devPtr, src.devPtr, C.size_t(src.rows*src.cols*2), C.cudaMemcpyDeviceToDevice, ctx.Ctx)
}

func Float32SliceToFloat16(data []float32) []uint16 {
	res := make([]uint16, len(data))
	for i, v := range data {
		res[i] = Float32ToFloat16(v)
	}
	return res
}
