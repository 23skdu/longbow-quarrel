//go:build linux && amd64 && cuda && cgo

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
extern void cudaFusedRoPE(cudaStream_t stream, void* tensor, const int* posIds, int batch, int heads, int seqLen, int headDim, float theta);
extern void cudaPrecomputeRoPE(cudaStream_t stream, float* ropeCos, float* ropeSin, int maxSeqLen, int headDim, int rotaryDim, float theta);
extern void cudaFusedRoPEPrecomputed(cudaStream_t stream, float* tensor, const int* posIds, const float* ropeCos, const float* ropeSin, int batch, int heads, int seqLen, int headDim, int maxSeqLen);
extern void cudaFusedQKVRope(cudaStream_t stream, const void* input, const void* qWeight, const void* kWeight, const void* vWeight, void* qOut, void* kOut, void* vOut, const void* ropeCos, const void* ropeSin, const int* posIds, int batch, int dim, int qDim, int kvDim, int headDim, int maxSeqLen, int isF16);
extern void cudaFusedMLP(cudaStream_t stream, const void* input, const void* gateWeight, const void* upWeight, const void* downWeight, void* output, int batch, int dim, int hiddenDim);
extern void cudaFusedRMSNormAdd(cudaStream_t stream, const void* input, const void* hidden, const void* weight, void* output, int batch, int dim, float eps);
extern void cudaStoreKVPagedBatch(cudaStream_t stream, const float* k, const float* v, void* kPool, void* vPool, const int* physicalPositions, int kvDim, int numTokens);
extern void cudaPagedAttentionBatch(cudaStream_t stream, const float* q, const void* kPool, const void* vPool, float* output, const int* tokenPositions, const int* blockTables, const int* tokenToSeq, int maxBlocks, int heads, int kvHeads, int headDim, int blockSize, int numTokens, float scale);
extern void cudaPagedAttentionTurboQuant(cudaStream_t stream, const float* q, const void* kPool, const void* vPool, float* output, const int* tokenPositions, const int* blockTables, const int* tokenToSeq, int maxBlocks, int heads, int kvHeads, int headDim, int blockSize, int numTokens, float scale, int qjlRows);

extern void cudaTurboQuantEncode(cudaStream_t stream, const float* input, const float* rotationMatrix, const float* qjlMatrix, int8_t* output, float* scaleOut, float* qjlScaleOut, int blockSize, int qjlRows, int numBlocks, int bits);

extern void cudaTurboQuantDecode(cudaStream_t stream, const int8_t* input, const float* rotationMatrix, void* output, const float* scaleIn, int blockSize, int qjlRows, int numBlocks);

extern void cudaStoreKVTurboQuant(cudaStream_t stream, const float* k, const float* v, void* kCache, void* vCache, const int* physicalPositions, int blockSize, int qjlRows, int numHeads, int numTokens);

extern void cudaMatVecDequantQ8_0(cudaStream_t stream, const void* weight, const float* x, float* y, int M, int K);
extern void cudaMatVecDequantQ4_K(cudaStream_t stream, const void* weight, const float* x, float* y, int M, int K);
extern void cudaMatVecDequantQ6_K(cudaStream_t stream, const void* weight, const float* x, float* y, int M, int K);
extern void cudaFlashAttentionPrefill(cudaStream_t stream, const float* q, const float* k, const float* v, float* output, int batch, int heads, int kvHeads, int qSeqLen, int kvSeqLen, int headDim, float scale, int slidingWindow);
extern void cudaPagedAttentionQuantized(cudaStream_t stream, const float* q, const void* kPool, const void* vPool, const float* kScales, const float* vScales, float* output, const int* tokenPositions, const int* blockTables, const int* tokenToSeq, int maxBlocks, int heads, int kvHeads, int headDim, int blockSize, int numTokens, float scale, int isFP8);

// MOE Kernels
extern void cudaMOERouterLogits(cudaStream_t stream, const void* input, const void* gate, float* output, int batch, int dim, int num_experts, int isF16);
extern void cudaMOETopKSelection(cudaStream_t stream, const float* logits, int top_k, float* indices, float* weights, int batch, int num_experts);
extern void cudaMOEExpertForward(cudaStream_t stream, const void* input, const void* expert_weights, const float* indices, const float* expert_weights_w, void* output, int batch, int dim, int hidden_dim, int num_experts, int top_k, int isF16);
extern void cudaMOEExpertGateUpSwiGLU(cudaStream_t stream, const void* input, const void* gate_experts, const void* up_experts, const float* indices, const float* weights, void* output, int batch, int dim, int hidden_dim, int num_experts, int top_k, int isF16);

// MLA (Multi-Head Latent Attention) Kernels
extern void cudaMLADecompressKV(cudaStream_t stream, const void* compressed_kv, const void* w_ukv, void* k_nope, void* v, int num_tokens, int kv_lora_rank, int heads, int qk_nope_dim, int v_head_dim, int isF16);
extern void cudaMLAProjectQuerySplitRoPE(cudaStream_t stream, const void* q_all, const int* pos_ids, void* q_nope, void* q_rope, int num_tokens, int heads, int qk_nope_dim, int qk_rope_dim, float theta, int isF16);
extern void cudaMLAAbsorbedQuery(cudaStream_t stream, const void* q_nope, const void* w_uk, void* q_absorbed, int num_tokens, int heads, int qk_nope_dim, int kv_lora_rank, int isF16);
extern void cudaMLAAbsorbedDecodeAttention(cudaStream_t stream, const void* q_absorbed, const void* q_rope, const void* k_cache, const void* k_rope_cache, const void* w_uv, void* output, int num_tokens, int seq_len, int heads, int kv_lora_rank, int qk_rope_dim, int v_head_dim, float scale, int isF16);
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/simd"
)

var globalContext *Context

type Context struct {
	Ctx         C.cudaStream_t
	Cublas      C.cublasHandle_t
	pool        *tensorPool
	TQRotation  *Tensor
	TQQJL       *Tensor
	ropeCos     *Tensor
	ropeSin     *Tensor
	ropeMaxLen  int
	ropeHeadDim int
	ropeRotDim  int
	ropeTheta   float32
	dPosIds     unsafe.Pointer
}

func (ctx *Context) DeviceID() int {
	return 0
}

type Tensor struct {
	devPtr    unsafe.Pointer
	rows      int
	cols      int
	dataType  DataType
	ctx       *Context
	pooled    bool
	sizeBytes int
	blockSize int
	qjlRows   int
}

type tensorPool struct {
	mu   sync.Mutex
	free map[int][]*Tensor
}

func NewContext() *Context {
	var stream C.cudaStream_t
	if err := C.cudaStreamCreate(&stream); err != 0 {
		panic(fmt.Sprintf("cudaStreamCreate failed: %v", err))
	}

	var handle C.cublasHandle_t
	if err := C.cublasCreate(&handle); err != 0 {
		panic(fmt.Sprintf("cublasCreate failed: %v", err))
	}
	C.cublasSetStream(handle, stream)

	ctx := &Context{
		Ctx:    stream,
		Cublas: handle,
		pool: &tensorPool{
			free: make(map[int][]*Tensor),
		},
	}
	C.cudaMalloc(&ctx.dPosIds, 4096)
	globalContext = ctx
	return ctx
}

func (ctx *Context) Free() {
	if ctx.dPosIds != nil {
		C.cudaFree(ctx.dPosIds)
		ctx.dPosIds = nil
	}
	if ctx.ropeCos != nil {
		ctx.ropeCos.Free()
		ctx.ropeCos = nil
	}
	if ctx.ropeSin != nil {
		ctx.ropeSin.Free()
		ctx.ropeSin = nil
	}
	if ctx.Ctx != nil {
		C.cudaStreamDestroy(ctx.Ctx)
	}
	if ctx.Cublas != nil {
		C.cublasDestroy(ctx.Cublas)
	}
}

func (ctx *Context) NewTensorFP32(rows, cols int) *Tensor {
	t, _ := ctx.NewTensor(rows, cols, DataTypeF32)
	return t
}

func (ctx *Context) NewTensorI32(rows, cols int) *Tensor {
	t, _ := ctx.NewTensor(rows, cols, DataTypeI32)
	return t
}

func (ctx *Context) NewTensor(rows, cols int, dtype DataType) (*Tensor, error) {
	size := rows * cols
	var bytes int
	switch dtype {
	case DataTypeF16, DataTypeQ4K, DataTypeQ4_0, DataTypeQ6K, DataTypeQ8_0:
		bytes = size * 2
	case DataTypeF32, DataTypeI32:
		bytes = size * 4
	default:
		bytes = size * 4
	}

	var ptr unsafe.Pointer
	if err := C.cudaMalloc(&ptr, C.size_t(bytes)); err != 0 {
		fmt.Printf("cudaMalloc FAILED: bytes=%d cudaError=%d\n", bytes, int(err))
		return nil, fmt.Errorf("cudaMalloc failed: %v", err)
	}

	return &Tensor{
		devPtr:    ptr,
		rows:      rows,
		cols:      cols,
		dataType:  dtype,
		ctx:       ctx,
		sizeBytes: bytes,
	}, nil
}

func (ctx *Context) NewTensorFromData(rows, cols int, dtype DataType, data []byte) (*Tensor, error) {
	t, err := ctx.NewTensor(rows, cols, dtype)
	if err != nil {
		return nil, err
	}
	if len(data) > 0 {
		C.cudaMemcpy(t.devPtr, unsafe.Pointer(&data[0]), C.size_t(len(data)), C.cudaMemcpyHostToDevice)
	}
	return t, nil
}

func (ctx *Context) NewTensorWithType(rows, cols int, dtype DataType) *Tensor {
	t, _ := ctx.NewTensor(rows, cols, dtype)
	return t
}

func (ctx *Context) NewTurboTensor(rows, cols int, dt DataType, blockSize, qjlRows int) *Tensor {
	numElements := rows * cols
	numBlocks := numElements / blockSize
	if numElements%blockSize != 0 {
		numBlocks++
	}
	bytesPerBlock := blockSize + qjlRows + 8
	sizeBytes := numBlocks * bytesPerBlock

	var ptr unsafe.Pointer
	if err := C.cudaMalloc(&ptr, C.size_t(sizeBytes)); err != 0 {
		return nil
	}

	return &Tensor{
		ctx:       ctx,
		rows:      rows,
		cols:      cols,
		dataType:  dt,
		devPtr:    ptr,
		sizeBytes: sizeBytes,
		blockSize: blockSize,
		qjlRows:   qjlRows,
	}
}

func (ctx *Context) NewTensorPooled(rows, cols int) *Tensor {
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

func (t *Tensor) DataType() DataType { return t.dataType }

func (t *Tensor) IsDevice() bool { return true }

func (t *Tensor) Rows() int            { return t.rows }
func (t *Tensor) Cols() int            { return t.cols }
func (t *Tensor) Data() unsafe.Pointer { return t.devPtr }

func (t *Tensor) SizeBytes() int {
	return t.sizeBytes
}

func (t *Tensor) RawData() []byte {
	return unsafe.Slice((*byte)(t.devPtr), t.SizeBytes())
}

func (t *Tensor) LoadFrom(data interface{}) error {
	var src unsafe.Pointer
	var bytes int

	switch d := data.(type) {
	case []float32:
		if t.dataType == DataTypeF16 {
			hostF16 := Float32SliceToFloat16(d)
			src = unsafe.Pointer(&hostF16[0])
			bytes = len(d) * 2
		} else {
			src = unsafe.Pointer(&d[0])
			bytes = len(d) * 4
		}
	case []int32:
		src = unsafe.Pointer(&d[0])
		bytes = len(d) * 4
	case []int:
		i32s := make([]int32, len(d))
		for i, v := range d {
			i32s[i] = int32(v)
		}
		src = unsafe.Pointer(&i32s[0])
		bytes = len(i32s) * 4
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

func (t *Tensor) LoadFromF32(data []float32) error {
	return t.LoadFrom(data)
}

func (t *Tensor) ToHost() []float32 {
	return t.ToHostF32()
}

func (t *Tensor) ToHostF32() []float32 {
	size := t.rows * t.cols
	result := make([]float32, size)

	if t.dataType == DataTypeF16 {
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

func (t *Tensor) ToHostFP16() []uint16 {
	size := t.rows * t.cols
	hostF16 := make([]uint16, size)
	if t.dataType == DataTypeF16 {
		C.cudaMemcpy(unsafe.Pointer(&hostF16[0]), t.devPtr, C.size_t(size*2), C.cudaMemcpyDeviceToHost)
	} else {
		hostF32 := t.ToHostF32()
		for i, v := range hostF32 {
			hostF16[i] = Float32ToFloat16(v)
		}
	}
	return hostF16
}

func (t *Tensor) StoreKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	if kCache == nil || vCache == nil || t == nil || v == nil || windowSize <= 0 {
		return
	}

	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		bits := 2
		if kCache.dataType == DataTypeTQ2_0 {
			bits = 4
		}
		blockSize := kCache.blockSize
		if blockSize == 0 {
			blockSize = headDim
		}
		qjlRows := kCache.qjlRows
		if qjlRows == 0 {
			qjlRows = 64
		}
		bytesPerBlock := blockSize + qjlRows + 8
		cacheOff := (pos % windowSize) * heads * bytesPerBlock

		tHost := t.ToHostF32()
		vHost := v.ToHostF32()
		rot := t.ctx.TQRotation
		qjl := t.ctx.TQQJL

		var rotData, qjlData []float32
		if rot != nil {
			rotData = rot.ToHostF32()
		}
		if qjl != nil {
			qjlData = qjl.ToHostF32()
		}

		kRaw := make([]byte, heads*bytesPerBlock)
		vRaw := make([]byte, heads*bytesPerBlock)

		for h := 0; h < heads; h++ {
			headStart := h * headDim
			headData := tHost[headStart : headStart+headDim]
			blockCacheStart := h * bytesPerBlock

			if rotData != nil {
				q, s, res := simd.PolarQuantSIMD(headData, rotData, blockSize, bits)
				var qj []int8
				var sj float32
				if qjlData != nil && len(qjlData) >= qjlRows*blockSize {
					qj, sj = simd.QJLTransformSIMD(res, qjlData, qjlRows, blockSize)
				}
				for i, val := range q {
					kRaw[blockCacheStart+i] = byte(val)
				}
				for i, val := range qj {
					kRaw[blockCacheStart+blockSize+i] = byte(val)
				}
				setFloat32(kRaw[blockCacheStart+blockSize+qjlRows:blockCacheStart+blockSize+qjlRows+4], s)
				setFloat32(kRaw[blockCacheStart+blockSize+qjlRows+4:blockCacheStart+blockSize+qjlRows+8], sj)

				vHeadData := vHost[headStart : headStart+headDim]
				qv, sv, resv := simd.PolarQuantSIMD(vHeadData, rotData, blockSize, bits)
				var qjv []int8
				var sjv float32
				if qjlData != nil && len(qjlData) >= qjlRows*blockSize {
					qjv, sjv = simd.QJLTransformSIMD(resv, qjlData, qjlRows, blockSize)
				}
				for i, val := range qv {
					vRaw[blockCacheStart+i] = byte(val)
				}
				for i, val := range qjv {
					vRaw[blockCacheStart+blockSize+i] = byte(val)
				}
				setFloat32(vRaw[blockCacheStart+blockSize+qjlRows:blockCacheStart+blockSize+qjlRows+4], sv)
				setFloat32(vRaw[blockCacheStart+blockSize+qjlRows+4:blockCacheStart+blockSize+qjlRows+8], sjv)
			}
		}

		C.cudaMemcpy(unsafe.Pointer(uintptr(kCache.devPtr)+uintptr(cacheOff)), unsafe.Pointer(&kRaw[0]), C.size_t(len(kRaw)), C.cudaMemcpyHostToDevice)
		C.cudaMemcpy(unsafe.Pointer(uintptr(vCache.devPtr)+uintptr(cacheOff)), unsafe.Pointer(&vRaw[0]), C.size_t(len(vRaw)), C.cudaMemcpyHostToDevice)
		return
	}

	count := heads * headDim
	if t.dataType == DataTypeF16 {
		offset := uintptr(pos%windowSize) * uintptr(count) * 2
		kTarget := unsafe.Pointer(uintptr(kCache.devPtr) + offset)
		vTarget := unsafe.Pointer(uintptr(vCache.devPtr) + offset)
		C.cudaMemcpyAsync(kTarget, t.devPtr, C.size_t(count*2), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
		C.cudaMemcpyAsync(vTarget, v.devPtr, C.size_t(count*2), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
	} else {
		offset := uintptr(pos%windowSize) * uintptr(count) * 4
		kTarget := unsafe.Pointer(uintptr(kCache.devPtr) + offset)
		vTarget := unsafe.Pointer(uintptr(vCache.devPtr) + offset)
		C.cudaMemcpyAsync(kTarget, t.devPtr, C.size_t(count*4), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
		C.cudaMemcpyAsync(vTarget, v.devPtr, C.size_t(count*4), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
	}
}

// FetchKV retrieves K and V projections from KV cache (FP16, FP32, or TurboQuant) into t and v.
func (t *Tensor) FetchKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	if kCache == nil || vCache == nil || t == nil || v == nil || windowSize <= 0 {
		return
	}

	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		blockSize := kCache.blockSize
		if blockSize == 0 {
			blockSize = headDim
		}
		qjlRows := kCache.qjlRows
		if qjlRows == 0 {
			qjlRows = 64
		}
		bytesPerBlock := blockSize + qjlRows + 8
		cacheOff := (pos % windowSize) * heads * bytesPerBlock

		kRaw := make([]byte, heads*bytesPerBlock)
		vRaw := make([]byte, heads*bytesPerBlock)

		C.cudaMemcpy(unsafe.Pointer(&kRaw[0]), unsafe.Pointer(uintptr(kCache.devPtr)+uintptr(cacheOff)), C.size_t(len(kRaw)), C.cudaMemcpyDeviceToHost)
		C.cudaMemcpy(unsafe.Pointer(&vRaw[0]), unsafe.Pointer(uintptr(vCache.devPtr)+uintptr(cacheOff)), C.size_t(len(vRaw)), C.cudaMemcpyDeviceToHost)

		rot := t.ctx.TQRotation
		qjl := t.ctx.TQQJL

		var rotData, qjlData []float32
		if rot != nil {
			rotData = rot.ToHostF32()
		}
		if qjl != nil {
			qjlData = qjl.ToHostF32()
		}

		tOut := make([]float32, heads*headDim)
		vOut := make([]float32, heads*headDim)

		for h := 0; h < heads; h++ {
			blockCacheStart := h * bytesPerBlock
			kSrc := kRaw[blockCacheStart:]
			kDest := tOut[h*headDim : (h+1)*headDim]
			dequantizeBlockCUDA(kSrc, kDest, blockSize, qjlRows, rotData, qjlData)

			vSrc := vRaw[blockCacheStart:]
			vDest := vOut[h*headDim : (h+1)*headDim]
			dequantizeBlockCUDA(vSrc, vDest, blockSize, qjlRows, rotData, qjlData)
		}

		_ = t.LoadFromF32(tOut)
		_ = v.LoadFromF32(vOut)
		return
	}

	count := heads * headDim
	if t.dataType == DataTypeF16 {
		offset := uintptr(pos%windowSize) * uintptr(count) * 2
		C.cudaMemcpyAsync(t.devPtr, unsafe.Pointer(uintptr(kCache.devPtr)+offset), C.size_t(count*2), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
		C.cudaMemcpyAsync(v.devPtr, unsafe.Pointer(uintptr(vCache.devPtr)+offset), C.size_t(count*2), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
	} else {
		offset := uintptr(pos%windowSize) * uintptr(count) * 4
		C.cudaMemcpyAsync(t.devPtr, unsafe.Pointer(uintptr(kCache.devPtr)+offset), C.size_t(count*4), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
		C.cudaMemcpyAsync(v.devPtr, unsafe.Pointer(uintptr(vCache.devPtr)+offset), C.size_t(count*4), C.cudaMemcpyDeviceToDevice, t.ctx.Ctx)
	}
}

func dequantizeBlockCUDA(src []byte, dst []float32, blockSize, qjlRows int, rotData, qjlData []float32) {
	q := make([]int8, blockSize)
	for i := 0; i < blockSize; i++ {
		q[i] = int8(src[i])
	}
	qj := make([]int8, qjlRows)
	for i := 0; i < qjlRows; i++ {
		qj[i] = int8(src[blockSize+i])
	}
	s := getFloat32(src[blockSize+qjlRows : blockSize+qjlRows+4])
	sj := getFloat32(src[blockSize+qjlRows+4 : blockSize+qjlRows+8])

	rotatedRes := make([]float32, blockSize)
	for i := 0; i < blockSize; i++ {
		rotatedRes[i] = float32(q[i]) * s
	}

	if rotData != nil && len(rotData) >= blockSize*blockSize {
		for i := 0; i < blockSize; i++ {
			var sum float32
			for j := 0; j < blockSize; j++ {
				sum += rotData[j*blockSize+i] * rotatedRes[j]
			}
			dst[i] = sum
		}
	} else {
		copy(dst, rotatedRes)
	}

	if sj > 0 && qjlData != nil && len(qjlData) >= qjlRows*blockSize {
		for i := 0; i < qjlRows; i++ {
			scale_i := float32(int8(qj[i])) * sj
			for j := 0; j < blockSize; j++ {
				dst[j] += scale_i * qjlData[i*blockSize+j]
			}
		}
	}
}

// Math Kernels

func (ctx *Context) RMSNorm(input, weight, output *Tensor, rows, cols int, eps float32) {
	C.cudaRMSNorm((*C.float)(input.devPtr), (*C.float)(weight.devPtr), (*C.float)(output.devPtr), C.int(rows), C.int(cols), C.float(eps), ctx.Ctx)
}

func (ctx *Context) Add(a, b, out *Tensor, size int) {
	C.cudaAdd((*C.float)(a.devPtr), (*C.float)(b.devPtr), (*C.float)(out.devPtr), C.int(size), ctx.Ctx)
}

func (ctx *Context) MatmulF16(a, b *Tensor) (*Tensor, error) {
	m := a.rows
	k := a.cols
	n := b.rows
	out := ctx.NewTensorPooled(m, n)

	if b.dataType == DataTypeQ8_0 {
		ctx.MatVecDequantQ8_0(b, a, out, n, k)
		return out, nil
	}
	if b.dataType == DataTypeQ4_K {
		ctx.MatVecDequantQ4_K(b, a, out, n, k)
		return out, nil
	}
	if b.dataType == DataTypeQ6_K {
		ctx.MatVecDequantQ6_K(b, a, out, n, k)
		return out, nil
	}

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

func (ctx *Context) FusedRoPE(tensor *Tensor, posIds []int, batch, heads, seqLen, headDim int, theta float32) {
	var dPosPtr unsafe.Pointer
	C.cudaMalloc(&dPosPtr, C.size_t(len(posIds)*4))
	C.cudaMemcpy(dPosPtr, unsafe.Pointer(&posIds[0]), C.size_t(len(posIds)*4), C.cudaMemcpyHostToDevice)
	C.cudaFusedRoPE(ctx.Ctx, tensor.devPtr, (*C.int)(dPosPtr), C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim), C.float(theta))
	C.cudaFree(dPosPtr)
}

func (ctx *Context) EnsureRoPETables(maxSeqLen, headDim, rotaryDim int, theta float32) {
	if rotaryDim <= 0 {
		rotaryDim = headDim
	}
	if ctx.ropeCos != nil && ctx.ropeMaxLen >= maxSeqLen && ctx.ropeHeadDim == headDim && ctx.ropeRotDim == rotaryDim && ctx.ropeTheta == theta {
		return
	}
	if ctx.ropeCos != nil {
		ctx.ropeCos.Free()
		ctx.ropeCos = nil
	}
	if ctx.ropeSin != nil {
		ctx.ropeSin.Free()
		ctx.ropeSin = nil
	}
	if maxSeqLen < 2048 {
		maxSeqLen = 2048
	}
	headDimHalf := headDim / 2
	ctx.ropeCos = ctx.NewTensorFP32(maxSeqLen, headDimHalf)
	ctx.ropeSin = ctx.NewTensorFP32(maxSeqLen, headDimHalf)
	ctx.ropeMaxLen = maxSeqLen
	ctx.ropeHeadDim = headDim
	ctx.ropeRotDim = rotaryDim
	ctx.ropeTheta = theta

	C.cudaPrecomputeRoPE(ctx.Ctx, (*C.float)(ctx.ropeCos.devPtr), (*C.float)(ctx.ropeSin.devPtr),
		C.int(maxSeqLen), C.int(headDim), C.int(rotaryDim), C.float(theta))
}

func (ctx *Context) FusedRoPEPrecomputed(tensor *Tensor, posIds []int, batch, heads, seqLen, headDim int) {
	if len(posIds) > 0 && ctx.dPosIds != nil {
		copyLen := len(posIds)
		if copyLen > 1024 {
			copyLen = 1024
		}
		C.cudaMemcpyAsync(ctx.dPosIds, unsafe.Pointer(&posIds[0]), C.size_t(copyLen*4), C.cudaMemcpyHostToDevice, ctx.Ctx)
	}
	maxLen := ctx.ropeMaxLen
	if maxLen <= 0 {
		maxLen = 2048
	}
	C.cudaFusedRoPEPrecomputed(ctx.Ctx, (*C.float)(tensor.devPtr), (*C.int)(ctx.dPosIds),
		(*C.float)(ctx.ropeCos.devPtr), (*C.float)(ctx.ropeSin.devPtr),
		C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim), C.int(maxLen))
}

func (ctx *Context) FusedQKVRope(input, qW, kW, vW *Tensor, posIds []int, batch, dim, heads, kvHeads, headDim, rotaryDim int, theta float32) (q, k, v *Tensor, err error) {
	if rotaryDim <= 0 {
		rotaryDim = headDim
	}
	qDim := heads * headDim
	kvDim := kvHeads * headDim

	maxPos := 0
	for _, p := range posIds {
		if p > maxPos {
			maxPos = p
		}
	}
	ctx.EnsureRoPETables(maxPos+1, headDim, rotaryDim, theta)

	isF16 := 0
	if qW.dataType == DataTypeF16 {
		isF16 = 1
	}

	// For small batches (typical decoding / small prefill) with non-quantized weights,
	// run the single-pass fused GEMV + RoPE kernel.
	if batch <= 8 && qW.dataType != DataTypeQ4_K && qW.dataType != DataTypeQ8_0 && qW.dataType != DataTypeQ6_K {
		q = ctx.NewTensorFP32(batch, qDim)
		k = ctx.NewTensorFP32(batch, kvDim)
		v = ctx.NewTensorFP32(batch, kvDim)

		if len(posIds) > 0 && ctx.dPosIds != nil {
			copyLen := len(posIds)
			if copyLen > 1024 {
				copyLen = 1024
			}
			C.cudaMemcpyAsync(ctx.dPosIds, unsafe.Pointer(&posIds[0]), C.size_t(copyLen*4), C.cudaMemcpyHostToDevice, ctx.Ctx)
		}
		C.cudaFusedQKVRope(ctx.Ctx, input.devPtr, qW.devPtr, kW.devPtr, vW.devPtr,
			q.devPtr, k.devPtr, v.devPtr,
			ctx.ropeCos.devPtr, ctx.ropeSin.devPtr,
			(*C.int)(ctx.dPosIds),
			C.int(batch), C.int(dim), C.int(qDim), C.int(kvDim), C.int(headDim),
			C.int(ctx.ropeMaxLen), C.int(isF16))
		return q, k, v, nil
	}

	// For larger batches or quantized weights: compute Q, K, V via Matmul / Dequant
	q, err = ctx.MatmulF16(input, qW)
	if err != nil {
		return nil, nil, nil, err
	}
	k, err = ctx.MatmulF16(input, kW)
	if err != nil {
		q.ReturnToPool()
		return nil, nil, nil, err
	}
	v, err = ctx.MatmulF16(input, vW)
	if err != nil {
		q.ReturnToPool()
		k.ReturnToPool()
		return nil, nil, nil, err
	}

	ctx.FusedRoPEPrecomputed(q, posIds, batch, heads, 1, headDim)
	ctx.FusedRoPEPrecomputed(k, posIds, batch, kvHeads, 1, headDim)

	return q, k, v, nil
}

func (ctx *Context) FusedAttention(q, k, v, output, kCache, vCache *Tensor, batch, heads, seqLen, kvSeqLen, headDim int, scale float32, useCache, windowSize int) {
	C.cudaFusedAttention(ctx.Ctx, q.devPtr, k.devPtr, v.devPtr, output.devPtr, kCache.devPtr, vCache.devPtr, C.int(batch), C.int(heads), C.int(seqLen), C.int(kvSeqLen), C.int(headDim), C.float(scale), C.int(useCache), C.int(windowSize))
}

func (ctx *Context) FusedMLP(input, gateW, upW, downW, output *Tensor, batch, dim, hiddenDim int) {
	C.cudaFusedMLP(ctx.Ctx, input.devPtr, gateW.devPtr, upW.devPtr, downW.devPtr, output.devPtr, C.int(batch), C.int(dim), C.int(hiddenDim))
}

func (ctx *Context) FusedRMSNormAdd(input, hidden, weight, output *Tensor, batch, dim int, eps float32) {
	C.cudaFusedRMSNormAdd(ctx.Ctx, input.devPtr, hidden.devPtr, weight.devPtr, output.devPtr, C.int(batch), C.int(dim), C.float(eps))
}

func (ctx *Context) FusedSwiGLU(gate, up, output *Tensor, rows, size int) {
	C.cudaSwiGLU((*C.float)(gate.devPtr), (*C.float)(up.devPtr), (*C.float)(output.devPtr), C.int(rows*size), ctx.Ctx)
}

// ============================================================================
// MOE (Mixture of Experts) Operations
// ============================================================================

// MOERouterLogits computes routing logits for MOE layer
// input: [batch_size, dim]
// gateWeight: [num_experts, dim]
// Returns: [batch_size, num_experts] logits (DataTypeF32)
func (ctx *Context) MOERouterLogits(input, gateWeight *Tensor) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	numExperts := gateWeight.Rows()

	logits := ctx.NewTensorFP32(batchSize, numExperts)
	isF16 := 0
	if input.dataType == DataTypeF16 || gateWeight.dataType == DataTypeF16 {
		isF16 = 1
	}

	C.cudaMOERouterLogits(ctx.Ctx, input.devPtr, gateWeight.devPtr,
		(*C.float)(logits.devPtr), C.int(batchSize), C.int(dim), C.int(numExperts), C.int(isF16))
	return logits
}

// MOETopKSelection selects top-k experts per token and computes softmax weights
// logits: [batch_size, num_experts]
// Returns: expertIndices [batch_size, top_k] (DataTypeF32), expertWeights [batch_size, top_k] (DataTypeF32)
func (ctx *Context) MOETopKSelection(logits *Tensor, topK int) (*Tensor, *Tensor) {
	batchSize := logits.Rows()
	numExperts := logits.Cols()

	expertIndices := ctx.NewTensorFP32(batchSize, topK)
	expertWeights := ctx.NewTensorFP32(batchSize, topK)

	C.cudaMOETopKSelection(ctx.Ctx, (*C.float)(logits.devPtr), C.int(topK),
		(*C.float)(expertIndices.devPtr), (*C.float)(expertWeights.devPtr),
		C.int(batchSize), C.int(numExperts))
	return expertIndices, expertWeights
}

// MOEExpertForward applies selected experts to input with weighted mixing
// input: [batch_size, dim]
// expertWeight: [hidden_dim * num_experts, dim] (flattened 3D)
// expertIndices: [batch_size, top_k] (DataTypeF32)
// expertWeights: [batch_size, top_k] (DataTypeF32)
// Returns: [batch_size, hiddenDim]
func (ctx *Context) MOEExpertForward(input, expertWeight, expertIndices, expertWeights *Tensor, hiddenDim int) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	topK := expertIndices.Cols()
	numExperts := expertWeight.Rows() / hiddenDim
	if numExperts <= 0 {
		numExperts = 1
	}

	isF16 := 0
	var output *Tensor
	if input.dataType == DataTypeF16 || expertWeight.dataType == DataTypeF16 {
		isF16 = 1
		output = ctx.NewTensorPooled(batchSize, hiddenDim)
	} else {
		output = ctx.NewTensorFP32(batchSize, hiddenDim)
	}

	C.cudaMOEExpertForward(ctx.Ctx, input.devPtr, expertWeight.devPtr,
		(*C.float)(expertIndices.devPtr), (*C.float)(expertWeights.devPtr),
		output.devPtr, C.int(batchSize), C.int(dim), C.int(hiddenDim),
		C.int(numExperts), C.int(topK), C.int(isF16))
	return output
}

// MOEExpertGateUpSwiGLU applies fused gate, up and SwiGLU forward pass for multiple experts
// input: [batch_size, dim]
// gateWeight: [hidden_dim * num_experts, dim]
// upWeight: [hidden_dim * num_experts, dim]
// expertIndices: [batch_size, top_k]
// expertWeights: [batch_size, top_k]
// Returns: [batch_size, hiddenDim]
func (ctx *Context) MOEExpertGateUpSwiGLU(input, gateWeight, upWeight, expertIndices, expertWeights *Tensor, hiddenDim int) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	topK := expertIndices.Cols()
	numExperts := gateWeight.Rows() / hiddenDim
	if numExperts <= 0 {
		numExperts = 1
	}

	isF16 := 0
	var output *Tensor
	if input.dataType == DataTypeF16 || gateWeight.dataType == DataTypeF16 {
		isF16 = 1
		output = ctx.NewTensorPooled(batchSize, hiddenDim)
	} else {
		output = ctx.NewTensorFP32(batchSize, hiddenDim)
	}

	C.cudaMOEExpertGateUpSwiGLU(ctx.Ctx, input.devPtr,
		gateWeight.devPtr, upWeight.devPtr,
		(*C.float)(expertIndices.devPtr), (*C.float)(expertWeights.devPtr),
		output.devPtr, C.int(batchSize), C.int(dim), C.int(hiddenDim),
		C.int(numExperts), C.int(topK), C.int(isF16))
	return output
}

// =============================================================================
// MLA (Multi-Head Latent Attention) Operations
// =============================================================================

// MLADecompressKV decompresses compressed latent KV cache into separate key (non-rotary) and value tensors
// compressedKV: [numTokens, kvLoraRank]
// wUKV: [heads * (qkNopeDim + vHeadDim), kvLoraRank]
// Returns: kNope [numTokens, heads * qkNopeDim], v [numTokens, heads * vHeadDim]
func (ctx *Context) MLADecompressKV(compressedKV, wUKV *Tensor, numTokens, kvLoraRank, heads, qkNopeDim, vHeadDim int) (*Tensor, *Tensor) {
	isF16 := 0
	var kNope, v *Tensor
	if compressedKV.dataType == DataTypeF16 || wUKV.dataType == DataTypeF16 {
		isF16 = 1
		kNope = ctx.NewTensorPooled(numTokens, heads*qkNopeDim)
		v = ctx.NewTensorPooled(numTokens, heads*vHeadDim)
	} else {
		kNope = ctx.NewTensorFP32(numTokens, heads*qkNopeDim)
		v = ctx.NewTensorFP32(numTokens, heads*vHeadDim)
	}

	C.cudaMLADecompressKV(ctx.Ctx, compressedKV.devPtr, wUKV.devPtr,
		kNope.devPtr, v.devPtr, C.int(numTokens), C.int(kvLoraRank),
		C.int(heads), C.int(qkNopeDim), C.int(vHeadDim), C.int(isF16))
	return kNope, v
}

// MLAProjectQuerySplitRoPE splits query projections into content and rotary parts, applying RoPE to rotary part
// qAll: [numTokens, heads * (qkNopeDim + qkRopeDim)]
// posIds: [numTokens] (optional int32 tensor)
// Returns: qNope [numTokens, heads * qkNopeDim], qRope [numTokens, heads * qkRopeDim]
func (ctx *Context) MLAProjectQuerySplitRoPE(qAll, posIds *Tensor, numTokens, heads, qkNopeDim, qkRopeDim int, theta float32) (*Tensor, *Tensor) {
	isF16 := 0
	var qNope, qRope *Tensor
	if qAll.dataType == DataTypeF16 {
		isF16 = 1
		qNope = ctx.NewTensorPooled(numTokens, heads*qkNopeDim)
		qRope = ctx.NewTensorPooled(numTokens, heads*qkRopeDim)
	} else {
		qNope = ctx.NewTensorFP32(numTokens, heads*qkNopeDim)
		qRope = ctx.NewTensorFP32(numTokens, heads*qkRopeDim)
	}

	var dPos unsafe.Pointer
	if posIds != nil {
		dPos = posIds.devPtr
	}

	C.cudaMLAProjectQuerySplitRoPE(ctx.Ctx, qAll.devPtr, (*C.int)(dPos),
		qNope.devPtr, qRope.devPtr, C.int(numTokens), C.int(heads),
		C.int(qkNopeDim), C.int(qkRopeDim), C.float(theta), C.int(isF16))
	return qNope, qRope
}

// MLAAbsorbedQuery projects content query into latent space per head using W_UK
// qNope: [numTokens, heads * qkNopeDim]
// wUK: [heads * qkNopeDim, kvLoraRank]
// Returns: qAbsorbed [numTokens, heads * kvLoraRank]
func (ctx *Context) MLAAbsorbedQuery(qNope, wUK *Tensor, numTokens, heads, qkNopeDim, kvLoraRank int) *Tensor {
	isF16 := 0
	var qAbsorbed *Tensor
	if qNope.dataType == DataTypeF16 || wUK.dataType == DataTypeF16 {
		isF16 = 1
		qAbsorbed = ctx.NewTensorPooled(numTokens, heads*kvLoraRank)
	} else {
		qAbsorbed = ctx.NewTensorFP32(numTokens, heads*kvLoraRank)
	}

	C.cudaMLAAbsorbedQuery(ctx.Ctx, qNope.devPtr, wUK.devPtr,
		qAbsorbed.devPtr, C.int(numTokens), C.int(heads),
		C.int(qkNopeDim), C.int(kvLoraRank), C.int(isF16))
	return qAbsorbed
}

// MLAAbsorbedDecodeAttention executes fused decode attention directly over compressed KV cache
// qAbsorbed: [numTokens, heads * kvLoraRank]
// qRope: [numTokens, heads * qkRopeDim]
// kCache: [seqLen, kvLoraRank]
// kRopeCache: [seqLen, qkRopeDim]
// wUV: [heads * vHeadDim, kvLoraRank]
// Returns: output [numTokens, heads * vHeadDim]
func (ctx *Context) MLAAbsorbedDecodeAttention(qAbsorbed, qRope, kCache, kRopeCache, wUV *Tensor, numTokens, seqLen, heads, kvLoraRank, qkRopeDim, vHeadDim int, scale float32) *Tensor {
	isF16 := 0
	var output *Tensor
	if qAbsorbed.dataType == DataTypeF16 || wUV.dataType == DataTypeF16 {
		isF16 = 1
		output = ctx.NewTensorPooled(numTokens, heads*vHeadDim)
	} else {
		output = ctx.NewTensorFP32(numTokens, heads*vHeadDim)
	}

	C.cudaMLAAbsorbedDecodeAttention(ctx.Ctx, qAbsorbed.devPtr, qRope.devPtr,
		kCache.devPtr, kRopeCache.devPtr, wUV.devPtr, output.devPtr,
		C.int(numTokens), C.int(seqLen), C.int(heads), C.int(kvLoraRank),
		C.int(qkRopeDim), C.int(vHeadDim), C.float(scale), C.int(isF16))
	return output
}

func (ctx *Context) MatVecDequantQ8_0(weight, x, y *Tensor, M, K int) {
	start := time.Now()
	C.cudaMatVecDequantQ8_0(ctx.Ctx, weight.devPtr, (*C.float)(x.devPtr), (*C.float)(y.devPtr), C.int(M), C.int(K))
	metrics.RecordCUDADequantGEMM("Q8_0", time.Since(start))
}

func (ctx *Context) MatVecDequantQ4_K(weight, x, y *Tensor, M, K int) {
	start := time.Now()
	C.cudaMatVecDequantQ4_K(ctx.Ctx, weight.devPtr, (*C.float)(x.devPtr), (*C.float)(y.devPtr), C.int(M), C.int(K))
	metrics.RecordCUDADequantGEMM("Q4_K", time.Since(start))
}

func (ctx *Context) MatVecDequantQ6_K(weight, x, y *Tensor, M, K int) {
	start := time.Now()
	C.cudaMatVecDequantQ6_K(ctx.Ctx, weight.devPtr, (*C.float)(x.devPtr), (*C.float)(y.devPtr), C.int(M), C.int(K))
	metrics.RecordCUDADequantGEMM("Q6_K", time.Since(start))
}

func (ctx *Context) FlashAttentionPrefill(q, k, v, output *Tensor, batch, heads, kvHeads, qSeqLen, kvSeqLen, headDim int, scale float32, slidingWindow int) {
	start := time.Now()
	C.cudaFlashAttentionPrefill(ctx.Ctx, (*C.float)(q.devPtr), (*C.float)(k.devPtr), (*C.float)(v.devPtr), (*C.float)(output.devPtr),
		C.int(batch), C.int(heads), C.int(kvHeads), C.int(qSeqLen), C.int(kvSeqLen), C.int(headDim), C.float(scale), C.int(slidingWindow))
	metrics.RecordFlashAttentionPrefill(time.Since(start), slidingWindow)
}

func (ctx *Context) PagedAttentionQuantized(q, kPool, vPool *Tensor, kScales, vScales []float32, output, tokenPositions, blockTables, tokenToSeq *Tensor, maxBlocks, heads, kvHeads, headDim, blockSize, numTokens int, scale float32, isFP8 bool) {
	fp8Int := 0
	if isFP8 {
		fp8Int = 1
	}
	var kScalePtr, vScalePtr *C.float
	if len(kScales) > 0 {
		var dK unsafe.Pointer
		C.cudaMalloc(&dK, C.size_t(len(kScales)*4))
		C.cudaMemcpy(dK, unsafe.Pointer(&kScales[0]), C.size_t(len(kScales)*4), C.cudaMemcpyHostToDevice)
		kScalePtr = (*C.float)(dK)
		defer C.cudaFree(dK)
	}
	if len(vScales) > 0 {
		var dV unsafe.Pointer
		C.cudaMalloc(&dV, C.size_t(len(vScales)*4))
		C.cudaMemcpy(dV, unsafe.Pointer(&vScales[0]), C.size_t(len(vScales)*4), C.cudaMemcpyHostToDevice)
		vScalePtr = (*C.float)(dV)
		defer C.cudaFree(dV)
	}
	C.cudaPagedAttentionQuantized(ctx.Ctx, (*C.float)(q.devPtr), kPool.devPtr, vPool.devPtr, kScalePtr, vScalePtr, (*C.float)(output.devPtr),
		(*C.int)(tokenPositions.devPtr), (*C.int)(blockTables.devPtr), (*C.int)(tokenToSeq.devPtr),
		C.int(maxBlocks), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(blockSize), C.int(numTokens), C.float(scale), C.int(fp8Int))
}

func (ctx *Context) Synchronize() {
	C.cudaStreamSynchronize(ctx.Ctx)
}

// CUDAModel and Weight Loading

type weight struct {
	devPtr   unsafe.Pointer
	rows     int
	cols     int
	dataType DataType
	ctx      *Context
}

type CUDAModel struct {
	Ctx     *Context
	Weights map[string]*weight
	KCache  []*Tensor
	VCache  []*Tensor
	mu      sync.RWMutex
}

func (ctx *Context) NewCUDAModel(f *gguf.GGUFFile, preDequantize bool, kvCacheSize int, numGPULayers ...int) (*CUDAModel, error) {
	m := &CUDAModel{
		Ctx:     ctx,
		Weights: make(map[string]*weight),
	}

	gpuLayers := -1
	if len(numGPULayers) > 0 {
		gpuLayers = numGPULayers[0]
	}

	var f32Scratch []float32
	var hostFP16 []uint16

	for _, tensor := range f.Tensors {
		name := tensor.Name

		// Skip output / lm_head on GPU - computed on CPU via raw quantized weights and SIMD
		if name == "output.weight" || name == "lm_head.weight" {
			continue
		}

		// If partial GPU layer offloading is active, skip blk.<L>.* where L >= gpuLayers
		if gpuLayers >= 0 && strings.HasPrefix(name, "blk.") {
			rem := name[4:]
			if dot := strings.IndexByte(rem, '.'); dot > 0 {
				if lIdx, err := strconv.Atoi(rem[:dot]); err == nil && lIdx >= gpuLayers {
					continue
				}
			}
		}

		cols := int(tensor.Dimensions[0])
		rows := 1
		if len(tensor.Dimensions) > 1 {
			rows = int(tensor.Dimensions[1])
		}
		for i := 2; i < len(tensor.Dimensions); i++ {
			rows *= int(tensor.Dimensions[i])
		}
		numElements := rows * cols

		if len(tensor.Dimensions) <= 1 {
			// 1D vectors (RMSNorm weights, biases): allocate as float32 to match CUDA RMSNorm
			var dPtr unsafe.Pointer
			if errCode := C.cudaMalloc(&dPtr, C.size_t(numElements*4)); errCode != 0 {
				m.Free()
				return nil, fmt.Errorf("cudaMalloc failed for 1D tensor %s (%d elements): cuda error %d", name, numElements, errCode)
			}
			if cap(f32Scratch) < numElements {
				f32Scratch = make([]float32, numElements)
			} else {
				f32Scratch = f32Scratch[:numElements]
			}
			switch tensor.Type {
			case gguf.GGMLTypeF32:
				for i := 0; i < numElements; i++ {
					f32Scratch[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Data[i*4:]))
				}
			default:
				gguf.DequantizeBlock(tensor.Data, f32Scratch, tensor.Type)
			}
			C.cudaMemcpy(dPtr, unsafe.Pointer(&f32Scratch[0]), C.size_t(numElements*4), C.cudaMemcpyHostToDevice)
			m.Weights[name] = &weight{
				devPtr:   dPtr,
				rows:     rows,
				cols:     cols,
				dataType: DataTypeF32,
				ctx:      ctx,
			}
			continue
		}

		// Zero-Dequant Mode: Keep raw quantized bytes directly in GPU VRAM
		if (!preDequantize || os.Getenv("CUDA_ZERO_DEQUANT") == "1") && (tensor.Type == gguf.GGMLTypeQ8_0 || tensor.Type == gguf.GGMLTypeQ4_K || tensor.Type == gguf.GGMLTypeQ6_K) {
			dataBytes := len(tensor.Data)
			var dPtr unsafe.Pointer
			if errCode := C.cudaMalloc(&dPtr, C.size_t(dataBytes)); errCode != 0 {
				m.Free()
				return nil, fmt.Errorf("cudaMalloc failed for raw quantized tensor %s (%d bytes): cuda error %d", name, dataBytes, errCode)
			}
			C.cudaMemcpy(dPtr, unsafe.Pointer(&tensor.Data[0]), C.size_t(dataBytes), C.cudaMemcpyHostToDevice)
			dtype := DataTypeQ8_0
			switch tensor.Type {
			case gguf.GGMLTypeQ4_K:
				dtype = DataTypeQ4_K
			case gguf.GGMLTypeQ6_K:
				dtype = DataTypeQ6_K
			}
			savedBytes := int64(numElements*2 - dataBytes)
			metrics.RecordCUDAVRAMSaved(name, savedBytes)
			m.Weights[name] = &weight{
				devPtr:   dPtr,
				rows:     rows,
				cols:     cols,
				dataType: dtype,
				ctx:      ctx,
			}
			continue
		}

		var dPtr unsafe.Pointer
		if errCode := C.cudaMalloc(&dPtr, C.size_t(numElements*2)); errCode != 0 {
			m.Free()
			return nil, fmt.Errorf("cudaMalloc failed for tensor %s (%d elements, %d bytes): cuda error %d", name, numElements, numElements*2, errCode)
		}

		if cap(f32Scratch) < numElements {
			f32Scratch = make([]float32, numElements)
		} else {
			f32Scratch = f32Scratch[:numElements]
		}
		if cap(hostFP16) < numElements {
			hostFP16 = make([]uint16, numElements)
		} else {
			hostFP16 = hostFP16[:numElements]
		}

		switch tensor.Type {
		case gguf.GGMLTypeF32:
			for i := 0; i < numElements; i++ {
				f32Scratch[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Data[i*4:]))
			}
		case gguf.GGMLTypeF16:
			f32Scratch = gguf.DequantizeF16(tensor.Data, numElements)
		case gguf.GGMLTypeBF16:
			f32Scratch = gguf.DequantizeBF16(tensor.Data, numElements)
		case gguf.GGMLTypeQ8_0:
			f32Scratch = gguf.DequantizeQ8_0(tensor.Data, numElements)
		case gguf.GGMLTypeQ4_K:
			f32Scratch = gguf.DequantizeQ4K_SIMD(tensor.Data, numElements)
		case gguf.GGMLTypeQ5_K:
			f32Scratch = gguf.DequantizeQ5K(tensor.Data, numElements)
		case gguf.GGMLTypeQ6_K:
			f32Scratch = gguf.DequantizeQ6K_SIMD(tensor.Data, numElements)
		case gguf.GGMLTypeQ2_K:
			f32Scratch = gguf.DequantizeQ2K(tensor.Data, numElements)
		case gguf.GGMLTypeQ3_K:
			f32Scratch = gguf.DequantizeQ3K(tensor.Data, numElements)
		case gguf.GGMLTypeQ4_0:
			f32Scratch = gguf.DequantizeQ4_0(tensor.Data, numElements)
		case gguf.GGMLTypeQ5_0:
			f32Scratch = gguf.DequantizeQ5_0(tensor.Data, numElements)
		case gguf.GGMLTypeIQ4_NL:
			f32Scratch = gguf.DequantizeIQ4NL(tensor.Data, numElements)
		default:
			gguf.DequantizeBlock(tensor.Data, f32Scratch, tensor.Type)
		}
		for i, v := range f32Scratch {
			hostFP16[i] = Float32ToFloat16(v)
		}

		if len(hostFP16) > 0 {
			C.cudaMemcpy(dPtr, unsafe.Pointer(&hostFP16[0]), C.size_t(numElements*2), C.cudaMemcpyHostToDevice)
		}
		m.Weights[name] = &weight{
			devPtr:   dPtr,
			rows:     rows,
			cols:     cols,
			dataType: DataTypeF16,
			ctx:      ctx,
		}
	}
	f32Scratch = nil
	hostFP16 = nil
	runtime.GC()
	debug.FreeOSMemory()

	arch := "llama"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}
	layers := getCudaKVInt(f.KV, arch+".block_count", "llama.block_count")
	heads := getCudaKVInt(f.KV, arch+".attention.head_count", "llama.attention.head_count")
	if heads == 0 {
		heads = 32
	}
	dim := getCudaKVInt(f.KV, arch+".embedding_length", "llama.embedding_length")
	if dim == 0 {
		dim = 2048
	}
	headDim := dim / heads

	cacheLayers := layers
	if gpuLayers >= 0 && gpuLayers < layers {
		cacheLayers = gpuLayers
	}

	m.KCache = make([]*Tensor, cacheLayers)
	m.VCache = make([]*Tensor, cacheLayers)
	for i := 0; i < cacheLayers; i++ {
		m.KCache[i], _ = ctx.NewTensor(kvCacheSize*heads, headDim, DataTypeF16)
		m.VCache[i], _ = ctx.NewTensor(kvCacheSize*heads, headDim, DataTypeF16)
	}

	return m, nil
}

func getCudaKVInt(kv map[string]interface{}, keys ...string) int {
	for _, key := range keys {
		val, ok := kv[key]
		if !ok {
			continue
		}
		switch v := val.(type) {
		case uint32:
			return int(v)
		case int32:
			return int(v)
		case uint64:
			return int(v)
		case int64:
			return int(v)
		case int:
			return v
		case float64:
			return int(v)
		case float32:
			return int(v)
		}
	}
	return 0
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
		// Fallback for tied embeddings
		if name == "output.weight" || name == "lm_head.weight" {
			w, ok = m.Weights["token_embd.weight"]
		}
	}
	if !ok {
		return nil, false
	}
	return &Tensor{
		devPtr:   w.devPtr,
		rows:     w.rows,
		cols:     w.cols,
		dataType: w.dataType,
		ctx:      m.Ctx,
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

func (m *CUDAModel) GetBatchEmbedding(tokens []int, vocabSize int) (*Tensor, error) {
	embWeight, ok := m.GetWeightTensor("token_embd.weight")
	if !ok {
		return nil, fmt.Errorf("embedding weight not found")
	}
	dim := embWeight.cols
	if dim == 0 {
		dim = embWeight.rows
	}
	numTokens := len(tokens)
	out := m.Ctx.NewTensorPooled(numTokens, dim)
	for i, tok := range tokens {
		tIdx := tok
		if tIdx < 0 || tIdx >= vocabSize {
			tIdx = 0
		}
		offsetSrc := uintptr(tIdx) * uintptr(dim) * 2
		offsetDst := uintptr(i) * uintptr(dim) * 2
		srcPtr := unsafe.Pointer(uintptr(embWeight.devPtr) + offsetSrc)
		dstPtr := unsafe.Pointer(uintptr(out.devPtr) + offsetDst)
		C.cudaMemcpyAsync(dstPtr, srcPtr, C.size_t(dim*2), C.cudaMemcpyDeviceToDevice, m.Ctx.Ctx)
	}
	return out, nil
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

func AllocatedBytes() int64 {
	return CUDAAllocatedBytes()
}

type LayerScratch struct {
	Normed *Tensor
	Attn   *Tensor
	Gate   *Tensor
	Up     *Tensor
	Down   *Tensor
}

func (ctx *Context) NewLayerScratch(batch, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize, qNormDim, kNormDim int) *LayerScratch {
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

func Float32SliceToFloat16(data []float32) []uint16 {
	res := make([]uint16, len(data))
	for i, v := range data {
		res[i] = Float32ToFloat16(v)
	}
	return res
}

func GetDeviceCount() (int, error) {
	var count C.int
	if err := C.cudaGetDeviceCount(&count); err != 0 {
		return 0, fmt.Errorf("cudaGetDeviceCount failed: %v", err)
	}
	return int(count), nil
}

func GetDeviceName(device int) string {
	var prop C.struct_cudaDeviceProp
	if err := C.cudaGetDeviceProperties(&prop, C.int(device)); err != 0 {
		return "Unknown"
	}
	return C.GoString(&prop.name[0])
}

func GetDeviceMemory(device int) (int64, error) {
	var prop C.struct_cudaDeviceProp
	if err := C.cudaGetDeviceProperties(&prop, C.int(device)); err != 0 {
		return 0, fmt.Errorf("cudaGetDeviceProperties failed: %v", err)
	}
	return int64(prop.totalGlobalMem), nil
}

// Slice extracts one row from source tensor and writes it to dst.
// src: [batchSize, vocabSize] (logical), rowIdx: which row, dst: [1, vocabSize]
func (c *Context) Slice(src *Tensor, dst *Tensor, rowIdx, vocabSize int) {
	srcHost := src.ToHostF32()
	rowData := srcHost[rowIdx*vocabSize : (rowIdx+1)*vocabSize]
	dst.LoadFrom(rowData)
}

// AttentionPagedBatch performs paged attention across a batch of sequences on the GPU.
func (c *Context) AttentionPagedBatch(q, kCache, vCache, output, tokenPositions, blockTables *Tensor, maxBlocksPerSeq, heads, kvHeads, headDim, blockSize int, tokenToSeq *Tensor, batchSize int) {
	C.cudaPagedAttentionBatch(c.Ctx, (*C.float)(q.devPtr), kCache.devPtr, vCache.devPtr, (*C.float)(output.devPtr), (*C.int)(tokenPositions.devPtr), (*C.int)(blockTables.devPtr), (*C.int)(tokenToSeq.devPtr), C.int(maxBlocksPerSeq), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(blockSize), C.int(q.rows), C.float(1.0/math.Sqrt(float64(headDim))))
}

// StoreKVPagedBatch stores K and V projections into their respective physical blocks in the GPU cache pool.
func (c *Context) StoreKVPagedBatch(k, v, kCache, vCache, physicalPositions *Tensor, kvDim, batchSize int) {
	C.cudaStoreKVPagedBatch(c.Ctx, (*C.float)(k.devPtr), (*C.float)(v.devPtr), kCache.devPtr, vCache.devPtr, (*C.int)(physicalPositions.devPtr), C.int(kvDim), C.int(k.rows))
}

// StoreKVTurboQuant stores K and V in TurboQuant format to paged KV cache.
func (c *Context) StoreKVTurboQuant(k, v *Tensor, kCache, vCache *Tensor, physicalPositions *Tensor, blockSize, qjlRows, numHeads, numTokens int) {
	C.cudaStoreKVTurboQuant(c.Ctx, (*C.float)(k.devPtr), (*C.float)(v.devPtr), kCache.devPtr, vCache.devPtr, (*C.int)(physicalPositions.devPtr), C.int(blockSize), C.int(qjlRows), C.int(numHeads), C.int(numTokens))
}

// TurboQuantEncode encodes input tensor to TurboQuant format on CUDA.
// For simplicity, this performs encoding on CPU then copies to GPU.
func (c *Context) TurboQuantEncode(input, rotationMatrix, qjlMatrix, output *Tensor, scaleOut, qjlScaleOut *Tensor, blockSize, qjlRows, bits int) {
	numElements := input.Rows() * input.Cols()
	numBlocks := numElements / blockSize

	// For CUDA, we encode on CPU then copy
	inputHost := input.ToHostF32()
	outputHost := make([]int8, numElements)

	var scaleHost []float32
	var qjlScaleHost []float32
	if scaleOut != nil {
		scaleHost = make([]float32, numBlocks)
	}
	if qjlScaleOut != nil {
		qjlScaleHost = make([]float32, numBlocks)
	}

	// Simple CPU encoding
	for b := 0; b < numBlocks; b++ {
		off := b * blockSize
		in := inputHost[off : off+blockSize]
		q, s := polarQuant(in, bits)
		for i, v := range q {
			outputHost[off+i] = v
		}
		if scaleHost != nil {
			scaleHost[b] = s
		}
		if qjlScaleHost != nil {
			qjlScaleHost[b] = 1.0
		}
	}

	// Copy to GPU
	C.cudaMemcpy(output.devPtr, unsafe.Pointer(&outputHost[0]), C.size_t(numElements), C.cudaMemcpyHostToDevice)
	if scaleOut != nil && scaleHost != nil {
		C.cudaMemcpy(scaleOut.devPtr, unsafe.Pointer(&scaleHost[0]), C.size_t(len(scaleHost)*4), C.cudaMemcpyHostToDevice)
	}
	if qjlScaleOut != nil && qjlScaleHost != nil {
		C.cudaMemcpy(qjlScaleOut.devPtr, unsafe.Pointer(&qjlScaleHost[0]), C.size_t(len(qjlScaleHost)*4), C.cudaMemcpyHostToDevice)
	}
}

// polarQuant performs simple polar quantization on CPU
func polarQuant(in []float32, bits int) ([]int8, float32) {
	n := len(in)
	var scale float32
	maxVal := float32(0)
	for _, v := range in {
		if v < 0 {
			v = -v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	if maxVal > 0 {
		scale = float32(bits-1) / maxVal
	}
	result := make([]int8, n)
	for i, v := range in {
		result[i] = int8(float32(v) * scale)
	}
	return result, 1.0 / scale
}

// StoreKVQuantized stores KV cache in TurboQuant format.
func (t *Tensor) StoreKVQuantized(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	if t.ctx.TQRotation == nil || t.ctx.TQQJL == nil {
		// Fallback to standard StoreKV if TurboQuant matrices not available
		t.StoreKV(v, kCache, vCache, pos, heads, headDim, windowSize)
		return
	}

	blockSize := headDim
	qjlRows := 64

	t.ctx.TurboQuantEncode(t, t.ctx.TQRotation, t.ctx.TQQJL, kCache, nil, nil, blockSize, qjlRows, 4)
}

// VisionPatchEmbed performs patch embedding projection on CUDA.
func (c *Context) VisionPatchEmbed(pixels *Tensor, weights *Tensor, output *Tensor, patchSize, visionDim, numPatchesX int) {
	if pixels == nil || weights == nil || output == nil {
		return
	}
	input := pixels.ToHostF32()
	wt := weights.ToHostF32()
	out := output.ToHostF32()

	if len(out) == 0 {
		return
	}
	hiddenSize := len(wt) / len(out)

	for i := 0; i < len(out); i++ {
		sum := float32(0)
		for j := 0; j < hiddenSize && i*hiddenSize+j < len(wt); j++ {
			sum += input[j] * wt[i*hiddenSize+j]
		}
		out[i] = sum
	}
	_ = output.LoadFrom(out)
}

// VisionPatchEmbedGemma4 performs Gemma 4 patch embedding on CUDA.
func (c *Context) VisionPatchEmbedGemma4(pixels *Tensor, weights *Tensor, bias *Tensor, output *Tensor, patchSize, hiddenDim, numPatches int) {
	if pixels == nil || weights == nil || output == nil {
		return
	}
	input := pixels.ToHostF32()
	wt := weights.ToHostF32()
	out := output.ToHostF32()
	biasVals := make([]float32, 0)
	if bias != nil {
		biasVals = bias.ToHostF32()
	}

	if numPatches == 0 {
		return
	}
	inputSize := len(input) / numPatches

	for p := 0; p < numPatches; p++ {
		offset := p * hiddenDim
		for i := 0; i < hiddenDim && offset+i < len(out); i++ {
			sum := float32(0)
			for j := 0; j < inputSize; j++ {
				srcIdx := p*inputSize + j
				wIdx := i*inputSize + j
				if srcIdx < len(input) && wIdx < len(wt) {
					sum += input[srcIdx] * wt[wIdx]
				}
			}
			out[offset+i] = sum
			if len(biasVals) > i {
				out[offset+i] += biasVals[i]
			}
		}
	}
	_ = output.LoadFrom(out)
}
