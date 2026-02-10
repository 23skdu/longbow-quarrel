//go:build linux && cuda

package device

/*
#cgo LDFLAGS: -L${SRCDIR} -lcuda_kernels -lcublas -lcudnn -lcuda -L/usr/local/cuda/lib64
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

extern void cudaDequantQ8_0(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ8_0ToBF16(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ4_K(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ4_KToBF16(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ6_K(cudaStream_t stream, void* src, void* dst, int numElements);
extern void cudaDequantQ6_KToBF16(cudaStream_t stream, void* src, void* dst, int numElements);

// Fused kernel exports
extern void cudaFusedAttention(cudaStream_t stream, const void* q, const void* k, const void* v, void* output, const void* kCache, const void* vCache, int batch, int heads, int seqLen, int kvSeqLen, int headDim, float scale, int useCache);
extern void cudaFlashFusedAttention(cudaStream_t stream, const void* q, const void* k, const void* v, void* output, int batch, int heads, int seqLen, int kvSeqLen, int headDim, float scale);
extern void cudaFusedRoPE(cudaStream_t stream, void* tensor, const int* posIds, int batch, int heads, int seqLen, int headDim, float theta);
extern void cudaFusedSwiGLU(cudaStream_t stream, const void* input, const void* gateWeight, const void* upWeight, const void* downWeight, void* output, int batch, int dim, int hiddenDim);
extern void cudaFusedMLP(cudaStream_t stream, const void* input, const void* gateWeight, const void* upWeight, const void* downWeight, void* output, int batch, int dim, int hiddenDim);
extern void cudaFusedRMSNormAdd(cudaStream_t stream, const void* input, const void* hidden, const void* weight, void* output, int batch, int dim, float eps);
*/
import "C"
import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

var cudaAllocatedBytes int64

func cudaTraceAlloc(delta int64) {
	newVal := atomic.AddInt64(&cudaAllocatedBytes, delta)
	metrics.RecordGPUMemory(newVal)
}

func CUDAAllocatedBytes() int64 {
	return atomic.LoadInt64(&cudaAllocatedBytes)
}

var MaxCUDAMemory int64 = 8 * 1024 * 1024 * 1024

type CUDAContext struct {
	device        int
	stream        C.cudaStream_t
	handle        C.cublasHandle_t
	mu            sync.Mutex
	pool          map[string][]*CUDATensor
	useTensorCore bool
}

var globalCUDAContext *CUDAContext

func NewCUDAContext() (*CUDAContext, error) {
	if globalCUDAContext != nil {
		return globalCUDAContext, nil
	}

	ctx := &CUDAContext{
		device:        0,
		stream:        nil,
		handle:        nil,
		pool:          make(map[string][]*CUDATensor),
		useTensorCore: true,
	}

	result := C.cudaSetDevice(C.int(ctx.device))
	if result != C.cudaSuccess {
		return nil, fmt.Errorf("cudaSetDevice failed: %v", result)
	}

	var cuDevice C.int
	C.cudaGetDevice(&cuDevice)
	ctx.device = int(cuDevice)

	C.cudaStreamCreate(&ctx.stream)
	if ctx.stream == nil {
		return nil, fmt.Errorf("cudaStreamCreate failed")
	}

	status := C.cublasCreate(&ctx.handle)
	if status != 0 {
		C.cudaStreamDestroy(ctx.stream)
		return nil, fmt.Errorf("cublasCreate failed with status: %d", status)
	}

	globalCUDAContext = ctx

	C.cublasSetStream(ctx.handle, ctx.stream)

	var version C.int
	C.cudaDriverGetVersion(&version)
	fmt.Printf("CUDA Driver Version: %d.%d\n", version/1000, (version%100)/10)

	var runtimeVersion C.int
	C.cudaRuntimeGetVersion(&runtimeVersion)
	fmt.Printf("CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10)

	var memInfo C.size_t
	C.cudaMemGetInfo(nil, &memInfo)
	fmt.Printf("GPU Memory: %.1f MB available\n", float64(memInfo)/1e6)

	runtime.SetFinalizer(ctx, func(c *CUDAContext) {
		c.Free()
	})

	return ctx, nil
}

func (c *CUDAContext) Free() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.handle != nil {
		C.cublasDestroy(c.handle)
		c.handle = nil
	}
	if c.stream != nil {
		C.cudaStreamDestroy(c.stream)
		c.stream = nil
	}

	for _, tensors := range c.pool {
		for _, t := range tensors {
			if t.devPtr != nil {
				C.cudaFree(t.devPtr)
				t.devPtr = nil
			}
		}
	}
	c.pool = make(map[string][]*CUDATensor)
}

func (c *CUDAContext) Synchronize() {
	C.cudaStreamSynchronize(c.stream)
}

type CUDADataType int

const (
	DataTypeInvalid CUDADataType = iota
	DataTypeF16
	DataTypeF32
	DataTypeQ4_0
	DataTypeQ4_K
	DataTypeQ6_K
	DataTypeQ8_0
)

type CUDATensor struct {
	ctx        *CUDAContext
	rows, cols int
	sizeBytes  int
	devPtr     unsafe.Pointer
	dataType   CUDADataType
	ggmlType   gguf.GGMLType
	HostData   []byte
}

func (t *CUDATensor) Rows() int { return t.rows }
func (t *CUDATensor) Cols() int { return t.cols }

func (t *CUDATensor) Free() {
	if t.devPtr != nil && t.ctx != nil {
		C.cudaFree(t.devPtr)
		t.devPtr = nil
	}
}

func (c *CUDAContext) NewTensor(rows, cols int, dt CUDADataType) (*CUDATensor, error) {
	elementSize := 2
	switch dt {
	case DataTypeF16:
		elementSize = 2
	case DataTypeF32:
		elementSize = 4
	case DataTypeQ4_0, DataTypeQ4_K, DataTypeQ6_K, DataTypeQ8_0:
		elementSize = 1
	}

	size := rows * cols * elementSize
	var devPtr unsafe.Pointer
	result := C.cudaMalloc(&devPtr, C.size_t(size))
	if result != C.cudaSuccess {
		return nil, fmt.Errorf("cudaMalloc failed: %v", result)
	}

	cudaTraceAlloc(int64(size))

	t := &CUDATensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: size,
		devPtr:    devPtr,
		dataType:  dt,
	}

	runtime.SetFinalizer(t, func(t *CUDATensor) {
		t.Free()
	})

	return t, nil
}

func (c *CUDAContext) NewTensorFP32(rows, cols int) (*CUDATensor, error) {
	elementSize := 4
	size := rows * cols * elementSize
	var devPtr unsafe.Pointer
	result := C.cudaMalloc(&devPtr, C.size_t(size))
	if result != C.cudaSuccess {
		return nil, fmt.Errorf("cudaMalloc failed: %v", result)
	}

	cudaTraceAlloc(int64(size))

	t := &CUDATensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: size,
		devPtr:    devPtr,
		dataType:  DataTypeF32,
	}

	runtime.SetFinalizer(t, func(t *CUDATensor) {
		t.Free()
	})

	return t, nil
}

func (c *CUDAContext) NewTensorFromData(rows, cols int, dt CUDADataType, data []byte) (*CUDATensor, error) {
	t, err := c.NewTensor(rows, cols, dt)
	if err != nil {
		return nil, err
	}
	t.HostData = data

	if len(data) > 0 {
		C.cudaMemcpyAsync(t.devPtr, unsafe.Pointer(&data[0]), C.size_t(len(data)), C.cudaMemcpyHostToDevice, c.stream)
	}

	return t, nil
}

func (c *CUDAContext) NewTensorPooled(rows, cols int) *CUDATensor {
	key := fmt.Sprintf("%dx%d", rows, cols)
	if tensors, ok := c.pool[key]; ok && len(tensors) > 0 {
		t := tensors[len(tensors)-1]
		c.pool[key] = tensors[:len(tensors)-1]
		return t
	}

	t, _ := c.NewTensor(rows, cols, DataTypeF16)
	return t
}

func (t *CUDATensor) ReturnToPool() {
	if t.devPtr == nil {
		return
	}
	key := fmt.Sprintf("%dx%d", t.rows, t.cols)
	t.ctx.mu.Lock()
	defer t.ctx.mu.Unlock()
	t.ctx.pool[key] = append(t.ctx.pool[key], t)
}

func (t *CUDATensor) ToHostF32() []float32 {
	result := make([]float32, t.rows*t.cols)
	C.cudaMemcpyAsync(unsafe.Pointer(&result[0]), t.devPtr, C.size_t(t.rows*t.cols*4), C.cudaMemcpyDeviceToHost, t.ctx.stream)
	t.ctx.Synchronize()
	return result
}

func (t *CUDATensor) LoadFrom(data []float32) error {
	size := len(data) * 4
	C.cudaMemcpyAsync(t.devPtr, unsafe.Pointer(&data[0]), C.size_t(size), C.cudaMemcpyHostToDevice, t.ctx.stream)
	return nil
}

func (c *CUDAContext) LinearF16(input, weight *CUDATensor) (*CUDATensor, error) {
	output, err := c.NewTensor(input.rows, weight.cols, DataTypeF16)
	if err != nil {
		return nil, err
	}

	var alpha C.float = 1.0
	var beta C.float = 0.0

	status := C.cublasSgemmEx(
		c.handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		C.int(weight.cols), C.int(input.rows), C.int(weight.rows),
		&alpha,
		weight.devPtr, C.CUDA_R_16F, C.int(weight.cols),
		input.devPtr, C.CUDA_R_16F, C.int(input.cols),
		&beta,
		output.devPtr, C.CUDA_R_16F, C.int(output.cols),
	)

	if status != 0 {
		fmt.Printf("cublasSgemmEx failed with status: %d\n", status)
	}

	return output, nil
}

func (c *CUDAContext) MatmulF16(input *CUDATensor, weight *CUDATensor) (*CUDATensor, error) {
	return c.LinearF16(input, weight)
}

func (c *CUDAContext) CopyF16(src, dst *CUDATensor) {
	size := src.rows * src.cols * 2
	C.cudaMemcpyAsync(dst.devPtr, src.devPtr, C.size_t(size), C.cudaMemcpyDeviceToDevice, c.stream)
}

func (c *CUDAContext) ZeroF16(t *CUDATensor) {
	C.cudaMemsetAsync(t.devPtr, 0, C.size_t(t.sizeBytes), c.stream)
}

func (c *CUDAContext) FusedAttention(q, k, v, output, kCache, vCache *CUDATensor, batch, heads, seqLen, kvSeqLen, headDim int, scale float32, useCache int) {
	C.cudaFusedAttention(
		c.stream,
		q.devPtr, k.devPtr, v.devPtr, output.devPtr,
		kCache.devPtr, vCache.devPtr,
		C.int(batch), C.int(heads), C.int(seqLen), C.int(kvSeqLen), C.int(headDim),
		C.float(scale), C.int(useCache))
}

func (c *CUDAContext) FlashFusedAttention(q, k, v, output *CUDATensor, batch, heads, seqLen, kvSeqLen, headDim int, scale float32) {
	C.cudaFlashFusedAttention(
		c.stream,
		q.devPtr, k.devPtr, v.devPtr, output.devPtr,
		C.int(batch), C.int(heads), C.int(seqLen), C.int(kvSeqLen), C.int(headDim),
		C.float(scale))
}

func (c *CUDAContext) FusedRoPE(tensor *CUDATensor, posIds []int, batch, heads, seqLen, headDim int, theta float32) {
	C.cudaFusedRoPE(
		c.stream,
		tensor.devPtr, (*C.int)(unsafe.Pointer(&posIds[0])),
		C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim),
		C.float(theta))
}

func (c *CUDAContext) FusedSwiGLU(input, gateWeight, upWeight, downWeight, output *CUDATensor, batch, dim, hiddenDim int) {
	C.cudaFusedSwiGLU(
		c.stream,
		input.devPtr, gateWeight.devPtr, upWeight.devPtr, downWeight.devPtr, output.devPtr,
		C.int(batch), C.int(dim), C.int(hiddenDim))
}

func (c *CUDAContext) FusedMLP(input, gateWeight, upWeight, downWeight, output *CUDATensor, batch, dim, hiddenDim int) {
	C.cudaFusedMLP(
		c.stream,
		input.devPtr, gateWeight.devPtr, upWeight.devPtr, downWeight.devPtr, output.devPtr,
		C.int(batch), C.int(dim), C.int(hiddenDim))
}

func (c *CUDAContext) FusedRMSNormAdd(input, hidden, weight, output *CUDATensor, batch, dim int, eps float32) {
	C.cudaFusedRMSNormAdd(
		c.stream,
		input.devPtr, hidden.devPtr, weight.devPtr, output.devPtr,
		C.int(batch), C.int(dim), C.float(eps))
}

type LayerScratch struct {
	Q, K, V     *CUDATensor
	Attn        *CUDATensor
	Normed      *CUDATensor
	Gate, Up    *CUDATensor
	Down        *CUDATensor
	Logits      []float32
	KVAllocated bool
}

func (c *CUDAContext) NewLayerScratch(maxTokens, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize int) *LayerScratch {
	scratch := &LayerScratch{}

	scratch.Q, _ = c.NewTensor(maxTokens, dim, DataTypeF16)
	scratch.K, _ = c.NewTensor(kvHeads*seqLen, headDim, DataTypeF16)
	scratch.V, _ = c.NewTensor(kvHeads*seqLen, headDim, DataTypeF16)
	scratch.Attn, _ = c.NewTensor(maxTokens, dim, DataTypeF16)
	scratch.Normed, _ = c.NewTensor(1, dim, DataTypeF16)
	scratch.Gate, _ = c.NewTensor(1, hiddenDim, DataTypeF16)
	scratch.Up, _ = c.NewTensor(1, hiddenDim, DataTypeF16)
	scratch.Down, _ = c.NewTensor(1, dim, DataTypeF16)

	scratch.Logits = make([]float32, vocabSize)

	return scratch
}

func (s *LayerScratch) Free() {
	resources := []*CUDATensor{s.Q, s.K, s.V, s.Attn, s.Normed, s.Gate, s.Up, s.Down}
	for _, r := range resources {
		if r != nil {
			r.Free()
		}
	}
}

type CUDAWeight struct {
	Name       string
	Rows, Cols int
	GGMLType   gguf.GGMLType
	DevPtr     unsafe.Pointer
	HostData   []byte
	DataBytes  int
	Dequanted  *CUDATensor
}

type CUDAModel struct {
	Ctx        *CUDAContext
	Weights    map[string]*CUDAWeight
	NumLayers  int
	NumHeads   int
	HeadDim    int
	KCache     []*CUDATensor
	VCache     []*CUDATensor
	OutputNorm *CUDATensor
	Output     *CUDATensor
	TokenEmb   *CUDATensor
}

func (m *CUDAModel) Free() {
	for _, w := range m.Weights {
		if w.DevPtr != nil {
			C.cudaFree(w.DevPtr)
			w.DevPtr = nil
		}
		if w.Dequanted != nil {
			w.Dequanted.Free()
			w.Dequanted = nil
		}
	}

	for i := range m.KCache {
		if m.KCache[i] != nil {
			m.KCache[i].Free()
		}
	}
	for i := range m.VCache {
		if m.VCache[i] != nil {
			m.VCache[i].Free()
		}
	}
}

func (c *CUDAContext) NewCUDAModel(f *gguf.GGUFFile, kvCache bool, maxSeqLen int) (*CUDAModel, error) {
	m := &CUDAModel{
		Ctx:     c,
		Weights: make(map[string]*CUDAWeight),
	}

	arch := "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}

	m.NumLayers = 1
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		m.NumLayers = int(v)
	}

	m.NumHeads = 32
	if v, ok := f.KV["llama.attention.head_count"].(uint32); ok {
		m.NumHeads = int(v)
	}

	dim := 2048
	if v, ok := f.KV["llama.embedding_length"].(uint32); ok {
		dim = int(v)
	}

	m.HeadDim = dim / m.NumHeads

	fmt.Printf("Loading %s with %d layers, %d heads, headDim=%d\n", arch, m.NumLayers, m.NumHeads, m.HeadDim)
	fmt.Printf("Loading %d tensors from GGUF...\n", len(f.Tensors))

	for i, t := range f.Tensors {
		name := t.Name

		numElements := uint64(1)
		for _, d := range t.Dimensions {
			numElements *= d
		}

		rows := int(t.Dimensions[0])
		cols := 1
		for d := 1; d < len(t.Dimensions); d++ {
			cols *= int(t.Dimensions[d])
		}

		if rows == 0 || cols == 0 {
			continue
		}

		fmt.Printf("GGUF: Found tensor %s (Type: %v, Dims: [%d %d], Elements: %d)\n", name, t.Type, rows, cols, numElements)

		dataBytes := int(t.SizeBytes())
		var devPtr unsafe.Pointer

		if dataBytes > 0 {
			result := C.cudaMalloc(&devPtr, C.size_t(dataBytes))
			if result != C.cudaSuccess {
				fmt.Printf("cudaMalloc failed for %s: %v (trying %d bytes)\n", name, result, dataBytes)
			} else {
				srcPtr := unsafe.Pointer(uintptr(unsafe.Pointer(&f.Data[0])) + uintptr(t.Offset))
				C.cudaMemcpyAsync(devPtr, srcPtr, C.size_t(dataBytes), C.cudaMemcpyHostToDevice, c.stream)
				cudaTraceAlloc(int64(dataBytes))
			}
		}

		if dataBytes > 0 {
			result := C.cudaMalloc(&devPtr, C.size_t(dataBytes))
			if result != C.cudaSuccess {
				fmt.Printf("cudaMalloc failed for %s: %v (trying %d bytes)\n", name, result, dataBytes)
			} else {
				C.cudaMemcpyAsync(devPtr, unsafe.Pointer(&t.Data[0]), C.size_t(dataBytes), C.cudaMemcpyHostToDevice, c.stream)
				cudaTraceAlloc(int64(dataBytes))
			}
		}

		if _, exists := m.Weights[name]; exists {
			continue
		}

		m.Weights[name] = &CUDAWeight{
			Name:      name,
			Rows:      rows,
			Cols:      cols,
			GGMLType:  t.Type,
			DevPtr:    devPtr,
			HostData:  t.Data,
			DataBytes: dataBytes,
		}

		if i >= 10 && i < len(f.Tensors)-10 {
			continue
		}
	}

	c.Synchronize()

	fmt.Printf("Allocating KV cache for %d layers, %d positions\n", m.NumLayers, maxSeqLen)

	if kvCache {
		cacheSize := maxSeqLen
		m.KCache = make([]*CUDATensor, m.NumLayers)
		m.VCache = make([]*CUDATensor, m.NumLayers)

		for i := 0; i < m.NumLayers; i++ {
			k, err := c.NewTensor(m.NumHeads, cacheSize*m.HeadDim, DataTypeF16)
			if err != nil {
				fmt.Printf("Warning: failed to allocate K cache for layer %d: %v\n", i, err)
				continue
			}
			m.KCache[i] = k

			v, err := c.NewTensor(m.NumHeads, cacheSize*m.HeadDim, DataTypeF16)
			if err != nil {
				fmt.Printf("Warning: failed to allocate V cache for layer %d: %v\n", i, err)
				continue
			}
			m.VCache[i] = v
		}
	}

	return m, nil
}

func (m *CUDAModel) GetWeight(name string) (*CUDAWeight, bool) {
	w, ok := m.Weights[name]
	return w, ok
}

func (m *CUDAModel) GetDequantedWeight(name string) (*CUDATensor, error) {
	w, ok := m.Weights[name]
	if !ok {
		return nil, fmt.Errorf("weight not found: %s", name)
	}

	if w.Dequanted != nil {
		return w.Dequanted, nil
	}

	numElements := w.Rows * w.Cols
	d, err := m.Ctx.NewTensorFP32(w.Rows, w.Cols)
	if err != nil {
		return nil, err
	}

	switch w.GGMLType {
	case gguf.GGMLTypeQ8_0:
		C.cudaDequantQ8_0(m.Ctx.stream, unsafe.Pointer(&w.HostData[0]), d.devPtr, C.int(numElements))
	case gguf.GGMLTypeQ4_K:
		C.cudaDequantQ4_K(m.Ctx.stream, unsafe.Pointer(&w.HostData[0]), d.devPtr, C.int(numElements))
	case gguf.GGMLTypeQ6_K:
		C.cudaDequantQ6_K(m.Ctx.stream, unsafe.Pointer(&w.HostData[0]), d.devPtr, C.int(numElements))
	case gguf.GGMLTypeF32, gguf.GGMLTypeF16:
		d.HostData = w.HostData
	default:
		return nil, fmt.Errorf("unsupported quantization type: %v", w.GGMLType)
	}

	w.Dequanted = d
	return d, nil
}

func (m *CUDAModel) GetEmbedding(token int) ([]float32, error) {
	emb, ok := m.GetWeight("token_embd.weight")
	if !ok {
		return nil, fmt.Errorf("embedding weight not found")
	}

	rows := emb.Rows
	cols := emb.Cols
	dataLen := len(emb.HostData)

	if rows < cols && rows <= 2048 {
		rows, cols = cols, rows
	}

	dim := cols
	result := make([]float32, dim)
	offset := token * dim * 4

	if offset+dim*4 > dataLen {
		return nil, fmt.Errorf("token index %d out of range (data=%d, offset=%d, dim=%d)", token, dataLen, offset, dim)
	}

	data := emb.HostData
	for i := 0; i < dim; i++ {
		bits := uint32(data[offset+i*4]) | uint32(data[offset+i*4+1])<<8 |
			uint32(data[offset+i*4+2])<<16 | uint32(data[offset+i*4+3])<<24
		result[i] = math.Float32frombits(bits)
	}

	return result, nil
}

func (m *CUDAModel) GetKCache(layer int) *CUDATensor {
	if layer < 0 || layer >= len(m.KCache) {
		return nil
	}
	return m.KCache[layer]
}

func (m *CUDAModel) GetVCache(layer int) *CUDATensor {
	if layer < 0 || layer >= len(m.VCache) {
		return nil
	}
	return m.VCache[layer]
}
