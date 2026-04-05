//go:build linux && cuda

package device

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR} -lcuda_kernels -lcublas -lcudnn -lcudart -L/usr/lib/x86_64-linux-gnu -l:libcuda.so.1
#cgo linux,amd64 CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
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

// TurboQuant kernel exports
extern void cudaTurboQuantPolarQuant(cudaStream_t stream, const float* input, const float* rotationMatrix, int8_t* quantized, float* scaleOut, float* residual, int n, int bits);
extern void cudaTurboQuantQJLTransform(cudaStream_t stream, const float* residual, const float* signMatrix, int8_t* quantized, float* scaleOut, int rows, int cols);
extern void cudaTurboQuantEncode(cudaStream_t stream, const float* input, const float* rotationMatrix, const float* qjlMatrix, int8_t* output, float* scaleOut, float* qjlScaleOut, int blockSize, int qjlRows, int numBlocks, int bits);
extern void cudaTurboQuantDecode(cudaStream_t stream, const int8_t* input, const float* rotationMatrix, void* output, const float* scaleIn, int blockSize, int qjlRows, int numBlocks);

// Device properties structure for multi-GPU support
typedef struct {
    char name[256];
    int totalGlobalMem;
    int sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int totalConstMem;
    int major;
    int minor;
    int multiGpuBoard;
    int memoryClockRate;
    int memoryBusWidth;
} cudaDevicePropFull;
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

type CUDADataType = DataType

const (
	CUDA_R_16F CUDADataType = DataTypeF16
	CUDA_R_32F CUDADataType = DataTypeF32
	CUDA_R_64F CUDADataType = 2
	CUDA_R_8I  CUDADataType = 3
	CUDA_R_8U  CUDADataType = 4
)

type Tensor struct {
	ctx       *Context
	rows      int
	cols      int
	sizeBytes int
	cudaPtr   *CUDATensor // The underlying GPU tensor
	name      string
	dataType  DataType
}

func NewTensor(name string, data []float32) *Tensor {
	if globalCUDAContext == nil {
		panic("CUDA context not initialized before NewTensor")
	}
	rows := 1
	cols := len(data)
	ct, err := globalCUDAContext.NewTensorFP32(rows, cols)
	if err != nil {
		panic(err)
	}
	err = ct.LoadFrom(data)
	if err != nil {
		panic(err)
	}
	return &Tensor{
		ctx:       &Context{cudaCtx: globalCUDAContext},
		rows:      rows,
		cols:      cols,
		cudaPtr:   ct,
		name:      name,
		dataType:  DataTypeF32,
		sizeBytes: len(data) * 4,
	}
}

func (t *Tensor) Rows() int { return t.rows }
func (t *Tensor) Cols() int { return t.cols }
func (t *Tensor) Free() {
	if t.cudaPtr != nil {
		t.cudaPtr.Free()
		t.cudaPtr = nil
	}
}
func (t *Tensor) Data() []float32 {
	if t.cudaPtr == nil {
		return nil
	}
	return t.cudaPtr.ToHostF32()
}
func (t *Tensor) Name() string { return t.name }
func (t *Tensor) ZeroInit() {
	if t.cudaPtr != nil {
		t.ctx.cudaCtx.ZeroF16(t.cudaPtr)
	}
}
func (t *Tensor) Dims() []int    { return []int{t.rows, t.cols} }
func (t *Tensor) Strides() []int { return []int{t.cols, 1} }
func (t *Tensor) ToHost() []float32 { return t.Data() }

type Context struct {
	device  int
	cudaCtx *CUDAContext
	mu      sync.Mutex
	pool    map[string][]*Tensor

	// TurboQuant Global Matrices
	TQRotation *Tensor
	TQQJL      *Tensor
}

func (c *Context) Synchronize() {
	if c.cudaCtx != nil {
		c.cudaCtx.Synchronize()
	}
}

func (c *Context) NewTensorRaw(sizeBytes int) (*Tensor, error) {
	ct, err := c.cudaCtx.NewTensorRaw(sizeBytes)
	if err != nil {
		return nil, err
	}
	return &Tensor{
		ctx:       c,
		sizeBytes: sizeBytes,
		cudaPtr:   ct,
	}, nil
}

func (t *Tensor) LoadFromRaw(data []byte) error {
	return t.cudaPtr.LoadFromRaw(data)
}

func (c *Context) NewTensorFP32(rows, cols int) *Tensor {
	ct, err := c.cudaCtx.NewTensorFP32(rows, cols)
	if err != nil {
		panic(err)
	}
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		cudaPtr:   ct,
		dataType:  DataTypeF32,
		sizeBytes: rows * cols * 4,
	}
}

func (c *Context) NewTensorFromData(rows, cols int, dt DataType, data []byte) (*Tensor, error) {
    ct, err := c.cudaCtx.NewTensorFromData(rows, cols, dt, data)
    if err != nil {
        return nil, err
    }
    return &Tensor{
        ctx:       c,
        rows:      rows,
        cols:      cols,
        cudaPtr:   ct,
        dataType:  dt,
        sizeBytes: len(data),
    }, nil
}

func (c *Context) NewTensorPooled(rows, cols int) *Tensor {
	ct := c.cudaCtx.NewTensorPooled(rows, cols)
	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		cudaPtr:   ct,
		dataType:  DataTypeF16,
		sizeBytes: rows * cols * 2,
	}
}

func (c *Context) NewTurboTensor(rows, cols int, dt DataType, blockSize, qjlRows int) *Tensor {
	numElements := rows * cols
	numBlocks := numElements / blockSize
	if numElements%blockSize != 0 {
		numBlocks++
	}
	bytesPerBlock := blockSize + qjlRows + 8
	sizeBytes := numBlocks * bytesPerBlock

	ct, err := c.cudaCtx.NewTensorRaw(sizeBytes)
	if err != nil {
		panic(fmt.Sprintf("Failed to allocate CUDA TurboTensor: %v", err))
	}

	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: sizeBytes,
		cudaPtr:   ct,
		dataType:  dt,
	}
}

func (c *Context) NewTensorWithType(rows, cols int, dt DataType) *Tensor {
	if dt == DataTypeTQ1_0 || dt == DataTypeTQ2_0 {
		return c.NewTurboTensor(rows, cols, dt, 256, 64)
	}

	sizeBytes := rows * cols * 2
	ct, err := c.cudaCtx.NewTensor(rows, cols, C.CUDA_DTYPE_F16)
	if err != nil {
		panic(fmt.Sprintf("Failed to allocate CUDA Tensor: %v", err))
	}

	return &Tensor{
		ctx:       c,
		rows:      rows,
		cols:      cols,
		sizeBytes: sizeBytes,
		cudaPtr:   ct,
		dataType:  dt,
	}
}

func (t *Tensor) StoreKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		t.storeKVTurbo(v, kCache, vCache, pos, heads, headDim, windowSize)
		return
	}

	// Standard F16 StoreKV
	C.cudaFusedAttention(t.ctx.cudaCtx.stream,
		nil, nil, nil, nil, // Q, K, V, output not used for pure store
		kCache.cudaPtr.devPtr, vCache.cudaPtr.devPtr,
		1, C.int(heads), 1, C.int(pos+1), C.int(headDim),
		1.0, 1) // useCache=1 signals store
}

func (t *Tensor) storeKVTurbo(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	ctx := t.ctx
	if ctx.TQRotation == nil || ctx.TQQJL == nil {
		panic("TurboQuant global matrices not initialized")
	}

	blockSize := 256
	qjlRows := 64
	numBlocksPerRow := (heads * headDim) / blockSize
	bytesPerBlock := blockSize + qjlRows + 8
	rowOffsetBytes := uintptr((pos % windowSize) * numBlocksPerRow * bytesPerBlock)
    fmt.Printf("DEBUG: StoreKV pos=%d, rowOffsetBytes=%d\n", pos, rowOffsetBytes)

    C.cudaTurboQuantEncode(ctx.cudaCtx.stream,
        (*C.float)(t.cudaPtr.devPtr),
        (*C.float)(ctx.TQRotation.cudaPtr.devPtr),
        (*C.float)(ctx.TQQJL.cudaPtr.devPtr),
        (*C.int8_t)(unsafe.Add(kCache.cudaPtr.devPtr, rowOffsetBytes)),
        nil, nil,
        C.int(blockSize), C.int(qjlRows), C.int(numBlocksPerRow), 4)

	// V Encode
    C.cudaTurboQuantEncode(ctx.cudaCtx.stream,
        (*C.float)(v.cudaPtr.devPtr),
        (*C.float)(ctx.TQRotation.cudaPtr.devPtr),
        (*C.float)(ctx.TQQJL.cudaPtr.devPtr),
        (*C.int8_t)(unsafe.Add(vCache.cudaPtr.devPtr, rowOffsetBytes)),
        nil, nil,
        C.int(blockSize), C.int(qjlRows), C.int(numBlocksPerRow), 4)
}

func (t *Tensor) FetchKV(kCache, vCache *Tensor, seqLen, heads, headDim int) {
    fmt.Printf("DEBUG: FetchKV called, seqLen=%d, heads=%d, headDim=%d, dt=%v\n", seqLen, heads, headDim, kCache.dataType)
	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		t.fetchKVTurbo(kCache, vCache, seqLen, heads, headDim)
		return
	}
	// F16 Fetch (copy cache to t)
	// Implementation depends on t's layout, usually used before attention.
}

func (t *Tensor) fetchKVTurbo(kCache, vCache *Tensor, seqLen, heads, headDim int) {
	ctx := t.ctx
	blockSize := 256
	qjlRows := 64
	numBlocksPerRow := (heads * headDim) / blockSize

	// Decompress all blocks for the sequence
	totalBlocks := seqLen * numBlocksPerRow
	vOffset := seqLen * heads * headDim * 2
	
	C.cudaTurboQuantDecode(ctx.cudaCtx.stream,
		(*C.int8_t)(kCache.cudaPtr.devPtr),
		(*C.float)(ctx.TQRotation.cudaPtr.devPtr),
		t.cudaPtr.devPtr,
		nil, C.int(blockSize), C.int(qjlRows), C.int(totalBlocks))

	// V Decode
	C.cudaTurboQuantDecode(ctx.cudaCtx.stream,
		(*C.int8_t)(vCache.cudaPtr.devPtr),
		(*C.float)(ctx.TQRotation.cudaPtr.devPtr),
		unsafe.Pointer(uintptr(t.cudaPtr.devPtr)+uintptr(vOffset)),
		nil, C.int(blockSize), C.int(qjlRows), C.int(totalBlocks))
}

var cudaAllocatedBytes int64

func cudaTraceAlloc(delta int64) {
	newVal := atomic.AddInt64(&cudaAllocatedBytes, delta)
	metrics.RecordGPUMemory(newVal)
}

func CUDAAllocatedBytes() int64 {
	return atomic.LoadInt64(&cudaAllocatedBytes)
}

var MaxCUDAMemory int64 = DefaultMaxMemoryCUDA

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

func GetDeviceCount() (int, error) {
	var count C.int
	result := C.cudaGetDeviceCount(&count)
	if result != C.cudaSuccess {
		return 0, fmt.Errorf("cudaGetDeviceCount failed: %v", result)
	}
	return int(count), nil
}

func GetDeviceName(device int) string {
	return fmt.Sprintf("GPU-%d", device)
}

func GetDeviceMemory(device int) (int64, error) {
	C.cudaSetDevice(C.int(device))
	var free, total C.size_t
	result := C.cudaMemGetInfo(&free, &total)
	if result != C.cudaSuccess {
		return 0, fmt.Errorf("cudaMemGetInfo failed: %v", result)
	}
	return int64(total), nil
}

type MultiGPUManager struct {
	devices    []int
	contexts   map[int]*CUDAContext
	currentIdx int
	mu         sync.Mutex
}

type DeviceInfo struct {
	ID       int
	Name     string
	MemoryMB int64
}

var multiGPU *MultiGPUManager

func GetMultiGPUManager() (*MultiGPUManager, error) {
	if multiGPU != nil {
		return multiGPU, nil
	}

	count, err := GetDeviceCount()
	if err != nil {
		return nil, err
	}

	if count <= 1 {
		return nil, fmt.Errorf("need multiple GPUs for multi-GPU support, found %d", count)
	}

	multiGPU = &MultiGPUManager{
		devices:  make([]int, count),
		contexts: make(map[int]*CUDAContext),
	}
	for i := 0; i < count; i++ {
		multiGPU.devices[i] = i
	}

	return multiGPU, nil
}

func (m *MultiGPUManager) GetContext(device int) (*CUDAContext, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if ctx, ok := m.contexts[device]; ok {
		return ctx, nil
	}

	if device >= len(m.devices) {
		return nil, fmt.Errorf("invalid device index: %d", device)
	}

	C.cudaSetDevice(C.int(device))

	ctx := &CUDAContext{
		device:        device,
		stream:        nil,
		handle:        nil,
		pool:          make(map[string][]*CUDATensor),
		useTensorCore: true,
	}

	var cuDevice C.int
	C.cudaGetDevice(&cuDevice)
	ctx.device = int(cuDevice)

	C.cudaStreamCreate(&ctx.stream)
	if ctx.stream == nil {
		return nil, fmt.Errorf("cudaStreamCreate failed for device %d", device)
	}

	status := C.cublasCreate(&ctx.handle)
	if status != 0 {
		C.cudaStreamDestroy(ctx.stream)
		return nil, fmt.Errorf("cublasCreate failed for device %d: %d", device, status)
	}

	C.cublasSetStream(ctx.handle, ctx.stream)
	m.contexts[device] = ctx

	return ctx, nil
}

func (m *MultiGPUManager) RoundRobinDevice() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	device := m.devices[m.currentIdx]
	m.currentIdx = (m.currentIdx + 1) % len(m.devices)
	return device
}

func (m *MultiGPUManager) GetDeviceInfo(device int) (*DeviceInfo, error) {
	return &DeviceInfo{
		ID:       device,
		Name:     GetDeviceName(device),
		MemoryMB: 0,
	}, nil
}

type CUDATensor struct {
	ctx        *CUDAContext
	rows, cols int
	sizeBytes  int
	devPtr     unsafe.Pointer
	dataType   DataType
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
	return c.NewTensor(rows, cols, DataTypeF32)
}

func (c *CUDAContext) NewTensorRaw(size int) (*CUDATensor, error) {
	var devPtr unsafe.Pointer
	result := C.cudaMalloc(&devPtr, C.size_t(size))
	if result != C.cudaSuccess {
		return nil, fmt.Errorf("cudaMalloc failed: %v", result)
	}
	cudaTraceAlloc(int64(size))
	return &CUDATensor{ctx: c, sizeBytes: size, devPtr: devPtr}, nil
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

func (t *CUDATensor) ToHostF16AsF32() []float32 {
	n := t.rows * t.cols
	if n == 0 {
		return nil
	}
	hostData := make([]uint16, n)
	C.cudaMemcpy(unsafe.Pointer(&hostData[0]), t.devPtr, C.size_t(n*2), C.cudaMemcpyDeviceToHost)
	
	// Convert to float32
	result := make([]float32, n)
	for i, v := range hostData {
		result[i] = math.Float32frombits(uint32(v) << 16) // Very crude half-to-float approximation for signs
        // Better: use a proper half-to-float conversion if needed
	}
	return result
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

func (t *CUDATensor) LoadFromRaw(data []byte) error {
	C.cudaMemcpyAsync(t.devPtr, unsafe.Pointer(&data[0]), C.size_t(len(data)), C.cudaMemcpyHostToDevice, t.ctx.stream)
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

func (c *CUDAContext) NewLayerScratch(maxTokens, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize, qNormDim, kNormDim int) *LayerScratch {
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
		// Host to device copy handled elsewhere or here
		d.LoadFromRaw(w.HostData)
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

func (ctx *CUDAContext) TurboQuantPolarQuant(input, rotationMatrix []float32, n, bits int) (quantized []int8, scale float32, residual []float32) {
	quantized = make([]int8, n)
	residual = make([]float32, n)
	return quantized, 1.0, residual
}

func (ctx *CUDAContext) TurboQuantQJLTransform(residual, signMatrix []float32, rows, cols int) (quantized []int8, scale float32) {
	quantized = make([]int8, rows)
	return quantized, 1.0
}

func (ctx *CUDAContext) TurboQuantEncode(input, rotationMatrix, qjlMatrix []float32, blockSize, qjlRows, bits int) (output []int8, scale, qjlScale float32) {
	outputSize := blockSize + qjlRows
	output = make([]int8, outputSize)
	return output, 1.0, 1.0
}

func (ctx *CUDAContext) TurboQuantDecode(input []int8, rotationMatrix, scaleIn []float32, blockSize, qjlRows int) []float32 {
	output := make([]float32, blockSize)
	return output
}
