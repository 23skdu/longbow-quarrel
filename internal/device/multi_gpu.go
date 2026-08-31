//go:build linux && cuda

package device

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR} -lcuda_kernels -lcublas -lcudnn -lcudart
#cgo linux,amd64 CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}
// NCCL support - uncomment if NCCL is available
// #cgo LDFLAGS: -lnccl
// #include <nccl.h>
// NCCL stubs when library not available
static void ncclAllReduceStub(void* sendBuf, void* recvBuf, size_t count, int datatype, int op, void* stream) {
    // Stub: copy own data
    if (sendBuf != recvBuf && count > 0) {
        // memcpy would be used here
    }
}
static void ncclBroadcastStub(void* sendBuf, void* recvBuf, size_t count, int datatype, int root, void* stream) {}
static void ncclAllGatherStub(void* sendBuf, void* recvBuf, size_t count, int datatype, void* stream) {}
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Peer-to-peer memory access
typedef struct {
    int canAccessPeer;
    size_t totalMem;
    size_t freeMem;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
} GPUDeviceInfo;
*/
import "C"
import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"
)

// =============================================================================
// Multi-GPU Configuration
// =============================================================================

type ParallelismMode int

const (
	TensorParallelism          ParallelismMode = 1 << iota // Split weights across GPUs
	PipelineParallelism                                    // Split layers across GPUs
	WeightStreamingParallelism                             // Stream large weights to multiple GPUs
)

type MultiGPUConfig struct {
	Mode               ParallelismMode
	NumGPUs            int
	TensorParallelSize int
	PipelineStages     int
	BatchSizePerGPU    int
	UseNCCL            bool
	UsePipelineBubbles bool
	PipelineDepth      int
}

var defaultMultiGPUConfig = &MultiGPUConfig{
	Mode:               TensorParallelism | PipelineParallelism,
	NumGPUs:            0,
	TensorParallelSize: 1,
	PipelineStages:     1,
	BatchSizePerGPU:    1,
	UseNCCL:            true,
	UsePipelineBubbles: true,
	PipelineDepth:      4,
}

// =============================================================================
// Tensor Parallelism
// =============================================================================

type TensorParallelManager struct {
	config    *MultiGPUConfig
	devices   []int
	contexts  map[int]*Context
	ranks     []int
	localRank int
	worldSize int
	mu        sync.RWMutex
}

var tensorParallel *TensorParallelManager
var multiGPUMu sync.Mutex // protects singleton initialization and config

func NewTensorParallelManager(config *MultiGPUConfig) (*TensorParallelManager, error) {
	multiGPUMu.Lock()
	defer multiGPUMu.Unlock()
	if tensorParallel != nil {
		return tensorParallel, nil
	}

	count, err := GetDeviceCount()
	if err != nil {
		return nil, fmt.Errorf("failed to get device count: %w", err)
	}

	if count < 2 {
		return nil, fmt.Errorf("tensor parallelism requires at least 2 GPUs, found %d", count)
	}

	if config.TensorParallelSize > count {
		return nil, fmt.Errorf("tensor parallel size %d exceeds device count %d", config.TensorParallelSize, count)
	}

	tp := &TensorParallelManager{
		config:    config,
		devices:   make([]int, 0, count),
		contexts:  make(map[int]*Context),
		ranks:     make([]int, count),
		localRank: 0,
		worldSize: count,
	}

	for i := 0; i < count; i++ {
		tp.devices = append(tp.devices, i)
		tp.ranks[i] = i
	}

	tensorParallel = tp
	return tp, nil
}

func (t *TensorParallelManager) GetContext(device int) (*Context, error) {
	t.mu.RLock()
	if ctx, ok := t.contexts[device]; ok {
		t.mu.RUnlock()
		return ctx, nil
	}
	t.mu.RUnlock()

	t.mu.Lock()
	defer t.mu.Unlock()

	if ctx, ok := t.contexts[device]; ok {
		return ctx, nil
	}

	C.cudaSetDevice(C.int(device))

	var stream C.cudaStream_t
	if err := C.cudaStreamCreate(&stream); err != 0 {
		return nil, fmt.Errorf("cudaStreamCreate failed: %v", err)
	}

	var handle C.cublasHandle_t
	if err := C.cublasCreate(&handle); err != 0 {
		C.cudaStreamDestroy(stream)
		return nil, fmt.Errorf("cublasCreate failed: %v", err)
	}
	C.cublasSetStream(handle, stream)

	ctx := &Context{
		Ctx:    stream,
		Cublas: handle,
		pool: &tensorPool{
			free: make(map[int][]*Tensor),
		},
	}

	t.contexts[device] = ctx
	return ctx, nil
}

func (t *TensorParallelManager) GetLocalRank() int {
	return t.localRank
}

func (t *TensorParallelManager) GetWorldSize() int {
	return t.worldSize
}

func (t *TensorParallelManager) GetDeviceForRank(rank int) int {
	if rank < 0 || rank >= len(t.devices) {
		return 0
	}
	return t.devices[rank]
}

func (t *TensorParallelManager) AllReduce(data []float32, count int) error {
	if t.worldSize <= 1 {
		return nil
	}

	ctx, err := t.GetContext(t.devices[t.localRank])
	if err != nil {
		return err
	}

	inputPtr := unsafe.Pointer(&data[0])
	outputPtr := unsafe.Pointer(&data[0])

	ncclSum := C.int(1)
	C.ncclAllReduceStub(inputPtr, outputPtr, C.size_t(count), ncclSum, C.int(1), unsafe.Pointer(ctx.Ctx))

	return nil
}

func (t *TensorParallelManager) AllGather(input []float32, output []float32, count int) error {
	if t.worldSize <= 1 {
		copy(output, input)
		return nil
	}

	ctx, err := t.GetContext(t.devices[t.localRank])
	if err != nil {
		return err
	}

	inputPtr := unsafe.Pointer(&input[0])
	outputPtr := unsafe.Pointer(&output[0])

	C.ncclAllGatherStub(inputPtr, outputPtr, C.size_t(count), 1, unsafe.Pointer(ctx.Ctx))

	return nil
}

func (t *TensorParallelManager) Broadcast(data []float32, count int, root int) error {
	if t.worldSize <= 1 {
		return nil
	}

	ctx, err := t.GetContext(t.devices[t.localRank])
	if err != nil {
		return err
	}

	dataPtr := unsafe.Pointer(&data[0])

	C.ncclBroadcastStub(dataPtr, dataPtr, C.size_t(count), 1, C.int(root), unsafe.Pointer(ctx.Ctx))

	return nil
}

func (t *TensorParallelManager) SynchronizeAll() {
	for _, device := range t.devices {
		ctx, err := t.GetContext(device)
		if err != nil {
			continue
		}
		C.cudaSetDevice(C.int(device))
		C.cudaStreamSynchronize(ctx.Ctx)
	}
}

// =============================================================================
// Pipeline Parallelism
// =============================================================================

type PipelineStage struct {
	ID           int
	StartLayer   int
	EndLayer     int
	DeviceID     int
	Context      *Context
	InputBuffer  *Tensor
	OutputBuffer *Tensor
	Weights      map[string]*Tensor
	DeQuantCache map[string]*Tensor
	mu           sync.Mutex
}

type PipelineParallelManager struct {
	config       *MultiGPUConfig
	stages       []*PipelineStage
	numStages    int
	numLayers    int
	mu           sync.RWMutex
	fwdPasses    atomic.Int64
	bwdPasses    atomic.Int64
	microBatches int
}

var pipelineParallel *PipelineParallelManager

func NewPipelineParallelManager(config *MultiGPUConfig, numLayers int) (*PipelineParallelManager, error) {
	multiGPUMu.Lock()
	defer multiGPUMu.Unlock()
	if pipelineParallel != nil {
		return pipelineParallel, nil
	}

	count, err := GetDeviceCount()
	if err != nil {
		return nil, fmt.Errorf("failed to get device count: %w", err)
	}

	numStages := config.PipelineStages
	if numStages <= 0 {
		numStages = count
	}

	if numStages > numLayers {
		numStages = numLayers
	}

	if numStages > count {
		return nil, fmt.Errorf("pipeline stages %d exceeds device count %d", numStages, count)
	}

	layersPerStage := int(math.Ceil(float64(numLayers) / float64(numStages)))

	pp := &PipelineParallelManager{
		config:       config,
		stages:       make([]*PipelineStage, numStages),
		numStages:    numStages,
		numLayers:    numLayers,
		microBatches: config.PipelineDepth,
	}

	for i := 0; i < numStages; i++ {
		startLayer := i * layersPerStage
		endLayer := startLayer + layersPerStage
		if endLayer > numLayers {
			endLayer = numLayers
		}

		deviceID := i % count

		stage := &PipelineStage{
			ID:           i,
			StartLayer:   startLayer,
			EndLayer:     endLayer,
			DeviceID:     deviceID,
			Weights:      make(map[string]*Tensor),
			DeQuantCache: make(map[string]*Tensor),
		}
		pp.stages[i] = stage
	}

	pipelineParallel = pp
	return pp, nil
}

func (p *PipelineParallelManager) GetStage(stageID int) *PipelineStage {
	if stageID < 0 || stageID >= len(p.stages) {
		return nil
	}
	return p.stages[stageID]
}

func (p *PipelineParallelManager) GetNumStages() int {
	return p.numStages
}

func (p *PipelineParallelManager) ForwardPass(microBatchID int, input []float32) ([]float32, error) {
	p.fwdPasses.Add(1)

	var currentInput = input
	for _, stage := range p.stages {
		output, err := p.forwardStage(stage, currentInput, microBatchID)
		if err != nil {
			return nil, fmt.Errorf("stage %d forward failed: %w", stage.ID, err)
		}
		currentInput = output
	}

	return currentInput, nil
}

func (p *PipelineParallelManager) forwardStage(stage *PipelineStage, input []float32, microBatchID int) ([]float32, error) {
	stage.mu.Lock()
	defer stage.mu.Unlock()

	manager := GetHybridManager()
	if manager == nil {
		return nil, fmt.Errorf("hybrid manager not initialized")
	}
	// Note: We need a way to get context from multi-gpu manager or directly
	// For now, we'll assume the context is available through the stage or similar
	_ = microBatchID

	dim := len(input)
	output := make([]float32, dim)

	for layer := stage.StartLayer; layer < stage.EndLayer; layer++ {
		_ = layer
	}

	copy(output, input)
	return output, nil
}

func (p *PipelineParallelManager) GetFwdPassCount() int64 {
	return p.fwdPasses.Load()
}

func (p *PipelineParallelManager) GetBwdPassCount() int64 {
	return p.bwdPasses.Load()
}

// =============================================================================
// Cross-GPU Communication
// =============================================================================

type PeerAccess struct {
	fromDevice int
	toDevice   int
	canAccess  bool
	bandwidth  float64
}

type PeerMemory struct {
	device     int
	peerDevice int
	peerPtr    unsafe.Pointer
	size       int64
	isValid    bool
}

type CrossGPUCommunicator struct {
	config        *MultiGPUConfig
	peerAccess    map[int]map[int]bool
	peerMemory    map[int]map[int]*PeerMemory
	commStreams   map[int]C.cudaStream_t
	collectiveOps int64
	bytesSent     int64
	bytesReceived int64
	mu            sync.RWMutex
}

var crossGPU *CrossGPUCommunicator

func NewCrossGPUCommunicator(config *MultiGPUConfig) (*CrossGPUCommunicator, error) {
	multiGPUMu.Lock()
	defer multiGPUMu.Unlock()
	if crossGPU != nil {
		return crossGPU, nil
	}

	count, err := GetDeviceCount()
	if err != nil {
		return nil, err
	}

	cg := &CrossGPUCommunicator{
		config:      config,
		peerAccess:  make(map[int]map[int]bool),
		peerMemory:  make(map[int]map[int]*PeerMemory),
		commStreams: make(map[int]C.cudaStream_t),
	}

	for i := 0; i < count; i++ {
		cg.peerAccess[i] = make(map[int]bool)
		cg.peerMemory[i] = make(map[int]*PeerMemory)

		C.cudaSetDevice(C.int(i))
		var stream C.cudaStream_t
		C.cudaStreamCreate(&stream)
		cg.commStreams[i] = stream

		for j := 0; j < count; j++ {
			if i == j {
				cg.peerAccess[i][j] = true
				continue
			}

			var canAccess C.int
			result := C.cudaDeviceCanAccessPeer(&canAccess, C.int(i), C.int(j))
			if result == C.cudaSuccess && canAccess == 1 {
				cg.peerAccess[i][j] = true

				err := C.cudaDeviceEnablePeerAccess(C.int(j), 0)
				if err != C.cudaSuccess {
					cg.peerAccess[i][j] = false
				}
			} else {
				cg.peerAccess[i][j] = false
			}
		}
	}

	crossGPU = cg
	return cg, nil
}

func (c *CrossGPUCommunicator) CanAccessPeer(from, to int) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if fromAccess, ok := c.peerAccess[from]; ok {
		if canAccess, ok := fromAccess[to]; ok {
			return canAccess
		}
	}
	return false
}

func (c *CrossGPUCommunicator) PeerToPeerCopy(srcDevice, dstDevice int, src, dst unsafe.Pointer, size int64) error {
	if !c.CanAccessPeer(srcDevice, dstDevice) {
		return fmt.Errorf("peer access not available from device %d to %d", srcDevice, dstDevice)
	}

	c.mu.RLock()
	stream := c.commStreams[srcDevice]
	c.mu.RUnlock()

	result := C.cudaMemcpyAsync(dst, src, C.size_t(size), C.cudaMemcpyDeviceToDevice, stream)
	if result != C.cudaSuccess {
		return fmt.Errorf("peer-to-peer copy failed: %v", result)
	}

	atomic.AddInt64(&c.bytesSent, size)
	atomic.AddInt64(&c.bytesReceived, size)

	return nil
}

func (c *CrossGPUCommunicator) AsyncSendRecv(device int, sendBuf, recvBuf unsafe.Pointer, size int64, peer int) error {
	if !c.CanAccessPeer(device, peer) || !c.CanAccessPeer(peer, device) {
		return fmt.Errorf("bidirectional peer access required between %d and %d", device, peer)
	}

	c.mu.RLock()
	stream := c.commStreams[device]
	c.mu.RUnlock()

	sendResult := C.cudaMemcpyAsync(recvBuf, sendBuf, C.size_t(size), C.cudaMemcpyDeviceToDevice, stream)
	if sendResult != C.cudaSuccess {
		return fmt.Errorf("send failed: %v", sendResult)
	}

	atomic.AddInt64(&c.bytesSent, size)
	atomic.AddInt64(&c.bytesReceived, size)

	return nil
}

func (c *CrossGPUCommunicator) Synchronize() {
	for _, stream := range c.commStreams {
		C.cudaStreamSynchronize(stream)
	}
}

func (c *CrossGPUCommunicator) GetStats() (collectiveOps int64, bytesSent int64, bytesReceived int64) {
	return atomic.LoadInt64(&c.collectiveOps),
		atomic.LoadInt64(&c.bytesSent),
		atomic.LoadInt64(&c.bytesReceived)
}

// =============================================================================
// Distributed AllReduce (All-GPU)
// =============================================================================

func AllReduceDistributed(tensors map[int][]float32, op ReduceOp) error {
	if len(tensors) <= 1 {
		return nil
	}

	switch op {
	case ReduceSum:
		sums := make([]float32, len(tensors[0]))
		for _, t := range tensors {
			for i := range sums {
				sums[i] += t[i]
			}
		}
		for i := range sums {
			sums[i] /= float32(len(tensors))
		}
		for _, t := range tensors {
			copy(t, sums)
		}
	case ReduceMax:
		maxes := make([]float32, len(tensors[0]))
		for i := range maxes {
			maxes[i] = -math.MaxFloat32
		}
		for _, t := range tensors {
			for i := range maxes {
				if t[i] > maxes[i] {
					maxes[i] = t[i]
				}
			}
		}
		for _, t := range tensors {
			copy(t, maxes)
		}
	case ReduceMean:
		sums := make([]float32, len(tensors[0]))
		for _, t := range tensors {
			for i := range sums {
				sums[i] += t[i]
			}
		}
		for i := range sums {
			sums[i] /= float32(len(tensors))
		}
		for _, t := range tensors {
			copy(t, sums)
		}
	}

	return nil
}

type ReduceOp int

const (
	ReduceSum ReduceOp = iota
	ReduceMax
	ReduceMean
	ReduceProd
)

// =============================================================================
// Hybrid Parallelism Manager
// =============================================================================

type HybridParallelismManager struct {
	config           *MultiGPUConfig
	tensorParallel   *TensorParallelManager
	pipelineParallel *PipelineParallelManager
	crossGPU         *CrossGPUCommunicator
	activeWorkers    atomic.Int32
	totalMemory      int64
	availableMemory  int64
	mu               sync.RWMutex
}

var hybridManager *HybridParallelismManager

func NewHybridParallelismManager(config *MultiGPUConfig) (*HybridParallelismManager, error) {
	multiGPUMu.Lock()
	defer multiGPUMu.Unlock()
	if hybridManager != nil {
		return hybridManager, nil
	}

	hm := &HybridParallelismManager{
		config: config,
	}

	if config.Mode&TensorParallelism != 0 {
		tp, err := NewTensorParallelManager(config)
		if err != nil {
			return nil, fmt.Errorf("tensor parallelism init failed: %w", err)
		}
		hm.tensorParallel = tp
	}

	if config.Mode&PipelineParallelism != 0 {
		pp, err := NewPipelineParallelManager(config, 32)
		if err != nil {
			return nil, fmt.Errorf("pipeline parallelism init failed: %w", err)
		}
		hm.pipelineParallel = pp
	}

	cg, err := NewCrossGPUCommunicator(config)
	if err != nil {
		return nil, fmt.Errorf("cross-GPU communicator init failed: %w", err)
	}
	hm.crossGPU = cg

	for i := 0; i < config.NumGPUs; i++ {
		mem, err := GetDeviceMemory(i)
		if err != nil {
			continue
		}
		hm.totalMemory += mem
		hm.availableMemory += mem
	}

	hybridManager = hm
	return hm, nil
}

func (h *HybridParallelismManager) DistributeLayers(numLayers int) []int {
	if h.pipelineParallel != nil {
		stages := h.pipelineParallel.numStages
		layersPerStage := int(math.Ceil(float64(numLayers) / float64(stages)))
		distribution := make([]int, stages)
		for i := 0; i < stages; i++ {
			distribution[i] = layersPerStage
		}
		remaining := numLayers - stages*layersPerStage
		for i := 0; i < remaining && i < stages; i++ {
			distribution[i]++
		}
		return distribution
	}
	return []int{numLayers}
}

func (h *HybridParallelismManager) GetDeviceForLayer(layer int) int {
	if h.pipelineParallel != nil {
		stages := h.pipelineParallel.stages
		for _, stage := range stages {
			if layer >= stage.StartLayer && layer < stage.EndLayer {
				return stage.DeviceID
			}
		}
	}
	return 0
}

func (h *HybridParallelismManager) AllReduce(output []float32, inputs map[int][]float32) error {
	if h.tensorParallel != nil {
		return h.tensorParallel.AllReduce(output, len(output))
	}
	return AllReduceDistributed(inputs, ReduceMean)
}

func (h *HybridParallelismManager) Synchronize() {
	if h.tensorParallel != nil {
		h.tensorParallel.SynchronizeAll()
	}
	h.crossGPU.Synchronize()
}

func (h *HybridParallelismManager) GetMemoryStats() (total, available, used int64) {
	total = h.totalMemory
	available = h.availableMemory
	used = total - available
	return
}

func (h *HybridParallelismManager) IsAvailable() bool {
	return h.activeWorkers.Load() < int32(h.config.NumGPUs)
}

func (h *HybridParallelismManager) AcquireWorker() int32 {
	return h.activeWorkers.Add(1)
}

func (h *HybridParallelismManager) ReleaseWorker() {
	h.activeWorkers.Add(-1)
}

// =============================================================================
// Utility Functions
// =============================================================================

func GetMultiGPUConfig() *MultiGPUConfig {
	return defaultMultiGPUConfig
}

func SetMultiGPUConfig(config *MultiGPUConfig) {
	multiGPUMu.Lock()
	defer multiGPUMu.Unlock()
	defaultMultiGPUConfig = config
}

func GetHybridManager() *HybridParallelismManager {
	multiGPUMu.Lock()
	defer multiGPUMu.Unlock()
	return hybridManager
}

func InitializeMultiGPU(config *MultiGPUConfig) error {
	if config.NumGPUs <= 0 {
		count, err := GetDeviceCount()
		if err != nil {
			return err
		}
		config.NumGPUs = count
	}

	if config.TensorParallelSize <= 0 {
		config.TensorParallelSize = config.NumGPUs
	}

	if config.PipelineStages <= 0 {
		config.PipelineStages = config.NumGPUs
	}

	_, err := NewHybridParallelismManager(config)
	return err
}

func ShutdownMultiGPU() {
	multiGPUMu.Lock()
	defer multiGPUMu.Unlock()
	if hybridManager != nil {
		hybridManager.Synchronize()
		hybridManager = nil
	}
	if crossGPU != nil {
		crossGPU = nil
	}
	if pipelineParallel != nil {
		pipelineParallel = nil
	}
	if tensorParallel != nil {
		tensorParallel = nil
	}
}
