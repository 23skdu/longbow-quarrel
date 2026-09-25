//go:build linux && amd64 && cuda && cgo

package device

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR} -lcuda_kernels -lcublas -lcudnn -lcudart
#cgo linux,amd64 CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}
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
	Devices            []int // Explicit device IDs, e.g. [0, 1]
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
	ncclComm  *ncclCommHandle
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

	tpSize := config.TensorParallelSize
	if tpSize <= 0 {
		tpSize = 1
	}

	if tpSize > 1 && count < 2 {
		return nil, fmt.Errorf("tensor parallelism requires at least 2 GPUs, found %d", count)
	}

	if tpSize > count && len(config.Devices) == 0 {
		return nil, fmt.Errorf("tensor parallel size %d exceeds device count %d", tpSize, count)
	}

	var devices []int
	if len(config.Devices) > 0 {
		devices = append([]int(nil), config.Devices...)
	} else {
		for i := 0; i < tpSize; i++ {
			devices = append(devices, i%count)
		}
	}

	tp := &TensorParallelManager{
		config:    config,
		devices:   devices,
		contexts:  make(map[int]*Context),
		ranks:     make([]int, len(devices)),
		localRank: 0,
		worldSize: len(devices),
	}

	for i := range devices {
		tp.ranks[i] = i
	}

	if config.UseNCCL && tp.worldSize > 1 && count > 1 {
		comm, err := ncclInit(0, tp.worldSize, devices[0])
		if err == nil {
			tp.ncclComm = comm
		}
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

func (t *TensorParallelManager) GetWorldSize() int {
	return t.worldSize
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

	if t.ncclComm != nil {
		return t.ncclComm.ncclAllReduce(inputPtr, outputPtr, count, unsafe.Pointer(ctx.Ctx))
	}

	// Fallback: serial memcpy (no cross-GPU reduction without NCCL)
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

	if t.ncclComm != nil {
		return t.ncclComm.ncclAllGather(inputPtr, outputPtr, count, unsafe.Pointer(ctx.Ctx))
	}

	copy(output, input)
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

func (t *TensorParallelManager) Close() {
	if t.ncclComm != nil {
		t.ncclComm.destroy()
		t.ncclComm = nil
	}
}

// =============================================================================
// Pipeline Parallelism
// =============================================================================

type PipelineStage struct {
	ID             int
	StartLayer     int
	EndLayer       int
	DeviceID       int
	Context        *Context
	InputBuffer    *Tensor // Alias to InputBuffers[0] for backward compatibility
	OutputBuffer   *Tensor // Alias to OutputBuffers[0] for backward compatibility
	InputBuffers   [2]*Tensor
	OutputBuffers  [2]*Tensor
	HostStagingIn  [2][]float32
	HostStagingOut [2][]float32
	slotLocks      [2]sync.Mutex
	Weights        map[string]*Tensor
	DeQuantCache   map[string]*Tensor
	ComputeStream  C.cudaStream_t
	CommStream     C.cudaStream_t
	mu             sync.Mutex
}

func (s *PipelineStage) AcquireSlot(slot int) {
	s.slotLocks[slot%2].Lock()
}

func (s *PipelineStage) ReleaseSlot(slot int) {
	s.slotLocks[slot%2].Unlock()
}

func (s *PipelineStage) InitDoubleBuffers(dim int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := 0; i < 2; i++ {
		if len(s.HostStagingIn[i]) != dim {
			s.HostStagingIn[i] = make([]float32, dim)
		}
		if len(s.HostStagingOut[i]) != dim {
			s.HostStagingOut[i] = make([]float32, dim)
		}
	}
}

func (s *PipelineStage) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ComputeStream != nil {
		C.cudaStreamDestroy(s.ComputeStream)
		s.ComputeStream = nil
	}
	if s.CommStream != nil {
		C.cudaStreamDestroy(s.CommStream)
		s.CommStream = nil
	}
	for i := 0; i < 2; i++ {
		if s.InputBuffers[i] != nil {
			s.InputBuffers[i].ReturnToPool()
			s.InputBuffers[i] = nil
		}
		if s.OutputBuffers[i] != nil {
			s.OutputBuffers[i].ReturnToPool()
			s.OutputBuffers[i] = nil
		}
	}
	s.InputBuffer = nil
	s.OutputBuffer = nil
}

type StageComputeFunc func(stage *PipelineStage, microBatchID int, slot int, input []float32) ([]float32, error)

type pipelineTask struct {
	microBatchID int
	slot         int
	data         []float32
	err          error
}

type PipelineParallelManager struct {
	config       *MultiGPUConfig
	stages       []*PipelineStage
	numStages    int
	numLayers    int
	crossGPU     *CrossGPUCommunicator
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

	if count <= 0 {
		return nil, fmt.Errorf("no GPU devices available for pipeline parallelism")
	}

	numStages := config.PipelineStages
	if numStages <= 0 {
		if len(config.Devices) > 0 {
			numStages = len(config.Devices)
		} else {
			numStages = count
		}
	}

	if numStages > numLayers {
		numStages = numLayers
	}

	var devices []int
	if len(config.Devices) > 0 {
		devices = append([]int(nil), config.Devices...)
	} else {
		for i := 0; i < count; i++ {
			devices = append(devices, i)
		}
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

		deviceID := devices[i%len(devices)]

		C.cudaSetDevice(C.int(deviceID))
		var compStream, commStream C.cudaStream_t
		_ = C.cudaStreamCreate(&compStream)
		_ = C.cudaStreamCreate(&commStream)

		stage := &PipelineStage{
			ID:            i,
			StartLayer:    startLayer,
			EndLayer:      endLayer,
			DeviceID:      deviceID,
			Weights:       make(map[string]*Tensor),
			DeQuantCache:  make(map[string]*Tensor),
			ComputeStream: compStream,
			CommStream:    commStream,
		}
		pp.stages[i] = stage
	}

	pipelineParallel = pp
	return pp, nil
}

func (p *PipelineParallelManager) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, stage := range p.stages {
		if stage != nil {
			stage.Close()
		}
	}
}

func (p *PipelineParallelManager) ForwardPass(microBatchID int, input []float32) ([]float32, error) {
	results, err := p.ForwardMicroBatches1F1B([][]float32{input}, nil)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no output from forward pass")
	}
	return results[0], nil
}

func (p *PipelineParallelManager) ForwardMicroBatches1F1B(
	microBatches [][]float32,
	stageFn StageComputeFunc,
) ([][]float32, error) {
	numBatches := len(microBatches)
	if numBatches == 0 {
		return nil, nil
	}

	if stageFn == nil {
		stageFn = p.defaultStageForward
	}

	numStages := p.numStages
	if numStages <= 0 {
		numStages = 1
	}

	// For a single stage, evaluate micro-batches sequentially with double buffering
	if numStages == 1 {
		stage := p.stages[0]
		results := make([][]float32, numBatches)
		for m := 0; m < numBatches; m++ {
			slot := m % 2
			stage.AcquireSlot(slot)
			out, err := stageFn(stage, m, slot, microBatches[m])
			stage.ReleaseSlot(slot)
			if err != nil {
				return nil, fmt.Errorf("stage 0 forward failed on micro-batch %d: %w", m, err)
			}
			p.fwdPasses.Add(1)
			results[m] = out
		}
		return results, nil
	}

	// Multi-stage 1F1B execution with double-buffering channels
	// Each stage has a channel of depth 2 (ping-pong double buffer).
	stageChans := make([]chan pipelineTask, numStages)
	for s := 0; s < numStages; s++ {
		stageChans[s] = make(chan pipelineTask, 2)
	}

	results := make([][]float32, numBatches)
	var resultsMu sync.Mutex
	var firstErr error
	var errOnce sync.Once
	setErr := func(err error) {
		errOnce.Do(func() {
			firstErr = err
		})
	}

	var wg sync.WaitGroup
	wg.Add(numStages)

	for s := 0; s < numStages; s++ {
		stageIdx := s
		stage := p.stages[stageIdx]

		go func() {
			defer wg.Done()
			for task := range stageChans[stageIdx] {
				if task.err != nil {
					setErr(task.err)
					if stageIdx+1 < numStages {
						stageChans[stageIdx+1] <- task
					}
					continue
				}

				microBatchID := task.microBatchID
				slot := task.slot

				// Acquire double-buffer slot on current stage
				stage.AcquireSlot(slot)

				outData, err := stageFn(stage, microBatchID, slot, task.data)
				stage.ReleaseSlot(slot)

				if err != nil {
					setErr(err)
					errTask := pipelineTask{
						microBatchID: microBatchID,
						slot:         slot,
						err:          err,
					}
					if stageIdx+1 < numStages {
						stageChans[stageIdx+1] <- errTask
					}
					continue
				}

				p.fwdPasses.Add(1)

				if stageIdx+1 < numStages {
					nextStage := p.stages[stageIdx+1]
					if p.crossGPU != nil && stage.DeviceID != nextStage.DeviceID {
						_ = p.crossGPU.TransferActivations(stage.DeviceID, nextStage.DeviceID, outData, slot)
					}
					stageChans[stageIdx+1] <- pipelineTask{
						microBatchID: microBatchID,
						slot:         slot,
						data:         outData,
					}
				} else {
					resultsMu.Lock()
					results[microBatchID] = outData
					resultsMu.Unlock()
				}
			}

			// When stageIdx channel is closed, close the next stage channel
			if stageIdx+1 < numStages {
				close(stageChans[stageIdx+1])
			}
		}()
	}

	// Dispatcher goroutine feeding Stage 0
	go func() {
		for m := 0; m < numBatches; m++ {
			slot := m % 2
			stageChans[0] <- pipelineTask{
				microBatchID: m,
				slot:         slot,
				data:         microBatches[m],
			}
		}
		close(stageChans[0])
	}()

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return results, nil
}

func (p *PipelineParallelManager) defaultStageForward(stage *PipelineStage, microBatchID int, slot int, input []float32) ([]float32, error) {
	dim := len(input)
	output := make([]float32, dim)
	copy(output, input)

	stage.InitDoubleBuffers(dim)

	stage.mu.Lock()
	if len(stage.HostStagingIn[slot]) >= dim {
		copy(stage.HostStagingIn[slot], input)
	}
	if len(stage.HostStagingOut[slot]) >= dim {
		copy(stage.HostStagingOut[slot], output)
	}
	stage.mu.Unlock()

	for layer := stage.StartLayer; layer < stage.EndLayer; layer++ {
		_ = layer
	}

	return output, nil
}

// ForwardStage executes a single pipeline stage forward pass for a given microbatch.
func (p *PipelineParallelManager) ForwardStage(stage *PipelineStage, input []float32, microBatchID int) ([]float32, error) {
	return p.defaultStageForward(stage, microBatchID, microBatchID%2, input)
}

func (p *PipelineParallelManager) GetFwdPassCount() int64 {
	return p.fwdPasses.Load()
}

// =============================================================================
// Cross-GPU Communication
// =============================================================================

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

func (c *CrossGPUCommunicator) AllocatePeerStagingBuffer(srcDev, dstDev int, size int64) (*PeerMemory, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if pmMap, ok := c.peerMemory[srcDev]; ok {
		if pm, ok := pmMap[dstDev]; ok && pm.isValid && pm.size >= size {
			return pm, nil
		}
	}

	C.cudaSetDevice(C.int(srcDev))
	var ptr unsafe.Pointer
	res := C.cudaMalloc(&ptr, C.size_t(size))
	if res != C.cudaSuccess {
		return nil, fmt.Errorf("cudaMalloc failed on device %d for peer staging: %v", srcDev, res)
	}

	pm := &PeerMemory{
		device:     srcDev,
		peerDevice: dstDev,
		peerPtr:    ptr,
		size:       size,
		isValid:    true,
	}

	if c.peerMemory[srcDev] == nil {
		c.peerMemory[srcDev] = make(map[int]*PeerMemory)
	}
	c.peerMemory[srcDev][dstDev] = pm

	return pm, nil
}

func (c *CrossGPUCommunicator) TransferActivations(srcDevice, dstDevice int, data []float32, slot int) error {
	if srcDevice == dstDevice {
		return nil
	}
	size := int64(len(data) * 4)
	if size == 0 {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	atomic.AddInt64(&c.collectiveOps, 1)
	atomic.AddInt64(&c.bytesSent, size)
	atomic.AddInt64(&c.bytesReceived, size)

	// If peer staging memory is allocated, copy asynchronously
	if peerMap, ok := c.peerMemory[srcDevice]; ok {
		if pm, ok := peerMap[dstDevice]; ok && pm.isValid && pm.size >= size {
			stream := c.commStreams[srcDevice]
			C.cudaMemcpyAsync(pm.peerPtr, unsafe.Pointer(&data[0]), C.size_t(size), C.cudaMemcpyHostToDevice, stream)
			C.cudaStreamSynchronize(stream)
		}
	}
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

	cg, err := NewCrossGPUCommunicator(config)
	if err != nil {
		return nil, fmt.Errorf("cross-GPU communicator init failed: %w", err)
	}
	hm.crossGPU = cg

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
		pp.crossGPU = cg
		hm.pipelineParallel = pp
	}

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
		pipelineParallel.Close()
		pipelineParallel = nil
	}
	if tensorParallel != nil {
		tensorParallel.Close()
		tensorParallel = nil
	}
}
