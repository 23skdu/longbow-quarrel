//go:build linux && amd64 && cuda && cgo

package device

/*
#include <cuda_runtime.h>

extern int cudaGraphBeginCaptureWrapper(cudaStream_t stream);
extern int cudaGraphEndCaptureWrapper(cudaStream_t stream, cudaGraph_t* pGraph);
extern int cudaGraphInstantiateWrapper(cudaGraphExec_t* pGraphExec, cudaGraph_t graph);
extern int cudaGraphLaunchWrapper(cudaGraphExec_t graphExec, cudaStream_t stream);
extern int cudaGraphExecDestroyWrapper(cudaGraphExec_t graphExec);
extern int cudaGraphDestroyWrapper(cudaGraph_t graph);
*/
import "C"
import (
	"fmt"
	"sync"
)

// CUDAGraphManager manages captured CUDA execution graphs for static decode passes.
// It eliminates CGo bridge overhead and CUDA driver kernel launch latency during single-token
// and fixed small-batch decode forward iterations.
type CUDAGraphManager struct {
	ctx            *Context
	mu             sync.RWMutex
	graphs         map[int]C.cudaGraph_t
	execs          map[int]C.cudaGraphExec_t
	isCapturing    bool
	capturingBatch int
}

// NewCUDAGraphManager initializes a new graph manager bound to a CUDA context.
func NewCUDAGraphManager(ctx *Context) *CUDAGraphManager {
	return &CUDAGraphManager{
		ctx:    ctx,
		graphs: make(map[int]C.cudaGraph_t),
		execs:  make(map[int]C.cudaGraphExec_t),
	}
}

// BeginCapture starts capturing CUDA stream operations into an execution graph for the given batch size.
func (m *CUDAGraphManager) BeginCapture(batchSize int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.isCapturing {
		return fmt.Errorf("cuda graph: capture already in progress for batch size %d", m.capturingBatch)
	}

	// Clean up previous graph for this batch size if it exists
	if exec, exists := m.execs[batchSize]; exists {
		C.cudaGraphExecDestroyWrapper(exec)
		delete(m.execs, batchSize)
	}
	if g, exists := m.graphs[batchSize]; exists {
		C.cudaGraphDestroyWrapper(g)
		delete(m.graphs, batchSize)
	}

	res := C.cudaGraphBeginCaptureWrapper(m.ctx.Ctx)
	if res != 0 {
		return fmt.Errorf("cudaStreamBeginCapture failed with error code: %d", int(res))
	}

	m.isCapturing = true
	m.capturingBatch = batchSize
	return nil
}

// EndCapture terminates capture on the stream, stores the graph, and instantiates it.
func (m *CUDAGraphManager) EndCapture(batchSize int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.isCapturing || m.capturingBatch != batchSize {
		return fmt.Errorf("cuda graph: no active capture matching batch size %d", batchSize)
	}

	var graph C.cudaGraph_t
	res := C.cudaGraphEndCaptureWrapper(m.ctx.Ctx, &graph)
	if res != 0 {
		m.isCapturing = false
		m.capturingBatch = 0
		return fmt.Errorf("cudaStreamEndCapture failed with error code: %d", int(res))
	}

	var graphExec C.cudaGraphExec_t
	instRes := C.cudaGraphInstantiateWrapper(&graphExec, graph)
	if instRes != 0 {
		C.cudaGraphDestroyWrapper(graph)
		m.isCapturing = false
		m.capturingBatch = 0
		return fmt.Errorf("cudaGraphInstantiate failed with error code: %d", int(instRes))
	}

	m.graphs[batchSize] = graph
	m.execs[batchSize] = graphExec
	m.isCapturing = false
	m.capturingBatch = 0
	return nil
}

// Launch replays the pre-captured graph for the specified batch size.
func (m *CUDAGraphManager) Launch(batchSize int) error {
	m.mu.RLock()
	exec, exists := m.execs[batchSize]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("cuda graph: no instantiated graph for batch size %d", batchSize)
	}

	res := C.cudaGraphLaunchWrapper(exec, m.ctx.Ctx)
	if res != 0 {
		return fmt.Errorf("cudaGraphLaunch failed with error code: %d", int(res))
	}
	return nil
}

// HasGraph checks whether a graph has been captured and instantiated for the given batch size.
func (m *CUDAGraphManager) HasGraph(batchSize int) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, exists := m.execs[batchSize]
	return exists
}

// IsCapturing returns true if the stream is currently undergoing graph capture.
func (m *CUDAGraphManager) IsCapturing() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.isCapturing
}

// Close destroys all allocated execution graphs and releases GPU resources.
func (m *CUDAGraphManager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for b, exec := range m.execs {
		C.cudaGraphExecDestroyWrapper(exec)
		delete(m.execs, b)
	}
	for b, g := range m.graphs {
		C.cudaGraphDestroyWrapper(g)
		delete(m.graphs, b)
	}
	m.isCapturing = false
	m.capturingBatch = 0
}
