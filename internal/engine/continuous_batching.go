//go:build darwin && metal

package engine

import (
	"fmt"
	"sync"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// InferenceRequest encapsulates a user generation request to be placed in the continuous batching queue.
type InferenceRequest struct {
	ID        uint64
	Prompt    []int
	MaxTokens int
	Config    SamplerConfig
	Result    chan []int
	Err       chan error

	TokenCallback  func(int)
	LogitsCallback func([]float32)
}

// RequestQueue handles thread-safe enqueuing and dequeuing of raw incoming requests.
type RequestQueue struct {
	mu       sync.Mutex
	requests []*InferenceRequest
}

func (q *RequestQueue) Push(req *InferenceRequest) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.requests = append(q.requests, req)
}

func (q *RequestQueue) PopUpTo(n int) []*InferenceRequest {
	q.mu.Lock()
	defer q.mu.Unlock()

	var popCount int
	if len(q.requests) > n {
		popCount = n
	} else {
		popCount = len(q.requests)
	}

	popped := q.requests[:popCount]
	q.requests = q.requests[popCount:]
	return popped
}

func (q *RequestQueue) Depth() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.requests)
}

// ContinuousBatchManager oversees the lifecycle of sequences during decoding.
type ContinuousBatchManager struct {
	waitingQueue *RequestQueue
	
	// running represents the set of sequences currently in the decoding iteration
	running map[uint64]*Sequence
	
	// prefill indicates sequences that need their prompt processed next tick
	prefill map[uint64]*Sequence
	
	mu sync.RWMutex
}

func NewContinuousBatchManager() *ContinuousBatchManager {
	return &ContinuousBatchManager{
		waitingQueue: &RequestQueue{},
		running:      make(map[uint64]*Sequence),
		prefill:      make(map[uint64]*Sequence),
	}
}

// Submit adds a new request to the waiting pool.
func (cm *ContinuousBatchManager) Submit(req *InferenceRequest) {
	cm.waitingQueue.Push(req)
	metrics.BatchQueueDepth.Set(float64(cm.waitingQueue.Depth()))
}

// Step advances the state of the batching iteration, pulling from the waiting queue if resources permit.
func (cm *ContinuousBatchManager) Step(maxBatchSize int, kvCache *PagedKVCache) ([]*Sequence, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	availableSlots := maxBatchSize - len(cm.running)

	// Pull new sequences into the prefill set if we have capacity
	if availableSlots > 0 {
		newReqs := cm.waitingQueue.PopUpTo(availableSlots)
		for _, req := range newReqs {
			// Allocate Sequence state
			seq := &Sequence{
				ID:        req.ID,
				PromptLen: len(req.Prompt),
				MaxTokens: req.MaxTokens,
				Tokens:    append([]int{}, req.Prompt...),
				Config:    req.Config,
				Result:    req.Result,
				Err:       req.Err,
				Pos:       0,
				Status:    SequenceStatusRunning, // Immediate transition for prefill
			}
			// Attach callbacks from existing request if needed
			// (Note: InferenceRequest should have these fields now)
			seq.TokenCallback = req.TokenCallback
			seq.LogitsCallback = req.LogitsCallback

			cm.prefill[seq.ID] = seq
		}
	}

	// For the active step, gather what needs computation
	var active []*Sequence
	// 1. Prefill sequences (batch them or handle first)
	for id, seq := range cm.prefill {
		active = append(active, seq)
		delete(cm.prefill, id)
		cm.running[id] = seq
	}

	// 2. Already running sequences
	for _, seq := range cm.running {
		// Only add to active batch if not already added by prefill logic in this tick
		found := false
		for _, a := range active {
			if a.ID == seq.ID {
				found = true
				break
			}
		}
		if !found {
			active = append(active, seq)
		}
	}

	metrics.RecordBatchStats(cm.waitingQueue.Depth(), len(cm.running), len(cm.prefill))
	return active, nil
}

// CompleteSequence removes a sequence from the active pools and cleans up its Paged Attention blocks.
func (cm *ContinuousBatchManager) CompleteSequence(id uint64, kvCache *PagedKVCache) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	delete(cm.running, id)
	delete(cm.prefill, id)
	
	seqIDStr := fmt.Sprintf("seq-%d", id)
	kvCache.FreeSequence(seqIDStr)
}
