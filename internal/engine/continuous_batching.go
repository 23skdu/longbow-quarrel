package engine

import (
	"fmt"
	"sync"
)

// InferenceRequest encapsulates a user generation request to be placed in the continuous batching queue.
type InferenceRequest struct {
	ID        uint64
	Prompt    []int
	MaxTokens int
	Config    SamplerConfig
	Result    chan []int
	Err       chan error
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
				Pos:       0,
				Status:    SequenceStatusPending,
			}
			
			// Under continuous batching PagedAttention, we don't need a single fat contiguous allocation.
			// We check block capacity on the fly. We'll simply prime the sequence state.
			cm.prefill[seq.ID] = seq
		}
	}

	// For the active step, gather what needs computation
	var active []*Sequence
	for _, seq := range cm.running {
		active = append(active, seq)
	}
	for _, seq := range cm.prefill {
		active = append(active, seq)
	}

	if len(active) == 0 {
		return nil, fmt.Errorf("no active sequences")
	}

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
