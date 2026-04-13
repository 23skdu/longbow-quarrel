//go:build metal



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

	PrefillCompleted bool // Set when the full prompt has been ingested into KV cache
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
func (cm *ContinuousBatchManager) Step(maxBatchSize int, kvCache *PagedKVCache, promptCache *PromptCache) ([]*Sequence, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	const PrefillChunkSize = 512
	availableSlots := maxBatchSize - len(cm.running)

	// Pull new sequences into the prefill set if we have capacity
	if availableSlots > 0 {
		newReqs := cm.waitingQueue.PopUpTo(availableSlots)
		for _, req := range newReqs {
			seqIDStr := fmt.Sprintf("seq-%d", req.ID)
			
			// 1. Check Prompt Cache for shared prefixes
			var matchedCount int
			var cachedBlocks []int32
			if promptCache != nil {
				matchedCount, cachedBlocks = promptCache.MatchPrefix(req.Prompt)
			}
			
			// 2. Allocate Sequence state
			seq := &Sequence{
				ID:        req.ID,
				PromptLen: len(req.Prompt),
				MaxTokens: req.MaxTokens,
				Tokens:    append([]int{}, req.Prompt...),
				Config:    req.Config,
				Result:    req.Result,
				Err:       req.Err,
				Pos:       0,
				Status:    SequenceStatusRunning,
			}
			seq.TokenCallback = req.TokenCallback
			seq.LogitsCallback = req.LogitsCallback

			// 3. If cache hit, adopt blocks and advance position
			if matchedCount > 0 {
				_ = kvCache.AttachPrefixBlocks(seqIDStr, cachedBlocks)
				seq.Pos = matchedCount
			}

			cm.prefill[seq.ID] = seq
		}
	}

	// For the active step, gather what needs computation
	var active []*Sequence
	
	// Process prefill sequences (with chunking)
	for id, seq := range cm.prefill {
		// Calculate how many tokens to process this step
		remainingPrompt := seq.PromptLen - seq.Pos
		if remainingPrompt > 0 {
			// Chunked prefill
			toProcess := remainingPrompt
			if toProcess > PrefillChunkSize {
				toProcess = PrefillChunkSize
			}
			
			// This sequence is active for this step
			active = append(active, seq)
			
			// If we fully processed the prompt, move to running
			if seq.Pos + toProcess >= seq.PromptLen {
				seq.PrefillCompleted = true
				delete(cm.prefill, id)
				cm.running[id] = seq
			}
		} else {
			// Already at decoding stage
			delete(cm.prefill, id)
			cm.running[id] = seq
		}
	}

	// Add already running sequences
	for _, seq := range cm.running {
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
	
	if kvCache != nil {
		seqIDStr := fmt.Sprintf("seq-%d", id)
		kvCache.FreeSequence(seqIDStr)
	}
}

// AbortAll notifies all tracks (waiting, prefill, running) with the given error and clears state.
func (cm *ContinuousBatchManager) AbortAll(err error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// 1. Abort Waiting Queue
	cm.waitingQueue.mu.Lock()
	for _, req := range cm.waitingQueue.requests {
		select {
		case req.Err <- err:
		default:
		}
	}
	cm.waitingQueue.requests = nil
	cm.waitingQueue.mu.Unlock()

	// 2. Abort Running
	for _, seq := range cm.running {
		select {
		case seq.Err <- err:
		default:
		}
	}
	cm.running = make(map[uint64]*Sequence)

	// 3. Abort Prefill
	for _, seq := range cm.prefill {
		select {
		case seq.Err <- err:
		default:
		}
	}
	cm.prefill = make(map[uint64]*Sequence)
}
