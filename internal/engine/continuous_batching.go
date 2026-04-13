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
	AdapterID      string // Active LoRA adapter for this request

	PrefillCompleted bool // Set when the full prompt has been ingested into KV cache
}

// BatchDescriptor describes a packed execution batch for the engine
type BatchDescriptor struct {
	Sequences []*Sequence
	// Tokens is a packed slice of all tokens to process this step
	Tokens []int
	// Offsets records the start index of each sequence in the Tokens slice
	Offsets []int
	// SequenceLengths records the current KV cache position for each sequence
	ContextLens []int
	// TokenToSeq maps each token in the packed Tokens slice to its index in the Sequences slice
	TokenToSeq []int
	// AdapterIDs maps each sequence to its active LoRA adapter ID
	AdapterIDs []string
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
func (cm *ContinuousBatchManager) Step(maxBatchSize int, kvCache *PagedKVCache, promptCache *PromptCache) (*BatchDescriptor, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	const PrefillChunkSize = 512
	availableSlots := maxBatchSize - len(cm.running) - len(cm.prefill)

	// Resource-aware Admission Control
	// Only pull if we have slots AND the KV cache isn't nearly exhausted
	// If kvCache is nil (tests), we bypass admission control
	canAdmit := availableSlots > 0
	if kvCache != nil {
		canAdmit = canAdmit && kvCache.FreeBlocksCount() > 32 // Lower threshold for admission
	}

	if canAdmit {
		newReqs := cm.waitingQueue.PopUpTo(availableSlots)
		for _, req := range newReqs {
			// Admittance check: can we even fit the first chunk?
			chunkSize := len(req.Prompt)
			if chunkSize > PrefillChunkSize {
				chunkSize = PrefillChunkSize
			}

			if kvCache != nil && !kvCache.HasCapacityFor(chunkSize) {
				// Re-queue and stop admitting for this tick
				cm.waitingQueue.Push(req)
				break
			}

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
			seq.AdapterID = req.AdapterID

			// 3. If cache hit, adopt blocks and advance position
			if matchedCount > 0 {
				_ = kvCache.AttachPrefixBlocks(seqIDStr, cachedBlocks)
				seq.Pos = matchedCount
			}

			cm.prefill[seq.ID] = seq
		}
	}

	desc := &BatchDescriptor{
		Sequences:   make([]*Sequence, 0),
		Tokens:      make([]int, 0),
		Offsets:     make([]int, 0),
		ContextLens: make([]int, 0),
		TokenToSeq:  make([]int, 0),
	}
	
	// Track added IDs to avoid duplicates
	added := make(map[uint64]bool)

	// Process prefill sequences (with chunking)
	for id, seq := range cm.prefill {
		remainingPrompt := seq.PromptLen - seq.Pos
		if remainingPrompt > 0 {
			toProcess := remainingPrompt
			if toProcess > PrefillChunkSize {
				toProcess = PrefillChunkSize
			}
			
			desc.Offsets = append(desc.Offsets, len(desc.Tokens))
			desc.Tokens = append(desc.Tokens, seq.Tokens[seq.Pos:seq.Pos+toProcess]...)
			desc.ContextLens = append(desc.ContextLens, seq.Pos)
			seqIdx := len(desc.Sequences)
			for j := 0; j < toProcess; j++ {
				desc.TokenToSeq = append(desc.TokenToSeq, seqIdx)
			}
			desc.Sequences = append(desc.Sequences, seq)
			added[id] = true
			
			// Update locally for next iteration (actual Pos update happens after kernel success in runBatchLoop)
			if seq.Pos + toProcess >= seq.PromptLen {
				seq.PrefillCompleted = true
				delete(cm.prefill, id)
				cm.running[id] = seq
			}
		} else {
			delete(cm.prefill, id)
			cm.running[id] = seq
		}
	}

	// Add already running sequences (decoding stage)
	for id, seq := range cm.running {
		if added[id] {
			continue
		}
		desc.AdapterIDs = append(desc.AdapterIDs, seq.AdapterID)
		
		desc.Offsets = append(desc.Offsets, len(desc.Tokens))
		desc.Tokens = append(desc.Tokens, seq.Tokens[len(seq.Tokens)-1])
		desc.ContextLens = append(desc.ContextLens, seq.Pos)
		desc.TokenToSeq = append(desc.TokenToSeq, len(desc.Sequences))
		desc.Sequences = append(desc.Sequences, seq)
		added[id] = true
	}

	metrics.RecordBatchStats(cm.waitingQueue.Depth(), len(cm.running), len(cm.prefill))
	
	if len(desc.Sequences) == 0 {
		return nil, nil
	}
	return desc, nil
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
