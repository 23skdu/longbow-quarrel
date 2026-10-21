package simd

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// Task represents a slice of parallel range work [start, end)
type task struct {
	start int
	end   int
	fn    func(start, end int)
	wg    *sync.WaitGroup
}

// WorkerPool manages a persistent set of worker goroutines to eliminate
// per-token goroutine spawn/teardown overhead in SIMD and MatVec operations.
type WorkerPool struct {
	numWorkers int
	tasks      chan task
	closed     atomic.Bool
	stopOnce   sync.Once
	stopChan   chan struct{}
}

var (
	defaultPool     *WorkerPool
	defaultPoolOnce sync.Once
)

// GetWorkerPool returns the global singleton WorkerPool initialized to GOMAXPROCS.
func GetWorkerPool() *WorkerPool {
	defaultPoolOnce.Do(func() {
		workers := runtime.GOMAXPROCS(0)
		if workers < 1 {
			workers = 1
		}
		defaultPool = NewWorkerPool(workers)
	})
	return defaultPool
}

// NewWorkerPool creates a persistent worker pool with the given worker count.
func NewWorkerPool(numWorkers int) *WorkerPool {
	if numWorkers < 1 {
		numWorkers = 1
	}
	p := &WorkerPool{
		numWorkers: numWorkers,
		tasks:      make(chan task, numWorkers*128),
		stopChan:   make(chan struct{}),
	}
	for i := 0; i < numWorkers; i++ {
		go p.workerLoop()
	}
	return p
}

func (p *WorkerPool) workerLoop() {
	for {
		select {
		case t, ok := <-p.tasks:
			if !ok {
				return
			}
			t.fn(t.start, t.end)
			t.wg.Done()
		case <-p.stopChan:
			return
		}
	}
}

// ParallelFor executes fn(start, end) partitioned across the pool's workers for range [0, total).
// The calling goroutine directly executes the first chunk to minimize scheduling latency.
func (p *WorkerPool) ParallelFor(total int, fn func(start, end int)) {
	if total <= 0 || fn == nil {
		return
	}
	if p == nil || p.numWorkers <= 1 || total < 2 || p.closed.Load() {
		fn(0, total)
		return
	}

	workers := p.numWorkers
	if workers > total {
		workers = total
	}

	chunkSize := (total + workers - 1) / workers
	firstEnd := chunkSize
	if firstEnd > total {
		firstEnd = total
	}

	var callWg sync.WaitGroup
	numDispatched := 0

	for w := 1; w < workers; w++ {
		start := w * chunkSize
		if start >= total {
			break
		}
		end := start + chunkSize
		if end > total {
			end = total
		}

		callWg.Add(1)
		numDispatched++
		select {
		case p.tasks <- task{start: start, end: end, fn: fn, wg: &callWg}:
		default:
			// Fallback: if task queue is full under heavy concurrency, execute inline
			callWg.Done()
			numDispatched--
			fn(start, end)
		}
	}

	// Calling thread executes chunk 0
	fn(0, firstEnd)

	// Wait for remote workers
	if numDispatched > 0 {
		callWg.Wait()
	}
}

// Close stops all worker goroutines in the pool.
func (p *WorkerPool) Close() {
	if p == nil {
		return
	}
	p.stopOnce.Do(func() {
		p.closed.Store(true)
		close(p.stopChan)
	})
}

// ParallelFor executes fn(start, end) on the global default worker pool.
func ParallelFor(total int, fn func(start, end int)) {
	GetWorkerPool().ParallelFor(total, fn)
}
