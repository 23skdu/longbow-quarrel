package vector

import (
	"container/heap"
	"fmt"
	"runtime"
	"sync"
)

// resultHeap implements a min-heap for keeping track of the top-k highest scoring items.
type resultHeap []SearchResult

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *resultHeap) Push(x interface{}) {
	*h = append(*h, x.(SearchResult))
}
func (h *resultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// FlatIndex holds vectors for exact linear scan search.
type FlatIndex struct {
	Dimension int
	DataType  DataType
	Float32   [][]float32
	Uint8     []Uint8Vector
	Complex   [][]complex128
	Turbo     []TurboQuantVector
}

// NewFlatIndex creates an empty FlatIndex.
func NewFlatIndex(dim int, dtype DataType) *FlatIndex {
	return &FlatIndex{
		Dimension: dim,
		DataType:  dtype,
	}
}

// AddFloat32 adds a batch of float32 vectors to the index.
func (idx *FlatIndex) AddFloat32(vectors [][]float32) {
	idx.Float32 = append(idx.Float32, vectors...)
}

// AddUint8 adds a batch of Uint8Vectors to the index.
func (idx *FlatIndex) AddUint8(vectors []Uint8Vector) {
	idx.Uint8 = append(idx.Uint8, vectors...)
}

// AddComplex128 adds a batch of complex128 vectors to the index.
func (idx *FlatIndex) AddComplex128(vectors [][]complex128) {
	idx.Complex = append(idx.Complex, vectors...)
}

// AddTurboQuant adds a batch of TurboQuantVectors to the index.
func (idx *FlatIndex) AddTurboQuant(vectors []TurboQuantVector) {
	idx.Turbo = append(idx.Turbo, vectors...)
}

// Count returns the number of vectors in the index.
func (idx *FlatIndex) Count() int {
	switch idx.DataType {
	case TypeFloat32:
		return len(idx.Float32)
	case TypeUint8:
		return len(idx.Uint8)
	case TypeComplex128:
		return len(idx.Complex)
	case TypeTurboQuant:
		return len(idx.Turbo)
	default:
		return 0
	}
}

// Search performs an exact brute-force search over all vectors using parallel chunks.
func (idx *FlatIndex) Search(query interface{}, k int, metric DistanceMetric) ([]SearchResult, error) {
	n := idx.Count()
	if n == 0 || k <= 0 {
		return nil, nil
	}
	if k > n {
		k = n
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 0 {
		workers = 1
	}

	chunkSize := (n + workers - 1) / workers
	workerHeaps := make([]resultHeap, workers)

	var wg sync.WaitGroup
	var workerErr error
	var errOnce sync.Once

	for w := 0; w < workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if start >= n {
			break
		}
		if end > n {
			end = n
		}

		workerIdx := w
		wg.Add(1)
		go func(s, e, wID int) {
			defer wg.Done()
			h := &resultHeap{}
			heap.Init(h)

			switch idx.DataType {
			case TypeFloat32:
				q, ok := query.([]float32)
				if !ok {
					errOnce.Do(func() { workerErr = fmt.Errorf("expected []float32 query") })
					return
				}
				for i := s; i < e; i++ {
					var score float32
					switch metric {
					case MetricDot:
						score = DotFloat32(q, idx.Float32[i])
					case MetricCosine:
						score = CosineFloat32(q, idx.Float32[i])
					case MetricL2:
						score = -L2Float32(q, idx.Float32[i]) // Negative distance so closer is higher
					default:
						score = DotFloat32(q, idx.Float32[i])
					}
					if h.Len() < k {
						heap.Push(h, SearchResult{ID: i, Score: score})
					} else if score > (*h)[0].Score {
						(*h)[0] = SearchResult{ID: i, Score: score}
						heap.Fix(h, 0)
					}
				}

			case TypeUint8:
				q, ok := query.(*Uint8Vector)
				if !ok {
					errOnce.Do(func() { workerErr = fmt.Errorf("expected *Uint8Vector query") })
					return
				}
				for i := s; i < e; i++ {
					var score float32
					switch metric {
					case MetricDot:
						score = DotUint8(q, &idx.Uint8[i])
					case MetricCosine:
						score = CosineUint8(q, &idx.Uint8[i])
					case MetricL2:
						score = -L2Uint8(q, &idx.Uint8[i])
					default:
						score = DotUint8(q, &idx.Uint8[i])
					}
					if h.Len() < k {
						heap.Push(h, SearchResult{ID: idx.Uint8[i].ID, Score: score})
					} else if score > (*h)[0].Score {
						(*h)[0] = SearchResult{ID: idx.Uint8[i].ID, Score: score}
						heap.Fix(h, 0)
					}
				}

			case TypeComplex128:
				q, ok := query.([]complex128)
				if !ok {
					errOnce.Do(func() { workerErr = fmt.Errorf("expected []complex128 query") })
					return
				}
				for i := s; i < e; i++ {
					var score float32
					switch metric {
					case MetricDot:
						score = float32(DotComplex128(q, idx.Complex[i]))
					case MetricCosine:
						score = float32(CosineComplex128(q, idx.Complex[i]))
					case MetricL2:
						score = -float32(L2Complex128(q, idx.Complex[i]))
					default:
						score = float32(DotComplex128(q, idx.Complex[i]))
					}
					if h.Len() < k {
						heap.Push(h, SearchResult{ID: i, Score: score})
					} else if score > (*h)[0].Score {
						(*h)[0] = SearchResult{ID: i, Score: score}
						heap.Fix(h, 0)
					}
				}

			case TypeTurboQuant:
				q, ok := query.(*TurboQuantVector)
				if !ok {
					errOnce.Do(func() { workerErr = fmt.Errorf("expected *TurboQuantVector query") })
					return
				}
				for i := s; i < e; i++ {
					var score float32
					switch metric {
					case MetricDot:
						score = DotTurboQuant(q, &idx.Turbo[i])
					case MetricCosine:
						score = CosineTurboQuant(q, &idx.Turbo[i])
					case MetricL2:
						score = -L2TurboQuant(q, &idx.Turbo[i])
					default:
						score = DotTurboQuant(q, &idx.Turbo[i])
					}
					if h.Len() < k {
						heap.Push(h, SearchResult{ID: idx.Turbo[i].ID, Score: score})
					} else if score > (*h)[0].Score {
						(*h)[0] = SearchResult{ID: idx.Turbo[i].ID, Score: score}
						heap.Fix(h, 0)
					}
				}
			}

			workerHeaps[wID] = *h
		}(start, end, workerIdx)
	}

	wg.Wait()

	if workerErr != nil {
		return nil, workerErr
	}

	// Merge worker heaps into final top-k
	finalHeap := &resultHeap{}
	heap.Init(finalHeap)

	for _, wh := range workerHeaps {
		for _, item := range wh {
			if finalHeap.Len() < k {
				heap.Push(finalHeap, item)
			} else if item.Score > (*finalHeap)[0].Score {
				(*finalHeap)[0] = item
				heap.Fix(finalHeap, 0)
			}
		}
	}

	results := make([]SearchResult, finalHeap.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(finalHeap).(SearchResult)
	}

	return results, nil
}
