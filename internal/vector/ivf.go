package vector

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// IVFIndex implements Inverted File indexing with multi-probe search.
type IVFIndex struct {
	Dimension     int
	NumCentroids  int
	NumProbes     int
	DataType      DataType
	Centroids     [][]float32
	InvertedLists [][]int // For each centroid, slice of vector IDs assigned to it

	// Backing vector store
	Float32 [][]float32
	Uint8   []Uint8Vector
	Complex [][]complex128
	Turbo   []TurboQuantVector
}

// NewIVFIndex creates an IVFIndex.
func NewIVFIndex(dim, numCentroids, numProbes int, dtype DataType) *IVFIndex {
	if numCentroids <= 0 {
		numCentroids = 64
	}
	if numProbes <= 0 {
		numProbes = 8
	}
	if numProbes > numCentroids {
		numProbes = numCentroids
	}
	return &IVFIndex{
		Dimension:     dim,
		NumCentroids:  numCentroids,
		NumProbes:     numProbes,
		DataType:      dtype,
		InvertedLists: make([][]int, numCentroids),
	}
}

// Build trains centroids and assigns vectors to their nearest cluster.
func (idx *IVFIndex) Build(floatVectors [][]float32) error {
	n := len(floatVectors)
	if n == 0 {
		return fmt.Errorf("cannot build IVF index with 0 vectors")
	}

	numCentroids := idx.NumCentroids
	if numCentroids > n {
		numCentroids = n
		idx.NumCentroids = n
		idx.InvertedLists = make([][]int, numCentroids)
	}

	// 1. Initialize centroids by sampling vectors
	rng := rand.New(rand.NewSource(42)) // #nosec G404 -- deterministic fixed-seed PRNG for reproducible index/benchmark data; not security-sensitive
	perm := rng.Perm(n)
	idx.Centroids = make([][]float32, numCentroids)
	for c := 0; c < numCentroids; c++ {
		idx.Centroids[c] = make([]float32, idx.Dimension)
		copy(idx.Centroids[c], floatVectors[perm[c]])
	}

	// 2. Perform 3 iterations of mini-batch k-means to settle centroids
	clusterSums := make([][]float32, numCentroids)
	clusterCounts := make([]int, numCentroids)
	for iter := 0; iter < 3; iter++ {
		for c := 0; c < numCentroids; c++ {
			clusterSums[c] = make([]float32, idx.Dimension)
			clusterCounts[c] = 0
		}

		sampleCount := n
		if sampleCount > 5000 {
			sampleCount = 5000
		}
		for s := 0; s < sampleCount; s++ {
			vIdx := (s * 31) % n
			v := floatVectors[vIdx]
			bestC := 0
			bestDist := float32(math.MaxFloat32)
			for c := 0; c < numCentroids; c++ {
				d := L2Float32(v, idx.Centroids[c])
				if d < bestDist {
					bestDist = d
					bestC = c
				}
			}
			for d := 0; d < idx.Dimension; d++ {
				clusterSums[bestC][d] += v[d]
			}
			clusterCounts[bestC]++
		}

		for c := 0; c < numCentroids; c++ {
			if clusterCounts[c] > 0 {
				invCnt := 1.0 / float32(clusterCounts[c])
				for d := 0; d < idx.Dimension; d++ {
					idx.Centroids[c][d] = clusterSums[c][d] * invCnt
				}
			}
		}
	}

	// 3. Assign all vectors to nearest centroid
	for c := 0; c < numCentroids; c++ {
		idx.InvertedLists[c] = make([]int, 0, n/numCentroids)
	}

	var assignMu sync.Mutex
	numWorkers := 4
	chunk := (n + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunk
		end := start + chunk
		if start >= n {
			break
		}
		if end > n {
			end = n
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			localLists := make([][]int, numCentroids)
			for i := s; i < e; i++ {
				v := floatVectors[i]
				bestC := 0
				bestDist := float32(math.MaxFloat32)
				for c := 0; c < numCentroids; c++ {
					d := L2Float32(v, idx.Centroids[c])
					if d < bestDist {
						bestDist = d
						bestC = c
					}
				}
				localLists[bestC] = append(localLists[bestC], i)
			}

			assignMu.Lock()
			for c := 0; c < numCentroids; c++ {
				idx.InvertedLists[c] = append(idx.InvertedLists[c], localLists[c]...)
			}
			assignMu.Unlock()
		}(start, end)
	}
	wg.Wait()

	idx.Float32 = floatVectors
	return nil
}

// Search performs a multi-probe search on the IVF index.
func (idx *IVFIndex) Search(query []float32, k int) ([]SearchResult, error) {
	if len(idx.Centroids) == 0 {
		return nil, fmt.Errorf("IVF index has not been built")
	}

	// 1. Find nearest centroids (probes)
	type centroidDist struct {
		id   int
		dist float32
	}
	cDistances := make([]centroidDist, len(idx.Centroids))
	for c := 0; c < len(idx.Centroids); c++ {
		cDistances[c] = centroidDist{id: c, dist: L2Float32(query, idx.Centroids[c])}
	}

	// Sort or select top numProbes centroids
	numProbes := idx.NumProbes
	if numProbes > len(cDistances) {
		numProbes = len(cDistances)
	}
	for i := 0; i < numProbes; i++ {
		minIdx := i
		for j := i + 1; j < len(cDistances); j++ {
			if cDistances[j].dist < cDistances[minIdx].dist {
				minIdx = j
			}
		}
		cDistances[i], cDistances[minIdx] = cDistances[minIdx], cDistances[i]
	}

	// 2. Scan only vectors inside the probed inverted lists
	h := &resultHeap{}
	heap.Init(h)

	for p := 0; p < numProbes; p++ {
		cID := cDistances[p].id
		list := idx.InvertedLists[cID]

		switch idx.DataType {
		case TypeFloat32:
			for _, vID := range list {
				score := DotFloat32(query, idx.Float32[vID])
				if h.Len() < k {
					heap.Push(h, SearchResult{ID: vID, Score: score})
				} else if score > (*h)[0].Score {
					(*h)[0] = SearchResult{ID: vID, Score: score}
					heap.Fix(h, 0)
				}
			}
		case TypeUint8:
			qQuant := QuantizeUint8(query, -1)
			for _, vID := range list {
				score := DotUint8(&qQuant, &idx.Uint8[vID])
				if h.Len() < k {
					heap.Push(h, SearchResult{ID: vID, Score: score})
				} else if score > (*h)[0].Score {
					(*h)[0] = SearchResult{ID: vID, Score: score}
					heap.Fix(h, 0)
				}
			}
		case TypeComplex128:
			qComp := Float32ToComplex128(query)
			for _, vID := range list {
				score := float32(DotComplex128(qComp, idx.Complex[vID]))
				if h.Len() < k {
					heap.Push(h, SearchResult{ID: vID, Score: score})
				} else if score > (*h)[0].Score {
					(*h)[0] = SearchResult{ID: vID, Score: score}
					heap.Fix(h, 0)
				}
			}
		case TypeTurboQuant:
			rot := CreateIdentityMatrix(idx.Dimension)
			qjl := CreateRandomQJLMatrix(32, idx.Dimension, 42)
			qTQ := QuantizeTurboQuant(query, -1, rot, qjl, 32)
			for _, vID := range list {
				score := DotTurboQuant(&qTQ, &idx.Turbo[vID])
				if h.Len() < k {
					heap.Push(h, SearchResult{ID: vID, Score: score})
				} else if score > (*h)[0].Score {
					(*h)[0] = SearchResult{ID: vID, Score: score}
					heap.Fix(h, 0)
				}
			}
		}
	}

	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}

	return results, nil
}
