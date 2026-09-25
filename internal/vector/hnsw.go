package vector

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
)

// HNSWNode represents a node in the HNSW graph.
type HNSWNode struct {
	ID        int
	Neighbors [][]int // Per-layer list of neighbor IDs
}

// HNSWIndex implements Hierarchical Navigable Small World graph search.
type HNSWIndex struct {
	Dimension int
	M         int     // Max outgoing edges per node
	M0        int     // Max outgoing edges on layer 0
	EfSearch  int     // Size of dynamic candidate list during search
	EfConst   int     // Size of dynamic candidate list during construction
	Ml        float64 // Normalization factor for layer assignment
	MaxLayer  int
	EnterNode int
	DataType  DataType
	Nodes     []*HNSWNode

	// Vector stores
	Float32 [][]float32
	Uint8   []Uint8Vector
	Complex [][]complex128
	Turbo   []TurboQuantVector
}

// NewHNSWIndex creates an empty HNSWIndex.
func NewHNSWIndex(dim, m, efSearch int, dtype DataType) *HNSWIndex {
	if m <= 0 {
		m = 16
	}
	if efSearch <= 0 {
		efSearch = 32
	}
	return &HNSWIndex{
		Dimension: dim,
		M:         m,
		M0:        2 * m,
		EfSearch:  efSearch,
		EfConst:   40,
		Ml:        1.0 / math.Log(float64(m)),
		MaxLayer:  -1,
		EnterNode: -1,
		DataType:  dtype,
		Nodes:     make([]*HNSWNode, 0),
	}
}

// computeDistance computes distance between vector node i and query.
func (h *HNSWIndex) computeDistance(nodeID int, query []float32) float32 {
	switch h.DataType {
	case TypeFloat32:
		return L2Float32(query, h.Float32[nodeID])
	case TypeUint8:
		qU8 := QuantizeUint8(query, -1)
		return L2Uint8(&qU8, &h.Uint8[nodeID])
	case TypeComplex128:
		qC := Float32ToComplex128(query)
		return float32(L2Complex128(qC, h.Complex[nodeID]))
	case TypeTurboQuant:
		rot := CreateIdentityMatrix(h.Dimension)
		qjl := CreateRandomQJLMatrix(32, h.Dimension, 42)
		qTQ := QuantizeTurboQuant(query, -1, rot, qjl, 32)
		return L2TurboQuant(&qTQ, &h.Turbo[nodeID])
	default:
		return L2Float32(query, h.Float32[nodeID])
	}
}

// computeDistanceNodes computes distance between node i and node j.
func (h *HNSWIndex) computeDistanceNodes(i, j int) float32 {
	if len(h.Float32) > i && len(h.Float32) > j {
		return L2Float32(h.Float32[i], h.Float32[j])
	}
	return 0
}

// Build constructs the HNSW graph over the provided float32 vectors.
func (h *HNSWIndex) Build(vectors [][]float32) error {
	n := len(vectors)
	if n == 0 {
		return fmt.Errorf("cannot build HNSW with 0 vectors")
	}

	h.Float32 = vectors
	h.Nodes = make([]*HNSWNode, n)
	rng := rand.New(rand.NewSource(12345)) // #nosec G404 -- deterministic fixed-seed PRNG for reproducible index/benchmark data; not security-sensitive

	for i := 0; i < n; i++ {
		// Sample layer for node
		layer := 0
		r := rng.Float64()
		if r > 0 {
			layer = int(-math.Log(r) * h.Ml)
		}
		if layer > 16 {
			layer = 16
		}

		node := &HNSWNode{
			ID:        i,
			Neighbors: make([][]int, layer+1),
		}
		for l := 0; l <= layer; l++ {
			node.Neighbors[l] = make([]int, 0, h.M0)
		}
		h.Nodes[i] = node

		if h.EnterNode == -1 {
			h.EnterNode = i
			h.MaxLayer = layer
			continue
		}

		// Connect node into existing graph
		currObj := h.EnterNode
		topL := h.MaxLayer

		// 1. Greedily navigate down to insertion layer
		for l := topL; l > layer; l-- {
			changed := true
			for changed {
				changed = false
				currDist := h.computeDistanceNodes(currObj, i)
				for _, neighbor := range h.Nodes[currObj].Neighbors[l] {
					d := h.computeDistanceNodes(neighbor, i)
					if d < currDist {
						currDist = d
						currObj = neighbor
						changed = true
					}
				}
			}
		}

		// 2. Connect in layers up to insertion layer
		for l := int(math.Min(float64(layer), float64(topL))); l >= 0; l-- {
			maxConn := h.M
			if l == 0 {
				maxConn = h.M0
			}

			// Add bidirectional edges
			node.Neighbors[l] = append(node.Neighbors[l], currObj)
			if len(h.Nodes[currObj].Neighbors[l]) < maxConn {
				h.Nodes[currObj].Neighbors[l] = append(h.Nodes[currObj].Neighbors[l], i)
			}
		}

		if layer > h.MaxLayer {
			h.MaxLayer = layer
			h.EnterNode = i
		}
	}

	return nil
}

// Search queries the HNSW graph using greedy beam search.
func (h *HNSWIndex) Search(query []float32, k int) ([]SearchResult, error) {
	if h.EnterNode == -1 || len(h.Nodes) == 0 {
		return nil, fmt.Errorf("HNSW index is empty")
	}

	currObj := h.EnterNode
	currDist := h.computeDistance(currObj, query)

	// 1. Navigate from top layer down to layer 1
	for l := h.MaxLayer; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range h.Nodes[currObj].Neighbors[l] {
				d := h.computeDistance(neighbor, query)
				if d < currDist {
					currDist = d
					currObj = neighbor
					changed = true
				}
			}
		}
	}

	// 2. Search layer 0 with dynamic candidate queue (efSearch)
	type candidate struct {
		id   int
		dist float32
	}

	visited := make(map[int]bool)
	visited[currObj] = true

	candidates := []candidate{{id: currObj, dist: currDist}}
	wPool := []candidate{{id: currObj, dist: currDist}}

	for len(candidates) > 0 {
		// Pop closest candidate
		minIdx := 0
		for idx, c := range candidates {
			if c.dist < candidates[minIdx].dist {
				minIdx = idx
			}
		}
		c := candidates[minIdx]
		candidates = append(candidates[:minIdx], candidates[minIdx+1:]...)

		// Furthest element in wPool
		maxWIdx := 0
		for idx, w := range wPool {
			if w.dist > wPool[maxWIdx].dist {
				maxWIdx = idx
			}
		}

		if c.dist > wPool[maxWIdx].dist && len(wPool) >= h.EfSearch {
			break
		}

		for _, neighbor := range h.Nodes[c.id].Neighbors[0] {
			if !visited[neighbor] {
				visited[neighbor] = true
				d := h.computeDistance(neighbor, query)

				if d < wPool[maxWIdx].dist || len(wPool) < h.EfSearch {
					candidates = append(candidates, candidate{id: neighbor, dist: d})
					wPool = append(wPool, candidate{id: neighbor, dist: d})

					if len(wPool) > h.EfSearch {
						// Remove furthest element
						newMaxIdx := 0
						for idx, w := range wPool {
							if w.dist > wPool[newMaxIdx].dist {
								newMaxIdx = idx
							}
						}
						wPool = append(wPool[:newMaxIdx], wPool[newMaxIdx+1:]...)
						maxWIdx = 0
						for idx, w := range wPool {
							if w.dist > wPool[maxWIdx].dist {
								maxWIdx = idx
							}
						}
					}
				}
			}
		}
	}

	// Extract top-k from wPool
	rH := &resultHeap{}
	heap.Init(rH)

	for _, w := range wPool {
		// Use negative distance for score so closest is highest
		score := -w.dist
		if rH.Len() < k {
			heap.Push(rH, SearchResult{ID: w.id, Score: score})
		} else if score > (*rH)[0].Score {
			(*rH)[0] = SearchResult{ID: w.id, Score: score}
			heap.Fix(rH, 0)
		}
	}

	results := make([]SearchResult, rH.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(rH).(SearchResult)
	}

	return results, nil
}
