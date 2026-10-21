package vector

import (
	"container/heap"
	"fmt"
	"runtime"
	"sync"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
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

// heapAdd inserts a candidate into a bounded top-k min-heap (higher
// scores win; the root is the current k-th best).
func heapAdd(h *resultHeap, id int, score float32, k int) {
	if h.Len() < k {
		heap.Push(h, SearchResult{ID: id, Score: score})
	} else if score > (*h)[0].Score {
		(*h)[0] = SearchResult{ID: id, Score: score}
		heap.Fix(h, 0)
	}
}

// mergeHeaps merges worker heaps into a final top-k, preserving the
// per-worker ordering semantics.
func mergeHeaps(heaps []resultHeap, k int) []SearchResult {
	finalHeap := &resultHeap{}
	heap.Init(finalHeap)
	for _, wh := range heaps {
		for _, item := range wh {
			heapAdd(finalHeap, item.ID, item.Score, k)
		}
	}
	results := make([]SearchResult, finalHeap.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(finalHeap).(SearchResult)
	}
	return results
}

// batchScoreBlock is the block size the batch kernels are called with.
// It keeps the score buffer on a small allocation the GC never has to
// churn, and the per-call overhead (one cgo transition) amortises out
// far below the vector work.
const batchScoreBlock = 256

// FlatIndex holds vectors for exact linear scan search.
type FlatIndex struct {
	Dimension int
	DataType  DataType
	Float32   [][]float32
	Uint8     []Uint8Vector
	Complex   [][]complex128
	Turbo     []TurboQuantVector

	// Accelerated views. Prepared once per index change:
	//
	//   - f32mirror: contiguous row-major copy of the float32 rows
	//     (stride == Dimension), what the batch float32 kernels score
	//     over. The cgo pointer rules forbid pointer arrays at the
	//     call boundary, so a ragged store cannot be batched: rows
	//     shorter than dim (or a zero dimension) leave the mirror nil
	//     and the chunk search falls back to the per-vector scalar
	//     path. The mirror doubles the float32 row memory while it is
	//     held and is rebuilt after every AddFloat32; the mmap-backed
	//     store (P0-05) reads rows already contiguous and needs no
	//     copy.
	//   - complexSoA: the split real/imag planes (P0-02).
	//   - tqPacked: the packed TurboQuant columns (P0-03).
	//   - rowNorms: cached vector norms for cosine over batch kernels.
	//   - gpuSoA / gpuF32: device staged copies (cuda build tag only).
	f32mirror    []float32
	f32Uniform   bool
	rowNorms     []float32
	complexSoA   *ComplexSoA
	complexNorms []float64
	// complexCount backs Count() for indexes whose interleaved rows
	// were never materialised (mmap-loaded SoA stores). AddComplex128
	// resets it to zero.
	complexCount int
	tqPacked     *TurboPackedColumns

	gpuSoA  interface{} // holds *gpuComplexStore under the cuda tag
	gpuF32  interface{} // holds *gpuFloat32Store under the cuda tag
	gpuOnce sync.Once
	gpuFail bool // staging failed; never retried
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
	idx.f32mirror = nil
	idx.f32Uniform = false
	idx.rowNorms = nil
}

// AddUint8 adds a batch of Uint8Vectors to the index.
func (idx *FlatIndex) AddUint8(vectors []Uint8Vector) {
	idx.Uint8 = append(idx.Uint8, vectors...)
}

// AddComplex128 adds a batch of complex128 vectors to the index. The
// SoA cache is marked dirty and rebuilt on the next search (or by
// calling PrepareComplex eagerly).
func (idx *FlatIndex) AddComplex128(vectors [][]complex128) {
	idx.Complex = append(idx.Complex, vectors...)
	idx.complexSoA = nil
	idx.complexNorms = nil
	idx.complexCount = 0
}

// AddTurboQuant adds a batch of TurboQuantVectors to the index. The
// packed-column cache is marked dirty and rebuilt on the next search
// (or by calling PrepareTurboQuant eagerly).
func (idx *FlatIndex) AddTurboQuant(vectors []TurboQuantVector) {
	idx.Turbo = append(idx.Turbo, vectors...)
	idx.tqPacked = nil
}

// PrepareFloat32 builds the contiguous row-major mirror the batch
// float32 kernels score over (stride == Dimension). Ragged rows, or a
// zero dimension, leave the mirror nil and keep the index searchable
// through the per-vector path used by Search. The mirror costs 2x the
// row memory while it is held — the mmap-backed store (P0-05) reads
// rows already contiguous and never needs it.
func (idx *FlatIndex) PrepareFloat32() {
	dim := idx.Dimension
	if dim <= 0 || len(idx.Float32) == 0 {
		idx.f32mirror = nil
		idx.f32Uniform = false
		return
	}
	for i := range idx.Float32 {
		if len(idx.Float32[i]) < dim {
			idx.f32mirror = nil
			idx.f32Uniform = false
			return
		}
	}
	mirror := make([]float32, len(idx.Float32)*dim)
	for i := range idx.Float32 {
		copy(mirror[i*dim:(i+1)*dim], idx.Float32[i][:dim])
	}
	idx.f32mirror = mirror
	idx.f32Uniform = true
}

// buildRowNorms computes sqrt(dot(row,row)) once for cosine scoring.
func (idx *FlatIndex) buildRowNorms() {
	if idx.rowNorms != nil || len(idx.Float32) == 0 {
		return
	}
	norms := make([]float32, len(idx.Float32))
	for i, row := range idx.Float32 {
		norms[i] = float32(safeSqrt(float64(dotFloat32Scalar(row, row))))
	}
	idx.rowNorms = norms
}

// PrepareComplex builds the split SoA representation of the stored
// complex rows (P0-02). Ragged rows fall back to the interleaved path.
func (idx *FlatIndex) PrepareComplex() *ComplexSoA {
	if idx.complexSoA != nil || len(idx.Complex) == 0 {
		return idx.complexSoA
	}
	soa := ComplexSoAFromInterleaved(idx.Complex, idx.Dimension)
	if soa == nil {
		return nil
	}
	norms := make([]float64, len(idx.Complex))
	for i := 0; i < soa.Count; i++ {
		re, im := soa.Row(i)
		norms[i] = dotComplex128Scalars(re, im, re, im)
	}
	idx.complexSoA = soa
	idx.complexNorms = norms
	return soa
}

// PrepareTurboQuant builds the packed-column view of the stored
// TurboQuant vectors (P0-03).
func (idx *FlatIndex) PrepareTurboQuant() *TurboPackedColumns {
	if idx.tqPacked != nil || len(idx.Turbo) == 0 {
		return idx.tqPacked
	}
	idx.tqPacked = NewTurboPackedColumns(idx.Turbo, idx.Dimension)
	return idx.tqPacked
}

// Count returns the number of vectors in the index.
func (idx *FlatIndex) Count() int {
	switch idx.DataType {
	case TypeFloat32:
		return len(idx.Float32)
	case TypeUint8:
		return len(idx.Uint8)
	case TypeComplex128:
		if idx.complexCount > 0 {
			return idx.complexCount
		}
		return len(idx.Complex)
	case TypeTurboQuant:
		return len(idx.Turbo)
	default:
		return 0
	}
}

// Search performs an exact brute-force search over all vectors using
// parallel chunks. The float32, complex128 and TurboQuant paths score
// with the native batch kernels when the runtime level supports it and
// fall back to the pure-Go reference otherwise.
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

	chunkSize := (n + workers - 1) / workers
	workerHeaps := make([]resultHeap, workers)

	// Accelerated-view preparation happens on this goroutine, before
	// the workers fan out, so the caches cannot race.
	ks := kernelSetFor()
	var soa *ComplexSoA
	var packed *TurboPackedColumns
	switch idx.DataType {
	case TypeFloat32:
		if idx.f32mirror == nil {
			idx.PrepareFloat32()
			idx.buildRowNorms()
		}
		if idx.rowNorms == nil {
			idx.buildRowNorms()
		}
	case TypeComplex128:
		soa = idx.PrepareComplex()
	case TypeTurboQuant:
		packed = idx.PrepareTurboQuant()
	}

	queryRe, queryIm := splitComplexQuery(query, idx.Dimension)

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
				idx.searchFloat32Chunk(ks, h, q, s, e, k, metric)
				_ = ok

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
					heapAdd(h, idx.Uint8[i].ID, score, k)
				}

			case TypeComplex128:
				q, ok := query.([]complex128)
				if !ok {
					errOnce.Do(func() { workerErr = fmt.Errorf("expected []complex128 query") })
					return
				}
				if soa != nil && queryRe != nil {
					idx.searchComplexSoAChunk(ks, h, soa, queryRe, queryIm, s, e, k, metric)
				} else {
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
						heapAdd(h, i, score, k)
					}
				}

			case TypeTurboQuant:
				q, ok := query.(*TurboQuantVector)
				if !ok {
					errOnce.Do(func() { workerErr = fmt.Errorf("expected *TurboQuantVector query") })
					return
				}
				if packed != nil {
					prep := PrepareTurboQuantQuery(q)
					idx.searchTurboChunk(ks, h, prep, packed, s, e, k, metric)
				} else {
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
						heapAdd(h, idx.Turbo[i].ID, score, k)
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

	return mergeHeaps(workerHeaps, k), nil
}

// splitComplexQuery extracts the split form of a complex query and
// reports nil when the query is not complex of the right length.
func splitComplexQuery(query interface{}, dim int) (re, im []float64) {
	q, ok := query.([]complex128)
	if !ok {
		return nil, nil
	}
	if dim <= 0 || len(q) < dim {
		dim = len(q)
	}
	re = make([]float64, dim)
	im = make([]float64, dim)
	for i := 0; i < dim; i++ {
		re[i] = real(q[i])
		im[i] = imag(q[i])
	}
	return re, im
}

// safeSqrt guards against tiny negative values from floating error.
// The Newton-loop version this replaces broke out after a single
// iteration whenever the x/2 seed sat below the true root (every
// x < 4), returning (x+4)/4 — a ~0.5% error at x=3 and up to 25% near
// x=1. That silently desynchronised the cached cosine norms from the
// CosineFloat32 reference far beyond the 1e-4 parity budget. mathx
// uses the stdlib Sqrt here, which is exact and a single machine
// instruction.
func safeSqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return mathx.Sqrt(x)
}

// flatMetricScore adapts (metric, raw score, query norm, row norm) into
// the comparable score with the sign convention Search uses (closer /
// higher-is-better is positive; distances are negated).
func flatMetricScore(metric DistanceMetric, score float32) float32 {
	switch metric {
	case MetricL2:
		return -score
	default:
		return score
	}
}
