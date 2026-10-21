package vector

import (
	"container/heap"
	"fmt"
	"math/rand"
	"runtime"
	"sync"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

// IVFIndex implements Inverted File indexing with multi-probe search.
//
// P0-01/P0-02/P0-03 wiring: the assignment pass and the list scans run
// through the native batch / multi-query kernels (with the pure-Go
// reference as fallback and parity oracle), the query-side quantisation
// happens once per Search instead of once per probe, and the cuda build
// can stage rows and run the k-means assignment passes on the GPU
// (P0-04). The public fields and semantics are unchanged: Search scores
// by dot product over the probed lists and returns vector positions
// (not stored IDs), as in the reference implementation.
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

	// Accelerated views (built lazily, single-goroutine).
	//
	// centroidsFlat is one contiguous row-major centroid buffer and
	// Centroids holds per-row views into it, so k-means updates through
	// the views write straight into the flat backing. The multi-query
	// L2 kernels need one base pointer per call (the cgo rules forbid
	// arrays of pointers), which this layout provides without copies.
	// centroidsUniform reports that every row is Dimension wide.
	centroidsFlat    []float32
	centroidsUniform bool
	f32RowsUniform   bool
	f32RowsChecked   bool

	// f32Flat is the rows of the index in one contiguous buffer, with
	// the rows at a fixed stride. The cgo rules forbid an array of row
	// pointers, so a kernel can only address the rows through one base
	// pointer and a stride.
	//
	// The rows are stored **in the order of the inverted lists**: the
	// vectors of a list are next to each other. The scan then reads a
	// contiguous range, the prefetcher follows it, and the memory
	// bandwidth of the scan is the bandwidth of a sequential read. The
	// rows of a list are otherwise scattered over the whole index, and
	// the scan was reading twelve megabytes per query out of memory
	// that the prefetcher could not follow.
	//
	// f32Order maps a position in the flat buffer to the id of the
	// vector it holds, and f32ListStart holds the position each list
	// starts at. The ids the search returns are the original ones.
	f32Flat      []float32
	f32FlatOK    bool
	f32FlatDim   int
	f32Order     []int
	f32ListStart []int
	f32Pos       []int

	// TurboQuant query prep cache. The reference implementation
	// created the seed-42 QJL matrix per probe; caching it yields the
	// byte-identical matrix per Search.
	qjlOnce   sync.Once
	qjlMatrix []float32

	// CUDA staging (cuda build tag only; never touched elsewhere).
	gpuListMu sync.Mutex
	gpuStore  interface{}
	gpuLists  interface{}
	gpuStaged bool
	gpuFailed bool
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

// buildCentroidViews fills the contiguous centroid buffer by sampling
// the given rows and exposes the public Centroids as per-row views, so
// k-means updates through the views write straight into the flat
// backing the multi-query L2 kernels read.
func (idx *IVFIndex) buildCentroidViews(floatVectors [][]float32, perm []int, numCentroids int) {
	dim := idx.Dimension
	flat := make([]float32, numCentroids*dim)
	idx.Centroids = make([][]float32, numCentroids)
	for c := 0; c < numCentroids; c++ {
		row := flat[c*dim : (c+1)*dim]
		idx.Centroids[c] = row
		copy(row, floatVectors[perm[c]])
	}
	idx.centroidsFlat = flat
	idx.centroidsUniform = dim > 0
}

// buildListPtrs was removed with the pointer-array design: the float32
// list scans now gather row blocks into contiguous scratch (see
// searchFloat32List below), which needs no cached pointer arrays.

// qjlMatrixCached returns the seed-42 QJL matrix used by the TurboQuant
// query prep (created once per index instead of once per probe).
func (idx *IVFIndex) qjlMatrixCached() []float32 {
	idx.qjlOnce.Do(func() {
		idx.qjlMatrix = CreateRandomQJLMatrix(32, idx.Dimension, 42)
	})
	return idx.qjlMatrix
}

// gpuTryStage stages the float32 rows for the CUDA build. It always
// returns false quickly when the hooks are absent (non-cuda builds) or
// after any failure; callers must not retry after a failure.
func (idx *IVFIndex) gpuTryStage(floatVectors [][]float32) bool {
	if gpuStageFloat32 == nil {
		return false
	}
	idx.gpuListMu.Lock()
	defer idx.gpuListMu.Unlock()
	if idx.gpuFailed {
		return false
	}
	if idx.gpuStaged {
		return true
	}
	store, ok := gpuStageFloat32(floatVectors, idx.Dimension)
	if !ok {
		idx.gpuFailed = true
		return false
	}
	idx.gpuStore = store
	idx.gpuStaged = true
	return true
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

	// 1. Initialize centroids by sampling vectors. The rows are views
	// into one contiguous buffer (centroidsFlat) so the multi-query L2
	// kernels score against a single base pointer.
	rng := rand.New(rand.NewSource(42)) // #nosec G404 -- deterministic fixed-seed PRNG for reproducible index/benchmark data; not security-sensitive
	perm := rng.Perm(n)
	idx.buildCentroidViews(floatVectors, perm, numCentroids)

	// Mini-batch sample positions: identical stride to the reference
	// implementation so both k-means paths walk the same samples.
	sampleCount := n
	if sampleCount > 5000 {
		sampleCount = 5000
	}
	sampleIdx := make([]int32, sampleCount)
	for s := 0; s < sampleCount; s++ {
		sampleIdx[s] = int32((s * 31) % n)
	}

	// 2. Mini-batch k-means (3 iterations). The CUDA build runs the
	// assignment passes on the device when the rows staged there;
	// otherwise the CPU path runs through the multi-query L2 kernels.
	gpuOK := idx.gpuTryStage(floatVectors) &&
		gpuKmeansAssign != nil && gpuFullAssign != nil
	if gpuOK {
		if !idx.gpuKmeans(sampleIdx, numCentroids) {
			gpuOK = false
		}
	}
	if !gpuOK {
		idx.cpuKmeans(floatVectors, sampleIdx, numCentroids)
	}

	// 3. Assign all vectors to nearest centroid.
	for c := 0; c < numCentroids; c++ {
		idx.InvertedLists[c] = make([]int, 0, n/numCentroids)
	}

	assigned := false
	if gpuOK {
		if best, ok := gpuFullAssign(idx.gpuStore, idx.Dimension, idx.Centroids); ok {
			for i, b := range best {
				if int(b) >= 0 && int(b) < numCentroids {
					idx.InvertedLists[b] = append(idx.InvertedLists[b], i)
				}
			}
			assigned = true
		}
	}
	if !assigned {
		idx.cpuAssignAll(floatVectors, numCentroids)
	}

	idx.Float32 = floatVectors

	// Record once whether the stored rows are scannable through the
	// gather+batch path (uniform dimension). Build is single-goroutine.
	idx.f32RowsUniform = idx.Dimension > 0
	for i := range idx.Float32 {
		if len(idx.Float32[i]) < idx.Dimension {
			idx.f32RowsUniform = false
			break
		}
	}

	// Build the contiguous copy the gather kernels address. The rows
	// themselves become views into it, so the data is stored once.
	idx.buildFlatFloat32()

	// Stage the freshly built lists for the CUDA scan hook.
	if idx.gpuStaged && gpuStageLists != nil {
		idx.gpuListMu.Lock()
		idx.gpuLists, _ = gpuStageLists(idx.gpuStore, idx.InvertedLists)
		idx.gpuListMu.Unlock()
	}
	return nil
}

// buildFlatFloat32 copies the rows into one contiguous buffer, in the
// order of the inverted lists, and makes the rows views into it.
//
// The copy is the price of the layout: the cgo rules forbid passing an
// array of row pointers to C, so a kernel can only address the rows
// through one base pointer and a stride. The copy is made once, at build
// time.
//
// The order is the point. The vectors of a list are next to each other in
// the buffer, so the scan of a list reads a contiguous range: the
// prefetcher follows it and the scan runs at the bandwidth of a
// sequential read. The ids the search returns are the original ones, and
// f32Order maps a position in the buffer back to the vector it holds.
func (idx *IVFIndex) buildFlatFloat32() {
	idx.f32Flat = nil
	idx.f32FlatOK = false
	idx.f32FlatDim = 0
	idx.f32Order = nil
	idx.f32ListStart = nil

	dim := idx.Dimension
	if dim <= 0 || len(idx.Float32) == 0 {
		return
	}
	for i := range idx.Float32 {
		if len(idx.Float32[i]) < dim {
			return
		}
	}

	// The order: the lists, one after the other.
	order := make([]int, 0, len(idx.Float32))
	starts := make([]int, len(idx.InvertedLists)+1)
	seen := make([]bool, len(idx.Float32))
	for c, list := range idx.InvertedLists {
		starts[c] = len(order)
		for _, id := range list {
			if id < 0 || id >= len(idx.Float32) || seen[id] {
				continue
			}
			seen[id] = true
			order = append(order, id)
		}
	}
	starts[len(idx.InvertedLists)] = len(order)

	// A vector that is in no list goes at the end: the search never
	// reaches it, but the store has to hold every row.
	for i := range idx.Float32 {
		if !seen[i] {
			order = append(order, i)
		}
	}

	flat := make([]float32, len(order)*dim)
	rows := make([][]float32, len(order))
	pos := make([]int, len(idx.Float32))
	for i, id := range order {
		row := flat[i*dim : (i+1)*dim : (i+1)*dim]
		copy(row, idx.Float32[id][:dim])
		rows[i] = row
		pos[id] = i
	}

	idx.Float32 = rows
	idx.f32Flat = flat
	idx.f32FlatDim = dim
	idx.f32FlatOK = true
	idx.f32Order = order
	idx.f32ListStart = starts
	idx.f32Pos = pos
}

// float32Row returns the row of a vector id.
//
// The rows are stored in the order of the inverted lists, so the position
// of a vector in the store is not its id. The mapping is what turns one
// into the other.
func (idx *IVFIndex) float32Row(id int) []float32 {
	if id < 0 || id >= len(idx.f32Pos) {
		return nil
	}
	pos := idx.f32Pos[id]
	if pos < 0 || pos >= len(idx.Float32) {
		return nil
	}
	return idx.Float32[pos]
}

// gpuKmeans runs the mini-batch iterations on the device. Returns false
// to fall back to the CPU path on any rejection.
func (idx *IVFIndex) gpuKmeans(sampleIdx []int32, numCentroids int) bool {
	dim := idx.Dimension
	for iter := 0; iter < 3; iter++ {
		sums := make([]float32, numCentroids*dim)
		counts := make([]int32, numCentroids)
		if !gpuKmeansAssign(idx.gpuStore, sampleIdx, dim, idx.Centroids, sums, counts) {
			return false
		}
		for c := 0; c < numCentroids; c++ {
			if counts[c] > 0 {
				invCnt := 1.0 / float32(counts[c])
				base := c * dim
				for d := 0; d < dim; d++ {
					idx.Centroids[c][d] = sums[base+d] * invCnt
				}
			}
		}
	}
	return true
}

// kmeansAcc holds one worker's partial mini-batch accumulators.
type kmeansAcc struct {
	sums   []float32
	counts []int32
}

// scoreSampleCentroids returns the nearest-centroid index for a block
// of samples: best[si] is the centroid with minimal squared L2. It
// returns false when the caller must use the scalar loop (no multi-query
// kernel, ragged centroids, mismatching dims).
//
// Orientation: every sample is a "query" (numQ = m in the kernel
// contract) and the centroids are the scored rows (count =
// numCentroids); out holds m * numCentroids squared distances.
func (idx *IVFIndex) scoreSampleCentroids(vs [][]float32, best []int, dim, numCentroids int, ks *distanceKernelSet, qs []float32, out []float32) bool {
	m := len(vs)
	if m == 0 || best == nil || len(best) < m {
		return false
	}
	if ks == nil || ks.mqL2Float32 == nil || !idx.centroidsUniform ||
		idx.centroidsFlat == nil || dim <= 0 || numCentroids <= 0 ||
		len(qs) < m*dim || len(out) < m*numCentroids ||
		len(idx.centroidsFlat) < numCentroids*dim {
		return false
	}
	for si := 0; si < m; si++ {
		if len(vs[si]) != dim {
			return false
		}
		copy(qs[si*dim:(si+1)*dim], vs[si])
	}
	if !ks.mqL2Float32(qs, m, idx.centroidsFlat, dim, numCentroids, out) {
		return false
	}
	for si := 0; si < m; si++ {
		base := si * numCentroids
		bestC := -1
		bestDist := float32(0)
		for c := 0; c < numCentroids; c++ {
			d := out[base+c]
			if bestC < 0 || d < bestDist {
				bestC = c
				bestDist = d
			}
		}
		best[si] = bestC
	}
	return true
}

// cpuKmeans is the parallel CPU mini-batch k-means: identical sample
// stride and iteration count as the reference implementation. The
// multi-query L2 kernels replace the two-level scalar loop when the
// runtime level provides them.
func (idx *IVFIndex) cpuKmeans(floatVectors [][]float32, sampleIdx []int32, numCentroids int) {
	dim := idx.Dimension
	ks := kernelSetFor()

	for iter := 0; iter < 3; iter++ {
		workers := runtime.GOMAXPROCS(0)
		if workers > len(sampleIdx) {
			workers = len(sampleIdx)
		}
		if workers == 0 {
			continue
		}
		chunk := (len(sampleIdx) + workers - 1) / workers
		accs := make([]kmeansAcc, workers)
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			s := w * chunk
			e := s + chunk
			if s >= len(sampleIdx) {
				break
			}
			if e > len(sampleIdx) {
				e = len(sampleIdx)
			}
			wg.Add(1)
			go func(s, e, w int) {
				defer wg.Done()
				acc := kmeansAcc{
					sums:   make([]float32, numCentroids*dim),
					counts: make([]int32, numCentroids),
				}
				accs[w] = acc

				// Block scratch: 256 samples at a time against all
				// centroids in one multi-query call.
				const mqBlock = 256
				best := make([]int, mqBlock)
				qs := make([]float32, mqBlock*dim)
				out := make([]float32, mqBlock*numCentroids)

				for bi := s; bi < e; bi += mqBlock {
					be := bi + mqBlock
					if be > e {
						be = e
					}
					block := make([][]float32, be-bi)
					for si := bi; si < be; si++ {
						block[si-bi] = floatVectors[sampleIdx[si]]
					}
					if len(block) > 0 && idx.scoreSampleCentroids(block, best, dim, numCentroids, ks, qs, out) {
						for si := bi; si < be; si++ {
							idx.kmeansAdd(acc.sums, acc.counts, best[si-bi], dim, block[si-bi])
						}
						continue
					}
					for si := bi; si < be; si++ {
						v := block[si-bi]
						bestC := 0
						bestDist := float32(mathx.MaxFloat32)
						for c := 0; c < numCentroids; c++ {
							d := l2Float32Scalar(v, idx.Centroids[c])
							if d < bestDist {
								bestDist = d
								bestC = c
							}
						}
						idx.kmeansAdd(acc.sums, acc.counts, bestC, dim, v)
					}
				}
			}(s, e, w)
		}
		wg.Wait()

		// Fold the worker accumulators, then update centroids exactly
		// like the reference implementation.
		var sums []float32
		counts := make([]int32, numCentroids)
		for w := range accs {
			if accs[w].counts == nil {
				continue
			}
			for c := 0; c < numCentroids; c++ {
				counts[c] += accs[w].counts[c]
			}
		}
		sums = make([]float32, numCentroids*dim)
		for w := range accs {
			if accs[w].sums == nil {
				continue
			}
			for i := range sums {
				sums[i] += accs[w].sums[i]
			}
		}
		for c := 0; c < numCentroids; c++ {
			if counts[c] > 0 {
				invCnt := 1.0 / float32(counts[c])
				base := c * dim
				for d := 0; d < dim; d++ {
					idx.Centroids[c][d] = sums[base+d] * invCnt
				}
			}
		}
	}
}

// kmeansAdd folds one sample into a worker's local accumulators.
func (idx *IVFIndex) kmeansAdd(sums []float32, counts []int32, bestC, dim int, v []float32) {
	if bestC < 0 || bestC >= len(counts) {
		return
	}
	base := bestC * dim
	for d := 0; d < dim; d++ {
		sums[base+d] += v[d]
	}
	counts[bestC]++
}

// cpuAssignAll assigns every vector to its nearest centroid through the
// multi-query L2 kernels (fallback: the scalar reference). Workers fill
// local lists; the merge runs in worker id order after the join, which
// keeps every vector in exactly one list (the reference merged under a
// mutex in completion order, so contents are identical and the order is
// now deterministic).
func (idx *IVFIndex) cpuAssignAll(floatVectors [][]float32, numCentroids int) {
	n := len(floatVectors)
	dim := idx.Dimension
	ks := kernelSetFor()

	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	localAll := make([][][]int, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if start >= n {
			break
		}
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(s, e, w int) {
			defer wg.Done()
			localLists := make([][]int, numCentroids)
			localAll[w] = localLists

			const mqBlock = 256
			best := make([]int, mqBlock)
			qs := make([]float32, mqBlock*dim)
			out := make([]float32, mqBlock*numCentroids)

			for bi := s; bi < e; bi += mqBlock {
				be := bi + mqBlock
				if be > e {
					be = e
				}
				block := make([][]float32, be-bi)
				for si := bi; si < be; si++ {
					block[si-bi] = floatVectors[si]
				}
				if len(block) > 0 && idx.scoreSampleCentroids(block, best, dim, numCentroids, ks, qs, out) {
					for si := bi; si < be; si++ {
						bestC := best[si-bi]
						if bestC < 0 || bestC >= numCentroids {
							bestC = 0
						}
						localLists[bestC] = append(localLists[bestC], si)
					}
					continue
				}
				for si := bi; si < be; si++ {
					v := block[si-bi]
					bestC := 0
					bestDist := float32(mathx.MaxFloat32)
					for c := 0; c < numCentroids; c++ {
						d := l2Float32Scalar(v, idx.Centroids[c])
						if d < bestDist {
							bestDist = d
							bestC = c
						}
					}
					localLists[bestC] = append(localLists[bestC], si)
				}
			}
		}(start, end, w)
	}
	wg.Wait()

	for w := range localAll {
		if localAll[w] == nil {
			continue
		}
		for c := 0; c < numCentroids; c++ {
			idx.InvertedLists[c] = append(idx.InvertedLists[c], localAll[w][c]...)
		}
	}
}

// Search performs a multi-probe search on the IVF index.
func (idx *IVFIndex) Search(query []float32, k int) ([]SearchResult, error) {
	if len(idx.Centroids) == 0 {
		return nil, fmt.Errorf("IVF index has not been built")
	}
	if k <= 0 {
		return make([]SearchResult, 0), nil
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

	// Sort or select top numProbes centroids (same selection network as
	// the reference implementation).
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

	probedIDs := make([]int32, numProbes)
	for p := 0; p < numProbes; p++ {
		probedIDs[p] = int32(cDistances[p].id)
	}

	// GPU list scan (cuda build; one launch covers all probed lists).
	if res, ok := idx.gpuSearchLists(query, k, probedIDs); ok {
		return res, nil
	}

	// 2. Scan the probed inverted lists. Query-side quantisation
	// happens once per Search (the reference re-did it per probe).
	ks := kernelSetFor()

	switch idx.DataType {
	case TypeComplex128:
		qComp := Float32ToComplex128(query)
		probeHeaps := make([]resultHeap, numProbes)
		idx.runProbes(numProbes, probedIDs, k, func(p, cID int, h *resultHeap) {
			for _, vID := range idx.InvertedLists[cID] {
				heapAdd(h, vID, float32(DotComplex128(qComp, idx.Complex[vID])), k)
			}
		}, probeHeaps)
		return mergeHeaps(probeHeaps, k), nil
	case TypeUint8:
		qQuant := QuantizeUint8(query, -1)
		probeHeaps := make([]resultHeap, numProbes)
		idx.runProbes(numProbes, probedIDs, k, func(p, cID int, h *resultHeap) {
			for _, vID := range idx.InvertedLists[cID] {
				heapAdd(h, vID, DotUint8(&qQuant, &idx.Uint8[vID]), k)
			}
		}, probeHeaps)
		return mergeHeaps(probeHeaps, k), nil
	case TypeTurboQuant:
		rot := CreateIdentityMatrix(idx.Dimension)
		qjl := idx.qjlMatrixCached()
		qTQ := QuantizeTurboQuant(query, -1, rot, qjl, 32)
		tqPrep := PrepareTurboQuantQuery(&qTQ)
		probeHeaps := make([]resultHeap, numProbes)
		idx.runProbes(numProbes, probedIDs, k, func(p, cID int, h *resultHeap) {
			for _, vID := range idx.InvertedLists[cID] {
				if vID < 0 || vID >= len(idx.Turbo) {
					continue
				}
				heapAdd(h, vID, scoreTurboPair(ks, tqPrep, &idx.Turbo[vID]), k)
			}
		}, probeHeaps)
		return mergeHeaps(probeHeaps, k), nil
	default:
		// float32 and unknown dtypes: gather-block batch scan. Rows of
		// each probed list are copied (≤ batchScoreBlock at a time)
		// into per-goroutine contiguous scratch and scored with one
		// batch-kernel call per block; the scratch is pooled, so
		// steady-state scans allocate nothing. Lists with out-of-range
		// ids, a ragged store, or a query longer than the stored dim
		// fall back to the scalar per-vector loop, matching the
		// reference result for result.
		probeHeaps := make([]resultHeap, numProbes)
		dim := idx.Dimension
		gatherable := idx.f32RowsUniform && dim > 0 && len(query) <= dim
		var scratch sync.Pool
		if gatherable {
			scratch.New = func() interface{} {
				return make([]float32, batchScoreBlock*dim)
			}
		}
		idx.runProbes(numProbes, probedIDs, k, func(p, cID int, h *resultHeap) {
			list := idx.InvertedLists[cID]
			if len(list) == 0 {
				return
			}
			if gatherable && idx.listIdsInBounds(list) {
				// The rows of a list are next to each other in the
				// store, so the scan of a list is a contiguous read:
				// the prefetcher follows it. The rows used to be
				// scattered over the whole index, and the scan read
				// twelve megabytes per query out of memory it could
				// not follow.
				if idx.f32FlatOK {
					idx.searchFloat32Contiguous(ks, h, query, cID, dim, k)
					return
				}
				g := scratch.Get().([]float32)
				idx.searchFloat32Gather(ks, h, query, list, g, dim, k)
				scratch.Put(g)
				return
			}
			for _, vID := range list {
				row := idx.float32Row(vID)
				if row == nil {
					continue
				}
				heapAdd(h, vID, DotFloat32(query, row), k)
			}
		}, probeHeaps)
		return mergeHeaps(probeHeaps, k), nil
	}
}

// listIdsInBounds reports that every id of the list indexes Float32.
func (idx *IVFIndex) listIdsInBounds(list []int) bool {
	for _, vID := range list {
		if vID < 0 || vID >= len(idx.Float32) {
			return false
		}
	}
	return true
}

// searchFloat32Contiguous scores one probed list out of the contiguous
// store.
//
// The rows of a list are next to each other in the store, so the scan is
// a contiguous read and the prefetcher follows it. The scores land in the
// heap with the original ids, so the heap contents are identical to the
// per-vector reference loop.
func (idx *IVFIndex) searchFloat32Contiguous(ks *distanceKernelSet, h *resultHeap, query []float32, cID, dim, k int) {
	list := idx.InvertedLists[cID]
	if len(list) == 0 {
		return
	}
	start := idx.f32ListStart[cID]
	if start < 0 || start+len(list) > len(idx.f32Order) {
		return
	}

	var buf [batchScoreBlock]float32
	for offset := 0; offset < len(list); offset += batchScoreBlock {
		count := len(list) - offset
		if count > batchScoreBlock {
			count = batchScoreBlock
		}
		pos := start + offset
		rows := idx.f32Flat[pos*dim : (pos+count)*dim]
		// The batch scorer falls back to the scalar reference when the
		// native path rejects the shape.
		scoreFloat32Batch(ks, true, query, rows, dim, count, buf[:])
		for j := 0; j < count; j++ {
			heapAdd(h, idx.f32Order[pos+j], buf[j], k)
		}
	}
}

// searchFloat32Gather scores one probed list through the gather+batch
// path. Rows are copied into gather (contiguous, stride dim) in blocks
// of batchScoreBlock; each block is scored with one kernel call and the
// scores land in the heap in list order, so the heap contents are
// identical to the per-vector reference loop.
func (idx *IVFIndex) searchFloat32Gather(ks *distanceKernelSet, h *resultHeap, query []float32, list []int, gather []float32, dim, k int) {
	var buf [batchScoreBlock]float32
	for start := 0; start < len(list); start += batchScoreBlock {
		end := start + batchScoreBlock
		if end > len(list) {
			end = len(list)
		}
		count := end - start
		for j := 0; j < count; j++ {
			row := idx.float32Row(list[start+j])
			if row == nil {
				continue
			}
			copy(gather[j*dim:(j+1)*dim], row[:dim])
		}
		scoreFloat32Batch(ks, true, query, gather[:count*dim], dim, count, buf[:])
		for j, sc := range buf[:count] {
			heapAdd(h, list[start+j], sc, k)
		}
	}
}

// runProbes fans the probe scans out over goroutines (one per probe,
// capped at NumProbes of work at a time) and collects every probe's
// heap. scan scores a single probed list into h and is data-type
// specific.
func (idx *IVFIndex) runProbes(numProbes int, probedIDs []int32, k int, scan func(p, cID int, h *resultHeap), probeHeaps []resultHeap) {
	var wg sync.WaitGroup
	// Cap the concurrency at the worker count; probes beyond that just
	// share the pool (the heap merge at the end is order-independent).
	limit := runtime.GOMAXPROCS(0)
	if limit > numProbes {
		limit = numProbes
	}
	sem := make(chan struct{}, limit)
	for p := 0; p < numProbes; p++ {
		cID := int(probedIDs[p])
		if cID < 0 || cID >= len(idx.InvertedLists) {
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(p, cID int) {
			defer wg.Done()
			defer func() { <-sem }()
			h := &resultHeap{}
			heap.Init(h)
			scan(p, cID, h)
			probeHeaps[p] = *h
		}(p, cID)
	}
	wg.Wait()
}

// gpuSearchLists scores the probed lists with one CUDA launch when the
// index staged for it. Falls back silently (ok=false) otherwise.
func (idx *IVFIndex) gpuSearchLists(query []float32, k int, probedIDs []int32) ([]SearchResult, bool) {
	if len(probedIDs) == 0 || k <= 0 || gpuScanListsFloat32 == nil || gpuQPackedFloat32 == nil {
		return nil, false
	}
	idx.gpuListMu.Lock()
	staged := idx.gpuStore != nil && idx.gpuLists != nil
	idx.gpuListMu.Unlock()
	if !staged {
		return nil, false
	}
	qs, ok := gpuQPackedFloat32([][]float32{query})
	if !ok {
		return nil, false
	}
	scores, counts, ok := gpuScanListsFloat32(idx.gpuStore, idx.gpuLists, qs, probedIDs)
	if !ok {
		return nil, false
	}
	offset := 0
	h := &resultHeap{}
	heap.Init(h)
	for i, cID := range probedIDs {
		count := int(counts[i])
		if int(cID) < len(idx.InvertedLists) {
			list := idx.InvertedLists[cID]
			for j := 0; j < count && j < len(list); j++ {
				heapAdd(h, list[j], scores[offset+j], k)
			}
		}
		offset += count
	}
	results := make([]SearchResult, h.Len())
	for ri := len(results) - 1; ri >= 0; ri-- {
		results[ri] = heap.Pop(h).(SearchResult)
	}
	return results, true
}
