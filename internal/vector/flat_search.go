package vector

import (
	"fmt"
	"runtime"
	"sync"
)

// flat_search.go carries the scoring loops of FlatIndex.
//
// Search splits rows over workers; each worker scores its range in
// blocks of batchScoreBlock rows through the native batch kernels (one
// cgo transition per block) and folds the scores into a bounded heap,
// which reproduces the per-row heap semantics of the reference
// implementation bit for bit.
//
// BatchSearch is the shared-scan path for many queries at once: the
// multi-query kernels score a whole row block for every query in one
// call, so the memory streams once for the whole batch instead of once
// per query. That is what carries the complex128 flat target (>500 QPS
// at 250k x 384): a single fp64 row sweep is memory-bound, and sharing
// it across 50 queries divides the sweep cost by the batch size.

// fixed score buffers; batchScoreBlock keeps these stack-sized so the
// hot paths do not allocate or churn the GC.

// searchFloat32Chunk scores rows [s, e) of Float32 against q and folds
// the top-k into h. Mirrors the reference loop in flat.go:
// dot / cosine (normalised) / L2 (negated). With a usable mirror the
// rows are scored in blocks of batchScoreBlock through the native batch
// kernels (one cgo transition per block); the blocks are slices of the
// contiguous mirror with stride == Dimension, the exact layout the
// mmap-backed store (P0-05) streams. Ragged rows, a zero dimension or
// a query longer than it fall back to the per-vector path, which is
// exactly the reference behaviour.
func (idx *FlatIndex) searchFloat32Chunk(ks *distanceKernelSet, h *resultHeap, q []float32, s, e, k int, metric DistanceMetric) {
	stride := idx.Dimension
	uniform := idx.f32Uniform && stride > 0 && idx.f32mirror != nil && len(q) <= stride
	if !uniform {
		for i := s; i < e; i++ {
			row := idx.Float32[i]
			var score float32
			switch metric {
			case MetricDot:
				score = DotFloat32(q, row)
			case MetricCosine:
				score = CosineFloat32(q, row)
			case MetricL2:
				score = -L2Float32(q, row)
			default:
				score = DotFloat32(q, row)
			}
			heapAdd(h, i, score, k)
		}
		return
	}
	switch metric {
	case MetricDot:
		var buf [batchScoreBlock]float32
		for start := s; start < e; start += batchScoreBlock {
			end := start + batchScoreBlock
			if end > e {
				end = e
			}
			scoreFloat32Batch(ks, true, q, idx.f32mirror[start*stride:], stride, end-start, buf[:])
			for j, sc := range buf[:end-start] {
				heapAdd(h, start+j, sc, k)
			}
		}
	case MetricL2:
		var buf [batchScoreBlock]float32
		for start := s; start < e; start += batchScoreBlock {
			end := start + batchScoreBlock
			if end > e {
				end = e
			}
			scoreFloat32Batch(ks, false, q, idx.f32mirror[start*stride:], stride, end-start, buf[:])
			for j, sc := range buf[:end-start] {
				heapAdd(h, start+j, flatMetricScore(MetricL2, sc), k)
			}
		}
	case MetricCosine:
		// Norms were prepared with the scalar reference on the search
		// goroutine; cosine keeps the dot kernel and normalises here,
		// which matches the reference within float error far inside
		// the 1e-4 parity budget.
		if idx.rowNorms == nil || len(idx.rowNorms) != len(idx.Float32) {
			// Norm cache unusable (ragged rebuild raced): fall back to
			// the scalar cosine path, which needs no norms.
			for i := s; i < e; i++ {
				heapAdd(h, i, CosineFloat32(q, idx.Float32[i]), k)
			}
			return
		}
		qNorm := float32(safeSqrt(float64(dotFloat32Scalar(q, q))))
		var buf [batchScoreBlock]float32
		for start := s; start < e; start += batchScoreBlock {
			end := start + batchScoreBlock
			if end > e {
				end = e
			}
			scoreFloat32Batch(ks, true, q, idx.f32mirror[start*stride:], stride, end-start, buf[:])
			for j, sc := range buf[:end-start] {
				heapAdd(h, start+j, cosineFromParts(sc, qNorm, idx.rowNorms[start+j]), k)
			}
		}
	default:
		// Unknown metric: the reference path scores dot.
		var buf [batchScoreBlock]float32
		for start := s; start < e; start += batchScoreBlock {
			end := start + batchScoreBlock
			if end > e {
				end = e
			}
			scoreFloat32Batch(ks, true, q, idx.f32mirror[start*stride:], stride, end-start, buf[:])
			for j, sc := range buf[:end-start] {
				heapAdd(h, start+j, sc, k)
			}
		}
	}
}

// cosineFromParts assembles dot/(|q| * |row|) with the zero-guard the
// reference CosineFloat32 uses.
func cosineFromParts(dot, qNorm, rowNorm float32) float32 {
	if qNorm <= 0 || rowNorm <= 0 {
		return 0
	}
	return dot / (qNorm * rowNorm)
}

// searchComplexSoAChunk scores SoA rows [s, e) against the split query.
// Only reached when the SoA cache exists, so all rows are uniform.
// Metric handling mirrors the reference: dot and cosine (normalised via
// the prepared norms), L2 negated, unknown metrics score dot.
func (idx *FlatIndex) searchComplexSoAChunk(ks *distanceKernelSet, h *resultHeap, soa *ComplexSoA, qRe, qIm []float64, s, e, k int, metric DistanceMetric) {
	l2 := metric == MetricL2
	cosine := metric == MetricCosine && idx.complexNorms != nil && len(idx.complexNorms) >= soa.Count
	var qNorm float64
	if cosine {
		qNorm = dotComplex128Scalars(qRe, qIm, qRe, qIm)
	}
	var buf [batchScoreBlock]float64
	for start := s; start < e; start += batchScoreBlock {
		end := start + batchScoreBlock
		if end > e {
			end = e
		}
		scoreComplexSoABatch(ks, !l2, qRe, qIm, soa.Re, soa.Im, soa.Dim, start, end, buf[:])
		for j, sc := range buf[:end-start] {
			score := float32(sc)
			switch {
			case l2:
				score = -score
			case cosine:
				rowNorm := idx.complexNorms[start+j]
				if qNorm <= 0 || rowNorm <= 0 {
					score = 0
				} else {
					score = float32(sc / safeSqrt(qNorm*rowNorm))
				}
			}
			heapAdd(h, start+j, score, k)
		}
	}
}

// searchTurboChunk scores packed TurboQuant rows [s, e) against a
// prepared query. Mirrors the reference semantics: dot by default,
// cosine normalised by the stored Norm fields, L2 via the
// ||a||^2+||b||^2-2*dot expansion (negated).
func (idx *FlatIndex) searchTurboChunk(ks *distanceKernelSet, h *resultHeap, prep *TurboPrep, packed *TurboPackedColumns, s, e, k int, metric DistanceMetric) {
	var buf [batchScoreBlock]float32
	for start := s; start < e; start += batchScoreBlock {
		end := start + batchScoreBlock
		if end > e {
			end = e
		}
		if !scoreTurboBatch(ks, prep, packed, start, end, buf[:]) {
			for j := range buf[:end-start] {
				buf[j] = 0
			}
		}
		for j, sc := range buf[:end-start] {
			row := start + j
			id := row
			if id < len(idx.Turbo) {
				id = idx.Turbo[row].ID
			}
			switch metric {
			case MetricL2:
				var dist float32
				if row < len(packed.Norm) {
					dist = (prep.Vec.Norm * prep.Vec.Norm) + (packed.Norm[row] * packed.Norm[row]) - (2.0 * sc)
				}
				if dist < 0 {
					dist = 0
				}
				heapAdd(h, id, -dist, k)
			case MetricCosine:
				var score float32
				if prep.Vec.Norm > 0 && row < len(packed.Norm) && packed.Norm[row] > 0 {
					score = sc / (prep.Vec.Norm * packed.Norm[row])
				}
				heapAdd(h, id, score, k)
			default:
				heapAdd(h, id, sc, k)
			}
		}
	}
}

// BatchSearch scores many queries in one pass. When the native
// multi-query kernels (or the CUDA build) are available the rows stream
// once for the whole batch; otherwise it degrades to per-query batch
// scoring with the same kernel set, which is still the accelerated
// path. Results come back in query order.
//
// All queries must be of the index's storage family ([]float32,
// []complex128, *Uint8Vector, *TurboQuantVector). Metric handling
// mirrors Search.
func (idx *FlatIndex) BatchSearch(queries []interface{}, k int, metric DistanceMetric) ([][]SearchResult, error) {
	n := idx.Count()
	numQ := len(queries)
	out := make([][]SearchResult, numQ)
	if n == 0 || numQ == 0 {
		return out, nil
	}
	if k <= 0 {
		return out, nil
	}
	if k > n {
		k = n
	}

	// Cache preparation happens here, single-goroutine, before the
	// workers fan out (same invariant as Search).
	ks := kernelSetFor()
	soa := idx.PrepareComplex()
	packed := idx.PrepareTurboQuant()
	if idx.DataType == TypeFloat32 && idx.f32mirror == nil {
		idx.PrepareFloat32()
	}
	if idx.rowNorms == nil && idx.DataType == TypeFloat32 {
		idx.buildRowNorms()
	}

	// 1. GPU shared-scan path (cuda build only; nil hooks elsewhere).
	gpuResults, gpuOK := idx.batchSearchGPU(queries, numQ, k, metric, n)
	if gpuOK {
		return gpuResults, nil
	}

	// 2. Native multi-query kernels: float32 and complex SoA. Queries
	// must be uniform to pack them into the [numQ][dim] matrix.
	switch idx.DataType {
	case TypeComplex128:
		if soa != nil && ks != nil && (ks.mqDotComplexSoA != nil || ks.mqL2ComplexSoA != nil) {
			qReM, qImM, dim, ok := packComplexQueries(queries, numQ)
			if ok && dim == soa.Dim {
				return idx.batchSearchComplexMQ(ks, soa, qReM, qImM, numQ, k, metric, n), nil
			}
		}
	case TypeFloat32:
		if idx.f32mirror != nil && ks != nil && (ks.mqDotFloat32 != nil || ks.mqL2Float32 != nil) {
			qm, dim, ok := packFloat32Queries(queries, numQ)
			// The multi-query kernel walks the mirror with stride ==
			// Dimension, so only shapes that match it take this path;
			// anything else keeps the per-query scoring below, which
			// still uses the batch kernels.
			if ok && dim == idx.Dimension {
				return idx.batchSearchFloat32MQ(ks, qm, numQ, k, metric, n), nil
			}
		}
	}

	// 3. Fallback: per-query batch scoring, rows split across workers.
	return idx.batchSearchFallback(queries, numQ, k, metric, n, ks, soa, packed)
}

// packFloat32Queries flattens []float32 queries into [numQ*dim]; ok is
// false when the queries differ in length (the caller then uses the
// per-query path).
func packFloat32Queries(queries []interface{}, numQ int) ([]float32, int, bool) {
	first, ok := queries[0].([]float32)
	if !ok || len(first) == 0 {
		return nil, 0, false
	}
	dim := len(first)
	for i := 1; i < numQ; i++ {
		q, ok := queries[i].([]float32)
		if !ok || len(q) != dim {
			return nil, 0, false
		}
	}
	qm := make([]float32, numQ*dim)
	for i := 0; i < numQ; i++ {
		copy(qm[i*dim:(i+1)*dim], queries[i].([]float32))
	}
	return qm, dim, true
}

// packComplexQueries flattens []complex128 queries into split planes.
func packComplexQueries(queries []interface{}, numQ int) (re, im []float64, dim int, ok bool) {
	first, ok := queries[0].([]complex128)
	if !ok || len(first) == 0 {
		return nil, nil, 0, false
	}
	dim = len(first)
	for i := 1; i < numQ; i++ {
		q, ok := queries[i].([]complex128)
		if !ok || len(q) != dim {
			return nil, nil, 0, false
		}
	}
	re = make([]float64, numQ*dim)
	im = make([]float64, numQ*dim)
	for i := 0; i < numQ; i++ {
		q := queries[i].([]complex128)
		base := i * dim
		for j, z := range q {
			re[base+j] = real(z)
			im[base+j] = imag(z)
		}
	}
	return re, im, dim, true
}

// batchSearchFloat32MQ scores the packed queries with the multi-query
// kernel over worker-split row ranges. Heap semantics match Search.
func (idx *FlatIndex) batchSearchFloat32MQ(ks *distanceKernelSet, qm []float32, numQ, k int, metric DistanceMetric, n int) [][]SearchResult {
	dim := len(qm) / numQ
	l2 := metric == MetricL2
	// The MQ kernels compute raw dot and L2 only. Anything else
	// (cosine needs the norm division, unknown metrics fall back to
	// dot) must take the per-query path that implements those
	// semantics; routing cosine through the dot kernel returned
	// unnormalised scores.
	useMQ := false
	if l2 && ks.mqL2Float32 != nil {
		useMQ = true
	}
	if metric == MetricDot && ks.mqDotFloat32 != nil {
		useMQ = true
	}
	if !useMQ {
		return idx.batchSearchCosineOrUnknownF32(qm, numQ, k, metric, n)
	}

	// The caller only routes here with dim == idx.Dimension, so the
	// mirror rows stride equals the packed query dim.
	stride := idx.Dimension

	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	const mqRows = 1024 // row block per kernel call
	bufRows := numQ * mqRows

	// heaps[w*numQ + q]
	heaps := make([]resultHeap, workers*numQ)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		s := w * chunk
		e := s + chunk
		if s >= n {
			break
		}
		if e > n {
			e = n
		}
		wg.Add(1)
		go func(s, e, w int) {
			defer wg.Done()
			localHeaps := heaps[w*numQ : (w+1)*numQ]
			out := make([]float32, bufRows)
			for start := s; start < e; start += mqRows {
				end := start + mqRows
				if end > e {
					end = e
				}
				m := end - start
				block := idx.f32mirror[start*stride : end*stride]
				view := out[:numQ*m]
				var ok bool
				if l2 {
					ok = ks.mqL2Float32(qm, numQ, dim, block, stride, m, view)
				} else {
					ok = ks.mqDotFloat32(qm, numQ, dim, block, stride, m, view)
				}
				for qi := 0; qi < numQ; qi++ {
					if !ok {
						// Kernel rejected the block: per-query batch path.
						scoreFloat32Batch(ks, !l2, qm[qi*dim:(qi+1)*dim], block, stride, m, out[:m])
					}
					row := out[:m]
					if ok {
						row = view[qi*m : qi*m+m]
					}
					hh := &localHeaps[qi]
					for j := range row {
						sc := row[j]
						if l2 {
							sc = -sc
						}
						heapAdd(hh, start+j, sc, k)
					}
				}
			}
		}(s, e, w)
	}
	wg.Wait()

	results := make([][]SearchResult, numQ)
	for qi := 0; qi < numQ; qi++ {
		perQ := make([]resultHeap, 0, workers)
		for w := 0; w < workers; w++ {
			perQ = append(perQ, heaps[w*numQ+qi])
		}
		results[qi] = mergeHeaps(perQ, k)
	}
	return results
}

// batchSearchCosineOrUnknownF32 handles the float32 metrics the
// multi-query kernels do not cover (cosine needs the norm division).
func (idx *FlatIndex) batchSearchCosineOrUnknownF32(qm []float32, numQ, k int, metric DistanceMetric, n int) [][]SearchResult {
	dim := len(qm) / numQ
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	heaps := make([]resultHeap, workers*numQ)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		s := w * chunk
		e := s + chunk
		if s >= n {
			break
		}
		if e > n {
			e = n
		}
		wg.Add(1)
		go func(s, e, w int) {
			defer wg.Done()
			localHeaps := heaps[w*numQ : (w+1)*numQ]
			ks := kernelSetFor()
			for qi := 0; qi < numQ; qi++ {
				q := qm[qi*dim : (qi+1)*dim]
				idx.searchFloat32Chunk(ks, &localHeaps[qi], q, s, e, k, metric)
			}
		}(s, e, w)
	}
	wg.Wait()

	results := make([][]SearchResult, numQ)
	for qi := 0; qi < numQ; qi++ {
		perQ := make([]resultHeap, 0, workers)
		for w := 0; w < workers; w++ {
			perQ = append(perQ, heaps[w*numQ+qi])
		}
		results[qi] = mergeHeaps(perQ, k)
	}
	return results
}

// batchSearchComplexMQ scores the packed split queries with the
// multi-query SoA kernel over worker-split row ranges.
func (idx *FlatIndex) batchSearchComplexMQ(ks *distanceKernelSet, soa *ComplexSoA, qReM, qImM []float64, numQ, k int, metric DistanceMetric, n int) [][]SearchResult {
	l2 := metric == MetricL2
	// Same metric guard as the float32 path: the SoA multi-query
	// kernels are dot/L2 only, so cosine and unknown metrics must go
	// through the per-query SoA scoring which normalises cosine.
	useMQ := false
	if l2 && ks.mqL2ComplexSoA != nil {
		useMQ = true
	}
	if metric == MetricDot && ks.mqDotComplexSoA != nil {
		useMQ = true
	}
	if !useMQ {
		return idx.batchSearchComplexPerQuery(qReM, qImM, numQ, k, metric, n)
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	const mqRows = 1024
	bufRows := numQ * mqRows
	dim := soa.Dim

	heaps := make([]resultHeap, workers*numQ)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		s := w * chunk
		e := s + chunk
		if s >= n {
			break
		}
		if e > n {
			e = n
		}
		wg.Add(1)
		go func(s, e, w int) {
			defer wg.Done()
			localHeaps := heaps[w*numQ : (w+1)*numQ]
			out := make([]float32, bufRows)
			var fbuf []float64
			for start := s; start < e; start += mqRows {
				end := start + mqRows
				if end > e {
					end = e
				}
				m := end - start
				view := out[:numQ*m]
				ok := false
				if l2 {
					ok = ks.mqL2ComplexSoA(qReM, qImM, numQ, soa.Re, soa.Im, dim, start, end, view)
				} else {
					ok = ks.mqDotComplexSoA(qReM, qImM, numQ, soa.Re, soa.Im, dim, start, end, view)
				}
				for qi := 0; qi < numQ; qi++ {
					if !ok {
						// Kernel rejected: per-query batch path (fp64).
						if fbuf == nil {
							fbuf = make([]float64, m)
						}
						scoreComplexSoABatch(ks, !l2, qReM[qi*dim:(qi+1)*dim], qImM[qi*dim:(qi+1)*dim],
							soa.Re, soa.Im, dim, start, end, fbuf)
						row := out[:m]
						for j := 0; j < m; j++ {
							row[j] = float32(fbuf[j])
						}
					}
					row := out[:m]
					if ok {
						row = view[qi*m : qi*m+m]
					}
					hh := &localHeaps[qi]
					for j := range row {
						sc := row[j]
						if l2 {
							heapAdd(hh, start+j, -sc, k)
						} else {
							heapAdd(hh, start+j, sc, k)
						}
					}
				}
			}
		}(s, e, w)
	}
	wg.Wait()

	results := make([][]SearchResult, numQ)
	for qi := 0; qi < numQ; qi++ {
		perQ := make([]resultHeap, 0, workers)
		for w := 0; w < workers; w++ {
			perQ = append(perQ, heaps[w*numQ+qi])
		}
		results[qi] = mergeHeaps(perQ, k)
	}
	return results
}

// batchSearchComplexPerQuery is the cosine/complex path: rows are split
// over workers and every query walks the same rows via the per-query SoA
// batch kernel. Cosine normalisation mirrors searchComplexSoAChunk.
func (idx *FlatIndex) batchSearchComplexPerQuery(qReM, qImM []float64, numQ, k int, metric DistanceMetric, n int) [][]SearchResult {
	dim := len(qReM) / maxInt(numQ, 1)
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	heaps := make([]resultHeap, workers*numQ)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		s := w * chunk
		e := s + chunk
		if s >= n {
			break
		}
		if e > n {
			e = n
		}
		wg.Add(1)
		go func(s, e, w int) {
			defer wg.Done()
			localHeaps := heaps[w*numQ : (w+1)*numQ]
			ks := kernelSetFor()
			for qi := 0; qi < numQ; qi++ {
				idx.searchComplexSoAChunk(ks, &localHeaps[qi], idx.complexSoA,
					qReM[qi*dim:(qi+1)*dim], qImM[qi*dim:(qi+1)*dim], s, e, k, metric)
			}
		}(s, e, w)
	}
	wg.Wait()

	results := make([][]SearchResult, numQ)
	for qi := 0; qi < numQ; qi++ {
		perQ := make([]resultHeap, 0, workers)
		for w := 0; w < workers; w++ {
			perQ = append(perQ, heaps[w*numQ+qi])
		}
		results[qi] = mergeHeaps(perQ, k)
	}
	return results
}

// batchSearchFallback covers uint8, turboquant (no multi-query kernel),
// non-uniform rows, and unknown query shapes with the worker/row split
// of Search applied over all queries per worker.
func (idx *FlatIndex) batchSearchFallback(queries []interface{}, numQ, k int, metric DistanceMetric, n int, ks *distanceKernelSet, soa *ComplexSoA, packed *TurboPackedColumns) ([][]SearchResult, error) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	heaps := make([]resultHeap, workers*numQ)
	var wg sync.WaitGroup
	var workerErr error
	var errOnce sync.Once
	for w := 0; w < workers; w++ {
		s := w * chunk
		e := s + chunk
		if s >= n {
			break
		}
		if e > n {
			e = n
		}
		wg.Add(1)
		go func(s, e, w int) {
			defer wg.Done()
			localHeaps := heaps[w*numQ : (w+1)*numQ]
			for qi := 0; qi < numQ; qi++ {
				hh := &localHeaps[qi]
				switch idx.DataType {
				case TypeFloat32:
					q, ok := queries[qi].([]float32)
					if !ok {
						errOnce.Do(func() { workerErr = fmt.Errorf("expected []float32 query") })
						return
					}
					idx.searchFloat32Chunk(ks, hh, q, s, e, k, metric)
				case TypeUint8:
					q, ok := queries[qi].(*Uint8Vector)
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
						heapAdd(hh, idx.Uint8[i].ID, score, k)
					}
				case TypeComplex128:
					q, ok := queries[qi].([]complex128)
					if !ok {
						errOnce.Do(func() { workerErr = fmt.Errorf("expected []complex128 query") })
						return
					}
					if soa != nil {
						qRe, qIm := splitComplexQuery(q, idx.Dimension)
						if qRe != nil {
							idx.searchComplexSoAChunk(ks, hh, soa, qRe, qIm, s, e, k, metric)
							continue
						}
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
						heapAdd(hh, i, score, k)
					}
				case TypeTurboQuant:
					q, ok := queries[qi].(*TurboQuantVector)
					if !ok {
						errOnce.Do(func() { workerErr = fmt.Errorf("expected *TurboQuantVector query") })
						return
					}
					if packed != nil {
						prep := PrepareTurboQuantQuery(q)
						idx.searchTurboChunk(ks, hh, prep, packed, s, e, k, metric)
						continue
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
						heapAdd(hh, idx.Turbo[i].ID, score, k)
					}
				}
			}
		}(s, e, w)
	}
	wg.Wait()

	if workerErr != nil {
		return nil, workerErr
	}

	results := make([][]SearchResult, numQ)
	for qi := 0; qi < numQ; qi++ {
		perQ := make([]resultHeap, 0, workers)
		for w := 0; w < workers; w++ {
			perQ = append(perQ, heaps[w*numQ+qi])
		}
		results[qi] = mergeHeaps(perQ, k)
	}
	return results, nil
}

// batchSearchGPU runs the CUDA shared-scan path when the cuda build
// staged this index and the queries fit. Returns ok=false on any
// fallback condition; the CPU paths always remain available.
func (idx *FlatIndex) batchSearchGPU(queries []interface{}, numQ, k int, metric DistanceMetric, n int) ([][]SearchResult, bool) {
	if n == 0 || numQ == 0 || k <= 0 {
		return nil, false
	}
	if metric != MetricDot && metric != MetricL2 {
		// The GPU multi-query kernels cover dot and L2 only.
		return nil, false
	}
	if idx.gpuFail {
		return nil, false
	}

	switch idx.DataType {
	case TypeFloat32:
		floatQueries := make([][]float32, numQ)
		dim := idx.Dimension
		if dim <= 0 {
			q0, ok := queries[0].([]float32)
			if !ok || len(q0) == 0 {
				return nil, false
			}
			dim = len(q0)
		}
		for i := range queries {
			q, ok := queries[i].([]float32)
			if !ok || len(q) != dim {
				return nil, false
			}
			floatQueries[i] = q
		}
		idx.gpuOnce.Do(func() {
			if gpuStageFloat32 == nil {
				idx.gpuFail = true
				return
			}
			store, ok := gpuStageFloat32(idx.Float32, idx.Dimension)
			if !ok {
				idx.gpuFail = true
				return
			}
			idx.gpuF32 = store
		})
		if idx.gpuFail || idx.gpuF32 == nil || gpuQPackedFloat32 == nil || gpuSearchFloat32 == nil {
			return nil, false
		}
		qs, ok := gpuQPackedFloat32(floatQueries)
		if !ok {
			return nil, false
		}
		if res, ok := gpuSearchFloat32(idx.gpuF32, qs, numQ, k, metric == MetricL2); ok {
			return res, true
		}
		return nil, false

	case TypeComplex128:
		complexQueries := make([][]complex128, numQ)
		dim := idx.Dimension
		if dim <= 0 {
			q0, ok := queries[0].([]complex128)
			if !ok || len(q0) == 0 {
				return nil, false
			}
			dim = len(q0)
		}
		for i := range queries {
			q, ok := queries[i].([]complex128)
			if !ok || len(q) != dim {
				return nil, false
			}
			complexQueries[i] = q
		}
		idx.gpuOnce.Do(func() {
			if gpuStageComplex == nil {
				idx.gpuFail = true
				return
			}
			store, ok := gpuStageComplex(idx.Complex, idx.Dimension)
			if !ok {
				idx.gpuFail = true
				return
			}
			idx.gpuSoA = store
		})
		if idx.gpuFail || idx.gpuSoA == nil || gpuQPackedComplex == nil || gpuSearchComplex == nil {
			return nil, false
		}
		qs, ok := gpuQPackedComplex(complexQueries)
		if !ok {
			return nil, false
		}
		if res, ok := gpuSearchComplex(idx.gpuSoA, qs, numQ, k, metric == MetricL2); ok {
			return res, true
		}
		return nil, false
	}
	return nil, false
}

// maxInt keeps the little helper local (the package avoids pulling
// builtins that older module graphs do not have).
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
