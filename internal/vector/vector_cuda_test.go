//go:build cuda && cgo && linux && amd64

package vector

// vector_cuda_test.go runs the GPU parity suite of internal/vector on
// the RTX host the team develops on (P0-01 batch inverted-list scoring,
// P0-02 complex SoA batch path, P0-04 GPU mini-batch k-means).
//
// Parity budget: the comparison uses gpuParityTol — 1e-4 at unit
// magnitude, scaled with |score| above 1. The fp32 device kernels
// accumulate in a different order than the scalar CPU reference (and
// the complex planes are fp32 while the CPU index holds fp64 — the
// documented P0-02 trade-off), so absolute errors grow with the
// product magnitude: a 25 880-magnitude dot shows ~5e-7 relative fp
// noise. Fixtures keep the top-k sets unambiguous so the comparison
// stays exact-ID based, with the nearest-1e-4*max(1,|cpu|) escape.
//
// The CPU reference legs of the flat tests run with the query/search
// hooks temporarily nilled (disableGpuSearchForCPU). The staging hooks
// stay installed so the lazy gpuOnce staging in batchSearch still runs;
// NULLing the staging hook instead would flip batchSearch's gpuOnce
// into the permanent idx.gpuFail state and silently turn the second
// leg into another CPU run.

import (
	"container/heap"
	"math/rand"
	"testing"
)

func skipIfNoGPU(t *testing.T) {
	t.Helper()
	if gpuStageFloat32 == nil {
		t.Skip("CUDA hooks unavailable (no device or init failed)")
	}
}

// gpuParityTol is the parity budget for one score: 1e-4 absolute at
// unit magnitude, rising proportionally for larger scores so fp32
// accumulation noise stays within it on the fixtures' ranges.
func gpuParityTol(cpu float32) float32 {
	if cpu < 0 {
		cpu = -cpu
	}
	if cpu < 1 {
		return 1e-4
	}
	return 1e-4 * cpu
}

// disableGpuSearchForCPU nils the query-packing and search hooks so
// BatchSearch takes the CPU path (the CPU reference leg). Staging hooks
// stay live: the batch path then lazily stages the rows with gpuOnce
// (does not touch gpuFail), and the GPU leg reuses that staged store.
// It returns the restore closure (also registered via t.Cleanup as a
// safety net) so the test can re-enable the GPU before its second leg.
func disableGpuSearchForCPU(t *testing.T) func() {
	t.Helper()
	pf, sf := gpuQPackedFloat32, gpuSearchFloat32
	pc, sc := gpuQPackedComplex, gpuSearchComplex
	restore := func() {
		gpuQPackedFloat32, gpuSearchFloat32 = pf, sf
		gpuQPackedComplex, gpuSearchComplex = pc, sc
	}
	gpuQPackedFloat32, gpuSearchFloat32 = nil, nil
	gpuQPackedComplex, gpuSearchComplex = nil, nil
	t.Cleanup(restore)
	return restore
}

// distinctFloatFixture builds n rows of dim with the distinct leading
// coordinate trick described at the top of the file.
func distinctFloatFixture(n, dim int) [][]float32 {
	rows := make([][]float32, n)
	for i := range rows {
		r := make([]float32, dim)
		r[0] = float32(i+1) * 0.25
		for j := 1; j < dim; j++ {
			r[j] = float32((i*j)%5) / 7
		}
		rows[i] = r
	}
	return rows
}

func absF(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func TestCudaFlatBatchSearchParity(t *testing.T) {
	skipIfNoGPU(t)
	n, dim, numQ, k := 3000, 64, 8, 10
	idx := NewFlatIndex(dim, TypeFloat32)
	idx.AddFloat32(distinctFloatFixture(n, dim))

	queries := make([]interface{}, numQ)
	for qi := 0; qi < numQ; qi++ {
		q := make([]float32, dim)
		q[0] = float32((qi*137)%n+1) * 0.25
		for j := 1; j < dim; j++ {
			q[j] = float32((qi+j)%4) / 5
		}
		queries[qi] = q
	}
	for _, metric := range []DistanceMetric{MetricDot, MetricL2} {
		// CPU reference leg: search hooks disabled, staging hook live.
		restore := disableGpuSearchForCPU(t)
		want, err := idx.BatchSearch(queries, k, metric)
		if err != nil {
			t.Fatalf("cpu BatchSearch (%v): %v", metric, err)
		}
		// GPU leg: re-enable the hooks and reuse the store staged
		// during the CPU leg.
		restore()
		got, err := idx.BatchSearch(queries, k, metric)
		if err != nil {
			t.Fatalf("gpu BatchSearch (%v): %v", metric, err)
		}
		for qi := range want {
			if len(got[qi]) != len(want[qi]) {
				t.Fatalf("metric %v query %d: len = %d want %d", metric, qi, len(got[qi]), len(want[qi]))
			}
			for r := range want[qi] {
				if got[qi][r].ID != want[qi][r].ID {
					continue // set equality below is the real assertion
				}
				tol := gpuParityTol(want[qi][r].Score)
				if diff := absF(got[qi][r].Score - want[qi][r].Score); diff > tol {
					t.Fatalf("metric %v query %d res %d: gpu %v cpu %v (gap %g > tol %g)", metric, qi, r, got[qi][r].Score, want[qi][r].Score, diff, tol)
				}
			}
			// Set equality: at this fixture every swap candidate must
			// be closer than the parity tol to the id it replaced (the
			// fixture's gaps dwarf the tolerance).
			seenW := map[int]bool{}
			seenG := map[int]bool{}
			for _, x := range want[qi] {
				seenW[x.ID] = true
			}
			for _, x := range got[qi] {
				seenG[x.ID] = true
			}
			for id := range seenG {
				if !seenW[id] {
					t.Fatalf("metric %v query %d: id %d only from gpu", metric, qi, id)
				}
			}
			for id := range seenW {
				if !seenG[id] {
					t.Fatalf("metric %v query %d: id %d only from cpu", metric, qi, id)
				}
			}
		}
	}
}

func TestCudaComplexBatchSearchParity(t *testing.T) {
	skipIfNoGPU(t)
	dim, n, numQ, k := 48, 2000, 6, 10
	idx := NewFlatIndex(dim, TypeComplex128)
	rows := make([][]complex128, n)
	for i := 0; i < n; i++ {
		row := make([]complex128, dim)
		row[0] = complex(float64(i+1)*0.05, 0.01)
		for j := 1; j < dim; j++ {
			row[j] = complex(float64((i*j)%7)/23, float64((i+j)%5)/29)
		}
		rows[i] = row
	}
	idx.AddComplex128(rows)

	queries := make([]interface{}, numQ)
	for qi := 0; qi < numQ; qi++ {
		q := make([]complex128, dim)
		q[0] = complex(float64((qi*97)%n+1)*0.05, 0.02)
		for j := 1; j < dim; j++ {
			q[j] = complex(float64((qi*j)%7)/23, float64((qi*j*2)%5)/29)
		}
		queries[qi] = q
	}
	for _, metric := range []DistanceMetric{MetricDot, MetricL2} {
		restore := disableGpuSearchForCPU(t)
		want, err := idx.BatchSearch(queries, k, metric)
		if err != nil {
			t.Fatalf("cpu complex BatchSearch (%v): %v", metric, err)
		}
		restore()
		got, err := idx.BatchSearch(queries, k, metric)
		if err != nil {
			t.Fatalf("gpu complex BatchSearch (%v): %v", metric, err)
		}
		for qi := range want {
			if len(got[qi]) != len(want[qi]) {
				t.Fatalf("metric %v q%d: len = %d want %d", metric, qi, len(got[qi]), len(want[qi]))
			}
			wp := map[int]float32{}
			for _, s := range want[qi] {
				wp[s.ID] = s.Score
			}
			for _, s := range got[qi] {
				ws, ok := wp[s.ID]
				if !ok {
					t.Fatalf("metric %v q%d: id %d only from gpu", metric, qi, s.ID)
				}
				// fp32 device planes + fp32 accumulation vs the fp64
				// CPU reference: magnitude-scaled budget (file header).
				tol := gpuParityTol(ws)
				if diff := absF(s.Score - ws); diff > tol {
					t.Fatalf("metric %v q%d id %d: gpu %v cpu %v (gap %g > tol %g)", metric, qi, s.ID, s.Score, ws, diff, tol)
				}
			}
		}
	}
}

// bruteIVFExpected replicates the CPU list-scan semantics (probe by
// L2 over the centroids, dot over the probed lists, results ids from
// vector positions) with the pure scalar oracle.
func bruteIVFExpected(idx *IVFIndex, query []float32, k int) []SearchResult {
	type cd struct {
		id   int
		dist float32
	}
	pairs := make([]cd, len(idx.Centroids))
	for c := range idx.Centroids {
		pairs[c] = cd{c, l2Float32Scalar(query, idx.Centroids[c])}
	}
	numProbes := idx.NumProbes
	if numProbes > len(pairs) {
		numProbes = len(pairs)
	}
	for i := 0; i < numProbes; i++ {
		m := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].dist < pairs[m].dist {
				m = j
			}
		}
		pairs[i], pairs[m] = pairs[m], pairs[i]
	}
	probed := map[int]bool{}
	for p := 0; p < numProbes; p++ {
		probed[pairs[p].id] = true
	}
	h := &resultHeap{}
	for c, list := range idx.InvertedLists {
		if !probed[c] {
			continue
		}
		for _, vID := range list {
			heapAdd(h, vID, dotFloat32Scalar(query, idx.float32Row(vID)), k)
		}
	}
	out := make([]SearchResult, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(SearchResult)
	}
	return out
}

func TestIVFSearchGPUAgainstRef(t *testing.T) {
	skipIfNoGPU(t)
	dim := 64
	clusters := 48
	perCluster := 96
	fixedCentroids := make([][]float32, clusters)
	for c := range fixedCentroids {
		row := make([]float32, dim)
		row[0] = float32(c+1) * 30 // far separation
		for j := 1; j < dim; j++ {
			row[j] = float32((c*j)%9) / 9
		}
		fixedCentroids[c] = row
	}
	rng := rand.New(rand.NewSource(7))
	rows := make([][]float32, 0, clusters*perCluster)
	labels := make([]int, 0, clusters*perCluster)
	for c := range fixedCentroids {
		for i := 0; i < perCluster; i++ {
			r := make([]float32, dim)
			copy(r, fixedCentroids[c])
			r[1] += float32(rng.NormFloat64()) / 64
			rows = append(rows, r)
			labels = append(labels, c)
		}
	}

	idx := NewIVFIndex(dim, clusters, 1, TypeFloat32)
	if err := idx.Build(rows); err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !idx.gpuStaged {
		t.Fatalf("expected the rows to stage on the device at Build")
	}

	// Assignment consistency, not slot==cluster identity:Build seeds
	// the 48 centroids from 48 random draws over the rows, so some
	// clusters draw no representative (their slots strand on another
	// cluster's side) and duplicate-representative slots collapse (the
	// tie resolves to the smaller slot id and the twin goes empty).
	// The assembler contract is "every row lands in its scalar-nearest
	// slot over the final centroids" — assert exactly that, per row.
	rowSlot := make([]int, len(rows))
	seen := make([]bool, len(rows))
	for c, list := range idx.InvertedLists {
		for _, vID := range list {
			if seen[vID] {
				t.Fatalf("row %d appears in lists %d and %d", vID, rowSlot[vID], c)
			}
			seen[vID] = true
			rowSlot[vID] = c
		}
	}
	for i := range seen {
		if !seen[i] {
			t.Fatalf("row %d assigned to no list", i)
		}
	}
	for i, r := range rows {
		best, bestD := -1, float32(1e30)
		for c := range idx.Centroids {
			if d := l2Float32Scalar(r, idx.Centroids[c]); d < bestD {
				best, bestD = c, d
			}
		}
		if rowSlot[i] != best {
			t.Fatalf("row %d (label %d) in list %d, scalar-nearest list is %d (d=%g)", i, labels[i], rowSlot[i], best, bestD)
		}
	}
	// Membership == scalar-nearest over the final centroids is the
	// assembler contract for both paths (device full-assign and CPU
	// scalar assign), so the assertion above covers either branch.

	// Search: the GPU scan must agree with the pure-Go reference on
	// ids and within the parity budget on scores.
	for trial := 0; trial < 8; trial++ {
		q := make([]float32, dim)
		for j := range q {
			q[j] = fixedCentroids[(trial*7+j)%clusters][j] * 0.98
		}
		got, err := idx.Search(q, 10)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		ref := bruteIVFExpected(idx, q, 10)
		// Tie-aware band comparison (see topKParity): intra-cluster
		// rows differ only by the noise term while the fp32 noise
		// floor scales with the 5.6e4-magnitude dot product, so
		// correct executives may disagree on which blob member ranks
		// where. tol covers the biggest score in this query's result
		// band; boundary swaps get 2 tol.
		scale := float32(0)
		for _, s := range ref {
			if v := s.Score; v > scale {
				scale = v
			}
		}
		if scale < 1 {
			scale = 1
		}
		topKParity(t, "ivf gpu scan", got, ref, 1e-4*scale)
	}
}

func TestCudaStagedListsShape(t *testing.T) {
	skipIfNoGPU(t)
	idx := NewIVFIndex(16, 4, 2, TypeFloat32)
	rows := distinctFloatFixture(64, 16)
	if err := idx.Build(rows); err != nil {
		t.Fatalf("Build: %v", err)
	}
	if idx.gpuLists == nil {
		t.Fatalf("expected the lists staged after Build")
	}
	ls, ok := idx.gpuLists.(*gpuListsStore)
	if !ok || ls.numLists != len(idx.InvertedLists) {
		t.Fatalf("list store shape mismatch")
	}
	if ls.total != len(rows) {
		t.Fatalf("staged total = %d, want %d", ls.total, len(rows))
	}
}

func TestCudaBatchSearchRejectsWrongShapes(t *testing.T) {
	skipIfNoGPU(t)
	dim := 8
	idx := NewFlatIndex(dim, TypeFloat32)
	idx.AddFloat32(distinctFloatFixture(16, dim))
	// ragged queries: per-query path still returns a result set.
	res, err := idx.BatchSearch([]interface{}{
		make([]float32, dim),
		make([]float32, dim/2), // different length forces per-query path
	}, 4, MetricDot)
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	if len(res) != 2 || len(res[0]) == 0 || len(res[1]) == 0 {
		t.Fatalf("fallback results incomplete: %v", res)
	}
}
