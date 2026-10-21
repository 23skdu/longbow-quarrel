package vector

// flat_batch_coverage_test.go exercises the BatchSearch routing
// corners: the per-query fallback for ragged/unknown stores, the
// cosine/unknown-metric branch of the multi-query path, the complex
// SoA per-query route, and the query-type errors. Every leg asserts
// the results against the pure scalar expectations, not just
// "some result came back".

import (
	"container/heap"
	"math/rand"
	"testing"
)

func scalarTopK(scores []float32, k int) []SearchResult {
	h := &resultHeap{}
	for i, sc := range scores {
		heapAdd(h, i, sc, k)
	}
	out := make([]SearchResult, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(SearchResult)
	}
	return out
}

func TestFlatBatchSearchRaggedRows(t *testing.T) {
	dim := 12
	idx := NewFlatIndex(dim, TypeFloat32)
	rng := rand.New(rand.NewSource(1234))
	rows := make([][]float32, 20)
	for i := range rows {
		rows[i] = make([]float32, dim-i%2) // ragged: last row one short
		for j := range rows[i] {
			rows[i][j] = float32(rng.NormFloat64())
		}
	}
	idx.AddFloat32(rows)
	if idx.f32mirror != nil {
		t.Fatalf("ragged rows must not build a mirror")
	}

	queries := make([]interface{}, 3)
	for qi := range queries {
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(rng.NormFloat64())
		}
		queries[qi] = q
	}
	got, err := idx.BatchSearch(queries, 5, MetricDot)
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	for qi, want := range got {
		exp := make([]float32, len(rows))
		for i, row := range rows {
			exp[i] = dotFloat32Scalar(queries[qi].([]float32), row)
		}
		ref := scalarTopK(exp, 5)
		if len(want) != len(ref) {
			t.Fatalf("q%d len = %d want %d", qi, len(want), len(ref))
		}
		for r := range ref {
			// Scores from one executor: exact equality is sound here.
			if want[r].ID != ref[r].ID {
				t.Fatalf("q%d r%d: id %d want %d", qi, r, want[r].ID, ref[r].ID)
			}
			if want[r].Score != ref[r].Score {
				t.Fatalf("q%d r%d: score %g want %g", qi, r, want[r].Score, ref[r].Score)
			}
		}
	}

	// The same store through Search (cosine and L2 exercise the ragged
	// per-vector branches).
	q := queries[0].([]float32)
	if _, err := idx.Search(q, 3, MetricCosine); err != nil {
		t.Fatalf("ragged cosine Search: %v", err)
	}
	if _, err := idx.Search(q, 3, MetricL2); err != nil {
		t.Fatalf("ragged L2 Search: %v", err)
	}
}

func TestFlatBatchSearchCosineAndUnknownMetric(t *testing.T) {
	dim := 20
	idx := NewFlatIndex(dim, TypeFloat32)
	rng := rand.New(rand.NewSource(5678))
	rows := make([][]float32, 40)
	for i := range rows {
		rows[i] = parityQuery(dim, rng)
		if rows[i][i%dim] == 0 {
			rows[i][i%dim] = 1 // keep every row normalisable
		}
	}
	idx.AddFloat32(rows)

	qm := make([]interface{}, 4)
	for qi := range qm {
		q := parityQuery(dim, rng)
		if q[0] == 0 {
			q[0] = 1
		}
		qm[qi] = q
	}

	metrics := []DistanceMetric{MetricCosine, DistanceMetric("nonsense")}
	for _, metric := range metrics {
		got, err := idx.BatchSearch(qm, 6, metric)
		if err != nil {
			t.Fatalf("BatchSearch(metric %v): %v", metric, err)
		}
		for qi, want := range got {
			q := qm[qi].([]float32)
			exp := make([]float32, len(rows))
			for i, row := range rows {
				if metric == MetricCosine {
					exp[i] = CosineFloat32(q, row)
				} else {
					// Unknown metrics score dot (reference semantics).
					exp[i] = dotFloat32Scalar(q, row)
				}
			}
			ref := scalarTopK(exp, 6)
			for r := range ref {
				if want[r].ID != ref[r].ID {
					// Exact ties resolve by heap order; allow a swap
					// only between equal scores.
					if ref[r].Score == want[r].Score {
						continue
					}
					t.Fatalf("metric %v q%d r%d: id %d want %d (scores %g vs %g)", metric, qi, r, want[r].ID, ref[r].ID, want[r].Score, ref[r].Score)
				}
				if diff := want[r].Score - ref[r].Score; diff > 1e-4 || diff < -1e-4 {
					t.Fatalf("metric %v q%d r%d: score %g want %g", metric, qi, r, want[r].Score, ref[r].Score)
				}
			}
		}
	}
}

func TestFlatBatchSearchComplexPerQueryRoute(t *testing.T) {
	dim := 24
	idx := NewFlatIndex(dim, TypeComplex128)
	rng := rand.New(rand.NewSource(6901))
	rows := make([][]complex128, 30)
	for i := range rows {
		row := make([]complex128, dim)
		for j := range row {
			row[j] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
		rows[i] = row
	}
	idx.AddComplex128(rows)

	qm := make([]interface{}, 3)
	for qi := range qm {
		q := make([]complex128, dim)
		for j := range q {
			q[j] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
		qm[qi] = q
	}
	// Cosine takes the per-query SoA route inside the complex MQ path
	// (the multi-query kernels cover dot and L2 only).
	got, err := idx.BatchSearch(qm, 5, MetricCosine)
	if err != nil {
		t.Fatalf("BatchSearch cosine: %v", err)
	}
	for qi, want := range got {
		exp := make([]float32, len(rows))
		for i, row := range rows {
			exp[i] = float32(CosineComplex128(qm[qi].([]complex128), row))
		}
		ref := scalarTopK(exp, 5)
		for r := range ref {
			if want[r].ID != ref[r].ID {
				t.Fatalf("cosine q%d r%d: id %d want %d", qi, r, want[r].ID, ref[r].ID)
			}
			if diff := want[r].Score - ref[r].Score; diff > 1e-4 || diff < -1e-4 {
				t.Fatalf("cosine q%d r%d: score %g want %g", qi, r, want[r].Score, ref[r].Score)
			}
		}
	}
}

func TestFlatBatchSearchRejectsUnknownQueryType(t *testing.T) {
	dim := 8
	f32Idx := NewFlatIndex(dim, TypeFloat32)
	f32Idx.AddFloat32(makeRowsFloat32(16, dim, 12345))
	_, err := f32Idx.BatchSearch([]interface{}{"bogus"}, 3, MetricDot)
	if err == nil {
		t.Fatalf("float32 index accepted a bogus query type")
	}
	if err.Error() != "expected []float32 query" {
		t.Fatalf("unexpected error: %v", err)
	}

	cIdx := NewFlatIndex(dim, TypeComplex128)
	cIdx.AddComplex128(makeRowsComplex128(16, dim))
	if _, err := cIdx.BatchSearch([]interface{}{make([]float32, dim)}, 3, MetricDot); err == nil {
		t.Fatalf("complex index accepted a float32 query")
	}
}

func makeRowsFloat32(n, dim int, seed int64) [][]float32 {
	rows := make([][]float32, n)
	rng := rand.New(rand.NewSource(seed))
	for i := range rows {
		rows[i] = parityQuery(dim, rng)
	}
	return rows
}

func makeRowsComplex128(n, dim int) [][]complex128 {
	rows := make([][]complex128, n)
	for i := range rows {
		row := make([]complex128, dim)
		for j := range row {
			row[j] = complex(float64(i), float64(j))
		}
		rows[i] = row
	}
	return rows
}
