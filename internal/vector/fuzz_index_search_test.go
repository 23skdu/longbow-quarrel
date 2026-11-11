package vector

import (
	"math"
	"testing"
)

func FuzzFlatIndexSearch(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		idx := buildFlatFixture(32, 8, 3)
		q := make([]float32, 8)
		for i := range q {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			q[i] = float32(v)/255.0*2 - 1
		}
		res, err := idx.Search(q, 5, MetricDot)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		if len(res) > 5 {
			t.Fatalf("too many results: %d", len(res))
		}
		for _, r := range res {
			if math.IsNaN(float64(r.Score)) || math.IsInf(float64(r.Score), 0) {
				t.Fatalf("non-finite score: %v", r.Score)
			}
		}
	})
}

func FuzzFlatIndexL2Search(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		idx := buildFlatFixture(32, 8, 3)
		q := make([]float32, 8)
		for i := range q {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			q[i] = float32(v)/255.0*2 - 1
		}
		res, err := idx.Search(q, 5, MetricL2)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		for _, r := range res {
			if math.IsNaN(float64(r.Score)) || math.IsInf(float64(r.Score), 0) {
				t.Fatalf("non-finite score: %v", r.Score)
			}
		}
	})
}

func makeFuzzRows(n, dim int) [][]float32 {
	rows := make([][]float32, n)
	for i := range rows {
		row := make([]float32, dim)
		for j := range row {
			row[j] = float32(i*dim+j) / float32(n*dim)
		}
		rows[i] = row
	}
	return rows
}

func FuzzIVFIndexSearch(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		idx := NewIVFIndex(8, 4, 2, TypeFloat32)
		if err := idx.Build(makeFuzzRows(32, 8)); err != nil {
			t.Fatalf("Build: %v", err)
		}
		q := make([]float32, 8)
		for i := range q {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			q[i] = float32(v)/255.0*2 - 1
		}
		res, err := idx.Search(q, 5)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		for _, r := range res {
			if math.IsNaN(float64(r.Score)) || math.IsInf(float64(r.Score), 0) {
				t.Fatalf("non-finite score: %v", r.Score)
			}
		}
	})
}

func FuzzMMapFlatRoundTrip(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		idx := buildFlatFixture(8, 4, 3)
		path := mmapTestPath(t, "fuzz.bin")
		if err := SaveFlatIndexFloat32(path, idx); err != nil {
			t.Fatalf("Save: %v", err)
		}
		loaded, store, err := LoadFlatIndexFloat32(path)
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		t.Cleanup(func() { _ = store.Close() })
		if loaded.Count() != idx.Count() {
			t.Fatalf("count mismatch")
		}
		_ = data
	})
}

func FuzzHNSWIndexSearch(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		h := NewHNSWIndex(8, 4, 10, TypeFloat32)
		if err := h.Build(makeFuzzRows(32, 8)); err != nil {
			t.Fatalf("Build: %v", err)
		}
		q := make([]float32, 8)
		for i := range q {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			q[i] = float32(v)/255.0*2 - 1
		}
		res, err := h.Search(q, 5)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		for _, r := range res {
			if math.IsNaN(float64(r.Score)) || math.IsInf(float64(r.Score), 0) {
				t.Fatalf("non-finite score: %v", r.Score)
			}
		}
	})
}
