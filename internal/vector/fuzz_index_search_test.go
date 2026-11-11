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
