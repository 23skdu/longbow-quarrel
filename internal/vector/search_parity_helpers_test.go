package vector

// search_parity_helpers_test.go holds the tie-aware top-k comparison
// shared by search parity tests. Executives that are numerically
// equivalent (scalar vs SIMD vs CUDA; fp64 index vs fp32 device
// planes) can resolve near-ties and exact ties differently, so
// per-position id equality is not a sound parity contract. What holds
// is: the two lists span the same score band — shared ids score
// identically within tol, and an id present on only one side scored
// within 2 tol of that side's k-th boundary score (one tol of
// deviation on each of the two scores forming the boundary).

import (
	"testing"
)

func topKParity(t *testing.T, name string, got, want []SearchResult, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len = %d, want %d", name, len(got), len(want))
	}
	if len(want) == 0 {
		return
	}
	wantScore := map[int]float32{}
	for _, s := range want {
		wantScore[s.ID] = s.Score
	}
	gotScore := map[int]float32{}
	for _, s := range got {
		gotScore[s.ID] = s.Score
	}
	wantBoundary := want[len(want)-1].Score
	gotBoundary := got[len(got)-1].Score
	for _, g := range got {
		ws, shared := wantScore[g.ID]
		if shared {
			if diff := g.Score - ws; diff > tol || diff < -tol {
				t.Fatalf("%s: id %d score %g vs want %g (diff %g > tol %g)", name, g.ID, g.Score, ws, diff, tol)
			}
			continue
		}
		// Missing from want: legal only within the tie budget of
		// want's boundary score.
		if g.Score < wantBoundary-2*tol {
			t.Fatalf("%s: id %d (score %g) scored below want's boundary %g by more than the tie budget", name, g.ID, g.Score, wantBoundary)
		}
	}
	for _, w := range want {
		if _, shared := gotScore[w.ID]; shared {
			continue
		}
		if w.Score < gotBoundary-2*tol {
			t.Fatalf("%s: id %d (score %g) scored below got's boundary %g by more than the tie budget", name, w.ID, w.Score, gotBoundary)
		}
	}
}
