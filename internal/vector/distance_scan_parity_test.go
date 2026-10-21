package vector

// The inverted-list scan.
//
// The scan used to copy every candidate into a scratch buffer before it
// could be scored. The copy cost as much as the distance: for a 250k index
// with sixteen probes it moves twelve megabytes per query. The rows of a
// list are now stored next to each other, so the scan reads them where
// they are.

import (
	"sort"
	"testing"
)

// The inverted-list scan reads the rows of a list out of the
// contiguous store: the rows of a list are next to each other, so the
// scan is a sequential read.
//
// The test below checks that the scan returns the same neighbours as the
// per-vector reference loop, which is what the gather kernels were
// checked for before the store was reordered.

// TestTheIVFScanchangesNothing checks that the gather scan returns the
// same neighbours as the per-vector reference loop.
//
// The scan is the one place the gather kernels are used, and the
// neighbours it returns are what a caller sees.
func TestTheIVFScanchangesNothing(t *testing.T) {
	if kernelSetFor() == nil {
		t.Skip("no native kernel set in this configuration")
	}

	const (
		count = 2000
		dim   = 64
		topK  = 10
	)

	vectors := GenerateRandomFloat32Dataset(count, dim, 11)

	idx := NewIVFIndex(dim, 16, 8, TypeFloat32)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("failed to build the index: %v", err)
	}

	query := GenerateRandomFloat32Dataset(1, dim, 12)[0]

	got, err := idx.Search(query, topK)
	if err != nil {
		t.Fatalf("the search failed: %v", err)
	}
	if len(got) != topK {
		t.Fatalf("the search returned %d neighbours, want %d", len(got), topK)
	}

	// The reference: the same candidates the scan looks at, scored one
	// by one. The search is approximate -- it only scans the probed lists
	// -- so the comparison is with the scan of those lists, not with a
	// brute-force search of the whole index.
	// The probed lists: the same selection the search makes.
	type centroidDist struct {
		id   int
		dist float32
	}
	dists := make([]centroidDist, len(idx.Centroids))
	for c := range idx.Centroids {
		dists[c] = centroidDist{id: c, dist: L2Float32(query, idx.Centroids[c])}
	}
	for i := 0; i < idx.NumProbes && i < len(dists); i++ {
		best := i
		for j := i + 1; j < len(dists); j++ {
			if dists[j].dist < dists[best].dist {
				best = j
			}
		}
		dists[i], dists[best] = dists[best], dists[i]
	}
	probed := make([]int, 0, idx.NumProbes)
	for i := 0; i < idx.NumProbes && i < len(dists); i++ {
		probed = append(probed, dists[i].id)
	}
	sort.Ints(probed)

	type scored struct {
		id    int
		score float32
	}
	all := make([]scored, 0, count)
	for _, c := range probed {
		for _, id := range idx.InvertedLists[c] {
			if id < 0 || id >= len(vectors) {
				continue
			}
			all = append(all, scored{id: id, score: dotFloat32Scalar(query, vectors[id])})
		}
	}
	sort.Slice(all, func(a, b int) bool {
		if all[a].score != all[b].score {
			return all[a].score > all[b].score
		}
		return all[a].id < all[b].id
	})
	if len(all) < topK {
		t.Fatalf("the probed lists hold %d candidates, want at least %d", len(all), topK)
	}

	for i, result := range got {
		if result.ID != all[i].id {
			t.Errorf("the neighbour %d is the vector %d, want %d (score %v against %v)",
				i, result.ID, all[i].id, result.Score, all[i].score)
		}
	}
}
