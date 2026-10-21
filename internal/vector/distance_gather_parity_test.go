package vector

// The gather kernels: they score the rows an index list names, which is
// what an inverted-list scan does.
//
// The scan used to copy every candidate into a scratch buffer before it
// could be scored. The copy cost as much as the distance: for a 250k
// index with sixteen probes it moves twelve megabytes per query.
//
// The tests below check that the gather kernels give the same numbers as
// the per-row reference, that they reject the shapes they must reject,
// and that the IVF scan still finds the same neighbours as the reference
// loop.

import (
	"math/rand"
	"sort"
	"testing"
)

func TestKernelParityGatherF32(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	for _, entry := range []struct {
		name string
		set  *distanceKernelSet
	}{
		{"avx2", avx2KernelSet},
		{"avx512", avx512KernelSet},
	} {
		t.Run(entry.name, func(t *testing.T) {
			ks := kernelOrSkip(t, entry.set, entry.name)

			for _, tc := range []struct {
				rows, dim, count int
			}{
				{rows: 8, dim: 16, count: 8},
				{rows: 64, dim: 63, count: 17},
				{rows: 300, dim: 384, count: 256},
			} {
				q := parityQuery(tc.dim, rng)
				rows := parityRows(tc.rows, tc.dim, rng)
				indices := make([]int32, tc.count)
				for i := range indices {
					indices[i] = int32(rng.Intn(tc.rows))
				}

				out := make([]float32, tc.count)
				if !ks.batchDotFloat32Gather(q, rows, indices, tc.dim, tc.count, out) {
					t.Fatalf("%s: the gather dot rejected a legal block", entry.name)
				}
				want := make([]float32, tc.count)
				for i, idx := range indices {
					want[i] = dotFloat32Scalar(q, rows[int(idx)*tc.dim:int(idx)*tc.dim+tc.dim])
				}
				closeBlock(t, "gatherDot", out, want, parityTolF32)

				if !ks.batchL2Float32Gather(q, rows, indices, tc.dim, tc.count, out) {
					t.Fatalf("%s: the gather L2 rejected a legal block", entry.name)
				}
				for i, idx := range indices {
					want[i] = l2Float32Scalar(q, rows[int(idx)*tc.dim:int(idx)*tc.dim+tc.dim])
				}
				closeBlock(t, "gatherL2", out, want, parityTolF32)
			}

			// The contract: a rejected shape returns false, it never
			// panics and it never reads out of bounds.
			q := parityQuery(16, rng)
			rows := parityRows(8, 16, rng)
			indices := make([]int32, 4)
			out := make([]float32, 4)

			if ks.batchDotFloat32Gather(nil, rows, indices, 16, 4, out) {
				t.Fatalf("%s: an empty query was accepted", entry.name)
			}
			if ks.batchDotFloat32Gather(q, rows, indices, 16, 0, out) {
				t.Fatalf("%s: a zero count was accepted", entry.name)
			}
			if ks.batchDotFloat32Gather(q, rows, indices[:2], 16, 4, out) {
				t.Fatalf("%s: too few indices were accepted", entry.name)
			}
			if ks.batchDotFloat32Gather(q, rows, indices, 16, 4, out[:2]) {
				t.Fatalf("%s: a small output was accepted", entry.name)
			}
		})
	}
}

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
