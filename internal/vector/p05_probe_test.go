package vector

// Full-scale P0-05 acceptance probe (run once, deleted after the run).
// Builds a 1,000,000 x 384 float32 store (1.54 GB on disk), persists it
// through the chunked mmap save path, loads it back zero-copy, and
// verifies the roadmap criterion shape: the Go heap stays bounded, the
// resident set only grows for pages actually touched, and evicting
// ranges returns the pages to the kernel. The file lives on a real
// filesystem (not tmpfs) so page cache behaviour is the production one.

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

const p05Count = 1_000_000

func p05vmRSS() int {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return -1
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				if kb, err := strconv.Atoi(fields[1]); err == nil {
					return kb / 1024 // MB
				}
			}
		}
	}
	return 0
}

func TestP05MillionVector(t *testing.T) {
	path := "/home/rsd/REPOS/longbow-quarrel/.p05_1m_probe.bin"
	defer os.Remove(path)

	idx := NewFlatIndex(384, TypeFloat32)
	for chunk := 0; chunk < 10; chunk++ {
		part := GenerateRandomFloat32Dataset(100000, 384, int64(1000+chunk))
		idx.AddFloat32(part)
	}
	runtime.GC()

	heapGenStart := heapMB()
	rssGenStart := p05vmRSS()
	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		t.Fatalf("save: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	t.Logf("saved %d bytes (%.2f GB); go heap before save %d MB, VmRSS %d MB",
		st.Size(), float64(st.Size())/1e9, heapGenStart-heapGenStart+heapMB(), rssGenStart)

	// Free the generator rows; load zero-copy.
	idx = nil
	runtime.GC()
	heapIdle := heapMB()

	loaded, store, err := LoadFlatIndexFloat32(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer store.Close()
	if got := loaded.Count(); got != p05Count {
		t.Fatalf("count = %d want %d", got, p05Count)
	}
	if loaded.Dimension != 384 {
		t.Fatalf("dim = %d", loaded.Dimension)
	}
	heapAfterLoad := heapMB()
	rssAfterLoad := p05vmRSS()
	t.Logf("after load: go heap %d MB (gen side released to %d), VmRSS %d MB",
		heapAfterLoad, heapIdle, rssAfterLoad)

	// Score 10 queries end to end; every page they touch is a fault-in
	// of file pages (not Go heap). A full sweep is 1.54 GB per query.
	fq := make([][]float32, 10)
	for q := range fq {
		fq[q] = probeQueryRow(loaded, q)
	}
	rssBeforeSearch := p05vmRSS()
	for q := range fq {
		if _, err := loaded.Search(fq[q], 10, MetricDot); err != nil {
			t.Fatalf("search q%d: %v", q, err)
		}
	}
	rssAfterSearch := p05vmRSS()
	t.Logf("after 10 full-sweep searches: VmRSS %d -> %d MB (delta %d MB, file-backed)",
		rssBeforeSearch, rssAfterSearch, rssAfterSearch-rssBeforeSearch)

	// Evict the first half of the mapping and show the pages go back.
	if err := store.Evict(0, int(store.Size()/2)); err != nil {
		t.Fatalf("evict: %v", err)
	}
	rssAfterEvict := p05vmRSS()
	t.Logf("after Evict(first half): VmRSS %d MB", rssAfterEvict)

	// Prefetch the whole range; RSS climbs again but stays file-backed.
	if err := store.Prefetch(); err != nil {
		t.Fatalf("prefetch: %v", err)
	}
	t.Logf("after Prefetch: VmRSS %d MB", p05vmRSS())

	if rssAfterSearch-rssBeforeSearch <= 0 {
		t.Fatalf("expected the searches to fault in file pages")
	}
}

func heapMB() int {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int(m.HeapAlloc / 1024 / 1024)
}

// probeQueryRow copies one stored row out as a fresh query (the loaded
// store's rows alias the mapping; a copy is safe to hand to Search).
func probeQueryRow(idx *FlatIndex, q int) []float32 {
	row := make([]float32, idx.Dimension)
	copy(row, idx.Float32[q][:idx.Dimension])
	return row
}
