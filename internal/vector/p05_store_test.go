package vector

// The acceptance test for the memory-mapped vector store (P0-05).
//
// The store keeps the vectors in a file and maps it, so the rows a search
// reads are file pages rather than Go heap. The test checks the property
// that matters: after the index is loaded, the Go heap holds a fraction
// of the data and the search does not grow it.
//
// The full-scale run (a million vectors of 384 dimensions, 1.54 GB) is in
// docs/performance.md; this is the same check at a size a test can run
// in a second.

import (
	"os"
	"runtime"
	"testing"
)

const (
	p05Rows = 100_000
	p05Dim  = 384
)

func heapBytes() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

func TestTheMillionVectorStoreStaysOutOfTheHeap(t *testing.T) {
	path := t.TempDir() + "/store.bin"

	idx := NewFlatIndex(p05Dim, TypeFloat32)
	for chunk := 0; chunk < 10; chunk++ {
		idx.AddFloat32(GenerateRandomFloat32Dataset(p05Rows/10, p05Dim, int64(1000+chunk)))
	}
	if idx.Count() != p05Rows {
		t.Fatalf("the index holds %d vectors, want %d", idx.Count(), p05Rows)
	}

	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		t.Fatalf("save: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	fileSize := info.Size()
	// The file holds the rows plus a small header.
	wantSize := int64(p05Rows*p05Dim*4) + 24
	if fileSize != wantSize {
		t.Fatalf("the store holds %d bytes, want %d", fileSize, wantSize)
	}

	// The generator's rows are released before the load, so the heap
	// starts small.
	idx = nil
	runtime.GC()
	beforeLoad := heapBytes()

	loaded, store, err := LoadFlatIndexFloat32(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer store.Close()

	if loaded.Count() != p05Rows {
		t.Fatalf("the loaded index holds %d vectors, want %d", loaded.Count(), p05Rows)
	}
	if loaded.Dimension != p05Dim {
		t.Fatalf("the loaded index has %d dimensions, want %d", loaded.Dimension, p05Dim)
	}

	afterLoad := heapBytes()
	if growth := afterLoad - beforeLoad; growth > uint64(fileSize)/4 {
		t.Errorf("the load grew the Go heap by %d bytes, want less than a quarter of the %d byte file",
			growth, fileSize)
	}

	// A search reads the mapped rows, so it does not grow the heap.
	query := make([]float32, p05Dim)
	copy(query, loaded.Float32[0][:p05Dim])

	beforeSearch := heapBytes()
	for i := 0; i < 10; i++ {
		if _, err := loaded.Search(query, 10, MetricDot); err != nil {
			t.Fatalf("search %d: %v", i, err)
		}
	}
	afterSearch := heapBytes()
	if growth := afterSearch - beforeSearch; growth > uint64(fileSize)/4 {
		t.Errorf("ten searches grew the Go heap by %d bytes, want less than a quarter of the %d byte file",
			growth, fileSize)
	}

	// The rows alias the mapping: the store does not copy them.
	if store.Size() != fileSize {
		t.Errorf("the mapping holds %d bytes, want %d", store.Size(), fileSize)
	}
}
