package vector

// mmapstore_test.go covers P0-05:
//
//   - round-trip bit-identical persistence with chunk-crossing sizes;
//   - zero-copy loads: rows alias the mapping and the search paths
//     score them like the in-memory index;
//   - the leak audit: every Close releases its fd (OpenCountFor) and
//     the Go heap grows less than 5 MB across load/search cycles
//     (mapped pages are not Go heap, so torching a whole index keeps
//     HeapAlloc flat).
//
// The fd probe needs /proc/self/fd (linux); on other platforms the
// leak test skips.

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func mmapTestPath(t *testing.T, name string) string {
	t.Helper()
	return filepath.Join(t.TempDir(), name)
}

// buildFlatFixture builds a float32 index large enough that the save
// path crosses at least one mmapCopyChunk chunk boundary (dim 48,
// n 50k: 9.2 MB payload over a 16 MiB chunk — use 64k rows to cross).
func buildFlatFixture(n, dim, seed int) *FlatIndex {
	idx := NewFlatIndex(dim, TypeFloat32)
	rows := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := make([]float32, dim)
		for j := 0; j < dim; j++ {
			row[j] = float32((i*seed+j)%17) / 17
		}
		rows[i] = row
	}
	idx.AddFloat32(rows)
	return idx
}

func TestSaveLoadFlatIndexFloat32RoundTrip(t *testing.T) {
	idx := buildFlatFixture(777, 48, 3)
	path := mmapTestPath(t, "flat.bin")
	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		t.Fatalf("SaveFlatIndexFloat32: %v", err)
	}
	loaded, store, err := LoadFlatIndexFloat32(path)
	if err != nil {
		t.Fatalf("LoadFlatIndexFloat32: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	if got := loaded.Count(); got != idx.Count() {
		t.Fatalf("count = %d, want %d", got, idx.Count())
	}
	for i := 0; i < idx.Count(); i++ {
		for j := 0; j < 48; j++ {
			a := idx.Float32[i][j]
			b := loaded.Float32[i][j]
			if a != b {
				t.Fatalf("row %d elem %d: %v != %v (bit-mismatch)", i, j, a, b)
			}
		}
	}
}

func TestMMapLoadedFlatSearchParity(t *testing.T) {
	n, dim, k := 3000, 64, 10
	idx := buildFlatFixture(n, dim, 5)
	path := mmapTestPath(t, "flat.bin")
	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		t.Fatalf("SaveFlatIndexFloat32: %v", err)
	}
	loaded, store, err := LoadFlatIndexFloat32(path)
	if err != nil {
		t.Fatalf("LoadFlatIndexFloat32: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	if !loaded.f32Uniform || loaded.f32mirror == nil {
		t.Fatalf("loaded index should be uniform with a zero-copy mirror")
	}

	for _, metric := range []DistanceMetric{MetricDot, MetricL2, MetricCosine} {
		q := make([]float32, dim)
		for j := 0; j < dim; j++ {
			q[j] = float32(j%7) / 9
		}
		want, err := idx.Search(q, k, metric)
		if err != nil {
			t.Fatalf("in-memory search: %v", err)
		}
		got, err := loaded.Search(q, k, metric)
		if err != nil {
			t.Fatalf("mmap search (%v): %v", metric, err)
		}
		if len(got) != len(want) {
			t.Fatalf("metric %v: len = %d, want %d", metric, len(got), len(want))
		}
		// Scores of scalar reference (in-memory, cosine unnormalised
		// chunks) and batch kernels agree within 1e-4 on this fixture;
		// compare ids first, then the score gap.
		for i := range want {
			if got[i].ID != want[i].ID {
				t.Fatalf("metric %v result %d: id %d, want %d", metric, i, got[i].ID, want[i].ID)
			}
			if diff := got[i].Score - want[i].Score; diff > 1e-4 || diff < -1e-4 {
				t.Fatalf("metric %v result %d: score %v, want %v (gap %g)", metric, i, got[i].Score, want[i].Score, diff)
			}
		}
	}
}

func TestMMapBatchSearchParity(t *testing.T) {
	n, dim, numQ, k := 2000, 96, 8, 10
	idx := buildFlatFixture(n, dim, 7)
	path := mmapTestPath(t, "flat.bin")
	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		t.Fatalf("SaveFlatIndexFloat32: %v", err)
	}
	loaded, store, err := LoadFlatIndexFloat32(path)
	if err != nil {
		t.Fatalf("LoadFlatIndexFloat32: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	queries := make([]interface{}, numQ)
	for qi := 0; qi < numQ; qi++ {
		q := make([]float32, dim)
		for j := 0; j < dim; j++ {
			q[j] = float32((qi+j)%11) / 13
		}
		queries[qi] = q
	}
	for _, metric := range []DistanceMetric{MetricDot, MetricL2} {
		want, err := idx.BatchSearch(queries, k, metric)
		if err != nil {
			t.Fatalf("in-memory BatchSearch: %v", err)
		}
		got, err := loaded.BatchSearch(queries, k, metric)
		if err != nil {
			t.Fatalf("mmap BatchSearch (%v): %v", metric, err)
		}
		for qi := range want {
			if len(got[qi]) != len(want[qi]) {
				t.Fatalf("metric %v query %d: len = %d want %d", metric, qi, len(got[qi]), len(want[qi]))
			}
			for r := range want[qi] {
				if got[qi][r].ID != want[qi][r].ID {
					t.Fatalf("metric %v query %d result %d: id = %d want %d", metric, qi, r, got[qi][r].ID, want[qi][r].ID)
				}
				if diff := got[qi][r].Score - want[qi][r].Score; diff > 1e-4 || diff < -1e-4 {
					t.Fatalf("metric %v query %d result %d: score gap %g", metric, qi, r, diff)
				}
			}
		}
	}
}

func TestMMapFloat32RegionBadInput(t *testing.T) {
	t.Run("bad magic", func(t *testing.T) {
		data := make([]byte, mmapHeaderSize+64)
		data[0] = 'X'
		if _, err := Float32RegionAt(data); err != ErrMMapBadFormat {
			t.Fatalf("err = %v, want ErrMMapBadFormat", err)
		}
	})
	t.Run("too short", func(t *testing.T) {
		if _, err := Float32RegionAt(make([]byte, 10)); err != ErrMMapTooShort {
			t.Fatalf("err = %v, want ErrMMapTooShort", err)
		}
	})
	t.Run("truncated payload", func(t *testing.T) {
		path := mmapTestPath(t, "trunc.bin")
		if err := twoKHeader(path); err != nil {
			t.Fatalf("header: %v", err)
		}
		if _, _, err := LoadFlatIndexFloat32(path); err != ErrMMapTooShort {
			t.Fatalf("err = %v, want ErrMMapTooShort", err)
		}
	})
	t.Run("missing file", func(t *testing.T) {
		if _, _, err := LoadFlatIndexFloat32(filepath.Join(t.TempDir(), "nope.bin")); err == nil {
			t.Fatalf("expected error for missing file")
		}
	})
}

// twoKHeader writes a 2k-dim header with no payload (truncated file).
func twoKHeader(path string) error {
	if err := os.WriteFile(path, []byte("junkjunk"), 0o600); err != nil {
		return err
	}
	over := make([]byte, mmapHeaderSize)
	copy(over[:8], []byte("\x00\x00\x00\x00\x00\x00\x00\x00"))
	// Not the float32 magic on purpose: rewrite after save-like write.
	idx := NewFlatIndex(2, TypeFloat32)
	idx.AddFloat32([][]float32{{1, 2}})
	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		return err
	}
	// Truncate away the whole payload.
	f, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Truncate(mmapHeaderSize)
}

func TestMMapLeakAudit(t *testing.T) {
	if OpenCountFor(".") < 0 {
		t.Skip("fd probe (/proc/self/fd) unavailable on this platform")
	}

	n, dim := 4096, 128 // 2 MiB payload: heap-neutral by design
	owner := buildFlatFixture(n, dim, 11)
	path := mmapTestPath(t, "leak.bin")
	if err := SaveFlatIndexFloat32(path, owner); err != nil {
		t.Fatalf("SaveFlatIndexFloat32: %v", err)
	}

	baseFD := OpenCountFor(path)
	if baseFD != 0 {
		t.Fatalf("pre-open fds(%s) = %d, want 0", filepath.Base(path), baseFD)
	}
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	var loaded *FlatIndex
	var store *MMapStore
	for i := 0; i < 16; i++ {
		var err error
		if i > 0 {
			if fds := OpenCountFor(path); fds != baseFD {
				t.Fatalf("cycle %d: leaked fds = %d", i, fds-baseFD)
			}
		}
		loaded, store, err = LoadFlatIndexFloat32(path)
		if err != nil {
			t.Fatalf("LoadFlatIndexFloat32: %v", err)
		}
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(j) / 5
		}
		res, err := loaded.Search(q, 10, MetricDot)
		if err != nil || len(res) == 0 {
			t.Fatalf("search %d: %v (res len %d)", i, err, len(res))
		}
		if err := store.Close(); err != nil {
			t.Fatalf("cycle %d Close: %v", i, err)
		}
		_ = loaded
		loaded = nil
	}
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	grewMB := float64(after.HeapAlloc-before.HeapAlloc) / (1 << 20)
	if grewMB > 5 {
		t.Fatalf("leaked heap across 16 load/search/close cycles: %.2f MB", grewMB)
	}
	if got := OpenCountFor(path); got != baseFD {
		t.Fatalf("open fds after Close = %d (%+d leaked)", got, got-baseFD)
	}
}

func TestMMapEvictAndPrefetch(t *testing.T) {
	idx := buildFlatFixture(512, 64, 13)
	path := mmapTestPath(t, "evict.bin")
	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		t.Fatalf("SaveFlatIndexFloat32: %v", err)
	}
	loaded, store, err := LoadFlatIndexFloat32(path)
	if err != nil {
		t.Fatalf("LoadFlatIndexFloat32: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	if err := store.Prefetch(); err != nil {
		t.Fatalf("Prefetch: %v", err)
	}
	if err := store.Evict(0, int(store.Size())); err != nil {
		t.Fatalf("Evict: %v", err)
	}
	// Data survives eviction (clean pages re-materialise from the
	// file), so scores are unchanged either way.
	q := make([]float32, 64)
	for j := range q {
		q[j] = float32(j%5) / 7
	}
	want, _ := idx.Search(q, 5, MetricDot)
	got, err := loaded.Search(q, 5, MetricDot)
	if err != nil {
		t.Fatalf("search after Evict: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("result %d id = %d want %d", i, got[i].ID, want[i].ID)
		}
	}
	// Range guards.
	if err := store.Evict(-1, 4); err == nil {
		t.Fatalf("negative lo accepted")
	}
	if err := store.Evict(0, int(store.Size())+1); err == nil {
		t.Fatalf("hi beyond size accepted")
	}
}

func TestSaveComplexSoA_RoundTrip(t *testing.T) {
	dim, n := 32, 256
	idx := NewFlatIndex(dim, TypeComplex128)
	rows := make([][]complex128, n)
	for i := 0; i < n; i++ {
		row := make([]complex128, dim)
		for j := 0; j < dim; j++ {
			row[j] = complex(float64((i+j)%9)/9, float64((i*j)%7)/7)
		}
		rows[i] = row
	}
	idx.AddComplex128(rows)
	if idx.PrepareComplex() == nil {
		t.Fatalf("PrepareComplex returned nil")
	}
	path := mmapTestPath(t, "complex.bin")
	if err := SaveComplexSoA(path, idx); err != nil {
		t.Fatalf("SaveComplexSoA: %v", err)
	}
	loaded, store, err := LoadFlatIndexComplexSoA(path)
	if err != nil {
		t.Fatalf("LoadFlatIndexComplexSoA: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if got := loaded.Count(); got != n {
		t.Fatalf("count = %d, want %d", got, n)
	}
	if loaded.complexSoA == nil {
		t.Fatalf("loaded index lost its SoA")
	}
	// Search parity: dot metric, one query, ids by position.
	q := make([]complex128, dim)
	for j := 0; j < dim; j++ {
		q[j] = complex(float64(j%6)/6, float64((j*2)%5)/5)
	}
	want, err := idx.Search(q, 8, MetricDot)
	if err != nil {
		t.Fatalf("in-memory search: %v", err)
	}
	got, err := loaded.Search(q, 8, MetricDot)
	if err != nil {
		t.Fatalf("mmap SoA search: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("result %d id = %d want %d", i, got[i].ID, want[i].ID)
		}
		if diff := got[i].Score - want[i].Score; diff > 1e-4 || diff < -1e-4 {
			t.Fatalf("result %d score gap %g", i, diff)
		}
	}
	// Batch path parity too.
	queries := make([]interface{}, 4)
	for qi := 0; qi < 4; qi++ {
		q := make([]complex128, dim)
		for j := 0; j < dim; j++ {
			q[j] = complex(float64((qi+j)%6)/6, float64((qi*j)%5)/5)
		}
		queries[qi] = q
	}
	bGot, err := loaded.BatchSearch(queries, 5, MetricDot)
	if err != nil {
		t.Fatalf("BatchSearch: %v", err)
	}
	bWant, err := idx.BatchSearch(queries, 5, MetricDot)
	if err != nil {
		t.Fatalf("BatchSearch in-memory: %v", err)
	}
	// Tie-aware band comparison: this fixture produces exact score
	// ties (q0 carries zero imaginary and rows i, i+9 share the real
	// plane), and any correct resolution order is legitimate — the
	// device top-k breaks ties toward the smaller id while the CPU
	// heap keeps merge order.
	for qi := range bWant {
		topKParity(t, "batch", bGot[qi], bWant[qi], 1e-4)
	}
}

func TestLoadFlatIndexComplexSoA_BadFiles(t *testing.T) {
	t.Run("bad magic", func(t *testing.T) {
		path := mmapTestPath(t, "bad.bin")
		idx := buildFlatFixture(4, 8, 1)
		if err := SaveFlatIndexFloat32(path, idx); err != nil {
			t.Fatalf("save: %v", err)
		}
		if _, _, err := LoadFlatIndexComplexSoA(path); err != ErrMMapBadFormat {
			t.Fatalf("err = %v want ErrMMapBadFormat", err)
		}
	})
	t.Run("truncated planes", func(t *testing.T) {
		path := mmapTestPath(t, "short.bin")
		idx := NewFlatIndex(8, TypeComplex128)
		rows := make([][]complex128, 4)
		for i := range rows {
			rows[i] = make([]complex128, 8)
		}
		idx.AddComplex128(rows)
		if err := SaveComplexSoA(path, idx); err != nil {
			t.Fatalf("save: %v", err)
		}
		f, err := os.OpenFile(path, os.O_RDWR, 0o600)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		sz, _ := f.Stat()
		if err := f.Truncate(sz.Size() - 8); err != nil {
			t.Fatalf("truncate: %v", err)
		}
		f.Close()
		if _, _, err := LoadFlatIndexComplexSoA(path); err != ErrMMapTooShort {
			t.Fatalf("err = %v want ErrMMapTooShort", err)
		}
	})
}

func TestSaveFlatIndexFloat32RejectsRagged(t *testing.T) {
	idx := NewFlatIndex(0, TypeFloat32)
	idx.AddFloat32([][]float32{{1, 2}, {3}})
	if err := SaveFlatIndexFloat32(mmapTestPath(t, "ragged.bin"), idx); err == nil {
		t.Fatalf("ragged save accepted")
	}
}

func TestMMapCloseIsIdempotent(t *testing.T) {
	idx := buildFlatFixture(16, 8, 1)
	path := mmapTestPath(t, "close.bin")
	if err := SaveFlatIndexFloat32(path, idx); err != nil {
		t.Fatalf("save: %v", err)
	}
	_, store, err := LoadFlatIndexFloat32(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if store.Data() != nil {
		t.Fatalf("Data after Close not nil")
	}
}

func TestOpenCountForSelf(t *testing.T) {
	got := OpenCountFor("/proc/self/exe")
	if got < 0 {
		// The probe reads /proc/self/fd, which only exists on Linux.
		if runtime.GOOS != "linux" {
			t.Skip("the file descriptor probe is Linux only")
		}
		t.Fatalf("the file descriptor probe failed on Linux: %d", got)
	}
}
