// Shape-matrix benchmarks for internal/vector.
//
// These mirror the methodology of cmd/vector_benchmark (50 sequential
// queries per operation, top-k = 10, 16 probes for IVF, HNSW capped at
// 50k nodes and sqrt(n) centroids clamped to [16, 512]) so the numbers
// here are directly comparable with the roadmap's blockers
// (docs/roadmap.md, "Vector search").
//
// The dataset sizes come from the environment so CI can run the default
// (50k) cheaply while the full shape matrix is reproducible:
//
//	VECTOR_BENCH_SIZES=50000,100000,250000 go test -bench=BenchmarkVectorSearch ./internal/vector/
//
// Every dataset is generated inside the sub-benchmark and freed before
// the next one runs.

package vector

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

var benchQueries = 50
var benchTopK = 10

const (
	benchHNSWMaxNodes   = 50000
	benchProbes         = 16
	benchCentroidsFloor = 16
	benchCentroidsCap   = 512
)

// benchDims is the shape matrix dimension axis.
var benchDims = []int{128, 384}

func benchSizes() []int {
	raw := strings.TrimSpace(os.Getenv("VECTOR_BENCH_SIZES"))
	var out []int
	seen := map[int]bool{}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil || v <= 0 || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out
}

func ivfCentroidCount(n int) int {
	c := n
	// Newton's iteration on a floating point seed approximates sqrt(n)
	// without pulling in math: x_{k+1} = (x_k + n/x_k) / 2.
	x := float64(n)
	if x > 1 {
		for i := 0; i < 24; i++ {
			x = (x + float64(n)/x) / 2
		}
	}
	c = int(x)
	if c < benchCentroidsFloor {
		c = benchCentroidsFloor
	}
	if c > benchCentroidsCap {
		c = benchCentroidsCap
	}
	return c
}

// benchBuilder generates one dataset of the matrix shape and converts it
// into the three storage representations. All fields are rebuilt fresh
// so a sub-benchmark never shares allocations with its neighbours.
type benchBuilder struct {
	n, d int

	f32  [][]float32
	cplx [][]complex128
	tq   []TurboQuantVector

	rot []float32
	qjl []float32
}

func (bb *benchBuilder) prepare(b *testing.B) {
	bStop(b)
	if bb.f32 == nil {
		bb.f32 = GenerateRandomFloat32Dataset(bb.n, bb.d, int64(bb.n+bb.d))
	}
	if bb.rot == nil {
		bb.rot = CreateIdentityMatrix(bb.d)
		bb.qjl = CreateRandomQJLMatrix(32, bb.d, 42)
	}
	bStart(b)
}

func (bb *benchBuilder) ensureComplex(b *testing.B) {
	bb.prepare(b)
	if bb.cplx == nil {
		bStop(b)
		bb.cplx = make([][]complex128, bb.n)
		for i, v := range bb.f32 {
			bb.cplx[i] = Float32ToComplex128(v)
		}
		bStart(b)
	}
}

func (bb *benchBuilder) ensureTurbo(b *testing.B) {
	bb.prepare(b)
	if bb.tq == nil {
		bStop(b)
		bb.tq = make([]TurboQuantVector, bb.n)
		for i, v := range bb.f32 {
			bb.tq[i] = QuantizeTurboQuant(v, i, bb.rot, bb.qjl, 32)
		}
		bStart(b)
	}
}

func (bb *benchBuilder) release() {
	bb.f32 = nil
	bb.cplx = nil
	bb.tq = nil
	bb.rot = nil
	bb.qjl = nil
	runtime.GC()
}

func bStop(b *testing.B) { b.StopTimer() }
func bStart(b *testing.B) {
	runtime.GC()
	b.StartTimer()
}

func BenchmarkFlatSearch(b *testing.B) {
	sizes := benchSizes()
	if len(sizes) == 0 {
		b.Skip("no VECTOR_BENCH_SIZES configured")
	}
	for _, n := range sizes {
		for _, d := range benchDims {
			// float32 flat: dot / l2 / cosine
			b.Run("float32_dot/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.prepare(b)
				idx := NewFlatIndex(d, TypeFloat32)
				idx.AddFloat32(bb.f32)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = idx.Search(queries[q], benchTopK, MetricDot)
					}
				}
			})
			b.Run("float32_l2/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.prepare(b)
				idx := NewFlatIndex(d, TypeFloat32)
				idx.AddFloat32(bb.f32)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = idx.Search(queries[q], benchTopK, MetricL2)
					}
				}
			})
			b.Run("float32_cosine/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.prepare(b)
				idx := NewFlatIndex(d, TypeFloat32)
				idx.AddFloat32(bb.f32)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = idx.Search(queries[q], benchTopK, MetricCosine)
					}
				}
			})
			// complex128 flat
			b.Run("complex128_dot/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.ensureComplex(b)
				idx := NewFlatIndex(d, TypeComplex128)
				idx.AddComplex128(bb.cplx)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = idx.Search(bb.cplx[q], benchTopK, MetricDot)
					}
				}
			})
			// turboquant flat
			b.Run("turboquant_dot/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.ensureTurbo(b)
				idx := NewFlatIndex(d, TypeTurboQuant)
				idx.AddTurboQuant(bb.tq)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = idx.Search(&bb.tq[q], benchTopK, MetricDot)
					}
				}
			})
		}
	}
}

func BenchmarkIVFSearch(b *testing.B) {
	sizes := benchSizes()
	if len(sizes) == 0 {
		b.Skip("no VECTOR_BENCH_SIZES configured")
	}
	for _, n := range sizes {
		for _, d := range benchDims {
			b.Run("float32/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				centroids := ivfCentroidCount(n)
				bb.prepare(b)
				ivf := NewIVFIndex(d, centroids, benchProbes, TypeFloat32)
				bStop(b)
				if err := ivf.Build(bb.f32); err != nil {
					b.Fatalf("ivf build: %v", err)
				}
				bStart(b)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = ivf.Search(queries[q], benchTopK)
					}
				}
			})
			b.Run("turboquant/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.ensureTurbo(b)
				ivf := NewIVFIndex(d, ivfCentroidCount(n), benchProbes, TypeTurboQuant)
				ivf.Turbo = bb.tq
				bStop(b)
				if err := ivf.Build(bb.f32); err != nil {
					b.Fatalf("ivf build: %v", err)
				}
				bStart(b)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = ivf.Search(queries[q], benchTopK)
					}
				}
			})
			b.Run("complex128/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.ensureComplex(b)
				ivf := NewIVFIndex(d, ivfCentroidCount(n), benchProbes, TypeComplex128)
				ivf.Complex = bb.cplx
				bStop(b)
				if err := ivf.Build(bb.f32); err != nil {
					b.Fatalf("ivf build: %v", err)
				}
				bStart(b)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = ivf.Search(queries[q], benchTopK)
					}
				}
			})
		}
	}
}

// BenchmarkIVFBuild measures the index construction itself (P0-04).
func BenchmarkIVFBuild(b *testing.B) {
	sizes := benchSizes()
	if len(sizes) == 0 {
		b.Skip("no VECTOR_BENCH_SIZES configured")
	}
	for _, n := range sizes {
		for _, d := range benchDims {
			b.Run("float32/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.prepare(b)
				centroids := ivfCentroidCount(n)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					ivf := NewIVFIndex(d, centroids, benchProbes, TypeFloat32)
					b.StartTimer()
					if err := ivf.Build(bb.f32); err != nil {
						b.Fatalf("ivf build: %v", err)
					}
				}
			})
		}
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	sizes := benchSizes()
	if len(sizes) == 0 {
		b.Skip("no VECTOR_BENCH_SIZES configured")
	}
	for _, n := range sizes {
		nodes := n
		if nodes > benchHNSWMaxNodes {
			nodes = benchHNSWMaxNodes
		}
		for _, d := range benchDims {
			b.Run("float32/"+strconv.Itoa(nodes)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.prepare(b)
				hnsw := NewHNSWIndex(d, 16, 32, TypeFloat32)
				bStop(b)
				if err := hnsw.Build(bb.f32[:nodes]); err != nil {
					b.Fatalf("hnsw build: %v", err)
				}
				bStart(b)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = hnsw.Search(queries[q], benchTopK)
					}
				}
			})
			b.Run("turboquant/"+strconv.Itoa(nodes)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.ensureTurbo(b)
				hnsw := NewHNSWIndex(d, 16, 32, TypeTurboQuant)
				hnsw.Turbo = bb.tq
				bStop(b)
				if err := hnsw.Build(bb.f32[:nodes]); err != nil {
					b.Fatalf("hnsw build: %v", err)
				}
				bStart(b)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = hnsw.Search(queries[q], benchTopK)
					}
				}
			})
			b.Run("complex128/"+strconv.Itoa(nodes)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.ensureComplex(b)
				hnsw := NewHNSWIndex(d, 16, 32, TypeComplex128)
				hnsw.Complex = bb.cplx
				bStop(b)
				if err := hnsw.Build(bb.f32[:nodes]); err != nil {
					b.Fatalf("hnsw build: %v", err)
				}
				bStart(b)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for q := 0; q < benchQueries; q++ {
						_, _ = hnsw.Search(queries[q], benchTopK)
					}
				}
			})
		}
	}
}

// Micro-benchmarks for the distance kernels themselves (P0-01 evidence).
func BenchmarkDistance(b *testing.B) {
	for _, d := range benchDims {
		a := GenerateRandomFloat32Dataset(2, d, 7)[0]
		cq := Float32ToComplex128(a)
		rot := CreateIdentityMatrix(d)
		qjl := CreateRandomQJLMatrix(32, d, 42)
		tqA := QuantizeTurboQuant(a, 0, rot, qjl, 32)
		tqB := QuantizeTurboQuant(a, 1, rot, qjl, 32)

		b.Run("L2Float32_"+strconv.Itoa(d), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = L2Float32(a, a)
			}
		})
		b.Run("DotFloat32_"+strconv.Itoa(d), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = DotFloat32(a, a)
			}
		})
		b.Run("L2Complex128_"+strconv.Itoa(d), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = L2Complex128(cq, cq)
			}
		})
		b.Run("DotComplex128_"+strconv.Itoa(d), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = DotComplex128(cq, cq)
			}
		})
		b.Run("DotTurboQuant_"+strconv.Itoa(d), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = DotTurboQuant(&tqA, &tqB)
			}
		})
		b.Run("L2TurboQuant_"+strconv.Itoa(d), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = L2TurboQuant(&tqA, &tqB)
			}
		})
	}
}

// BenchmarkFlatBatchSearch measures BatchSearch over the same 50-query
// batches (P0-02 evidence for the multi-query executive). One cycle
// scores one 50-query mount against the store, so the numbers are
// directly comparable with the BenchmarkFlatSearch legs above (50
// sequential single-query sweeps there, one multi-query pass here —
// the same 50 x n distance evaluations per operation).
func BenchmarkFlatBatchSearch(b *testing.B) {
	sizes := benchSizes()
	if len(sizes) == 0 {
		b.Skip("no VECTOR_BENCH_SIZES configured")
	}
	for _, n := range sizes {
		for _, d := range benchDims {
			b.Run("float32_dot/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.prepare(b)
				idx := NewFlatIndex(d, TypeFloat32)
				idx.AddFloat32(bb.f32)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				qs := make([]interface{}, benchQueries)
				for q := range qs {
					qs[q] = queries[q]
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = idx.BatchSearch(qs, benchTopK, MetricDot)
				}
			})
			b.Run("complex128_dot/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.ensureComplex(b)
				idx := NewFlatIndex(d, TypeComplex128)
				idx.AddComplex128(bb.cplx)
				qs := make([]interface{}, benchQueries)
				for q := range qs {
					qs[q] = bb.cplx[q]
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = idx.BatchSearch(qs, benchTopK, MetricDot)
				}
			})
			b.Run("float32_cosine/"+strconv.Itoa(n)+"_"+strconv.Itoa(d), func(b *testing.B) {
				bb := &benchBuilder{n: n, d: d}
				defer bb.release()
				bb.prepare(b)
				idx := NewFlatIndex(d, TypeFloat32)
				idx.AddFloat32(bb.f32)
				queries := GenerateRandomFloat32Dataset(benchQueries, d, 999)
				qs := make([]interface{}, benchQueries)
				for q := range qs {
					qs[q] = queries[q]
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = idx.BatchSearch(qs, benchTopK, MetricCosine)
				}
			})
		}
	}
}
