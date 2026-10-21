package vector

// distance_alloc_test.go pins the no-allocation contract of the hot
// kernels (P0 block): the batch/multi-query call sites and their
// scalar fallbacks must score preallocated buffers without touching
// the heap, so steady-state Search/BatchSearch runs GC-silent. Run
// with an env the whole package shares (see the P0 report) — no global
// dispatch forcing here: the tests pass either kernel set (or nil, the
// scalar fallback) so every build configuration is covered.

import (
	"math/rand"
	"testing"
)

func runKernelAllocs(ks *distanceKernelSet) func() {
	// Fixtures built once, OUTSIDE the measured closure: only the
	// scoring calls themselves are measured.
	rng := rand.New(rand.NewSource(1))
	q64 := parityQuery(64, rng)
	rows64 := parityRows(32, 64, rng)
	rows16 := parityRows(16, 64, rng)
	qs4 := parityRows(4, 64, rng)
	outF32 := make([]float32, 256)
	re := parityRows(16, 32, rng)
	im := parityRows(16, 32, rng)
	qRe := f32ToF64(parityRows(1, 32, rng))
	qIm := f32ToF64(parityRows(1, 32, rng))
	reF64, imF64 := f32ToF64(re), f32ToF64(im)
	outF64 := make([]float64, 16)
	a := parityQuery(384, rng)
	row := parityRows(1, 384, rng)
	qRe64, rRe64 := f32ToF64(a), f32ToF64(row)
	qIm64, rIm64 := f32ToF64(a[:64]), f32ToF64(row[:64])

	count, dim := 8, 32
	rot := CreateIdentityMatrix(dim)
	tvecs := make([]TurboQuantVector, count)
	for i := range tvecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64()) * 0.4
		}
		tvecs[i] = QuantizeTurboQuant(v, i, rot, nil, 0)
		RefreshTurboQuantWords(&tvecs[i])
	}
	tpacked := NewTurboPackedColumns(tvecs, dim)
	tq := QuantizeTurboQuant(parityQuery(dim, rng), -1, rot, nil, 0)
	RefreshTurboQuantWords(&tq)
	tprep := PrepareTurboQuantQuery(&tq)
	tout := make([]float32, count)

	hotF32 := func() {
		scoreFloat32Batch(ks, true, q64, rows64, 64, 32, outF32)
		scoreFloat32Batch(ks, false, q64, rows64, 64, 32, outF32)
	}
	mqF32 := func() {
		if ks != nil {
			scoreMQFloat32(ks, false, qs4, 4, rows16, 64, 16, outF32)
		} else {
			scoreMQFloat32All(false, qs4, 4, rows16, 64, 16, outF32)
		}
	}
	csoa := func() {
		scoreComplexSoABatch(ks, true, qRe, qIm, reF64, imF64, 32, 0, 16, outF64)
		scoreComplexSoABatch(ks, false, qRe, qIm, reF64, imF64, 32, 0, 16, outF64)
	}
	scalarOracles := func() {
		_ = dotFloat32Scalar(a, row)
		_ = l2Float32Scalar(a, row)
		_ = dotComplex128Scalars(qRe64, qIm64, rRe64, rIm64)
		_ = l2Complex128Scalars(qRe64, qIm64, rRe64, rIm64)
	}
	turbo := func() {
		scoreTurboBatch(ks, tprep, tpacked, 0, count, tout)
	}
	return func() {
		hotF32()
		mqF32()
		csoa()
		scalarOracles()
		turbo()
	}
}

func TestHotKernelsAllocateNothing(t *testing.T) {
	for _, tc := range []struct {
		name string
		ks   *distanceKernelSet
	}{
		{"native", kernelSetFor()},
		{"scalar-fallback", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fn := runKernelAllocs(tc.ks)
			if avg := testing.AllocsPerRun(50, fn); avg != 0 {
				t.Fatalf("%s site allocates %g per run", tc.name, avg)
			}
		})
	}
}

func TestSingleVectorScalarsAllocateNothing(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	a := parityQuery(384, rng)
	b := parityRows(1, 384, rng)
	fn := func() {
		_ = dotFloat32Scalar(a, b)
		_ = l2Float32Scalar(a, b)
	}
	if avg := testing.AllocsPerRun(50, fn); avg != 0 {
		t.Fatalf("scalar kernels allocate %g per run", avg)
	}
}
