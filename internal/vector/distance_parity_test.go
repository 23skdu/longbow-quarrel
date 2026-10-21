package vector

// distance_parity_test.go asserts the scalar-oracle contract of the
// kernel plumbing (distance_dispatch.go):
//
//   - every native kernel agrees with the pure-Go reference to 1e-4
//     (fp64 layouts to 1e-9, turbo to its exact integer reduction);
//   - every kernel rejects malformed shapes with ok=false so the
//     callers keep their fallback paths intact (the cgo pointer rules
//     ban pointer-array contracts, so all row blocks are contiguous
//     with an explicit stride — shapes are checked before any pointer
//     escape);
//   - the hot kernels allocate nothing (Testing.AllocsPerRun == 0).
//
// Kernels are invoked through their distanceKernelSet directly, never
// via the global vecLevel() forcing, so tests stay independent of
// global dispatch state. On builds without the cgo kernels the parity
// tests skip but the allocation tests still run (the scalar fallback
// must be allocation-free too).

import (
	"math/rand"
	"testing"
)

// parityTolF32 is the parity budget of the fp32 kernels against the
// fp32 scalar reference (unit-scaled fixtures).
const parityTolF32 = 1e-4

// parityTolF64 is the budget of the fp64 kernels against the fp64
// scalar reference.
const parityTolF64 = 1e-9

// kernelOrSkip returns the given kernel set, skipping when nil.
func kernelOrSkip(t *testing.T, ks *distanceKernelSet, name string) *distanceKernelSet {
	t.Helper()
	if ks == nil {
		t.Skipf("%s kernels not compiled in this configuration", name)
	}
	return ks
}

// parityRows builds a count x stride row-major block with values in
// (-1, 1) and a deterministic seed.
func parityRows(n, stride int, rng *rand.Rand) []float32 {
	rows := make([]float32, n*stride)
	for i := range rows {
		rows[i] = float32(rng.NormFloat64()) * 0.5
	}
	return rows
}

func parityQuery(dim int, rng *rand.Rand) []float32 {
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(rng.NormFloat64()) * 0.5
	}
	return q
}

// scalarBlockScores scores every row of the block with the scalar
// reference (the caller decides the oracle order).
func scalarBlockScores(dot bool, q, rows []float32, stride, count int) []float32 {
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		if dot {
			out[i] = dotFloat32Scalar(q, rows[i*stride:(i+1)*stride])
		} else {
			out[i] = l2Float32Scalar(q, rows[i*stride:(i+1)*stride])
		}
	}
	return out
}

func closeBlock(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len = %d want %d", name, len(got), len(want))
	}
	for i := range want {
		if d := absF32(got[i] - want[i]); d > tol {
			t.Fatalf("%s[%d]: got %g want %g (diff %g > tol %g)", name, i, got[i], want[i], d, tol)
		}
	}
}

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func absF64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestScalarOraclesSelfConsistent(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, dim := range []int{1, 3, 8, 9, 16, 63, 64, 384} {
		a := parityQuery(dim, rng)
		b := parityQuery(dim, rng)

		var dotNaive float32
		var l2Naive float32
		for i := range a {
			dotNaive += a[i] * b[i]
			d := a[i] - b[i]
			l2Naive += d * d
		}
		if diff := absF32(dotFloat32Scalar(a, b) - dotNaive); diff > 1e-2*absMax32(1, absF32(dotNaive)) {
			t.Fatalf("dim %d: dotFloat32Scalar diverges from naive: %g vs %g", dim, dotFloat32Scalar(a, b), dotNaive)
		}
		if diff := absF32(l2Float32Scalar(a, b) - l2Naive); diff > 1e-2*absMax32(1, absF32(l2Naive)) {
			t.Fatalf("dim %d: l2Float32Scalar diverges from naive: %g vs %g", dim, l2Float32Scalar(a, b), l2Naive)
		}
		// fp64 scalar oracles against the interleaved reference.
		qRe := make([]float64, dim)
		qIm := make([]float64, dim)
		re := make([]float64, dim)
		im := make([]float64, dim)
		var dotNaiveF64, l2NaiveF64 float64
		for i := 0; i < dim; i++ {
			qRe[i] = float64(a[i]) + 0.1
			qIm[i] = float64(b[i]) - 0.2
			re[i] = float64(a[i]) - 0.05
			im[i] = float64(b[i]) + 0.15
			dr := qRe[i] - re[i]
			di := qIm[i] - im[i]
			dotNaiveF64 += qRe[i]*re[i] + qIm[i]*im[i]
			l2NaiveF64 += dr*dr + di*di
		}
		if diff := absF64(dotComplex128Scalars(qRe, qIm, re, im) - dotNaiveF64); diff > 1e-9 {
			t.Fatalf("dim %d: dotComplex128Scalars diverges: %g vs %g", dim, dotComplex128Scalars(qRe, qIm, re, im), dotNaiveF64)
		}
		if diff := absF64(l2Complex128Scalars(qRe, qIm, re, im) - l2NaiveF64); diff > 1e-9 {
			t.Fatalf("dim %d: l2Complex128Scalars diverges: %g vs %g", dim, l2Complex128Scalars(qRe, qIm, re, im), l2NaiveF64)
		}
		// The interleaved public functions agree with the split oracles
		// on reconstructed rows (SoA contract).
		rowC := make([]complex128, dim)
		for j := 0; j < dim; j++ {
			rowC[j] = complex(re[j], im[j])
		}
		qC := make([]complex128, dim)
		for j := 0; j < dim; j++ {
			qC[j] = complex(qRe[j], qIm[j])
		}
		if got := DotComplex128(qC, rowC); absF64(got-dotNaiveF64) > 1e-9 {
			t.Fatalf("dim %d: DotComplex128 vs split oracle: %g vs %g", dim, got, dotNaiveF64)
		}
		if got := L2Complex128(qC, rowC); absF64(got-l2NaiveF64) > 1e-9 {
			t.Fatalf("dim %d: L2Complex128 vs split oracle: %g vs %g", dim, got, l2NaiveF64)
		}
	}
}

func absMax32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func TestKernelParitySingleF32(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, entry := range []struct {
		name string
		set  *distanceKernelSet
	}{
		{"avx2", avx2KernelSet},
		{"avx512", avx512KernelSet},
	} {
		t.Run(entry.name, func(t *testing.T) {
			ks := kernelOrSkip(t, entry.set, entry.name)
			for _, dim := range []int{16, 17, 63, 64, 384} {
				a := parityQuery(dim, rng)
				b := parityQuery(dim, rng)
				wantD := dotFloat32Scalar(a, b)
				wantL := l2Float32Scalar(a, b)
				if got := ks.dotFloat32(a, b); absF32(got-wantD) > parityTolF32 {
					t.Fatalf("%s dim %d: dot %g vs scalar %g", entry.name, dim, got, wantD)
				}
				if got := ks.l2Float32(a, b); absF32(got-wantL) > parityTolF32 {
					t.Fatalf("%s dim %d: l2 %g vs scalar %g", entry.name, dim, got, wantL)
				}
			}
		})
	}
}

func TestKernelParityBatchF32(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
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
				count, dim, stride int
			}{
				{1, 16, 16}, {5, 17, 17}, {7, 63, 64}, {16, 64, 64}, {63, 384, 384},
			} {
				q := parityQuery(tc.dim, rng)
				rows := parityRows(tc.count, tc.stride, rng)
				out := make([]float32, tc.count)
				if !ks.batchDotFloat32(q, rows, tc.stride, tc.count, out) {
					t.Fatalf("%s: batchDot rejected a legal block (count=%d dim=%d stride=%d)", entry.name, tc.count, tc.dim, tc.stride)
				}
				closeBlock(t, "batchDot", out, scalarBlockScores(true, q, rows, tc.stride, tc.count), parityTolF32)
				if !ks.batchL2Float32(q, rows, tc.stride, tc.count, out) {
					t.Fatalf("%s: batchL2 rejected a legal block", entry.name)
				}
				closeBlock(t, "batchL2", out, scalarBlockScores(false, q, rows, tc.stride, tc.count), parityTolF32)
			}
			// Contract rejects (must return false, never panic).
			q := parityQuery(16, rng)
			rows := parityRows(4, 16, rng)
			out := make([]float32, 4)
			if ks.batchDotFloat32(nil, rows, 16, 4, out) {
				t.Fatalf("%s: empty query accepted", entry.name)
			}
			if ks.batchDotFloat32(parityQuery(24, rng), rows, 16, 4, out) {
				t.Fatalf("%s: dim larger than stride accepted", entry.name)
			}
			if ks.batchDotFloat32(q, rows, 24, 4, out) {
				t.Fatalf("%s: rows shorter than count*stride accepted", entry.name)
			}
			if ks.batchDotFloat32(q, rows, 16, 4, out[:2]) {
				t.Fatalf("%s: small out accepted", entry.name)
			}
			if ks.batchDotFloat32(q, rows, 16, 0, out) {
				t.Fatalf("%s: zero count accepted", entry.name)
			}
		})
	}
}

func TestKernelParityMQF32(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
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
				numQ, count, dim, stride int
			}{
				{1, 16, 16, 16}, {3, 9, 17, 17}, {4, 15, 63, 63}, {2, 512, 64, 64},
			} {
				rows := parityRows(tc.count, tc.stride, rng)
				qs := make([]float32, 0, tc.numQ*tc.dim)
				want := make([]float32, 0, tc.numQ*tc.count)
				for qi := 0; qi < tc.numQ; qi++ {
					q := parityQuery(tc.dim, rng)
					qs = append(qs, q...)
					want = append(want, scalarBlockScores(true, q, rows, tc.stride, tc.count)...)
				}
				out := make([]float32, tc.numQ*tc.count)
				if !scoreMQFloat32(ks, false, qs, tc.numQ, rows, tc.stride, tc.count, out) {
					t.Fatalf("%s: mqDot rejected a legal block", entry.name)
				}
				closeBlock(t, "mqDot", out, want, parityTolF32)
				if !scoreMQFloat32(ks, true, qs, tc.numQ, rows, tc.stride, tc.count, out) {
					t.Fatalf("%s: mqL2 rejected a legal block", entry.name)
				}
				for qi := 0; qi < tc.numQ; qi++ {
					q := qs[qi*tc.dim : (qi+1)*tc.dim]
					wantL := scalarBlockScores(false, q, rows, tc.stride, tc.count)
					closeBlock(t, "mqL2", out[qi*tc.count:(qi+1)*tc.count], wantL, parityTolF32)
				}
			}
			// The scalar MQ fallback is the oracle for the same shapes.
			rows := parityRows(4, 16, rng)
			qs := parityRows(3, 16, rng)
			out := make([]float32, 12)
			if !scoreMQFloat32All(false, qs, 3, rows, 16, 4, out) {
				t.Fatalf("scalar MQ fallback rejected a legal block")
			}
			closeBlock(t, "mqScalar", out, func() []float32 {
				w := make([]float32, 0, 12)
				for qi := 0; qi < 3; qi++ {
					w = append(w, scalarBlockScores(true, qs[qi*16:(qi+1)*16], rows, 16, 4)...)
				}
				return w
			}(), 1e-9) // same arithmetic: exact
			if scoreMQFloat32All(true, qs[:17], 2, rows, 16, 4, out) {
				t.Fatalf("scalar MQ accepted len(qs)%%numQ != 0")
			}
			if scoreMQFloat32All(false, qs, 3, rows, 16, 4, out[:2]) {
				t.Fatalf("scalar MQ accepted small out")
			}
			// Kernel rejects.
			if scoreMQFloat32(ks, false, qs, 5, rows, 16, 4, out) {
				t.Fatalf("kernel MQ accepted len(qs)%%numQ != 0")
			}
			if scoreMQFloat32(ks, false, qs, 3, rows, 16, 4, out[:11]) {
				t.Fatalf("kernel MQ accepted small out")
			}
			if scoreMQFloat32(ks, false, qs, 3, rows, 12, 4, out) {
				t.Fatalf("kernel MQ accepted stride < dim")
			}
		})
	}
}

func TestKernelParityComplexSoA(t *testing.T) {
	rng := rand.New(rand.NewSource(51))
	for _, entry := range []struct {
		name string
		set  *distanceKernelSet
	}{
		{"avx2", avx2KernelSet},
		{"avx512", avx512KernelSet},
	} {
		t.Run(entry.name, func(t *testing.T) {
			ks := kernelOrSkip(t, entry.set, entry.name)
			count, dim := 7, 32
			re := make([]float64, count*dim)
			im := make([]float64, count*dim)
			for i := range re {
				re[i] = rng.NormFloat64()
				im[i] = rng.NormFloat64()
			}
			qRe := make([]float64, dim)
			qIm := make([]float64, dim)
			for j := range qRe {
				qRe[j] = rng.NormFloat64()
				qIm[j] = rng.NormFloat64()
			}
			outDot := make([]float64, count)
			outL2 := make([]float64, count)
			if !ks.dotComplexSoA(qRe, qIm, re, im, dim, 0, count, outDot) {
				t.Fatalf("%s: dotComplexSoA rejected a legal block", entry.name)
			}
			for i := 0; i < count; i++ {
				want := dotComplex128Scalars(qRe, qIm, re[i*dim:(i+1)*dim], im[i*dim:(i+1)*dim])
				if diff := absF64(outDot[i] - want); diff > parityTolF64 {
					t.Fatalf("%s: dot SoA[%d]: got %.17g want %.17g", entry.name, i, outDot[i], want)
				}
			}
			if !ks.l2ComplexSoA(qRe, qIm, re, im, dim, 0, count, outL2) {
				t.Fatalf("%s: l2ComplexSoA rejected a legal block", entry.name)
			}
			for i := 0; i < count; i++ {
				want := l2Complex128Scalars(qRe, qIm, re[i*dim:(i+1)*dim], im[i*dim:(i+1)*dim])
				if diff := absF64(outL2[i] - want); diff > parityTolF64 {
					t.Fatalf("%s: l2 SoA[%d]: got %.17g want %.17g", entry.name, i, outL2[i], want)
				}
			}
			// A start offset must land on the same row values.
			out2 := make([]float64, count-3)
			if !ks.dotComplexSoA(qRe, qIm, re, im, dim, 3, count, out2) {
				t.Fatalf("%s: dotComplexSoA rejected offset block", entry.name)
			}
			for i := range out2 {
				if diff := absF64(out2[i] - outDot[3+i]); diff > 1e-12 {
					t.Fatalf("%s: offset block row %d diverged: %g vs %g", entry.name, 3+i, out2[i], outDot[3+i])
				}
			}

			// Contract rejects.
			if ks.dotComplexSoA(qRe[:dim-1], qIm, re, im, dim, 0, count, outDot) {
				t.Fatalf("%s: short query accepted", entry.name)
			}
			if ks.dotComplexSoA(qRe, qIm, re, im, dim, 0, count, outDot[:2]) {
				t.Fatalf("%s: small out accepted", entry.name)
			}
			if ks.dotComplexSoA(qRe, qIm, re[:count*dim-1], im, dim, 0, count, outDot) {
				t.Fatalf("%s: short re plane accepted", entry.name)
			}
			if ks.dotComplexSoA(qRe, qIm, re, im, dim, -1, count, outDot) {
				t.Fatalf("%s: negative start accepted", entry.name)
			}
		})
	}
}

func f32ToF64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func TestKernelParityMQComplexSoA(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	for _, entry := range []struct {
		name string
		set  *distanceKernelSet
	}{
		{"avx2", avx2KernelSet},
		{"avx512", avx512KernelSet},
	} {
		t.Run(entry.name, func(t *testing.T) {
			ks := kernelOrSkip(t, entry.set, entry.name)
			numQ, count, dim := 3, 11, 24
			re := make([]float64, count*dim)
			im := make([]float64, count*dim)
			for i := range re {
				re[i] = rng.NormFloat64() * 0.5
				im[i] = rng.NormFloat64() * 0.5
			}
			qsRe := make([]float64, numQ*dim)
			qsIm := make([]float64, numQ*dim)
			for i := range qsRe {
				qsRe[i] = rng.NormFloat64() * 0.5
				qsIm[i] = rng.NormFloat64() * 0.5
			}
			out := make([]float32, numQ*count)
			if !ks.mqDotComplexSoA(qsRe, qsIm, numQ, re, im, dim, 0, count, out) {
				t.Fatalf("%s: mqDotComplexSoA rejected a legal block", entry.name)
			}
			for qi := 0; qi < numQ; qi++ {
				for i := 0; i < count; i++ {
					want := float32(dotComplex128Scalars(qsRe[qi*dim:(qi+1)*dim], qsIm[qi*dim:(qi+1)*dim], re[i*dim:(i+1)*dim], im[i*dim:(i+1)*dim]))
					if diff := absF32(out[qi*count+i] - want); diff > parityTolF32 {
						t.Fatalf("%s: mq dot q%d r%d: got %g want %g", entry.name, qi, i, out[qi*count+i], want)
					}
				}
			}
			if !ks.mqL2ComplexSoA(qsRe, qsIm, numQ, re, im, dim, 0, count, out) {
				t.Fatalf("%s: mqL2ComplexSoA rejected a legal block", entry.name)
			}
			for qi := 0; qi < numQ; qi++ {
				for i := 0; i < count; i++ {
					want := float32(l2Complex128Scalars(qsRe[qi*dim:(qi+1)*dim], qsIm[qi*dim:(qi+1)*dim], re[i*dim:(i+1)*dim], im[i*dim:(i+1)*dim]))
					if diff := absF32(out[qi*count+i] - want); diff > parityTolF32 {
						t.Fatalf("%s: mq l2 q%d r%d: got %g want %g", entry.name, qi, i, out[qi*count+i], want)
					}
				}
			}
			// Contract rejects.
			if ks.mqDotComplexSoA(qsRe, qsIm, 0, re, im, dim, 0, count, out) {
				t.Fatalf("%s: zero queries accepted", entry.name)
			}
			if ks.mqDotComplexSoA(qsRe[:numQ*dim-1], qsIm, numQ, re, im, dim, 0, count, out) {
				t.Fatalf("%s: short query planes accepted", entry.name)
			}
			if ks.mqDotComplexSoA(qsRe, qsIm, numQ, re, im, dim, 0, count, out[:numQ]) {
				t.Fatalf("%s: small out accepted", entry.name)
			}
			if ks.mqDotComplexSoA(qsRe, qsIm, numQ, re, im, dim, 0, count+1, out) {
				t.Fatalf("%s: range past the plane accepted", entry.name)
			}
		})
	}
}

// turboFixture builds quantized vectors with QJL words refreshed, the
// exact state the packed and pair kernels require.
func turboFixture(dim, count int, rng *rand.Rand, qjlRows int, seed int64) ([]TurboQuantVector, []float32, []float32) {
	rot := CreateIdentityMatrix(dim)
	qjl := CreateRandomQJLMatrix(qjlRows, dim, seed)
	vecs := make([]TurboQuantVector, count)
	for i := 0; i < count; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64()) * 0.4
		}
		vecs[i] = QuantizeTurboQuant(v, i, rot, qjl, qjlRows)
		RefreshTurboQuantWords(&vecs[i])
	}
	return vecs, rot, qjl
}

func TestKernelParityTurbo(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	for _, entry := range []struct {
		name string
		set  *distanceKernelSet
	}{
		{"avx2", avx2KernelSet},
		{"avx512", avx512KernelSet},
	} {
		t.Run(entry.name, func(t *testing.T) {
			ks := kernelOrSkip(t, entry.set, entry.name)
			if ks.batchTurboDot == nil && ks.turboDot == nil {
				t.Skipf("%s: no turbo kernels in set", entry.name)
			}
			count, dim := 9, 48
			vecs, _, _ := turboFixture(dim, count, rng, 32, 77)
			packed := NewTurboPackedColumns(vecs, dim)
			qVec, _, _ := turboFixture(dim, 1, rng, 32, 78)
			q := PrepareTurboQuantQuery(&qVec[0])

			start, end := 2, 7
			out := make([]float32, end-start)
			if !scoreTurboBatch(ks, q, packed, start, end, out) {
				t.Fatalf("%s: batchTurboDot rejected a legal block", entry.name)
			}
			want := make([]float32, end-start)
			if !scoreTurboBatchScalar(q, packed, start, end, want) {
				t.Fatalf("scalar turbo batch reference rejected")
			}
			for i := range want {
				if diff := absF32(out[i] - want[i]); diff > 1e-4*absMax32(1, absF32(want[i])) {
					t.Fatalf("%s: turbo batch[%d]: kernel %g scalar %g", entry.name, i, out[i], want[i])
				}
			}
			// Pair form (QJL words present): scalar oracle dotTurboQuantPrepared.
			if ks.turboDot != nil {
				for i := 0; i < count; i += 2 {
					want := dotTurboQuantPrepared(q, &vecs[i])
					if diff := absF32(ks.turboDot(q, &vecs[i]) - want); diff > 1e-4*absMax32(1, absF32(want)) {
						t.Fatalf("%s: turbo pair[%d]: kernel %g scalar %g", entry.name, i, ks.turboDot(q, &vecs[i]), want)
					}
				}
			}
			// Packed Row accessor exposes the same data (zero-copy views).
			row := packed.Row(0)
			if row.ID != 0 || len(row.Codes) != dim {
				t.Fatalf("packed Row(0) shape wrong")
			}
		})
	}
}

func TestTurboScoresAgainstReference(t *testing.T) {
	// scoreTurboBatchScalar must reproduce DotTurboQuant bit for bit —
	// the packed path is the parity oracle for the kernels and this is
	// the fixture that pins it to the public reference.
	rng := rand.New(rand.NewSource(81))
	count, dim := 6, 32
	vecs, _, _ := turboFixture(dim, count, rng, 16, 99)
	packed := NewTurboPackedColumns(vecs, dim)
	qVec, _, _ := turboFixture(dim, 1, rng, 16, 100)
	q := PrepareTurboQuantQuery(&qVec[0])
	out := make([]float32, count)
	if !scoreTurboBatchScalar(q, packed, 0, count, out) {
		t.Fatalf("scalar packed scoring rejected")
	}
	for i := 0; i < count; i++ {
		want := DotTurboQuant(&q.Vec, &vecs[i])
		if out[i] != want {
			t.Fatalf("row %d: packed %g vs DotTurboQuant %g", i, out[i], want)
		}
	}
}
