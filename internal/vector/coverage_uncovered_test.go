package vector

// coverage_uncovered_test.go covers the small public surfaces no other
// test exercises per function: the ComplexSoA standalone API, the
// MMapStore/Float32Region accessors, CosineFloat32Dispatch,
// TurboPackedColumns.QueryPrepFor, and the cpuinfo test-hook reset.
// Every assertion is against the scalar reference, not just
// "it returned".

import (
	"math/rand"
	"testing"
)

func TestComplexSoAStandaloneAPI(t *testing.T) {
	if s := NewComplexSoA(-1, 4); s != nil {
		t.Fatalf("negative count must be rejected, got %v", s)
	}
	if s := NewComplexSoA(2, -3); s != nil {
		t.Fatalf("negative dim must be rejected, got %v", s)
	}

	dim := 6
	rng := rand.New(rand.NewSource(9001))
	rows := make([][]complex128, 5)
	for i := range rows {
		row := make([]complex128, dim)
		for j := range row {
			row[j] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
		rows[i] = row
	}

	soa := ComplexSoAFromInterleaved(rows, dim)
	if soa == nil {
		t.Fatalf("uniform rows rejected")
	}
	if soa.Count != len(rows) || soa.Dim != dim {
		t.Fatalf("shape %d x %d want %d x %d", soa.Count, soa.Dim, len(rows), dim)
	}

	// Round-trip is exact: the planes are float64 mirrors of the
	// complex128 components.
	rt := soa.Interleave()
	for i := range rows {
		for j := range rows[i] {
			if rt[i][j] != rows[i][j] {
				t.Fatalf("interleave round-trip row %d elem %d: %v want %v", i, j, rt[i][j], rows[i][j])
			}
		}
	}

	// SetVector overwrites slot 2; the Row view reflects it and the
	// rejections do not touch anything.
	re, im := soa.Row(2)
	for j := range re {
		re[j], im[j] = 0, 0
	}
	v := make([]complex128, dim)
	for j := range v {
		v[j] = complex(float64(j)+1.5, float64(j)-0.5)
	}
	if !soa.SetVector(2, v) {
		t.Fatalf("SetVector(valid) returned false")
	}
	for j := range re {
		if re[j] != real(v[j]) || im[j] != imag(v[j]) {
			t.Fatalf("SetVector row 2 elem %d: (%g,%g) want (%g,%g)",
				j, re[j], im[j], real(v[j]), imag(v[j]))
		}
	}
	if soa.SetVector(soa.Count, v) {
		t.Fatalf("SetVector accepted i == Count")
	}
	if soa.SetVector(-1, v) {
		t.Fatalf("SetVector accepted i == -1")
	}
	if soa.SetVector(1, v[:dim-1]) {
		t.Fatalf("SetVector accepted a short vector")
	}

	// DotRow/L2Row against the whole-form references. The scalar
	// accumulators behind both agree well inside 1e-9 on fp64.
	q := rows[0]
	qRe, qIm := splitComplexQuery(q, dim)
	if qRe == nil {
		t.Fatalf("splitComplexQuery rejected a uniform query")
	}
	for i := range rows {
		row := rows[i]
		if i == 2 {
			row = v // slot 2 was overwritten above
		}
		wantDot := DotComplex128(q, row)
		if got := soa.DotRow(qRe, qIm, i); got-wantDot > 1e-9 || wantDot-got > 1e-9 {
			t.Fatalf("DotRow row %d: %g want %g", i, got, wantDot)
		}
		wantL2 := L2Complex128(q, row)
		if got := soa.L2Row(qRe, qIm, i); got-wantL2 > 1e-9 || wantL2-got > 1e-9 {
			t.Fatalf("L2Row row %d: %g want %g", i, got, wantL2)
		}
	}

	// Ragged rows are rejected by the interleaved conversion so the
	// caller keeps its per-vector fallback.
	if ComplexSoAFromInterleaved([][]complex128{rows[0], rows[1][:dim-1]}, dim) != nil {
		t.Fatalf("ragged rows accepted")
	}
	if ComplexSoAFromInterleaved(nil, dim) != nil {
		t.Fatalf("nil rows accepted")
	}
	if ComplexSoAFromInterleaved(rows, 0) != nil {
		t.Fatalf("dim == 0 accepted")
	}
}

func TestMMapAccessors(t *testing.T) {
	var nilStore *MMapStore
	if nilStore.Path() != "" {
		t.Fatalf("nil store Path() = %q", nilStore.Path())
	}
	if nilStore.Data() != nil {
		t.Fatalf("nil store Data() not nil")
	}
	st := &MMapStore{path: "somewhere", data: []byte{1, 2, 3}}
	if st.Path() != "somewhere" {
		t.Fatalf("Path() = %q", st.Path())
	}
	if len(st.Data()) != 3 {
		t.Fatalf("Data() len = %d", len(st.Data()))
	}

	empty := Float32Region{}
	if empty.Count() != 0 {
		t.Fatalf("dim 0 region Count() = %d", empty.Count())
	}
	plane := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	reg := Float32Region{plane: plane, dim: 4}
	if reg.Count() != 3 || reg.Dim() != 4 {
		t.Fatalf("region shape Count=%d Dim=%d", reg.Count(), reg.Dim())
	}
	if &reg.Plane()[0] != &plane[0] {
		t.Fatalf("Plane() must alias the mapping")
	}
	row1 := reg.Row(1)
	for j := range row1 {
		if row1[j] != float32(5+j) {
			t.Fatalf("Row(1)[%d] = %g", j, row1[j])
		}
	}
	rows := reg.Rows()
	if len(rows) != 3 || &rows[2][0] != &plane[8] {
		t.Fatalf("Rows() lost stride: %d rows, third row start mismatch", len(rows))
	}
}

func TestCosineFloat32DispatchMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(40001))
	for _, dim := range []int{8, 24, 64} { // below and above the batch threshold
		a := parityQuery(dim, rng)
		b := parityQuery(dim, rng)
		a[0] = 1
		b[1] = 1
		got := CosineFloat32Dispatch(a, b)
		want := CosineFloat32(a, b)
		// The dispatch may take the batch kernel where the reference
		// is scalar; the documented parity budget is 1e-4.
		diff := got - want
		if diff > 1e-4 || diff < -1e-4 {
			t.Fatalf("dim %d: dispatch %g want %g", dim, got, want)
		}
	}
	if got := CosineFloat32Dispatch(make([]float32, 24), parityQuery(24, rand.New(rand.NewSource(7)))); got != 0 {
		t.Fatalf("zero-norm guard returned %g", got)
	}
}

func TestTurboQueryPrepFor(t *testing.T) {
	dim := 32
	rot := CreateIdentityMatrix(dim)
	qjl := CreateRandomQJLMatrix(16, dim, 77)
	v := parityQuery(dim, rand.New(rand.NewSource(11)))
	tq := QuantizeTurboQuant(v, 1, rot, qjl, 16)
	p := &TurboPackedColumns{}
	prep := p.QueryPrepFor(&tq)
	ref := PrepareTurboQuantQuery(&tq)
	if prep == nil || ref == nil {
		t.Fatalf("query prep nil: %v/%v", prep, ref)
	}
	if len(prep.Words) != len(ref.Words) || prep.TotalBits != ref.TotalBits {
		t.Fatalf("prep shape %d/%d want %d/%d", len(prep.Words), prep.TotalBits, len(ref.Words), ref.TotalBits)
	}
	for i := range ref.Words {
		if prep.Words[i] != ref.Words[i] {
			t.Fatalf("Words[%d] = %x want %x", i, prep.Words[i], ref.Words[i])
		}
	}
}

func TestResetVectorCPUForTestingRestoresDetection(t *testing.T) {
	resetVectorCPUForTesting(-1)
	lvl := vecLevel()
	if lvl < VecLevelScalar || lvl > VecLevelAVX512 {
		t.Fatalf("vecLevel after reset = %d", lvl)
	}
	// Leave the package state exactly as detected.
	resetVectorCPUForTesting(-1)
}
