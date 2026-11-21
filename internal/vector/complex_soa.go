package vector

// ComplexSoA stores complex128 vectors in Structure-of-Arrays (split
// real and imaginary float64 planes) layout so batch distance kernels
// can stream contiguous real and imaginary lanes without deinterleaving.
type ComplexSoA struct {
	Count int
	Dim   int
	Re    []float64
	Im    []float64
}

// NewComplexSoA allocates an empty ComplexSoA for count vectors of
// dimension dim. Negative count or dim is rejected with nil.
func NewComplexSoA(count, dim int) *ComplexSoA {
	if count < 0 || dim < 0 {
		return nil
	}
	n := count * dim
	return &ComplexSoA{
		Count: count,
		Dim:   dim,
		Re:    make([]float64, n),
		Im:    make([]float64, n),
	}
}

// ComplexSoAFromInterleaved converts uniform-length interleaved
// complex128 rows into a split ComplexSoA layout. Returns nil if rows is
// empty, dim <= 0, or any row does not match dim.
func ComplexSoAFromInterleaved(rows [][]complex128, dim int) *ComplexSoA {
	if len(rows) == 0 || dim <= 0 {
		return nil
	}
	for _, r := range rows {
		if len(r) != dim {
			return nil
		}
	}
	soa := NewComplexSoA(len(rows), dim)
	if soa == nil {
		return nil
	}
	for i, r := range rows {
		base := i * dim
		for j, c := range r {
			soa.Re[base+j] = real(c)
			soa.Im[base+j] = imag(c)
		}
	}
	return soa
}

// Interleave reconstructs interleaved [][]complex128 rows from the split
// planes.
func (s *ComplexSoA) Interleave() [][]complex128 {
	if s == nil || s.Count <= 0 || s.Dim <= 0 {
		return nil
	}
	out := make([][]complex128, s.Count)
	for i := 0; i < s.Count; i++ {
		row := make([]complex128, s.Dim)
		base := i * s.Dim
		for j := 0; j < s.Dim; j++ {
			row[j] = complex(s.Re[base+j], s.Im[base+j])
		}
		out[i] = row
	}
	return out
}

// Row returns the real and imaginary slice views for row i.
func (s *ComplexSoA) Row(i int) (re, im []float64) {
	if s == nil || i < 0 || i >= s.Count || s.Dim <= 0 {
		return nil, nil
	}
	base := i * s.Dim
	if base+s.Dim > len(s.Re) || base+s.Dim > len(s.Im) {
		return nil, nil
	}
	return s.Re[base : base+s.Dim], s.Im[base : base+s.Dim]
}

// SetVector writes v into row i of the split planes. Returns false if i
// is out of bounds or v is shorter than s.Dim.
func (s *ComplexSoA) SetVector(i int, v []complex128) bool {
	if s == nil || i < 0 || i >= s.Count || len(v) < s.Dim {
		return false
	}
	base := i * s.Dim
	if base+s.Dim > len(s.Re) || base+s.Dim > len(s.Im) {
		return false
	}
	for j := 0; j < s.Dim; j++ {
		s.Re[base+j] = real(v[j])
		s.Im[base+j] = imag(v[j])
	}
	return true
}

// DotRow computes the real Hermitian inner product between the split
// query (qRe, qIm) and row i.
func (s *ComplexSoA) DotRow(qRe, qIm []float64, i int) float64 {
	re, im := s.Row(i)
	if re == nil {
		return 0
	}
	return dotComplex128Scalars(qRe, qIm, re, im)
}

// L2Row computes the squared Euclidean distance between the split query
// (qRe, qIm) and row i.
func (s *ComplexSoA) L2Row(qRe, qIm []float64, i int) float64 {
	re, im := s.Row(i)
	if re == nil {
		return 0
	}
	return l2Complex128Scalars(qRe, qIm, re, im)
}
