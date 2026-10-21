package vector

import (
	"math/bits"
)

// distance_dispatch.go holds the runtime kernel plumbing.
//
// The per-architecture kernels live in build-tagged files that follow
// the internal/simd house style:
//
//	distance_avx.go    (amd64 && cgo && !avx512): AVX2 kernels
//	distance_avx512.go (amd64 && cgo && avx512): AVX-512 (+VPOPCNTDQ)
//	                     and AVX2 kernels, mirroring internal/simd/avx512.go
//	                     which declares both sets in its preamble.
//
// Without a matching tag the sets below stay nil and every dispatcher
// falls through to the pure-Go reference implementation in distance.go,
// which is also the parity oracle for the kernels.

// distanceKernelSet groups the native implementations of one ISA level.
// All batch functions return false when the input does not satisfy the
// kernel contract (uniform, non-nil rows with at least dim elements),
// in which case the caller must fall back to the scalar loop.
type distanceKernelSet struct {
	l2Float32  func(a, b []float32) float32
	dotFloat32 func(a, b []float32) float32

	// Batch float32 over a contiguous row block (row-major): row j of
	// the block lives at rows[j*stride : j*stride+dim]. One Go pointer
	// crosses into C (the cgo pointer rules ban arrays that contain
	// pointers), which is also the layout the mmap-backed store
	// (P0-05) streams directly.
	batchL2Float32  func(q []float32, rows []float32, stride, count int, out []float32) bool
	batchDotFloat32 func(q []float32, rows []float32, stride, count int, out []float32) bool

	// complex128 stored split (SoA). re/im are contiguous arrays of
	// stride dim: row i lives at re[i*dim:(i+1)*dim].
	dotComplexSoA func(qRe, qIm []float64, re, im []float64, dim int, start, end int, out []float64) bool
	l2ComplexSoA  func(qRe, qIm []float64, re, im []float64, dim int, start, end int, out []float64) bool

	// TurboQuant asymmetric distance. The batch form scores a packed
	// TurboPackedColumns block (built once per index); the pair form
	// scores against a raw stored vector (HNSW edge traversal).
	batchTurboDot func(q *TurboPrep, packed *TurboPackedColumns, start, end int, out []float32) bool
	turboDot      func(q *TurboPrep, v *TurboQuantVector) float32

	// Multi-query shared-scan kernels (BatchSearch): one call scores
	// numQ queries against count contiguous rows and writes out in
	// [numQ][count] order (out[q*count+r]). qs holds numQ queries of
	// dim floats laid out back to back; the buffer may be longer than
	// the block of queries in it.
	mqDotFloat32    func(qs []float32, numQ, dim int, rows []float32, stride, count int, out []float32) bool
	mqL2Float32     func(qs []float32, numQ, dim int, rows []float32, stride, count int, out []float32) bool
	mqDotComplexSoA func(qsRe, qsIm []float64, numQ int, re, im []float64, dim, start, end int, out []float32) bool
	mqL2ComplexSoA  func(qsRe, qsIm []float64, numQ int, re, im []float64, dim, start, end int, out []float32) bool
}

var (
	// Set from distance_avx.go / distance_avx512.go after init().
	avx2KernelSet   *distanceKernelSet
	avx512KernelSet *distanceKernelSet
)

// kernelSetFor returns the native kernel set for the effective level,
// or nil when the runtime level is scalar or the kernels are missing.
func kernelSetFor() *distanceKernelSet {
	switch vecLevel() {
	case VecLevelAVX512:
		if avx512KernelSet != nil {
			return avx512KernelSet
		}
		if avx2KernelSet != nil {
			return avx2KernelSet
		}
	case VecLevelAVX2:
		if avx2KernelSet != nil {
			return avx2KernelSet
		}
	}
	return nil
}

// Kernel availability flags. The *_compiled flags are set by init() in
// the build-tagged files that carry the cgo kernels; without the right
// tag the C objects do not exist and the dispatcher must not touch the
// function pointers. (They are declared in cpuinfo.go.)

// DotFloat32 / L2Float32 dispatchers live in distance.go; this file
// holds only the plumbing (kernel sets, batch scoring helpers, scalar
// oracles for the structured layouts).

// scoreFloat32Batch scores a contiguous row block against q with the
// native kernel, or with the scalar reference. Row j of the block is
// rows[j*stride : j*stride+stride]; the kernel needs len(q) == stride
// and full rows, everything else falls back scalar-per-row (the mirror
// of the public []float32 loops).
func scoreFloat32Batch(ks *distanceKernelSet, dot bool, q []float32, rows []float32, stride, count int, out []float32) {
	if ks != nil {
		if dot && ks.batchDotFloat32 != nil && ks.batchDotFloat32(q, rows, stride, count, out) {
			return
		}
		if !dot && ks.batchL2Float32 != nil && ks.batchL2Float32(q, rows, stride, count, out) {
			return
		}
	}
	dim := stride
	if len(q) < dim {
		dim = len(q)
	}
	for i := 0; i < count; i++ {
		lo := i * stride
		hi := lo + dim
		if lo < 0 || hi > len(rows) {
			out[i] = 0
			continue
		}
		row := rows[lo:hi]
		if dot {
			out[i] = dotFloat32Scalar(q, row)
		} else {
			out[i] = l2Float32Scalar(q, row)
		}
	}
}

// scoreMQFloat32runs the multi-query float32 block kernel; returns
// false when the native path is rejected so the caller scores with the
// per-query path.
//
// The dimension is a parameter, not something derived from the length of
// the query buffer: a caller that packs fewer queries than the buffer
// holds would otherwise have the kernel read the queries of the next
// block.
func scoreMQFloat32(ks *distanceKernelSet, l2 bool, qs []float32, numQ, dim int, rows []float32, stride, count int, out []float32) bool {
	if ks == nil || numQ <= 0 || dim <= 0 || count <= 0 || len(qs) < numQ*dim {
		return false
	}
	if l2 && ks.mqL2Float32 != nil {
		return ks.mqL2Float32(qs, numQ, dim, rows, stride, count, out)
	}
	if !l2 && ks.mqDotFloat32 != nil {
		return ks.mqDotFloat32(qs, numQ, dim, rows, stride, count, out)
	}
	return false
}

// scoreMQFloat32All is the scalar multi-query fallback: every query
// scores every row with the scalar reference, results out[q*count+r].
func scoreMQFloat32All(l2 bool, qs []float32, numQ, dim int, rows []float32, stride, count int, out []float32) bool {
	if numQ <= 0 || dim <= 0 || count <= 0 || len(qs) < numQ*dim || len(out) < numQ*count {
		return false
	}
	if dim != stride {
		return false
	}
	for qi := 0; qi < numQ; qi++ {
		q := qs[qi*dim : (qi+1)*dim]
		for i := 0; i < count; i++ {
			lo := i * stride
			hi := lo + stride
			if hi > len(rows) {
				out[qi*count+i] = 0
				continue
			}
			row := rows[lo:hi]
			if l2 {
				out[qi*count+i] = l2Float32Scalar(q, row)
			} else {
				out[qi*count+i] = dotFloat32Scalar(q, row)
			}
		}
	}
	return true
}

// scoreComplexSoABatch scores SoA rows [start, end) against a query held
// as split real/imag arrays.
func scoreComplexSoABatch(ks *distanceKernelSet, dot bool, qRe, qIm []float64, re, im []float64, dim int, start, end int, out []float64) {
	if ks != nil {
		if dot && ks.dotComplexSoA != nil && ks.dotComplexSoA(qRe, qIm, re, im, dim, start, end, out) {
			return
		}
		if !dot && ks.l2ComplexSoA != nil && ks.l2ComplexSoA(qRe, qIm, re, im, dim, start, end, out) {
			return
		}
	}
	for i := start; i < end; i++ {
		row := i * dim
		if row+dim <= len(re) && row+dim <= len(im) {
			rr := re[row : row+dim]
			ii := im[row : row+dim]
			if dot {
				out[i-start] = dotComplex128Scalars(qRe, qIm, rr, ii)
			} else {
				out[i-start] = l2Complex128Scalars(qRe, qIm, rr, ii)
			}
			continue
		}
		out[i-start] = 0
	}
}

// scoreTurboBatch scores stored TurboQuant vectors [start, end) against
// a prepared query with the native kernel or the scalar reference.
func scoreTurboBatch(ks *distanceKernelSet, q *TurboPrep, packed *TurboPackedColumns, start, end int, out []float32) bool {
	if ks != nil && ks.batchTurboDot != nil && packed != nil {
		if ks.batchTurboDot(q, packed, start, end, out) {
			return true
		}
	}
	return scoreTurboBatchScalar(q, packed, start, end, out)
}

// scoreTurboBatchScalar is the pure-Go scoring over packed columns. It
// reproduces the reduction of DotTurboQuant (int32 code accumulation,
// one float32 multiplies, one correlation add) so the packed path is
// the parity oracle for the kernels.
func scoreTurboBatchScalar(q *TurboPrep, packed *TurboPackedColumns, start, end int, out []float32) bool {
	if packed == nil || packed.Count < end {
		return false
	}
	wpr := packed.WordsPerRow
	dim := packed.Dim
	total := packed.TotalBits
	for i := start; i < end; i++ {
		var polarSum int32
		codes := packed.Codes[i*dim : (i+1)*dim]
		for j := 0; j < dim; j++ {
			polarSum += int32(q.Vec.Codes[j]) * int32(codes[j])
		}
		polarScore := float32(polarSum) * (q.Vec.Scale * packed.Scale[i])
		if wpr > 0 && q.TotalBits > 0 && total > 0 && q.Vec.QJLScale > 0 && packed.QJLScale[i] > 0 {
			vw := packed.Words[i*wpr : (i+1)*wpr]
			diff := 0
			for w := 0; w < wpr; w++ {
				diff += bits.OnesCount64(q.Words[w] ^ vw[w])
			}
			corr := float32(total-2*diff) / float32(total)
			polarScore += corr * (q.Vec.QJLScale * packed.QJLScale[i])
		}
		out[i-start] = polarScore
	}
	return true
}

// dotComplex128Scalars is the scalar Hermitian dot over split arrays;
// the reference shape used by scoreComplexSoABatch and the kernels.
func dotComplex128Scalars(qRe, qIm, re, im []float64) float64 {
	n := len(qRe)
	if len(re) < n {
		n = len(re)
	}
	if len(qIm) < n {
		n = len(qIm)
	}
	if len(im) < n {
		n = len(im)
	}
	var sum0, sum1, sum2, sum3 float64
	i := 0
	for ; i <= n-4; i += 4 {
		sum0 += qRe[i]*re[i] + qIm[i]*im[i]
		sum1 += qRe[i+1]*re[i+1] + qIm[i+1]*im[i+1]
		sum2 += qRe[i+2]*re[i+2] + qIm[i+2]*im[i+2]
		sum3 += qRe[i+3]*re[i+3] + qIm[i+3]*im[i+3]
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	for ; i < n; i++ {
		sum += qRe[i]*re[i] + qIm[i]*im[i]
	}
	return sum
}

// l2Complex128Scalars is the scalar squared L2 over split arrays.
func l2Complex128Scalars(qRe, qIm, re, im []float64) float64 {
	n := len(qRe)
	if len(re) < n {
		n = len(re)
	}
	if len(qIm) < n {
		n = len(qIm)
	}
	if len(im) < n {
		n = len(im)
	}
	var sum0, sum1, sum2, sum3 float64
	i := 0
	for ; i <= n-4; i += 4 {
		dr0 := qRe[i] - re[i]
		di0 := qIm[i] - im[i]
		dr1 := qRe[i+1] - re[i+1]
		di1 := qIm[i+1] - im[i+1]
		dr2 := qRe[i+2] - re[i+2]
		di2 := qIm[i+2] - im[i+2]
		dr3 := qRe[i+3] - re[i+3]
		di3 := qIm[i+3] - im[i+3]
		sum0 += dr0*dr0 + di0*di0
		sum1 += dr1*dr1 + di1*di1
		sum2 += dr2*dr2 + di2*di2
		sum3 += dr3*dr3 + di3*di3
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	for ; i < n; i++ {
		dr := qRe[i] - re[i]
		di := qIm[i] - im[i]
		sum += dr*dr + di*di
	}
	return sum
}
