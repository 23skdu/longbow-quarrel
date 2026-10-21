//go:build amd64 && cgo && !avx512

package vector

/*
#include <stdint.h>

float vec_l2_f32_avx2(const float* a, const float* b, int n);
float vec_dot_f32_avx2(const float* a, const float* b, int n);
void vec_batch_l2_f32_avx2(const float* q, const float* rows, long rowStride, int dim, int m, float* out);
void vec_batch_dot_f32_avx2(const float* q, const float* rows, long rowStride, int dim, int m, float* out);
void vec_batch_dot_c128_soa_avx2(const double* qRe, const double* qIm, const double* re, const double* im, int dim, int m, double* out);
void vec_batch_l2_c128_soa_avx2(const double* qRe, const double* qIm, const double* re, const double* im, int dim, int m, double* out);
float vec_turbo_dot_avx2(const int8_t* aCodes, const int8_t* bCodes, const uint64_t* aWords, const uint64_t* bWords, int dim, int wordsPerRow, int totalBits, float scaleA, float scaleB, float qjlScaleA, float qjlScaleB);
void vec_batch_turbo_dot_avx2(const int8_t* qCodes, const uint64_t* qWords, float qScale, float qQJLScale, const int8_t* codes, const uint64_t* words, const float* scale, const float* qjlScale, int dim, int wordsPerRow, int totalBits, int m, float* out);
void vec_mq_dot_f32_avx2(const float* qs, int numQ, const float* rows, long rowStride, int dim, int m, float* out);
void vec_mq_l2_f32_avx2(const float* qs, int numQ, const float* rows, long rowStride, int dim, int m, float* out);
void vec_mq_dot_c128_soa_avx2(const double* qsRe, const double* qsIm, int numQ, const double* re, const double* im, int dim, int m, float* out);
void vec_mq_l2_c128_soa_avx2(const double* qsRe, const double* qsIm, int numQ, const double* re, const double* im, int dim, int m, float* out);
*/
import "C"
import (
	"unsafe"
)

func init() {
	kernelsCompiledAVX2 = true

	vecDetectCPU()
	if !vecHasAVX2 {
		// The C functions would fault (announced by #pragma GCC
		// target) on a CPU without AVX2; keep the set nil so the
		// dispatcher stays on the scalar reference.
		return
	}

	avx2KernelSet = &distanceKernelSet{
		l2Float32: func(a, b []float32) float32 {
			return float32(C.vec_l2_f32_avx2(
				(*C.float)(unsafe.Pointer(&a[0])),
				(*C.float)(unsafe.Pointer(&b[0])),
				C.int(minFast(len(a), len(b))),
			))
		},
		dotFloat32: func(a, b []float32) float32 {
			return float32(C.vec_dot_f32_avx2(
				(*C.float)(unsafe.Pointer(&a[0])),
				(*C.float)(unsafe.Pointer(&b[0])),
				C.int(minFast(len(a), len(b))),
			))
		},
		batchL2Float32: func(q []float32, rows []float32, stride, count int, out []float32) bool {
			return callBatchF32AVX2(true, q, rows, stride, count, out)
		},
		batchDotFloat32: func(q []float32, rows []float32, stride, count int, out []float32) bool {
			return callBatchF32AVX2(false, q, rows, stride, count, out)
		},
		dotComplexSoA: func(qRe, qIm []float64, re, im []float64, dim int, start, end int, out []float64) bool {
			return callBatchCSoAAVX2(true, qRe, qIm, re, im, dim, start, end, out)
		},
		l2ComplexSoA: func(qRe, qIm []float64, re, im []float64, dim int, start, end int, out []float64) bool {
			return callBatchCSoAAVX2(false, qRe, qIm, re, im, dim, start, end, out)
		},
		turboDot: func(q *TurboPrep, v *TurboQuantVector) float32 {
			return callTurboDotAVX2(q, v)
		},
		batchTurboDot: func(q *TurboPrep, packed *TurboPackedColumns, start, end int, out []float32) bool {
			return callBatchTurboAVX2(q, packed, start, end, out)
		},
		mqDotFloat32: func(qs []float32, numQ int, rows []float32, stride, count int, out []float32) bool {
			return callMQF32AVX2(false, qs, numQ, rows, stride, count, out)
		},
		mqL2Float32: func(qs []float32, numQ int, rows []float32, stride, count int, out []float32) bool {
			return callMQF32AVX2(true, qs, numQ, rows, stride, count, out)
		},
		mqDotComplexSoA: func(qsRe, qsIm []float64, numQ int, re, im []float64, dim, start, end int, out []float32) bool {
			return callMQCSoAAVX2(false, qsRe, qsIm, numQ, re, im, dim, start, end, out)
		},
		mqL2ComplexSoA: func(qsRe, qsIm []float64, numQ int, re, im []float64, dim, start, end int, out []float32) bool {
			return callMQCSoAAVX2(true, qsRe, qsIm, numQ, re, im, dim, start, end, out)
		},
	}
}

// callBatchF32AVX2 batches rows [start, end) against q. The row pointer
// array is threaded straight through to C: the pointers are only
// dereferenced during the call and never retained.
func callBatchF32AVX2(l2 bool, q, rows []float32, stride, count int, out []float32) bool {
	dim := len(q)
	if dim == 0 || stride < dim || count <= 0 || len(rows) < count*stride || len(out) < count {
		return false
	}
	if l2 {
		C.vec_batch_l2_f32_avx2(
			(*C.float)(unsafe.Pointer(&q[0])),
			(*C.float)(unsafe.Pointer(&rows[0])),
			C.long(stride), C.int(dim), C.int(count),
			(*C.float)(unsafe.Pointer(&out[0])),
		)
	} else {
		C.vec_batch_dot_f32_avx2(
			(*C.float)(unsafe.Pointer(&q[0])),
			(*C.float)(unsafe.Pointer(&rows[0])),
			C.long(stride), C.int(dim), C.int(count),
			(*C.float)(unsafe.Pointer(&out[0])),
		)
	}
	return true
}

func callBatchCSoAAVX2(dot bool, qRe, qIm []float64, re, im []float64, dim int, start, end int, out []float64) bool {
	if dim <= 0 || dim > len(qRe) || dim > len(qIm) {
		return false
	}
	if start < 0 || end < start || end*dim > len(re) || end*dim > len(im) {
		return false
	}
	if len(out) < end-start {
		return false
	}
	m := end - start
	reOff := re[start*dim:]
	imOff := im[start*dim:]
	if dot {
		C.vec_batch_dot_c128_soa_avx2(
			(*C.double)(unsafe.Pointer(&qRe[0])),
			(*C.double)(unsafe.Pointer(&qIm[0])),
			(*C.double)(unsafe.Pointer(&reOff[0])),
			(*C.double)(unsafe.Pointer(&imOff[0])),
			C.int(dim), C.int(m),
			(*C.double)(unsafe.Pointer(&out[0])),
		)
	} else {
		C.vec_batch_l2_c128_soa_avx2(
			(*C.double)(unsafe.Pointer(&qRe[0])),
			(*C.double)(unsafe.Pointer(&qIm[0])),
			(*C.double)(unsafe.Pointer(&reOff[0])),
			(*C.double)(unsafe.Pointer(&imOff[0])),
			C.int(dim), C.int(m),
			(*C.double)(unsafe.Pointer(&out[0])),
		)
	}
	return true
}

func callTurboDotAVX2(q *TurboPrep, v *TurboQuantVector) float32 {
	n := minFast(len(q.Vec.Codes), len(v.Codes))
	wpr := minFast(len(q.Words), len(v.QJLWords))
	total := q.TotalBits
	if vb := len(v.QJLBits) * 8; vb < total {
		total = vb
	}
	if n == 0 {
		return 0
	}
	return float32(C.vec_turbo_dot_avx2(
		(*C.int8_t)(unsafe.Pointer(&q.Vec.Codes[0])),
		(*C.int8_t)(unsafe.Pointer(&v.Codes[0])),
		(*C.uint64_t)(qWordsPtr(q)), (*C.uint64_t)(vWordsPtr(v)),
		C.int(n), C.int(wpr), C.int(total),
		C.float(q.Vec.Scale), C.float(v.Scale),
		C.float(q.Vec.QJLScale), C.float(v.QJLScale),
	))
}

// qWordsPtr / vWordsPtr hand C the word arrays; empty arrays pass a
// pointer the kernel will not dereference (totalBits guards them).
func qWordsPtr(q *TurboPrep) unsafe.Pointer {
	if len(q.Words) == 0 {
		return unsafe.Pointer(&q.Vec) // any address; wordsPerRow == 0 keeps it unused
	}
	return unsafe.Pointer(&q.Words[0])
}

func vWordsPtr(v *TurboQuantVector) unsafe.Pointer {
	if len(v.QJLWords) == 0 {
		return unsafe.Pointer(v) // any address; wordsPerRow == 0 keeps it unused
	}
	return unsafe.Pointer(&v.QJLWords[0])
}

// callBatchTurboAVX2 scores packed TurboQuant columns [start, end) with
// the AVX2 kernel. Contiguous columns, one call, zero allocations.
func callBatchTurboAVX2(q *TurboPrep, packed *TurboPackedColumns, start, end int, out []float32) bool {
	if packed == nil {
		return false
	}
	if start < 0 || end < start || end > packed.Count {
		return false
	}
	if len(out) < end-start {
		return false
	}
	if packed.Dim > 0 && len(q.Vec.Codes) < packed.Dim {
		return false
	}
	total := q.TotalBits
	if packed.TotalBits > 0 && (total == 0 || packed.TotalBits < total) {
		total = packed.TotalBits
	}
	qWpr := len(q.Words)
	if qWpr > 0 && qWpr < packed.WordsPerRow {
		return false
	}
	codesOff := packed.Codes[start*packed.Dim:]
	scaleOff := packed.Scale[start:]
	qjlOff := packed.QJLScale[start:]
	var wordsOff []uint64
	var wordsPtr unsafe.Pointer
	if packed.WordsPerRow > 0 {
		wordsOff = packed.Words[start*packed.WordsPerRow:]
		wordsPtr = unsafe.Pointer(&wordsOff[0])
	} else {
		wordsPtr = unsafe.Pointer(packed) // unused; wordsPerRow == 0
	}
	C.vec_batch_turbo_dot_avx2(
		(*C.int8_t)(unsafe.Pointer(&q.Vec.Codes[0])),
		(*C.uint64_t)(qWordsPtr(q)),
		C.float(q.Vec.Scale), C.float(q.Vec.QJLScale),
		(*C.int8_t)(unsafe.Pointer(&codesOff[0])),
		(*C.uint64_t)(wordsPtr),
		(*C.float)(unsafe.Pointer(&scaleOff[0])),
		(*C.float)(unsafe.Pointer(&qjlOff[0])),
		C.int(packed.Dim), C.int(packed.WordsPerRow), C.int(total),
		C.int(end-start),
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	return true
}

func minFast(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// callMQF32AVX2 runs the AVX-2 multi-query shared-scan kernel. qs holds
// numQ queries laid out back to back (dim = len(qs)/numQ); out is
// [numQ][end-start].
func callMQF32AVX2(l2 bool, qs []float32, numQ int, rows []float32, stride, count int, out []float32) bool {
	if numQ <= 0 || numQ > len(qs) || len(qs)%numQ != 0 {
		return false
	}
	dim := len(qs) / numQ
	if dim == 0 || stride < dim || count <= 0 || len(rows) < count*stride || len(out) < numQ*count {
		return false
	}
	if l2 {
		C.vec_mq_l2_f32_avx2(
			(*C.float)(unsafe.Pointer(&qs[0])),
			C.int(numQ),
			(*C.float)(unsafe.Pointer(&rows[0])),
			C.long(stride), C.int(dim), C.int(count),
			(*C.float)(unsafe.Pointer(&out[0])),
		)
	} else {
		C.vec_mq_dot_f32_avx2(
			(*C.float)(unsafe.Pointer(&qs[0])),
			C.int(numQ),
			(*C.float)(unsafe.Pointer(&rows[0])),
			C.long(stride), C.int(dim), C.int(count),
			(*C.float)(unsafe.Pointer(&out[0])),
		)
	}
	return true
}

// callMQCSoAAVX2 runs the complex SoA multi-query kernel; re/im point
// at the first row of the block.
func callMQCSoAAVX2(l2 bool, qsRe, qsIm []float64, numQ int, re, im []float64, dim, start, end int, out []float32) bool {
	if dim <= 0 || numQ <= 0 {
		return false
	}
	if len(qsRe) < numQ*dim || len(qsIm) < numQ*dim {
		return false
	}
	if start < 0 || end < start || end*dim > len(re) || end*dim > len(im) {
		return false
	}
	if len(out) < numQ*(end-start) {
		return false
	}
	m := end - start
	reOff := re[start*dim:]
	imOff := im[start*dim:]
	if l2 {
		C.vec_mq_l2_c128_soa_avx2(
			(*C.double)(unsafe.Pointer(&qsRe[0])),
			(*C.double)(unsafe.Pointer(&qsIm[0])),
			C.int(numQ),
			(*C.double)(unsafe.Pointer(&reOff[0])),
			(*C.double)(unsafe.Pointer(&imOff[0])),
			C.int(dim), C.int(m),
			(*C.float)(unsafe.Pointer(&out[0])),
		)
	} else {
		C.vec_mq_dot_c128_soa_avx2(
			(*C.double)(unsafe.Pointer(&qsRe[0])),
			(*C.double)(unsafe.Pointer(&qsIm[0])),
			C.int(numQ),
			(*C.double)(unsafe.Pointer(&reOff[0])),
			(*C.double)(unsafe.Pointer(&imOff[0])),
			C.int(dim), C.int(m),
			(*C.float)(unsafe.Pointer(&out[0])),
		)
	}
	return true
}
