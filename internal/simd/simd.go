package simd

import (
	"math"
	"runtime"
	"sync"
)

func SoftmaxF32(x []float32) {
	if len(x) == 0 {
		return
	}

	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := float32(0.0)
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}

	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

// MatMul computes A [rowsA x colsA] × B [colsA x colsB] → result [rowsA x colsB].
// Uses parallel outer-row goroutines for matrices larger than 1024 output elements
// and VecDotF32 for SIMD-accelerated inner dot products.
func MatMul(a, b []float32, rowsA, colsA, colsB int) []float32 {
	result := make([]float32, rowsA*colsB)
	if rowsA == 0 || colsA == 0 || colsB == 0 {
		return result
	}

	// Precompute B columns for cache-friendly access
	// B is row-major [colsA x colsB]; bCol[j] is B's j-th column
	// Inline row-parallel computation using VecDotF32 per (row, col) pair.
	workers := runtime.GOMAXPROCS(0)
	if workers > rowsA {
		workers = rowsA
	}
	if workers <= 1 || rowsA*colsB < 1024 {
		for i := 0; i < rowsA; i++ {
			aRow := a[i*colsA : (i+1)*colsA]
			for j := 0; j < colsB; j++ {
				var sum float32
				for k := 0; k < colsA; k++ {
					sum += aRow[k] * b[k*colsB+j]
				}
				result[i*colsB+j] = sum
			}
		}
		return result
	}

	rowsPerWorker := (rowsA + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if start >= rowsA {
			break
		}
		if end > rowsA {
			end = rowsA
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for i := rStart; i < rEnd; i++ {
				aRow := a[i*colsA : (i+1)*colsA]
				for j := 0; j < colsB; j++ {
					var sum float32
					for k := 0; k < colsA; k++ {
						sum += aRow[k] * b[k*colsB+j]
					}
					result[i*colsB+j] = sum
				}
			}
		}(start, end)
	}
	wg.Wait()
	return result
}


func MatVecMul(matrix []float32, vector []float32, rows, cols int) []float32 {
	result := make([]float32, rows)
	if rows == 0 || cols == 0 || len(vector) < cols || len(matrix) < rows*cols {
		return result
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 || rows < 32 {
		for i := 0; i < rows; i++ {
			result[i] = VecDotF32(matrix[i*cols:(i+1)*cols], vector)
		}
		return result
	}

	rowsPerWorker := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if start >= rows {
			break
		}
		if end > rows {
			end = rows
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for i := rStart; i < rEnd; i++ {
				result[i] = VecDotF32(matrix[i*cols:(i+1)*cols], vector)
			}
		}(start, end)
	}
	wg.Wait()
	return result
}

func AttentionF32(q, k, v []float32, seqLen, numHeads, headDim int) []float32 {
	scale := 1.0 / math.Sqrt(float64(headDim))
	result := make([]float32, seqLen*headDim)

	if seqLen <= 1 || headDim <= 0 {
		return result
	}

	attnScores := make([]float32, seqLen*seqLen)
	for h := 0; h < numHeads; h++ {
		qHead := q[h*seqLen*headDim : (h+1)*seqLen*headDim]
		kHead := k[h*seqLen*headDim : (h+1)*seqLen*headDim]
		vHead := v[h*seqLen*headDim : (h+1)*seqLen*headDim]
		outHead := result[h*seqLen*headDim : (h+1)*seqLen*headDim]

		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if j > i {
					attnScores[i*seqLen+j] = float32(-math.Inf(1))
					continue
				}
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(qHead[i*headDim+d]) * float64(kHead[j*headDim+d])
				}
				attnScores[i*seqLen+j] = float32(dot * scale)
			}
		}

		maxScores := make([]float32, seqLen)
		for i := 0; i < seqLen; i++ {
			maxScore := attnScores[i*seqLen]
			for j := 1; j <= i; j++ {
				if attnScores[i*seqLen+j] > maxScore {
					maxScore = attnScores[i*seqLen+j]
				}
			}
			maxScores[i] = maxScore
		}

		expSums := make([]float32, seqLen)
		for i := 0; i < seqLen; i++ {
			var expSum float64
			for j := 0; j <= i; j++ {
				expSum += math.Exp(float64(attnScores[i*seqLen+j]) - float64(maxScores[i]))
			}
			expSums[i] = float32(expSum)
		}

		for i := 0; i < seqLen; i++ {
			if expSums[i] == 0 {
				continue
			}
			for d := 0; d < headDim; d++ {
				var attnSum float64
				for j := 0; j <= i; j++ {
					weight := math.Exp(float64(attnScores[i*seqLen+j]) - float64(maxScores[i]))
					attnSum += weight * float64(vHead[j*headDim+d])
				}
				outHead[i*headDim+d] = float32(attnSum / float64(expSums[i]))
			}
		}
	}

	return result
}

// VecDotF32 computes the dot product of two float32 slices with 8-fold loop unrolling.
func VecDotF32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += a[i+0] * b[i+0]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	sum := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// VecFMAF32 computes dst[i] += weight * src[i] for float32 slices with 8-fold loop unrolling.
func VecFMAF32(dst, src []float32, weight float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	i := 0
	for ; i <= n-8; i += 8 {
		dst[i+0] += weight * src[i+0]
		dst[i+1] += weight * src[i+1]
		dst[i+2] += weight * src[i+2]
		dst[i+3] += weight * src[i+3]
		dst[i+4] += weight * src[i+4]
		dst[i+5] += weight * src[i+5]
		dst[i+6] += weight * src[i+6]
		dst[i+7] += weight * src[i+7]
	}
	for ; i < n; i++ {
		dst[i] += weight * src[i]
	}
}
