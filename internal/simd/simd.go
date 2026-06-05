package simd

import (
	"math"
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

func MatMul(a, b []float32, rowsA, colsA, colsB int) []float32 {
	result := make([]float32, rowsA*colsB)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			var sum float32
			for k := 0; k < colsA; k++ {
				sum += a[i*colsA+k] * b[k*colsB+j]
			}
			result[i*colsB+j] = sum
		}
	}
	return result
}

func MatVecMul(matrix []float32, vector []float32, rows, cols int) []float32 {
	result := make([]float32, rows)
	for i := 0; i < rows; i++ {
		var sum float32
		for j := 0; j < cols; j++ {
			sum += matrix[i*cols+j] * vector[j]
		}
		result[i] = sum
	}
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
