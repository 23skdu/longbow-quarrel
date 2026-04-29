package simd

import (
	"math"
)

func Softmax(x []float64) {
	if len(x) == 0 {
		return
	}

	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := 0.0
	for i := range x {
		x[i] = math.Exp(x[i] - max)
		sum += x[i]
	}

	if sum > 0 {
		invSum := 1.0 / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

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

func SwiGLU(gate, up, out []float32) {
	n := len(gate)
	if n != len(up) || n != len(out) {
		return
	}

	for i := 0; i < n; i++ {
		g := gate[i]
		if g > 10.0 {
			g = 10.0
		}
		if g < -10.0 {
			g = -10.0
		}
		sigmoid := float32(1.0) / (float32(1.0) + float32(math.Exp(float64(-g))))
		out[i] = up[i] * g * sigmoid
	}
}

func Fp16ToFp32(src []uint16, dst []float32) {
	n := len(src)
	if n != len(dst) {
		return
	}

	for i := 0; i < n; i++ {
		h := src[i]
		sign := uint32(h>>15) & 0x1
		exp := uint32(h>>10) & 0x1F
		mant := uint32(h) & 0x3FF

		var f32 uint32
		switch exp {
		case 0:
			if mant == 0 {
				f32 = sign << 31
			} else {
				shift := uint32(0)
				m := mant
				for m < 0x400 {
					m <<= 1
					shift++
				}
				m = (m & 0x3FF) << 13
				exp = 127 - 14 - shift
				f32 = (sign << 31) | (exp << 23) | m
			}
		case 31:
			if mant == 0 {
				f32 = (sign << 31) | 0x7F800000
			} else {
				f32 = (sign << 31) | 0x7F800000 | (mant << 13)
			}
		default:
			newExp := exp - 15 + 127
			f32 = (sign << 31) | (newExp << 23) | (mant << 13)
		}
		dst[i] = math.Float32frombits(f32)
	}
}

func Fp32ToFp16(src []float32, dst []uint16) {
	n := len(src)
	if n != len(dst) {
		return
	}

	for i := 0; i < n; i++ {
		f := src[i]
		bits := math.Float32bits(f)
		sign := bits >> 31
		exp := (bits >> 23) & 0xFF
		mant := bits & 0x7FFFFF

		var h uint16
		switch exp {
		case 0:
			h = 0
		case 255:
			h = uint16(sign<<15) | 0x7C00 | uint16(mant>>9)
		default:
			newExp := exp - 127 + 15
			if newExp >= 31 {
				h = uint16(sign<<15) | 0x7C00
			} else if newExp <= 0 {
				shift := uint32(1 - newExp)
				m := mant | 0x800000
				h = uint16(sign<<15) | uint16(m>>(9+shift))
			} else {
				h = uint16(sign<<15) | uint16(newExp<<10) | uint16(mant>>13)
			}
		}
		dst[i] = h
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
