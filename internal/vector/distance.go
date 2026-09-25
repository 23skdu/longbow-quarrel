package vector

import (
	"math"
	"math/bits"
)

// DotFloat32 computes the dot product of two float32 vectors with 8-way loop unrolling.
func DotFloat32(a, b []float32) float32 {
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

// L2Float32 computes squared Euclidean distance of two float32 vectors with 8-way loop unrolling.
func L2Float32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		d0 := a[i+0] - b[i+0]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]

		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	sum := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// CosineFloat32 computes cosine similarity between two float32 vectors.
func CosineFloat32(a, b []float32) float32 {
	dot := DotFloat32(a, b)
	normA := DotFloat32(a, a)
	normB := DotFloat32(b, b)
	if normA <= 0 || normB <= 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(normA*normB)))
}

// DotComplex128 computes the real part of the complex Hermitian inner product: Re(<a, b>) = sum(Re(a)*Re(b) + Im(a)*Im(b)).
func DotComplex128(a, b []complex128) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s0, s1, s2, s3 float64
	i := 0
	for ; i <= n-4; i += 4 {
		s0 += real(a[i+0])*real(b[i+0]) + imag(a[i+0])*imag(b[i+0])
		s1 += real(a[i+1])*real(b[i+1]) + imag(a[i+1])*imag(b[i+1])
		s2 += real(a[i+2])*real(b[i+2]) + imag(a[i+2])*imag(b[i+2])
		s3 += real(a[i+3])*real(b[i+3]) + imag(a[i+3])*imag(b[i+3])
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += real(a[i])*real(b[i]) + imag(a[i])*imag(b[i])
	}
	return sum
}

// L2Complex128 computes the squared Euclidean distance between two complex128 vectors: sum(|a_k - b_k|^2).
func L2Complex128(a, b []complex128) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s0, s1, s2, s3 float64
	i := 0
	for ; i <= n-4; i += 4 {
		dr0 := real(a[i+0]) - real(b[i+0])
		di0 := imag(a[i+0]) - imag(b[i+0])
		dr1 := real(a[i+1]) - real(b[i+1])
		di1 := imag(a[i+1]) - imag(b[i+1])
		dr2 := real(a[i+2]) - real(b[i+2])
		di2 := imag(a[i+2]) - imag(b[i+2])
		dr3 := real(a[i+3]) - real(b[i+3])
		di3 := imag(a[i+3]) - imag(b[i+3])

		s0 += dr0*dr0 + di0*di0
		s1 += dr1*dr1 + di1*di1
		s2 += dr2*dr2 + di2*di2
		s3 += dr3*dr3 + di3*di3
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		dr := real(a[i]) - real(b[i])
		di := imag(a[i]) - imag(b[i])
		sum += dr*dr + di*di
	}
	return sum
}

// CosineComplex128 computes the normalized Hermitian cosine similarity for complex128 vectors.
func CosineComplex128(a, b []complex128) float64 {
	dot := DotComplex128(a, b)
	normA := DotComplex128(a, a)
	normB := DotComplex128(b, b)
	if normA <= 0 || normB <= 0 {
		return 0
	}
	return dot / math.Sqrt(normA*normB)
}

// DotUint8 computes the dot product of two Uint8Vectors reconstructed into continuous scale.
func DotUint8(a, b *Uint8Vector) float32 {
	n := len(a.Data)
	if len(b.Data) < n {
		n = len(b.Data)
	}
	var sumInt uint64
	var s0, s1, s2, s3 uint32
	i := 0
	for ; i <= n-4; i += 4 {
		s0 += uint32(a.Data[i+0]) * uint32(b.Data[i+0])
		s1 += uint32(a.Data[i+1]) * uint32(b.Data[i+1])
		s2 += uint32(a.Data[i+2]) * uint32(b.Data[i+2])
		s3 += uint32(a.Data[i+3]) * uint32(b.Data[i+3])
	}
	sumInt = uint64(s0 + s1 + s2 + s3)
	for ; i < n; i++ {
		sumInt += uint64(a.Data[i]) * uint64(b.Data[i])
	}
	return float32(sumInt) * (a.Scale * b.Scale)
}

// L2Uint8 computes the squared Euclidean distance between two Uint8Vectors.
func L2Uint8(a, b *Uint8Vector) float32 {
	n := len(a.Data)
	if len(b.Data) < n {
		n = len(b.Data)
	}
	scaleAvg := (a.Scale + b.Scale) * 0.5
	var sumDist uint64
	for i := 0; i < n; i++ {
		var diff uint64
		if a.Data[i] >= b.Data[i] {
			diff = uint64(a.Data[i] - b.Data[i])
		} else {
			diff = uint64(b.Data[i] - a.Data[i])
		}
		sumDist += diff * diff
	}
	return float32(sumDist) * (scaleAvg * scaleAvg)
}

// CosineUint8 computes the cosine similarity between two Uint8Vectors.
func CosineUint8(a, b *Uint8Vector) float32 {
	dot := DotUint8(a, b)
	normA := DotUint8(a, a)
	normB := DotUint8(b, b)
	if normA <= 0 || normB <= 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(normA*normB)))
}

// DotTurboQuant computes similarity using polar integer dot product + QJL 1-bit Hamming correlation.
func DotTurboQuant(a, b *TurboQuantVector) float32 {
	n := len(a.Codes)
	if len(b.Codes) < n {
		n = len(b.Codes)
	}

	var polarSum int32
	i := 0
	for ; i <= n-8; i += 8 {
		polarSum += int32(a.Codes[i+0])*int32(b.Codes[i+0]) +
			int32(a.Codes[i+1])*int32(b.Codes[i+1]) +
			int32(a.Codes[i+2])*int32(b.Codes[i+2]) +
			int32(a.Codes[i+3])*int32(b.Codes[i+3]) +
			int32(a.Codes[i+4])*int32(b.Codes[i+4]) +
			int32(a.Codes[i+5])*int32(b.Codes[i+5]) +
			int32(a.Codes[i+6])*int32(b.Codes[i+6]) +
			int32(a.Codes[i+7])*int32(b.Codes[i+7])
	}
	for ; i < n; i++ {
		polarSum += int32(a.Codes[i]) * int32(b.Codes[i])
	}

	polarScore := float32(polarSum) * (a.Scale * b.Scale)

	// QJL residual correction using bitwise XOR and Popcount
	qjlBytes := len(a.QJLBits)
	if len(b.QJLBits) < qjlBytes {
		qjlBytes = len(b.QJLBits)
	}

	if qjlBytes > 0 && a.QJLScale > 0 && b.QJLScale > 0 {
		totalBits := qjlBytes * 8
		diffBits := 0
		for bIdx := 0; bIdx < qjlBytes; bIdx++ {
			diffBits += bits.OnesCount8(a.QJLBits[bIdx] ^ b.QJLBits[bIdx])
		}
		// Hamming correlation: (same - diff) / total = (total - 2*diff) / total
		qjlCorr := float32(totalBits-2*diffBits) / float32(totalBits)
		qjlScore := qjlCorr * (a.QJLScale * b.QJLScale)
		return polarScore + qjlScore
	}

	return polarScore
}

// L2TurboQuant approximates squared L2 distance for TurboQuant vectors: ||a||^2 + ||b||^2 - 2*<a, b>.
func L2TurboQuant(a, b *TurboQuantVector) float32 {
	dot := DotTurboQuant(a, b)
	dist := (a.Norm * a.Norm) + (b.Norm * b.Norm) - (2.0 * dot)
	if dist < 0 {
		return 0
	}
	return dist
}

// CosineTurboQuant computes cosine similarity for TurboQuant vectors.
func CosineTurboQuant(a, b *TurboQuantVector) float32 {
	dot := DotTurboQuant(a, b)
	if a.Norm <= 0 || b.Norm <= 0 {
		return 0
	}
	return dot / (a.Norm * b.Norm)
}
