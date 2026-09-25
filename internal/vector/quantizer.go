package vector

import (
	"math"
	"math/rand"

	"github.com/23skdu/longbow-quarrel/internal/simd"
)

// QuantizeUint8 quantizes a float32 vector into an 8-bit affine Uint8Vector [0, 255].
func QuantizeUint8(v []float32, id int) Uint8Vector {
	n := len(v)
	if n == 0 {
		return Uint8Vector{ID: id, Data: []uint8{}, Scale: 1, Offset: 0}
	}

	minVal := v[0]
	maxVal := v[0]
	for _, x := range v {
		if x < minVal {
			minVal = x
		}
		if x > maxVal {
			maxVal = x
		}
	}

	diff := maxVal - minVal
	if diff == 0 {
		diff = 1.0
	}
	scale := diff / 255.0

	data := make([]uint8, n)
	invScale := 1.0 / scale
	for i, x := range v {
		val := (x - minVal) * invScale
		if val < 0 {
			val = 0
		} else if val > 255 {
			val = 255
		}
		data[i] = uint8(val)
	}

	return Uint8Vector{
		ID:     id,
		Data:   data,
		Scale:  scale,
		Offset: minVal,
	}
}

// Float32ToComplex128 converts a float32 vector to complex128 by setting the imaginary part to zero or using analytic split.
func Float32ToComplex128(v []float32) []complex128 {
	n := len(v)
	out := make([]complex128, n)
	for i, x := range v {
		out[i] = complex(float64(x), 0)
	}
	return out
}

// QuantizeTurboQuant encodes a float32 vector into a TurboQuantVector using PolarQuant and QJL transform.
func QuantizeTurboQuant(v []float32, id int, rotation []float32, qjlMatrix []float32, qjlRows int) TurboQuantVector {
	n := len(v)
	if n == 0 {
		return TurboQuantVector{ID: id}
	}

	norm := float32(math.Sqrt(float64(DotFloat32(v, v))))
	if norm == 0 {
		norm = 1.0
	}

	// 1. PolarQuant (4-bit default)
	codes, scale, residual := simd.PolarQuantSIMD(v, rotation, n, 4)

	// 2. QJL Transform on residual
	var qjlBytes []byte
	var qjlScale float32
	if qjlRows > 0 && len(qjlMatrix) >= qjlRows*n {
		signs, sj := simd.QJLTransformSIMD(residual, qjlMatrix, qjlRows, n)
		qjlScale = sj

		// Pack binary signs (float32 > 0 -> 1, <= 0 -> 0) into packed byte array
		numBytes := (qjlRows + 7) / 8
		qjlBytes = make([]byte, numBytes)
		for r, s := range signs {
			if s > 0 {
				qjlBytes[r/8] |= 1 << (r % 8)
			}
		}
	}

	return TurboQuantVector{
		ID:       id,
		Codes:    codes,
		Scale:    scale,
		QJLBits:  qjlBytes,
		QJLScale: qjlScale,
		Norm:     norm,
	}
}

// GenerateRandomFloat32Dataset creates N normalized float32 vectors of dimension D.
func GenerateRandomFloat32Dataset(n, d int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed)) // #nosec G404 -- deterministic fixed-seed PRNG for reproducible index/benchmark data; not security-sensitive
	dataset := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, d)
		var sumSq float64
		for j := 0; j < d; j++ {
			val := rng.Float32()*2.0 - 1.0
			vec[j] = val
			sumSq += float64(val * val)
		}
		invNorm := float32(1.0 / math.Sqrt(sumSq))
		for j := 0; j < d; j++ {
			vec[j] *= invNorm
		}
		dataset[i] = vec
	}
	return dataset
}

// CreateIdentityMatrix creates an orthogonal NxN identity matrix for testing / TurboQuant rotation.
func CreateIdentityMatrix(n int) []float32 {
	mat := make([]float32, n*n)
	for i := 0; i < n; i++ {
		mat[i*n+i] = 1.0
	}
	return mat
}

// CreateRandomQJLMatrix creates a random m x n projection matrix with elements in {-1, +1}.
func CreateRandomQJLMatrix(m, n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed)) // #nosec G404 -- deterministic fixed-seed PRNG for reproducible index/benchmark data; not security-sensitive
	mat := make([]float32, m*n)
	for i := range mat {
		if rng.Float32() > 0.5 {
			mat[i] = 1.0
		} else {
			mat[i] = -1.0
		}
	}
	return mat
}
