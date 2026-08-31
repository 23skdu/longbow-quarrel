package gguf

import (
	"errors"
	"math"
)

// GenerateRandomOrthogonalMatrix generates a random orthogonal matrix of size n x n
// (In a real implementation, this would use QR decomposition of a Gaussian matrix, or similar)
func GenerateRandomOrthogonalMatrix(n int) []float32 {
	// MVP: return identity matrix scaled or a pseudo-random rotation for testing
	res := make([]float32, n*n)
	for i := 0; i < n; i++ {
		res[i*n+i] = 1.0 // Simple fallback for now
	}
	return res
}

// GenerateRandomSignMatrix generates a matrix with random +1/-1 for QJL
func GenerateRandomSignMatrix(rows, cols int) []float32 {
	res := make([]float32, rows*cols)
	for i := range res {
		if i%2 == 0 {
			res[i] = 1.0
		} else {
			res[i] = -1.0
		}
	}
	return res
}

// PolarQuant applies random orthogonal rotation to gaussianize the distribution,
// then performs scalar quantization, returning the quantized data, a scale factor, and the residual error.
func PolarQuant(input []float32, rotationMatrix []float32, n int, bits int) ([]int8, float32, []float32, error) {
	if len(input) != n || len(rotationMatrix) != n*n {
		return nil, 0, nil, errors.New("size mismatch in PolarQuant")
	}

	// 1. Rotation: y = R * x
	rotated := make([]float32, n)
	maxAbs := float32(0.0)

	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rotationMatrix[i*n+j] * input[j]
		}
		rotated[i] = sum
		if abs := float32(math.Abs(float64(sum))); abs > maxAbs {
			maxAbs = abs
		}
	}

	// 2. Scalar Quantization
	quantized := make([]int8, n)
	shiftAmount := uint(bits - 1)
	maxQuantVal := float32(int(1<<shiftAmount) - 1)
	
	scale := maxAbs / maxQuantVal
	if scale == 0 {
		scale = 1.0
	}

	inverseScale := 1.0 / scale
	residual := make([]float32, n)

	for i := 0; i < n; i++ {
		q := float32(math.Round(float64(rotated[i] * inverseScale)))
		if q > maxQuantVal {
			q = maxQuantVal
		} else if q < -maxQuantVal {
			q = -maxQuantVal
		}
		quantized[i] = int8(q)

		// Calculate residual in rotated space
		resRotated := rotated[i] - (q * scale)
		residual[i] = resRotated
	}

	// Inverse rotate residual: R^T * resRotated (assuming R is orthogonal)
	finalResidual := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ { // R^T multiply
			sum += rotationMatrix[j*n+i] * residual[j]
		}
		finalResidual[i] = sum
	}

	return quantized, scale, finalResidual, nil
}

// QJLTransform applies 1-bit quantization to the residual using a random sign matrix
func QJLTransform(residual []float32, signMatrix []float32, rows, cols int) ([]int8, float32, error) {
	if len(residual) != cols || len(signMatrix) != rows*cols {
		return nil, 0, errors.New("size mismatch in QJLTransform")
	}

	// z = P * e
	projected := make([]float32, rows)
	normSq := float32(0.0)

	for i := 0; i < rows; i++ {
		var sum float32
		for j := 0; j < cols; j++ {
			sum += signMatrix[i*cols+j] * residual[j]
		}
		projected[i] = sum
		normSq += sum * sum
	}

	// Scale factor to preserve norm
	scale := float32(math.Sqrt(float64(normSq / float32(rows))))

	quantized := make([]int8, rows)
	for i := 0; i < rows; i++ {
		if projected[i] >= 0 {
			quantized[i] = 1
		} else {
			quantized[i] = -1
		}
	}

	return quantized, scale, nil
}

// QuantizeTurboQuant compresses an array into TurboQuant blocks
func QuantizeTurboQuant(input []float32, rotationMatrix []float32, qjlMatrix []float32, blockSize int, bits int) ([]byte, error) {
	numElements := len(input)
	if numElements%blockSize != 0 {
		return nil, errors.New("input size must be a multiple of blockSize")
	}
	numBlocks := numElements / blockSize
	qjlRows := 64 // Consistent with CPU device logic
	bytesPerBlock := blockSize + qjlRows + 8
	result := make([]byte, numBlocks*bytesPerBlock)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		blockData := input[start : start+blockSize]
		
		q, s, residual, err := PolarQuant(blockData, rotationMatrix, blockSize, bits)
		if err != nil {
			return nil, err
		}
		
		qj, sj, err := QJLTransform(residual, qjlMatrix, qjlRows, blockSize)
		if err != nil {
			return nil, err
		}
		
		// Fill the result block
		off := b * bytesPerBlock
		for i := 0; i < blockSize; i++ {
			result[off+i] = byte(q[i]) // #nosec G115 -- int8 to byte for quantized data
		}
		for i := 0; i < qjlRows; i++ {
			result[off+blockSize+i] = byte(qj[i]) // #nosec G115 -- int8 to byte for quantized data
		}
		
		setFloat32(result[off+blockSize+qjlRows:off+blockSize+qjlRows+4], s)
		setFloat32(result[off+blockSize+qjlRows+4:off+blockSize+qjlRows+8], sj)
	}

	return result, nil
}

// DequantizeTurboQuant decompresses an array from TurboQuant blocks
func DequantizeTurboQuant(data []byte, rotationMatrix []float32, qjlMatrix []float32, blockSize int) ([]float32, error) {
	qjlRows := 64
	bytesPerBlock := blockSize + qjlRows + 8
	numBlocks := len(data) / bytesPerBlock
	result := make([]float32, numBlocks*blockSize)

	for b := 0; b < numBlocks; b++ {
		off := b * bytesPerBlock
		q := data[off : off+blockSize]
		qj := data[off+blockSize : off+blockSize+qjlRows]
		s := getFloat32(data[off+blockSize+qjlRows : off+blockSize+qjlRows+4])
		sj := getFloat32(data[off+blockSize+qjlRows+4 : off+blockSize+qjlRows+8])
		
		for i := 0; i < blockSize; i++ {
			val := float32(int8(q[i])) * s // #nosec G115 -- byte to int8 for quantized data
			if i < qjlRows {
				val += float32(int8(qj[i])) * sj // #nosec G115 -- byte to int8 for quantized data
			}
			result[b*blockSize+i] = val
		}
	}

	return result, nil
}

func getFloat32(b []byte) float32 {
	bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return math.Float32frombits(bits)
}

func setFloat32(b []byte, f float32) {
	bits := math.Float32bits(f)
	b[0] = byte(bits)      // #nosec G115 -- byte extraction from uint32
	b[1] = byte(bits >> 8)  // #nosec G115 -- byte extraction from uint32
	b[2] = byte(bits >> 16) // #nosec G115 -- byte extraction from uint32
	b[3] = byte(bits >> 24) // #nosec G115 -- byte extraction from uint32
}
