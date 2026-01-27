//go:build darwin && metal

package device

import (
	"math"
)

// CPU Reference Implementation for GQA attention comparison
func CPUAttentionGQA(q, k, v []float32, numHeads, kvHeads, headDim int) ([]float32, []float32) {
	// Compute QK scores
	scores := make([]float32, numHeads)
	output := make([]float32, numHeads*headDim)

	// Compute expected QK scores
	for h := 0; h < numHeads; h++ {
		kvh := h / (numHeads / kvHeads)
		score := float32(0.0)
		for i := 0; i < headDim; i++ {
			score += q[h*headDim+i] * k[kvh*headDim+i]
		}
		scores[h] = score / float32(math.Sqrt(float64(headDim)))
	}

	// Simple softmax (since seqLen=1)
	expScores := make([]float32, numHeads)
	maxScore := scores[0]
	for i := range scores {
		if scores[i] > maxScore {
			maxScore = scores[i]
		}
	}

	var expSum float32
	for i := range scores {
		expScores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
		expSum += expScores[i]
	}

	if expSum == 0 {
		expSum = 1e-6
	}

	// Compute attention weights and output
	attentionWeights := make([]float32, numHeads)
	for h := 0; h < numHeads; h++ {
		attentionWeights[h] = expScores[h] / expSum
	}

	for h := 0; h < numHeads; h++ {
		kvh := h / (numHeads / kvHeads)
		weight := attentionWeights[h]

		for i := 0; i < headDim; i++ {
			output[h*headDim+i] = weight * v[kvh*headDim+i]
		}
	}

	return output, scores
}

// CPU RMSNorm reference implementation
func CPURMSNorm(input, weight []float32, eps float32) []float32 {
	dim := len(weight)
	output := make([]float32, len(input))

	for i := 0; i < len(input)/dim; i++ {
		// Compute RMS
		sumSquares := float32(0.0)
		for j := 0; j < dim; j++ {
			val := input[i*dim+j]
			sumSquares += val * val
		}
		rms := float32(math.Sqrt(float64(sumSquares/float32(dim)) + float64(eps)))

		// Apply normalization
		for j := 0; j < dim; j++ {
			output[i*dim+j] = (input[i*dim+j] / rms) * weight[j]
		}
	}

	return output
}

// CPU MatMul reference implementation (A * B^T)
func CPUMatMul(a, b []float32, m, n, k int) []float32 {
	output := make([]float32, m*n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0.0)
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * b[j*k+l] // B is transposed
			}
			output[i*n+j] = sum
		}
	}

	return output
}

// CPU RoPE reference implementation
func CPURoPE(input []float32, pos, heads, headDim int, theta float32) []float32 {
	output := make([]float32, len(input))
	copy(output, input)

	halfDim := headDim / 2

	for h := 0; h < heads; h++ {
		headOffset := h * headDim
		for i := 0; i < halfDim; i++ {
			idx0 := headOffset + i
			idx1 := headOffset + i + halfDim

			// theta_i = pos * theta^(-2*i/headDim)
			freq := float64(pos) * math.Pow(float64(theta), -2.0*float64(i)/float64(headDim))
			cosVal := float32(math.Cos(freq))
			sinVal := float32(math.Sin(freq))

			x := output[idx0]
			y := output[idx1]

			output[idx0] = x*cosVal - y*sinVal
			output[idx1] = x*sinVal + y*cosVal
		}
	}

	return output
}

// CPU SwiGLU reference implementation
func CPUSwiGLU(gate, up []float32) []float32 {
	if len(gate) != len(up) {
		panic("gate and up must have same length")
	}

	output := make([]float32, len(gate)/2)

	for i := 0; i < len(output); i++ {
		gateVal := gate[i]
		upVal := up[i]
		// Swish activation: x * sigmoid(x)
		sigmoid := float32(1.0) / (float32(1.0) + float32(math.Exp(-float64(gateVal))))
		swish := gateVal * sigmoid
		output[i] = swish * upVal
	}

	return output
}

// CPU Q4K MatMul reference implementation (simplified version for testing)
func CPUQ4KMatMul(input []float32, q4kData []byte, m, n, k int) []float32 {
	// This is a simplified version - in reality Q4K would need dequantization
	// For testing purposes, we'll just do a basic matrix multiply with safe bounds
	output := make([]float32, m*n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0.0)
			for l := 0; l < k; l++ {
				// Calculate Q4K data position safely
				// Q4K uses block quantization, but for testing we'll use simple indexing
				dataPos := j*k + l
				if dataPos < len(q4kData) {
					// Simplified: convert byte to float32 for testing
					q4kVal := float32(q4kData[dataPos])
					sum += input[i*k+l] * q4kVal
				}
			}
			output[i*n+j] = sum
		}
	}

	return output
}
