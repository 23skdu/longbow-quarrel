package device

import (
	"math"
)

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

	output := make([]float32, len(gate))

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
