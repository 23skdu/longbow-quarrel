package device

import (
	"math"
)

// CPU Reference Implementations

func CPURMSNorm(data []float32, weight []float32, eps float32) []float32 {
	out := make([]float32, len(data))
	cols := len(weight)
	rows := len(data) / cols
	
	for i := 0; i < rows; i++ {
		rowOff := i * cols
		// Calculate sum of squares
		var sum float64 = 0
		for j := 0; j < cols; j++ {
			val := data[rowOff+j]
			sum += float64(val * val)
		}
		mean := sum / float64(cols)
		scale := float32(1.0 / math.Sqrt(mean+float64(eps)))
		
		for j := 0; j < cols; j++ {
			out[rowOff+j] = data[rowOff+j] * scale * weight[j]
		}
	}
	return out
}

// CPUMatMul computes A * B^T
// A: [M, K], B: [N, K]. Output: [M, N]
func CPUMatMul(a []float32, b []float32, M, N, K int) []float32 {
	out := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32 = 0
			for k := 0; k < K; k++ {
				sum += a[i*K+k] * b[j*K+k]
			}
			out[i*N+j] = sum
		}
	}
	return out
}

func CPURoPE(data []float32, pos, heads, headDim int, theta float32) []float32 {
	out := make([]float32, len(data))
	copy(out, data)
	// data is row-major flatten: [Batch, Heads, HeadDim] or similar?
	// Our tensor is usually [Batch * Heads * HeadDim] flat.
	// But the Kernel treats it as rows of [Heads * HeadDim].
	// Batch size? Let's assume input is 2D [Batch, Cols]
	
	cols := heads * headDim
	rows := len(data) / cols
	
	for r := 0; r < rows; r++ {
		rowOff := r * cols
		p := pos + r // Pos increments?
		// Wait, Mistral RoPE: usually applied to query/key states.
		// If input is [Batch, SeqLen, Heads, HeadDim], then pos changes per seq.
		// If input is [Batch*SeqLen, Heads*HeadDim], pos logic depends on context.
		// Our implementation assumes rows == tokens in batch/seq.
		
		for h := 0; h < heads; h++ {
			headOff := rowOff + h*headDim
			for i := 0; i < headDim/2; i++ {
				freq := float64(p) * math.Pow(float64(theta), -2.0*float64(i)/float64(headDim))
				cos := float32(math.Cos(freq))
				sin := float32(math.Sin(freq))
				
				idx1 := headOff + i
				idx2 := headOff + i + headDim/2
				
				x1 := data[idx1]
				x2 := data[idx2]
				
				out[idx1] = x1*cos - x2*sin
				out[idx2] = x1*sin + x2*cos
			}
		}
	}
	return out
}

func CPUSoftmax(data []float32, rows, cols int) []float32 {
	out := make([]float32, len(data))
	
	for i := 0; i < rows; i++ {
		rowOff := i * cols
		// Find max
		maxVal := float32(-math.MaxFloat32)
		for j := 0; j < cols; j++ {
			if data[rowOff+j] > maxVal {
				maxVal = data[rowOff+j]
			}
		}
		
		// Exp and Sum
		var sum float32 = 0
		for j := 0; j < cols; j++ {
			val := float32(math.Exp(float64(data[rowOff+j] - maxVal)))
			out[rowOff+j] = val
			sum += val
		}
		
		// Normalize
		for j := 0; j < cols; j++ {
			out[rowOff+j] /= sum
		}
	}
	return out
}

func cpuSilu(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

func CPUSwiGLU(gate, up []float32) []float32 {
	// Out = silu(gate) * up
	if len(gate) != len(up) {
		panic("SwiGLU len mismatch")
	}
	out := make([]float32, len(gate))
	for i := range gate {
		out[i] = cpuSilu(gate[i]) * up[i]
	}
	return out
}

// Q4K Constants
const BlockSize = 256
const Q4KBlockSize = 144

// CPUQ4KMatMul computes A * B^T where B is Q4K quantized byte array.
// A: [M, K] (F32), B: [N, K] (Q4K bytes). Out: [M, N]
// Input A is F32 here for simplicity, but we can verify F16 accuracy.
func CPUQ4KMatMul(a []float32, b []byte, M, N, K int) []float32 {
	if len(b) != (N*K/BlockSize)*Q4KBlockSize {
		// panic(fmt.Sprintf("Q4K size mismatch: %d vs expected %d", len(b), (N*K/BlockSize)*Q4KBlockSize))
	}
	out := make([]float32, M*N)
	
	// For each row in A
	for i := 0; i < M; i++ {
		// For each row in B (output dim)
		for j := 0; j < N; j++ {
			// Compute dot product
			// B row j starts at j * (K/256 * 144)
			bRowStart := j * (K / BlockSize) * Q4KBlockSize
			
			var sum float32 = 0
			
			// Iterate over blocks
			numBlocks := K / BlockSize
			for bIdx := 0; bIdx < numBlocks; bIdx++ {
				blockOff := bRowStart + bIdx*Q4KBlockSize
				
				// Decode Block
				// 0-2: d (F16)
				// 2-4: dmin (F16)
				// 4-16: scales (12 bytes)
				// 16-144: qs (128 bytes)
				
				// We need binary reader
				d16 := uint16(b[blockOff]) | (uint16(b[blockOff+1]) << 8)
				dm16 := uint16(b[blockOff+2]) | (uint16(b[blockOff+3]) << 8)
				
				d := Float16ToFloat32(d16)
				dmin := Float16ToFloat32(dm16)
				
				scales := b[blockOff+4 : blockOff+16]
				qs := b[blockOff+16 : blockOff+144]
				
				// Decode scales (Llama.cpp logic)
				var sc [8]uint8
				var m [8]uint8
				
				// Correct loop for scales decode to match Metal:
				for k := 0; k < 8; k++ {
					if k < 4 {
						sc[k] = scales[k] & 63
						m[k] = scales[k+4] & 63
					} else {
						sc[k] = (scales[k+4] & 0xF) | ((scales[k-4] >> 6) << 4)
						m[k] = (scales[k+4] >> 4) | ((scales[k] >> 6) << 4)
					}
				}
				
				// Dot product for block (256 elements)
				// Sub-blocks of 32
				aBlockOff := i*K + bIdx*BlockSize
				
				for sb := 0; sb < 8; sb++ {
					dVal := d * float32(sc[sb])
					mVal := dmin * float32(m[sb])
					
					// 32 weights
					for l := 0; l < 16; l++ {
						// 2 weights per byte in qs
						// qs index: sb*16 + l
						val := qs[sb*16+l]
						w0 := dVal * float32(val & 0xF) - mVal
						w1 := dVal * float32(val >> 4) - mVal
						
						// Multiply with A
						idx := aBlockOff + sb*32 + l*2
						sum += a[idx] * w0
						sum += a[idx+1] * w1
					}
				}
			}
			out[i*N+j] = sum
		}
	}
	return out
}
