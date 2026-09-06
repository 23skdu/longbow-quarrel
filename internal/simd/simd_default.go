//go:build !avx512

package simd

import "math"

// RMSNorm scalar fallback
func RMSNorm(input, weight, output []float32, rows, cols int, eps float32) {
	for r := 0; r < rows; r++ {
		offset := r * cols
		sum := float32(0.0)
		for c := 0; c < cols; c++ {
			v := input[offset+c]
			sum += v * v
		}
		invNorm := float32(1.0 / math.Sqrt(float64(sum)/float64(cols) + float64(eps)))
		for c := 0; c < cols; c++ {
			output[offset+c] = input[offset+c] * invNorm * weight[c]
		}
	}
}

// Matmul scalar fallback (C = A * B, A: m×k, B: k×n)
func Matmul(a, b, c []float32, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0.0)
			for kk := 0; kk < k; kk++ {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// RoPE scalar fallback
func RoPE(tensor []float32, positions []int, batch, heads, seqLen, headDim int, theta float32) {
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for s := 0; s < seqLen; s++ {
				pos := 0
				if s < len(positions) {
					pos = positions[s]
				}
				half := headDim / 2
				for d := 0; d < half; d++ {
					offset := b*heads*seqLen*headDim + h*seqLen*headDim + s*headDim
					freq := float32(pos) / float32(math.Pow(float64(theta), float64(2*d)/float64(headDim)))
					cos := float32(math.Cos(float64(freq)))
					sin := float32(math.Sin(float64(freq)))
					ei := offset + d
					oi := offset + d + half
					ev := tensor[ei]
					od := tensor[oi]
					tensor[ei] = ev*cos - od*sin
					tensor[oi] = ev*sin + od*cos
				}
			}
		}
	}
}

// FusedAttention scalar fallback
func FusedAttention(q, k, v, output []float32, batch, heads, seqLen, kvSeqLen, headDim int, scale float32) {
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for s := 0; s < seqLen; s++ {
				offset := b*heads*seqLen*headDim + h*seqLen*headDim + s*headDim
				maxVal := float32(-math.MaxFloat32)
				for kv := 0; kv < kvSeqLen; kv++ {
					kvOff := b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim
					dot := float32(0.0)
					for d := 0; d < headDim; d++ {
						dot += q[offset+d] * k[kvOff+d]
					}
					dot *= scale
					if kv == 0 || dot > maxVal {
						maxVal = dot
					}
				}
				expSum := float32(0.0)
				for kv := 0; kv < kvSeqLen; kv++ {
					kvOff := b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim
					dot := float32(0.0)
					for d := 0; d < headDim; d++ {
						dot += q[offset+d] * k[kvOff+d]
					}
					expSum += float32(math.Exp(float64(dot*scale - maxVal)))
				}
				if expSum > 0 {
					for d := 0; d < headDim; d++ {
						sum := float32(0.0)
						for kv := 0; kv < kvSeqLen; kv++ {
							kvOff := b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim
							dot := float32(0.0)
							for dd := 0; dd < headDim; dd++ {
								dot += q[offset+dd] * k[kvOff+dd]
							}
							weight := float32(math.Exp(float64(dot*scale-maxVal))) / expSum
							sum += weight * v[kvOff+d]
						}
						output[offset+d] = sum
					}
				}
			}
		}
	}
}

// FusedMLP scalar fallback
func FusedMLP(input, gateW, upW, downW, output []float32, batch, dim, hiddenDim int) {
	for b := 0; b < batch; b++ {
		inOff := b * dim
		for d := 0; d < dim; d++ {
			sum := float32(0.0)
			for h := 0; h < hiddenDim; h++ {
				g := gateW[h]
				if g > 10.0 {
					g = 10.0
				}
				if g < -10.0 {
					g = -10.0
				}
				sig := float32(1.0) / (float32(1.0) + float32(math.Exp(float64(-g))))
				val := upW[h] * g * sig
				sum += val * downW[h*dim+d]
			}
			output[inOff+d] = sum
		}
	}
}

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
				h = uint16(sign<<15) | uint16(m>>(9+shift)) // #nosec G115 -- safe bit manipulation for float16
			} else {
				h = uint16(sign<<15) | uint16(newExp<<10) | uint16(mant>>13)
			}
		}
		dst[i] = h
	}
}
