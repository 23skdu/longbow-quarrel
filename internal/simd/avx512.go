//go:build amd64 && cgo && avx512

package simd

/*
#cgo CFLAGS: -mavx512f -mavx512bw -mavx512dq -mavx512vl
#include <stdint.h>
#include <immintrin.h>

// AVX-512 Softmax
void softmax_avx512(float* x, int n);
void softmax_avx2(float* x, int n);

// AVX-512 SwiGLU
void swiglu_avx512(const float* gate, const float* up, float* out, int n);
void swiglu_avx2(const float* gate, const float* up, float* out, int n);

// AVX-512 FP16 conversion
void fp16_to_fp32_avx512(const uint16_t* src, float* dst, int n);
void fp32_to_fp16_avx512(const float* src, uint16_t* dst, int n);

// AVX-512 fallback to AVX2
void fp16_to_fp32_avx2(const uint16_t* src, float* dst, int n);
void fp32_to_fp16_avx2(const float* src, uint16_t* dst, int n);

// AVX-512 RMSNorm
void rmsnorm_avx512(const float* input, const float* weight, float* output, int rows, int cols, float eps);

// AVX-512 Matmul
void matmul_avx512(const float* a, const float* b, float* c, int m, int n, int k);
void matmul_avx2(const float* a, const float* b, float* c, int m, int n, int k);

// AVX-512 RoPE
void rope_avx512(float* tensor, const int* posIds, int batch, int heads, int seqLen, int headDim, float theta);
void rope_avx2(float* tensor, const int* posIds, int batch, int heads, int seqLen, int headDim, float theta);

// AVX-512 Fused Attention
void fused_attention_avx512(const float* q, const float* k, const float* v, float* output,
                          int batch, int heads, int seqLen, int kvSeqLen, int headDim, float scale);
void fused_attention_avx2(const float* q, const float* k, const float* v, float* output,
                         int batch, int heads, int seqLen, int kvSeqLen, int headDim, float scale);

// AVX-512 Fused MLP
void fused_mlp_avx512(const float* input, const float* gateWeight, const float* upWeight,
                     const float* downWeight, float* output, int batch, int dim, int hiddenDim);
void fused_mlp_avx2(const float* input, const float* gateWeight, const float* upWeight,
                   const float* downWeight, float* output, int batch, int dim, int hiddenDim);
*/
import "C"
import (
	"runtime"
	"unsafe"
)

var (
	simdInitDone bool
	simdLevel    int // 0=scalar, 1=avx2, 2=avx512
)

func initSIMD() {
	if simdInitDone {
		return
	}
	simdInitDone = true

	detectCPU()

	if hasAVX512 {
		simdLevel = 2
	} else if hasAVX2 {
		simdLevel = 1
	} else {
		simdLevel = 0
	}
}

// GetSIMDLevel returns the current SIMD level (0=scalar, 1=AVX2, 2=AVX512)
func GetSIMDLevel() int {
	if !simdInitDone {
		initSIMD()
	}
	return simdLevel
}

// Softmax selects the best available implementation
func Softmax(x []float32) {
	if !simdInitDone {
		initSIMD()
	}

	if len(x) == 0 {
		return
	}

	// Use AVX512 if available, otherwise fall back to AVX2 or scalar
	switch simdLevel {
	case 2:
		if len(x) >= 32 {
			C.softmax_avx512((*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
			return
		}
		// Fall through to AVX2 for smaller arrays
		fallthrough
	case 1:
		if len(x) >= 16 {
			C.softmax_avx2((*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
			return
		}
		fallthrough
	default:
		softmaxScalar(x)
	}
}

// SwiGLU selects the best available implementation
func SwiGLU(gate, up, out []float32) {
	if !simdInitDone {
		initSIMD()
	}

	n := len(gate)
	if n == 0 || n != len(up) || n != len(out) {
		return
	}

	switch simdLevel {
	case 2:
		if n >= 32 {
			C.swiglu_avx512(
				(*C.float)(unsafe.Pointer(&gate[0])),
				(*C.float)(unsafe.Pointer(&up[0])),
				(*C.float)(unsafe.Pointer(&out[0])),
				C.int(n),
			)
			return
		}
		fallthrough
	case 1:
		if n >= 16 {
			C.swiglu_avx2(
				(*C.float)(unsafe.Pointer(&gate[0])),
				(*C.float)(unsafe.Pointer(&up[0])),
				(*C.float)(unsafe.Pointer(&out[0])),
				C.int(n),
			)
			return
		}
		fallthrough
	default:
		swigluScalar(gate, up, out)
	}
}

// Fp16ToFp32 selects the best available implementation
func Fp16ToFp32(src []uint16, dst []float32) {
	if !simdInitDone {
		initSIMD()
	}

	n := len(src)
	if n == 0 || n != len(dst) {
		return
	}

	switch simdLevel {
	case 2:
		if n >= 32 {
			C.fp16_to_fp32_avx512(
				(*C.uint16_t)(unsafe.Pointer(&src[0])),
				(*C.float)(unsafe.Pointer(&dst[0])),
				C.int(n),
			)
			return
		}
		fallthrough
	case 1:
		if n >= 16 {
			C.fp16_to_fp32_avx2(
				(*C.uint16_t)(unsafe.Pointer(&src[0])),
				(*C.float)(unsafe.Pointer(&dst[0])),
				C.int(n),
			)
			return
		}
		fallthrough
	default:
		fp16ToFp32Scalar(src, dst)
	}
}

// Fp32ToFp16 selects the best available implementation
func Fp32ToFp16(src []float32, dst []uint16) {
	if !simdInitDone {
		initSIMD()
	}

	n := len(src)
	if n == 0 || n != len(dst) {
		return
	}

	switch simdLevel {
	case 2:
		if n >= 32 {
			C.fp32_to_fp16_avx512(
				(*C.float)(unsafe.Pointer(&src[0])),
				(*C.uint16_t)(unsafe.Pointer(&dst[0])),
				C.int(n),
			)
			return
		}
		fallthrough
	case 1:
		if n >= 16 {
			C.fp32_to_fp16_avx2(
				(*C.float)(unsafe.Pointer(&src[0])),
				(*C.uint16_t)(unsafe.Pointer(&dst[0])),
				C.int(n),
			)
			return
		}
		fallthrough
	default:
		fp32ToFp16Scalar(src, dst)
	}
}

// RMSNorm uses AVX512 with AVX2 fallback
func RMSNorm(input, weight, output []float32, rows, cols int, eps float32) {
	if !simdInitDone {
		initSIMD()
	}

	if rows*cols == 0 {
		return
	}

	switch simdLevel {
	case 2:
		if rows*cols >= 64 {
			C.rmsnorm_avx512(
				(*C.float)(unsafe.Pointer(&input[0])),
				(*C.float)(unsafe.Pointer(&weight[0])),
				(*C.float)(unsafe.Pointer(&output[0])),
				C.int(rows), C.int(cols),
				C.float(eps),
			)
			return
		}
		fallthrough
	default:
		rmsnormScalar(input, weight, output, rows, cols, eps)
	}
}

// Matmul uses AVX512 with AVX2 fallback
func Matmul(a, b, c []float32, m, n, k int) {
	if !simdInitDone {
		initSIMD()
	}

	if m*n*k == 0 {
		return
	}

	switch simdLevel {
	case 2:
		if m >= 8 && n >= 8 && k >= 16 {
			C.matmul_avx512(
				(*C.float)(unsafe.Pointer(&a[0])),
				(*C.float)(unsafe.Pointer(&b[0])),
				(*C.float)(unsafe.Pointer(&c[0])),
				C.int(m), C.int(n), C.int(k),
			)
			return
		}
		fallthrough
	case 1:
		if m >= 4 && n >= 4 && k >= 8 {
			C.matmul_avx2(
				(*C.float)(unsafe.Pointer(&a[0])),
				(*C.float)(unsafe.Pointer(&b[0])),
				(*C.float)(unsafe.Pointer(&c[0])),
				C.int(m), C.int(n), C.int(k),
			)
			return
		}
		fallthrough
	default:
		matmulScalar(a, b, c, m, n, k)
	}
}

// RoPE uses AVX512 with AVX2 fallback
func RoPE(tensor []float32, positions []int, batch, heads, seqLen, headDim int, theta float32) {
	if !simdInitDone {
		initSIMD()
	}

	if batch*heads*seqLen*headDim == 0 {
		return
	}

	posSlice := make([]C.int, len(positions))
	for i, p := range positions {
		posSlice[i] = C.int(p)
	}

	switch simdLevel {
	case 2:
		C.rope_avx512(
			(*C.float)(unsafe.Pointer(&tensor[0])),
			&posSlice[0],
			C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim),
			C.float(theta),
		)
	case 1:
		C.rope_avx2(
			(*C.float)(unsafe.Pointer(&tensor[0])),
			&posSlice[0],
			C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim),
			C.float(theta),
		)
	default:
		ropeScalar(tensor, positions, batch, heads, seqLen, headDim, theta)
	}
}

// FusedAttention uses AVX512 with AVX2 fallback
func FusedAttention(q, k, v, output []float32, batch, heads, seqLen, kvSeqLen, headDim int, scale float32) {
	if !simdInitDone {
		initSIMD()
	}

	if batch*heads*seqLen*headDim == 0 {
		return
	}

	switch simdLevel {
	case 2:
		C.fused_attention_avx512(
			(*C.float)(unsafe.Pointer(&q[0])),
			(*C.float)(unsafe.Pointer(&k[0])),
			(*C.float)(unsafe.Pointer(&v[0])),
			(*C.float)(unsafe.Pointer(&output[0])),
			C.int(batch), C.int(heads), C.int(seqLen), C.int(kvSeqLen), C.int(headDim),
			C.float(scale),
		)
	case 1:
		C.fused_attention_avx2(
			(*C.float)(unsafe.Pointer(&q[0])),
			(*C.float)(unsafe.Pointer(&k[0])),
			(*C.float)(unsafe.Pointer(&v[0])),
			(*C.float)(unsafe.Pointer(&output[0])),
			C.int(batch), C.int(heads), C.int(seqLen), C.int(kvSeqLen), C.int(headDim),
			C.float(scale),
		)
	default:
		fusedAttentionScalar(q, k, v, output, batch, heads, seqLen, kvSeqLen, headDim, scale)
	}
}

// FusedMLP uses AVX512 with AVX2 fallback
func FusedMLP(input, gateW, upW, downW, output []float32, batch, dim, hiddenDim int) {
	if !simdInitDone {
		initSIMD()
	}

	if batch*dim*hiddenDim == 0 {
		return
	}

	switch simdLevel {
	case 2:
		C.fused_mlp_avx512(
			(*C.float)(unsafe.Pointer(&input[0])),
			(*C.float)(unsafe.Pointer(&gateW[0])),
			(*C.float)(unsafe.Pointer(&upW[0])),
			(*C.float)(unsafe.Pointer(&downW[0])),
			(*C.float)(unsafe.Pointer(&output[0])),
			C.int(batch), C.int(dim), C.int(hiddenDim),
		)
	case 1:
		C.fused_mlp_avx2(
			(*C.float)(unsafe.Pointer(&input[0])),
			(*C.float)(unsafe.Pointer(&gateW[0])),
			(*C.float)(unsafe.Pointer(&upW[0])),
			(*C.float)(unsafe.Pointer(&downW[0])),
			(*C.float)(unsafe.Pointer(&output[0])),
			C.int(batch), C.int(dim), C.int(hiddenDim),
		)
	default:
		fusedMLPScalar(input, gateW, upW, downW, output, batch, dim, hiddenDim)
	}
}

// Scalar fallback implementations
func softmaxScalar(x []float32) {
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
		x[i] = fastExp(x[i] - max)
		sum += x[i]
	}

	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

func swigluScalar(gate, up, out []float32) {
	for i := 0; i < len(gate); i++ {
		g := gate[i]
		if g > 10.0 {
			g = 10.0
		}
		if g < -10.0 {
			g = -10.0
		}
		sigmoid := float32(1.0) / (float32(1.0) + fastExp(-g))
		out[i] = up[i] * g * sigmoid
	}
}

func fp16ToFp32Scalar(src []uint16, dst []float32) {
	for i := 0; i < len(src); i++ {
		dst[i] = fp16ToFp32(src[i])
	}
}

func fp32ToFp16Scalar(src []float32, dst []uint16) {
	for i := 0; i < len(src); i++ {
		dst[i] = fp32ToFp16(src[i])
	}
}

func rmsnormScalar(input, weight, output []float32, rows, cols int, eps float32) {
	for r := 0; r < rows; r++ {
		offset := r * cols

		// Compute sum of squares
		sum := float32(0.0)
		for c := 0; c < cols; c++ {
			v := input[offset+c]
			sum += v * v
		}
		sum = float32(1.0) / (float32(math.Sqrt(float64(sum)/float64(cols)) + float64(eps)))

		// Normalize and scale
		for c := 0; c < cols; c++ {
			output[offset+c] = input[offset+c] * sum * weight[c]
		}
	}
}

func matmulScalar(a, b, c []float32, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0.0)
			for l := 0; l < k; l++ {
				sum += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

func ropeScalar(tensor []float32, positions []int, batch, heads, seqLen, headDim int, theta float32) {
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for s := 0; s < seqLen; s++ {
				pos := 0
				if s < len(positions) {
					pos = positions[s]
				}
				for d := 0; d < headDim/2; d++ {
					offset := b*heads*seqLen*headDim + h*seqLen*headDim + s*headDim
					freq := float32(pos) / fastPow(theta, float32(2*d)/float32(headDim))
					cos := float32(math.Cos(float64(freq)))
					sin := float32(math.Sin(float64(freq)))

					evenIdx := offset + d
					oddIdx := offset + d + headDim/2

					even := tensor[evenIdx]
					odd := tensor[oddIdx]

					tensor[evenIdx] = even*cos - odd*sin
					tensor[oddIdx] = even*sin + odd*cos
				}
			}
		}
	}
}

func fusedAttentionScalar(q, k, v, output []float32, batch, heads, seqLen, kvSeqLen, headDim int, scale float32) {
	// Simple attention for fallback
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for s := 0; s < seqLen; s++ {
				sum := float32(0.0)
				offset := b*heads*seqLen*headDim + h*seqLen*headDim + s*headDim

				// Compute attention scores
				for kv := 0; kv < kvSeqLen; kv++ {
					kvOffset := b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim

					dot := float32(0.0)
					for d := 0; d < headDim; d++ {
						dot += q[offset+d] * k[kvOffset+d]
					}
					dot *= scale
					sum += fastExp(dot)
				}

				// Apply softmax and compute output
				for d := 0; d < headDim; d++ {
					val := float32(0.0)
					for kv := 0; kv < kvSeqLen; kv++ {
						kvOffset := b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim

						dot := float32(0.0)
						for dd := 0; dd < headDim; dd++ {
							dot += q[offset+dd] * k[kvOffset+dd]
						}
						dot *= scale

						weight := fastExp(dot) / sum
						val += weight * v[kvOffset+d]
					}
					output[offset+d] = val
				}
			}
		}
	}
}

func fusedMLPScalar(input, gateW, upW, downW, output []float32, batch, dim, hiddenDim int) {
	temp := make([]float32, hiddenDim)
	for b := 0; b < batch; b++ {
		inOffset := b * dim
		outOffset := b * hiddenDim

		// Gate + SiLU
		for h := 0; h < hiddenDim; h++ {
			g := gateW[h]
			if g > 10.0 {
				g = 10.0
			}
			if g < -10.0 {
				g = -10.0
			}
			sigmoid := float32(1.0) / (float32(1.0) + fastExp(-g))
			temp[h] = upW[inOffset+h] * g * sigmoid
		}

		// Down projection
		for d := 0; d < dim; d++ {
			sum := float32(0.0)
			for h := 0; h < hiddenDim; h++ {
				sum += temp[h] * downW[h*dim+d]
			}
			output[outOffset+d] = sum
		}
	}
}

// Helper functions
func fastExp(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func fastPow(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func fp16ToFp32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	var f32 uint32
	if exp == 0 {
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
			e := uint32(127 - 14 - shift)
			f32 = (sign << 31) | (e << 23) | m
		}
	} else if exp == 31 {
		if mant == 0 {
			f32 = (sign << 31) | 0x7F800000
		} else {
			f32 = (sign << 31) | 0x7F800000 | (mant << 13)
		}
	} else {
		newExp := exp - 15 + 127
		f32 = (sign << 31) | (newExp << 23) | (mant << 13)
	}

	return *(*float32)(unsafe.Pointer(&f32))
}

func fp32ToFp16(f float32) uint16 {
	bits := *(*uint32)(unsafe.Pointer(&f))
	sign := bits >> 31
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF

	var h uint16
	if exp == 0 {
		h = 0
	} else if exp == 255 {
		h = uint16(sign<<15) | 0x7C00 | uint16(mant>>9)
	} else {
		newExp := int(exp) - 127 + 15
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
	return h
}

// Force compiler to not optimize away the SIMD functions
var _ = runtime.GOARCH