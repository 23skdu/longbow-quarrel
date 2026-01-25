//go:build amd64 && !noasm

package simd

func init() {
	softmaxImpl = softmaxAVX2
}

func softmaxAVX2(x []float64) {
	// TODO: Implement AVX2 version
	softmaxFallback(x)
}
